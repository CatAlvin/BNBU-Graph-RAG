
from __future__ import annotations

import os
import re
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, confloat
from typing import Annotated

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough


def remove_lucene_chars(text: str) -> str:
    """Simple Lucene-unfriendly characters cleaner."""
    return re.sub(r'[+\-!(){}[\]^"~*?:\\/]|&&|\|\|', ' ', text)

def generate_full_text_query(input_text: str) -> str:
    words = [el for el in remove_lucene_chars(input_text).split() if el]
    if not words:
        return ""
    return " AND ".join([f"{word}~2" for word in words]).strip()



class Entities(BaseModel):
    person: List[str] = Field(default_factory=list)
    organization: List[str] = Field(default_factory=list)
    topic: List[str] = Field(default_factory=list)
    publication: List[str] = Field(default_factory=list)
    location: List[str] = Field(default_factory=list)
    date: List[str] = Field(default_factory=list)
    course: List[str] = Field(default_factory=list)
    assessment: List[str] = Field(default_factory=list)




class EvidenceSpan(BaseModel):
    source: str = Field(description="Which PDF/source this evidence came from.")
    page: Optional[int] = Field(default=None, description="Page number if available.")
    text: str = Field(description="Short supporting excerpt or faithful paraphrase.")
class Claim(BaseModel):
    claim: str = Field(description="A checkable statement answering part of the question.")
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    confidence: confloat(ge=0, le=1) = Field(default=0.6)


class RAGSpecialistOutput(BaseModel):
    answer_draft: str = Field(description="A concise draft answer based ONLY on given context.")
    claims: List[Claim] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    recommend_refuse: bool = False
    refusal_reason: Optional[str] = None
    used_sources: List[str] = Field(default_factory=list)




def ground_citations(output: RAGSpecialistOutput, records: List[Dict[str, Any]]) -> RAGSpecialistOutput:
    """Resolve quoted evidence to retrieved documents, never model-invented filenames."""
    normalize = lambda text: " ".join((text or "").split()).casefold()
    for claim in output.claims:
        verified = []
        for span in claim.evidence:
            quote = normalize(span.text)
            candidates = [r for r in records if quote and quote in normalize(r.get("text"))]
            locations = {(r.get("source"), r.get("page")) for r in candidates if r.get("source") not in (None, "unknown")}
            declared = (span.source, span.page)
            if declared in locations:
                verified.append(span)
            elif len(locations) == 1:
                span.source, span.page = next(iter(locations))
                verified.append(span)
        claim.evidence = verified
    output.claims = [claim for claim in output.claims if claim.evidence]
    output.used_sources = sorted({span.source for claim in output.claims for span in claim.evidence})
    if not output.claims:
        output.recommend_refuse = True
        output.refusal_reason = "No quoted evidence could be traced to the retrieved documents."
    return output


def build_graph() -> Neo4jGraph:
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri or not user or not pwd:
        raise ValueError(
            "Missing Neo4j env vars. Please set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )

    return Neo4jGraph(
        url=uri,
        username=user,
        password=pwd,
        database=db,
    )

def build_llm(
    model: Optional[str] = None,
    temperature: float = 0.0
) -> ChatOllama:
    load_dotenv()
    model = model or os.getenv("RAG_OLLAMA_CHAT_MODEL", "llama3.1:8b")
    return ChatOllama(model=model, temperature=temperature)

def build_entity_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an expert in academic/campus documents. "
             "Your task is to extract entities from the text."),
            ("human",
             "Use the given format to extract information from the following input: {question}")
        ]
    )
    return prompt | llm.with_structured_output(Entities)

def build_vector_retriever(
    *,
    embedding_model: Optional[str] = None,
    search_type: str = "hybrid",
    node_label: str = "Document",
    text_node_properties: Optional[List[str]] = None,
    embedding_node_property: str = "embedding",
    keyword_index_name: str = "documentFullTextIndex",
):
    load_dotenv()
    embedding_model = embedding_model or os.getenv(
        "RAG_OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"
    )
    text_node_properties = text_node_properties or ["text", "source", "title"]

    embedding = OllamaEmbeddings(model=embedding_model)

    # Support the installed LangChain connection signature.
    try:
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embedding,
            search_type=search_type,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
            keyword_index_name=keyword_index_name,
            retrieval_query="RETURN node.text AS text, score, node {.source, .page, .title} AS metadata",
        )
        return vector_index.as_retriever()
    except TypeError:
        # Fallback: pass explicit connection args if required by your langchain build
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        db = os.getenv("NEO4J_DATABASE", "neo4j")

        vector_index = Neo4jVector.from_existing_graph(
            embedding=embedding,
            search_type=search_type,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
            keyword_index_name=keyword_index_name,
            retrieval_query="RETURN node.text AS text, score, node {.source, .page, .title} AS metadata",
            url=uri,
            username=user,
            password=pwd,
            database=db,
        )
        return vector_index.as_retriever()



class RAGSpecialistAgent:

    def __init__(
        self,
        *,
        graph: Neo4jGraph,
        vector_retriever,
        llm: ChatOllama,
        entity_chain,
        fulltext_index_name: str = "documentFullTextIndex",
        top_k_vector: int = 5,
        limit_per_entity: int = 4,
        max_entities_each_type: int = 1,
        high_confidence_threshold: float = 0.75,
        min_high_confidence_claims: int = 1,
    ):
        self.graph = graph
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.entity_chain = entity_chain
        self.fulltext_index_name = fulltext_index_name
        self.top_k_vector = top_k_vector
        self.limit_per_entity = limit_per_entity
        self.max_entities_each_type = max_entities_each_type
        self.high_confidence_threshold = high_confidence_threshold
        self.min_high_confidence_claims = min_high_confidence_claims

        self.parser = PydanticOutputParser(pydantic_object=RAGSpecialistOutput)

        self.prompt = ChatPromptTemplate.from_template(
            """You are the RAG Specialist for a university QA system.

Your job:
1) Use ONLY the given context to draft an answer.
2) Provide checkable claims with short evidence spans.
3) If the context is insufficient, list unknowns and recommend_refuse=true.

Rules:
- Do NOT invent facts.
- Evidence text must be a short verbatim excerpt from the supplied context.
- Copy the source filename and integer page number exactly; GRAPH and VECTOR are retrieval channels, not sources.
- used_sources should list unique source file names you relied on.
- If recommend_refuse=true, provide a brief refusal_reason.

{format_instructions}

Context:
{context}

Question:
{question}
"""
        ).partial(format_instructions=self.parser.get_format_instructions())

    # ---- Graph retrieval (entity-driven fulltext) ----
    def _graph_records(self, question: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        entities = self.entity_chain.invoke({"question": question})

        seen_entities = set()

        for entity_type, values in entities.dict().items():
            if not values:
                continue

            for val in values[: self.max_entities_each_type]:
                val = (val or "").strip()
                if not val or val in seen_entities:
                    continue
                seen_entities.add(val)

                query = generate_full_text_query(val)
                if not query:
                    continue

                response = self.graph.query(
                    f"""
                    CALL db.index.fulltext.queryNodes('{self.fulltext_index_name}', $query, {{limit: $limit}})
                    YIELD node, score
                    RETURN 
                        node.title AS title,
                        node.text AS text,
                        node.source AS source,
                        node.page AS page,
                        score
                    ORDER BY score DESC
                    """,
                    {"query": query, "limit": self.limit_per_entity},
                )

                for r in response:
                    records.append(
                        {
                            "entity_type": entity_type,
                            "entity": val,
                            "title": r.get("title"),
                            "text": r.get("text"),
                            "source": r.get("source"),
                            "page": r.get("page"),
                            "score": float(r.get("score", 0.0)),
                        }
                    )

        # Light de-dup
        uniq = []
        seen = set()
        for r in records:
            key = (r.get("source"), r.get("page"), (r.get("text") or "")[:120])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        return uniq

    # ---- Vector retrieval ----
    def _vector_docs(self, question: str):
        try:
            return self.vector_retriever.invoke(question, top_k=self.top_k_vector)
        except TypeError:
            try:
                return self.vector_retriever.invoke(question)
            except Exception:
                try:
                    return self.vector_retriever.get_relevant_documents(question)
                except Exception:
                    return []


    # ---- Build unified context string ----
    def _build_context(self, question: str) -> str:
        graph_records = self._graph_records(question)
        vector_docs = self._vector_docs(question)

        self._evidence_records = list(graph_records)
        graph_part = []
        for r in graph_records:
            text = (r.get("text") or "")
            score = float(r.get("score", 0.0) or 0.0)
            graph_part.append(
                f"- [GRAPH] source={r.get('source')} page={r.get('page')} score={score:.2f} "
                f"entity={r.get('entity')} | {text[:600]}"
            )


        vector_part = []
        for d in vector_docs:
            md = getattr(d, "metadata", {}) or {}
            src = md.get("source") or md.get("source_file") or md.get("file_name") or "unknown"
            page = md.get("page")
            content = getattr(d, "page_content", "") or ""
            self._evidence_records.append({"source": src, "page": page, "text": content})
            vector_part.append(
                f"- [VECTOR] source={src} page={page} | {content[:700]}"
            )

        context = "GRAPH EVIDENCE:\n" + "\n".join(graph_part) + "\n\n" + \
                  "VECTOR EVIDENCE:\n" + "\n".join(vector_part)

        return context.strip()

    # ---- Public API ----
    def invoke(self, question: str) -> RAGSpecialistOutput:
        context = self._build_context(question)

        chain = (
            {
                "context": lambda _: context,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | self.parser
        )

        output: RAGSpecialistOutput = chain.invoke(question)

        output = ground_citations(output, self._evidence_records)

        # Auto-fill used_sources if empty
        if not output.used_sources:
            sources = set()
            for line in context.splitlines():
                if "source=" in line:
                    chunk = line.split("source=")[1]
                    src = chunk.split()[0].strip()
                    if src and src != "unknown":
                        sources.add(src)
            output.used_sources = sorted(list(sources))

        # Conservative guard: no claims + has unknowns => recommend refuse
        if len(output.claims) == 0 and len(output.unknowns) > 0:
            output.recommend_refuse = True
            if not output.refusal_reason:
                output.refusal_reason = "Insufficient evidence in RAG context."
    
        # 强规则：找不到“高置信度条目”就拒答
        high_conf_claims = [
            c for c in (output.claims or [])
            if (float(getattr(c, "confidence", 0.0) or 0.0) >= self.high_confidence_threshold)
            and (c.evidence is not None) and (len(c.evidence) > 0)
        ]

        if len(high_conf_claims) < self.min_high_confidence_claims:
            output.recommend_refuse = True
            if not output.refusal_reason:
                output.refusal_reason = (
                    f"No high-confidence claims found "
                    f"(>= {self.high_confidence_threshold})."
                )
        return output

def create_rag_specialist_agent(
    *,
    chat_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    temperature: float = 0.0,
    fulltext_index_name: str = "documentFullTextIndex",
    top_k_vector: int = 5,
    limit_per_entity: int = 4,
    max_entities_each_type: int = 1,
    high_confidence_threshold: float = 0.75,
    min_high_confidence_claims: int = 1,
):
    graph = build_graph()
    llm = build_llm(model=chat_model, temperature=temperature)
    entity_chain = build_entity_chain(llm)

    vector_retriever = build_vector_retriever(
        embedding_model=embed_model,
        keyword_index_name=fulltext_index_name,
    )

    return RAGSpecialistAgent(
        graph=graph,
        vector_retriever=vector_retriever,
        llm=llm,
        entity_chain=entity_chain,
        fulltext_index_name=fulltext_index_name,
        top_k_vector=top_k_vector,
        limit_per_entity=limit_per_entity,
        max_entities_each_type=max_entities_each_type,
        high_confidence_threshold=high_confidence_threshold,
        min_high_confidence_claims=min_high_confidence_claims,
    )
