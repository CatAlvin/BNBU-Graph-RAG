from dotenv import load_dotenv
from neo4j.exceptions import ClientError
from sentence_transformers import CrossEncoder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

from utils import GraphRetriever, Entities
from config import CFG

# load the env variables
load_dotenv()


def _sanitize_for_lucene(q: str) -> str:
    if not q:
        return q
    cleaned = remove_lucene_chars(q)
    return " ".join(cleaned.split())


def _min_max_norm(values):
    values = list(values)
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


class WeightedFusionRetriever:
    def __init__(
        self,
        embedding_model,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha_retrieval: float = 0.4,
        k: int = 5,
        rerank_top_n: int = 20,
    ):
        self.embedding_model = embedding_model
        self.alpha = alpha_retrieval
        self.k = k
        self.rerank_top_n = rerank_top_n

        # Neo4j hybrid & vector indexes
        self.hybrid_store = Neo4jVector.from_existing_graph(
            embedding=self.embedding_model,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text", "source", "title"],
            embedding_node_property="embedding",
        )
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embedding_model,
            search_type="vector",
            node_label="Document",
            text_node_properties=["text", "source", "title"],
            embedding_node_property="embedding",
        )

        # CrossEncoder reranker
        self.reranker = CrossEncoder(reranker_model)

    def _similarity_search_hybrid_with_score_safe(self, query: str):
        try:
            docs_scores = self.hybrid_store.similarity_search_with_score(
                query, k=self.rerank_top_n
            )
            return docs_scores, "hybrid_raw"
        except ClientError:
            pass

        safe_q = _sanitize_for_lucene(query)
        if safe_q:
            try:
                docs_scores = self.hybrid_store.similarity_search_with_score(
                    safe_q, k=self.rerank_top_n
                )
                return docs_scores, "hybrid_sanitized"
            except ClientError:
                pass

        # fallback
        docs_scores = self.vector_store.similarity_search_with_score(
            query, k=self.rerank_top_n
        )
        return docs_scores, "vector_fallback"

    def __call__(self, query: str):
        # top-k Document 
        docs_scores, mode = self._similarity_search_hybrid_with_score_safe(query)
        if not docs_scores:
            return []

        docs, retr_scores = zip(*docs_scores)
        docs = list(docs)
        retr_scores = list(retr_scores)
        docs_text = [d.page_content for d in docs]

        # CrossEncoder 
        pairs = [(query, t) for t in docs_text]
        ce_scores = self.reranker.predict(pairs)

        retr_norm = _min_max_norm(retr_scores)
        ce_norm = _min_max_norm(list(ce_scores))

        alpha = self.alpha
        fusion_scores = [
            alpha * r + (1.0 - alpha) * c
            for r, c in zip(retr_norm, ce_norm)
        ]

        order = sorted(
            range(len(docs)),
            key=lambda i: fusion_scores[i],
            reverse=True,
        )
        top_idx = order[: self.k]
        top_docs = [docs[i] for i in top_idx]
        return top_docs


def build_chain():
    entity_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in scientific papers. Your task is to extract entities from the text. "
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}"
            )
        ]
    )
    entity_chain = entity_prompt | CFG.llm.with_structured_output(Entities)

    graph_retriever = GraphRetriever(
        graph=CFG.graph,
        entity_chain=entity_chain,
        embedding_model=CFG.embedding_model,
    )

    fusion_retriever = WeightedFusionRetriever(
        embedding_model=CFG.embedding_model,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha_retrieval=0.3,  # 0.3/0.4/0.5
        k=5,
        rerank_top_n=20,
    )

    qa_template = """Answer the question based on the provided data.

Graph Data:
{graph_data}

Retrieved Documents:
{retrieved_docs}

Question: {question}

Use the context to answer the question as accurately as possible. Be concise and to the point.

Answer:
"""
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    def build_context(question: str) -> dict:
        graph_data = graph_retriever._graph_retriever(question=question)

        fusion_docs = fusion_retriever(query=question)
        docs_text = [d.page_content for d in fusion_docs]
        retrieved_docs = "\n\n".join(docs_text) if docs_text else ""

        return {
            "graph_data": graph_data,
            "retrieved_docs": retrieved_docs,
        }

    chain = (
        {
            "graph_data": lambda q: build_context(q)["graph_data"],
            "retrieved_docs": lambda q: build_context(q)["retrieved_docs"],
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | CFG.llm
        | StrOutputParser()
    )

    return chain
