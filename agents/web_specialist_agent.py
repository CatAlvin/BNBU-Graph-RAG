from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough


from .rag_specialist_agent import (
    Entities,
    EvidenceSpan,
    Claim,
    RAGSpecialistOutput,
)


from web_search import web_search, tavily_extract_top


def build_llm(
    model: Optional[str] = None,
    temperature: float = 0.0
) -> ChatOllama:
    load_dotenv()
    model = model or os.getenv("WEB_OLLAMA_CHAT_MODEL", "llama3.1:8b")
    return ChatOllama(model=model, temperature=temperature)

def build_entity_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an expert in academic/campus queries. "
             "Your task is to extract key entities from the question."),
            ("human",
             "Use the given format to extract information from the following input: {question}")
        ]
    )
    return prompt | llm.with_structured_output(Entities)



class WebSpecialistAgent:
    """
    Web Specialist:
    - Searches the web using web_search(...)
    - Extracts top results with tavily_extract_top(...)
    - Produces structured evidence output aligned with RAG Specialist
    """

    def __init__(
        self,
        *,
        llm: ChatOllama,
        entity_chain,
        max_results: int = 5,
        search_depth: str = "basic",
        use_deepseek: bool = False,
        top_k_evidence: int = 3,
    ):
        self.llm = llm
        self.entity_chain = entity_chain

        self.max_results = max_results
        self.search_depth = search_depth
        self.use_deepseek = use_deepseek
        self.top_k_evidence = top_k_evidence

        self.parser = PydanticOutputParser(pydantic_object=RAGSpecialistOutput)

        self.prompt = ChatPromptTemplate.from_template(
            """You are the Web Specialist for a university QA system.

Your job:
1) Use ONLY the given web context to draft an answer.
2) Provide checkable claims with short evidence spans (source should be URL).
3) If the context is insufficient, conflicting or just not about BNBU(UIC), list unknowns and recommend_refuse=true.

Rules:
- Do NOT invent facts.
- Evidence text should be short and directly supportive.
- used_sources should list unique URLs you relied on.
- If recommend_refuse=true, provide a brief refusal_reason.

{format_instructions}

Web Context:
{context}

Question:
{question}
"""
        ).partial(format_instructions=self.parser.get_format_instructions())

    # ---- Query expansion (lightweight, entity-aware) ----
    def _build_queries(self, question: str) -> List[str]:
        queries = [question]

        try:
            ents: Entities = self.entity_chain.invoke({"question": question})
            # Pick one salient entity as a lightweight expansion
            for _, values in ents.dict().items():
                if values:
                    v = values[0].strip()
                    if v and v not in question:
                        queries.append(f"{v} {question}")
                        break
        except Exception:
            pass

        # Deduplicate
        uniq = []
        seen = set()
        for q in queries:
            qn = q.strip()
            if not qn or qn in seen:
                continue
            seen.add(qn)
            uniq.append(qn)

        return uniq[:2]

    # ---- Web search ----
    def _search(self, q: str) -> Dict[str, Any]:
        return web_search(
            q,
            require_ans=True,
            require_raw=False,
            max_results=self.max_results,
            search_depth=self.search_depth,
            use_deepseek=self.use_deepseek
        )

    # ---- Build web context ----
    def _build_context(self, question: str) -> str:
        queries = self._build_queries(question)
        all_blocks: List[str] = []

        for qi, q in enumerate(queries):
            res = self._search(q)
            answer = res.get("answer")

            all_blocks.append(f"=== QUERY {qi+1} ===\n{q}")

            if answer:
                all_blocks.append(f"[SEARCH_ANSWER]\n{answer}")

            # Extract top-k scored results
            # tavily_extract_top(res, k=1/2/3...) per your example
            for k in range(1, self.top_k_evidence + 1):
                try:
                    url, title, content, score = tavily_extract_top(res, k=k)
                except Exception:
                    continue

                snippet = (content or "").strip()
                if len(snippet) > 800:
                    snippet = snippet[:800]

                all_blocks.append(
                    f"[TOP_{k}] score={score}\n"
                    f"title={title}\n"
                    f"url={url}\n"
                    f"content={snippet}"
                )

        return "\n\n".join(all_blocks).strip()

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

        # Auto-fill used_sources if empty:
        # Collect URLs from the raw context lines
        if not output.used_sources:
            urls = set()
            for line in context.splitlines():
                line = line.strip()
                if line.startswith("url="):
                    u = line.replace("url=", "").strip()
                    if u:
                        urls.add(u)
            output.used_sources = sorted(list(urls))

        # Conservative guard:
        if len(output.claims) == 0 and len(output.unknowns) > 0:
            output.recommend_refuse = True
            if not output.refusal_reason:
                output.refusal_reason = "Insufficient evidence from web search context."

        return output


def create_web_specialist_agent(
    *,
    chat_model: Optional[str] = None,
    temperature: float = 0.0,
    max_results: Optional[int] = None,
    search_depth: Optional[str] = None,
    use_deepseek: Optional[bool] = None,
    top_k_evidence: int = 3,
):
    load_dotenv()

    llm = build_llm(model=chat_model, temperature=temperature)
    entity_chain = build_entity_chain(llm)

    # Allow env override
    if max_results is None:
        max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    if search_depth is None:
        search_depth = os.getenv("WEB_SEARCH_DEPTH", "basic")

    if use_deepseek is None:
        ud = os.getenv("WEB_USE_DEEPSEEK", "false").lower().strip()
        use_deepseek = ud in ("1", "true", "yes", "y")

    return WebSpecialistAgent(
        llm=llm,
        entity_chain=entity_chain,
        max_results=max_results,
        search_depth=search_depth,
        use_deepseek=use_deepseek,
        top_k_evidence=top_k_evidence,
    )
