from __future__ import annotations

import os
from typing import Optional, Dict, Any, Literal, Union, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, confloat

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from .rag_specialist_agent import (
    RAGSpecialistOutput,
    Claim,
    EvidenceSpan,
    create_rag_specialist_agent,
)

from .web_specialist_agent import (
    create_web_specialist_agent,
)

from .verifier_agent import (
    VerifierOutput,
    create_verifier_agent,
)

import json
from openai import OpenAI



RouteType = Literal["RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE", "AUTO"]

class ScoreBreakdown(BaseModel):
    routing_score: confloat(ge=0, le=1) = 0.5
    evidence_score: confloat(ge=0, le=1) = 0.5
    faithfulness: confloat(ge=0, le=1) = 0.5
    helpfulness: confloat(ge=0, le=1) = 0.5
    refusal_appropriateness: confloat(ge=0, le=1) = 0.5

class AggregatorOutput(BaseModel):
    final_answer: str
    final_refuse: bool = False
    refusal_reason: Optional[str] = None

    preferred_source: Optional[str] = None  # copied from verifier if present
    used_sources: List[str] = Field(default_factory=list)

    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    # for debugging / analysis
    issues: List[Dict[str, Any]] = Field(default_factory=list)




def build_llm(model: Optional[str] = None, temperature: float = 0.2) -> ChatOllama:
    load_dotenv()
    model = model or os.getenv("AGG_OLLAMA_CHAT_MODEL", "llama3.1:8b")
    return ChatOllama(model=model, temperature=temperature)

def use_deepseek_final_default() -> bool:
    load_dotenv()
    v = os.getenv("AGG_USE_DEEPSEEK_FINAL", "true").lower().strip()
    return v in ("1", "true", "yes", "y")

load_dotenv(".local.env")

def _get_env_var(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def ask_deepseek(question: str, context: str | None = None) -> str:
    deepseek_key = _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
    if not deepseek_key:
        return "抱歉，当前没有足够的可验证证据回答这个问题。 / I do not have enough verified evidence to answer."
    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    system_prompt = (
        "You are the final Aggregator/Judge for a multi-agent university QA system of BNBU.\n"
        "You MUST base your answer strictly on the provided context when context is given.\n"
        "Follow any STYLE_HINT instructions in the context.\n"
        "Do not invent facts. If the context indicates refusal, insufficient evidence, or uncertainty, "
        "you must produce a polite refusal.\n"
        "Your final output should be concise and in the same language as the user's question."
    )

    user_payload = {
        "question": question,
        "context": context or "",
    }
    user_prompt = (
        "User question:\n"
        f"{question}\n\n"
        "Context (may include RAG/WEB claims and verifier verdict):\n"
        f"{json.dumps(user_payload, ensure_ascii=False)}\n\n"
        "Now, based ONLY on this context, write the best possible final answer.\n"
        "If the context is insufficient or recommends refusal, refuse politely."
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip() or "抱歉，我无法基于现有证据安全回答这个问题。"
    except Exception:
        return "抱歉，我无法基于现有证据安全回答这个问题。"


def _issue_to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    # pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            return {}
    # pydantic v1
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            return {}
    return {"raw": str(x)}


def _has_nonfatal_conflict(verdict: Optional[VerifierOutput]) -> bool:
    if not verdict:
        return False
    if getattr(verdict, "recommend_refuse", False):
        return False

    conflict_keywords = {
        "conflict", "contradict", "inconsistent", "disagree",
        "冲突", "矛盾", "不一致", "互相否定", "相互冲突"
    }

    # 1) 看 summary
    summary = (getattr(verdict, "summary", "") or "").lower()
    if any(k in summary for k in conflict_keywords):
        return True

    # 2) 看 issues
    issues = getattr(verdict, "issues", None) or []
    for it in issues:
        d = _issue_to_dict(it)
        # 把 dict 里的文本字段拼起来做关键词匹配
        blob = " ".join([str(v) for v in d.values() if isinstance(v, (str, int, float))]).lower()
        if any(k in blob for k in conflict_keywords):
            return True

    return False


def _normalize_pack(
    pack: Optional[Union[RAGSpecialistOutput, Dict[str, Any]]]
) -> Optional[RAGSpecialistOutput]:
    if pack is None:
        return None
    if isinstance(pack, RAGSpecialistOutput):
        return pack
    try:
        return RAGSpecialistOutput.model_validate(pack)
    except Exception:
        return None

def _normalize_verdict(
    verdict: Optional[Union[VerifierOutput, Dict[str, Any]]]
) -> Optional[VerifierOutput]:
    if verdict is None:
        return None
    if isinstance(verdict, VerifierOutput):
        return verdict
    try:
        return VerifierOutput.model_validate(verdict)
    except Exception:
        return None

def _collect_sources(rag_pack: Optional[RAGSpecialistOutput], web_pack: Optional[RAGSpecialistOutput]) -> List[str]:
    s = []
    if rag_pack and rag_pack.used_sources:
        s.extend(rag_pack.used_sources)
    if web_pack and web_pack.used_sources:
        s.extend(web_pack.used_sources)
    # unique preserve order
    seen = set()
    out = []
    for x in s:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _claims_brief(pack: Optional[RAGSpecialistOutput], label: str, max_claims: int = 6) -> str:
    if not pack:
        return f"{label}: <NONE>"
    lines = [f"{label} answer_draft: {pack.answer_draft}"]
    for i, c in enumerate(pack.claims[:max_claims], start=1):
        evs = []
        for e in (c.evidence or [])[:2]:
            src = e.source
            pg = e.page
            txt = (e.text or "")[:120]
            evs.append(f"[{src} p={pg}] {txt}")
        ev_str = " | ".join(evs) if evs else "NO_EVIDENCE"
        lines.append(f"{label} claim_{i} (conf={c.confidence:.2f}): {c.claim} :: {ev_str}")
    if pack.unknowns:
        lines.append(f"{label} unknowns: {pack.unknowns}")
    if pack.recommend_refuse:
        lines.append(f"{label} recommends refuse: {pack.refusal_reason}")
    return "\n".join(lines)

def _compose_context_for_final(
    question: str,
    rag_pack: Optional[RAGSpecialistOutput],
    web_pack: Optional[RAGSpecialistOutput],
    verdict: Optional[VerifierOutput],
) -> str:
    parts = [f"Question: {question}"]

    parts.append(_claims_brief(rag_pack, "RAG"))
    parts.append(_claims_brief(web_pack, "WEB"))

    if verdict:
        parts.append(
            "VERIFIER:\n"
            f"- preferred_source: {verdict.preferred_source}\n"
            f"- recommend_refuse: {verdict.recommend_refuse} ({verdict.refusal_reason})\n"
            f"- rag_trust: {verdict.rag_trust}\n"
            f"- web_trust: {verdict.web_trust}\n"
            f"- summary: {verdict.summary}\n"
            f"- issues: {[i.model_dump() for i in (verdict.issues or [])]}"
        )
        # ---- Cautious / downgrade phrasing hint ----
    if _has_nonfatal_conflict(verdict):
        parts.append(
            "STYLE_HINT (IMPORTANT):\n"
            "- The verifier detected non-fatal conflicts.\n"
            "- Use cautious/downgraded wording.\n"
            "- Prefer templates like:\n"
            "  1) According to the available information that can be retrieved at present，……\n"
            "  2) There may be differences in the current publicly available information，……\n"
            "  3) It is recommended to refer to the latest announcements on the official website of BNBU or from relevant departments.\n"
            "  4) For precise details, it is recommended to contact the corresponding college/functional department for confirmation.\n"
            "- Do NOT overstate certainty."
        )

    return "\n\n".join(parts).strip()

def _light_score_heuristic(
    route: RouteType,
    rag_pack: Optional[RAGSpecialistOutput],
    web_pack: Optional[RAGSpecialistOutput],
    verdict: Optional[VerifierOutput],
    final_refuse: bool
) -> ScoreBreakdown:

    routing = 0.6 if route != "AUTO" else 0.5
    evidence = 0.5
    faith = 0.5
    helpful = 0.5
    refusal_app = 0.5

    rag_claims = len(rag_pack.claims) if rag_pack else 0
    web_claims = len(web_pack.claims) if web_pack else 0

    if rag_claims + web_claims >= 3:
        evidence = 0.75
        faith = 0.7
        helpful = 0.7
    elif rag_claims + web_claims == 0:
        evidence = 0.2
        faith = 0.3
        helpful = 0.3

    if verdict:
        # trust 作为 evidence 的微调
        evidence = max(0.1, min(0.95, (verdict.rag_trust + verdict.web_trust) / 2))
        if verdict.recommend_refuse:
            refusal_app = 0.85 if final_refuse else 0.25
        else:
            refusal_app = 0.75 if not final_refuse else 0.2

    if final_refuse:
        helpful = min(helpful, 0.5)

    return ScoreBreakdown(
        routing_score=routing,
        evidence_score=evidence,
        faithfulness=faith,
        helpfulness=helpful,
        refusal_appropriateness=refusal_app
    )



class AggregatorJudgeAgent:
    """
    Aggregator/Judge:
    - Can be used in two modes:
      A) Pure aggregation: aggregate(question, rag_pack, web_pack, verdict, route)
      B) Orchestrated answering: answer(question, route)
         which internally calls RAG/Web specialists + Verifier with safe try/except.

    Sensitivity-safe requirement:
    - If rag_agent.invoke or web_agent.invoke raises exception (often JSON parse failure),
      directly use ask_deepseek to produce a final refusal.
    """

    def __init__(
        self,
        *,
        llm: Optional[ChatOllama] = None,
        use_deepseek_final: Optional[bool] = None,
    ):
        self.llm = llm or build_llm()
        self.use_deepseek_final = use_deepseek_final_default() if use_deepseek_final is None else use_deepseek_final

        self.parser = PydanticOutputParser(pydantic_object=AggregatorOutput)

        self.prompt = ChatPromptTemplate.from_template(
            """You are the Aggregator/Judge for a multi-agent university QA system.

You receive:
- user question
- RAG specialist pack (structured)
- WEB specialist pack (structured)
- Verifier verdict (structured)
- route label

Your tasks:
1) Decide whether to refuse.
   - If verifier recommends refusal, follow it unless you can safely soften the answer. If the obtained information is found to have no connection with BNBU (or UIC) or is incorrect, please filter it out or choose not to answer.
2) Produce the final answer using claims and evidence.
3) Be concise, factual, and avoid invented details.
4) Include used_sources (merge of RAG/WEB used_sources).
5) Provide a lightweight score_breakdown.

{format_instructions}

Route: {route}

Context:
{context}

Question:
{question}
"""
        ).partial(format_instructions=self.parser.get_format_instructions())

    # ---------------------
    # A) Pure aggregation
    # ---------------------
    def aggregate(
        self,
        *,
        question: str,
        rag_pack: Optional[Union[RAGSpecialistOutput, Dict[str, Any]]] = None,
        web_pack: Optional[Union[RAGSpecialistOutput, Dict[str, Any]]] = None,
        verdict: Optional[Union[VerifierOutput, Dict[str, Any]]] = None,
        route: RouteType = "AUTO",
    ) -> AggregatorOutput:

        rag_pack_n = _normalize_pack(rag_pack)
        web_pack_n = _normalize_pack(web_pack)
        verdict_n = _normalize_verdict(verdict)

        # If route says REFUSE, obey
        if route == "REFUSE":
            text = ask_deepseek(question, context="Route indicates refusal.")
            out = AggregatorOutput(
                final_answer=text,
                final_refuse=True,
                refusal_reason="Router decided to refuse.",
                preferred_source=verdict_n.preferred_source if verdict_n else None,
                used_sources=_collect_sources(rag_pack_n, web_pack_n),
            )
            out.score_breakdown = _light_score_heuristic(route, rag_pack_n, web_pack_n, verdict_n, True)
            if verdict_n and verdict_n.issues:
                out.issues = [i.model_dump() for i in verdict_n.issues]
            return out

        # If verifier recommends refusal, default to refuse
        if verdict_n and verdict_n.recommend_refuse:
            ctx = _compose_context_for_final(question, rag_pack_n, web_pack_n, verdict_n)
            text = ask_deepseek(question, context=ctx)
            out = AggregatorOutput(
                final_answer=text,
                final_refuse=True,
                refusal_reason=verdict_n.refusal_reason or "Verifier recommends refusal.",
                preferred_source=verdict_n.preferred_source,
                used_sources=_collect_sources(rag_pack_n, web_pack_n),
            )
            out.score_breakdown = _light_score_heuristic(route, rag_pack_n, web_pack_n, verdict_n, True)
            if verdict_n.issues:
                out.issues = [i.model_dump() for i in verdict_n.issues]
            return out

        # Otherwise generate final answer
        sources = _collect_sources(rag_pack_n, web_pack_n)
        ctx = _compose_context_for_final(question, rag_pack_n, web_pack_n, verdict_n)

        # Use DeepSeek for final synthesis by default
        if self.use_deepseek_final:
            text = ask_deepseek(question, context=ctx)
            out = AggregatorOutput(
                final_answer=text,
                final_refuse=False,
                refusal_reason=None,
                preferred_source=verdict_n.preferred_source if verdict_n else None,
                used_sources=sources,
            )
            out.score_breakdown = _light_score_heuristic(route, rag_pack_n, web_pack_n, verdict_n, False)
            if verdict_n and verdict_n.issues:
                out.issues = [i.model_dump() for i in verdict_n.issues]
            return out

        # LLM (Ollama) structured generation path
        chain = (
            {
                "route": lambda _: route,
                "context": lambda _: ctx,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | self.parser
        )

        out: AggregatorOutput = chain.invoke(question)

        # Fill used_sources if missing
        if not out.used_sources:
            out.used_sources = sources
        out.preferred_source = out.preferred_source or (verdict_n.preferred_source if verdict_n else None)

        # Fill scores by heuristic if model didn't populate properly
        out.score_breakdown = _light_score_heuristic(route, rag_pack_n, web_pack_n, verdict_n, out.final_refuse)

        if verdict_n and verdict_n.issues:
            out.issues = [i.model_dump() for i in verdict_n.issues]

        return out

    def answer(
        self,
        question: str,
        route: RouteType = "AUTO",
        *,
        run_verifier: bool = True,
    ) -> AggregatorOutput:

        # Instantiate specialists on demand
        rag_agent = None
        web_agent = None
        verifier = None

        if route in ("RAG_ONLY", "HYBRID", "AUTO"):
            rag_agent = create_rag_specialist_agent()
        if route in ("WEB_ONLY", "HYBRID", "AUTO"):
            web_agent = create_web_specialist_agent()
        if run_verifier:
            verifier = create_verifier_agent()

        # 1) Safe invoke packs
        rag_pack = None
        web_pack = None

        try:
            if rag_agent:
                rag_pack = rag_agent.invoke(question)
        except Exception:
            text = ask_deepseek(question, context="RAG specialist failed to produce structured output. Provide a polite refusal.")
            out = AggregatorOutput(
                final_answer=text,
                final_refuse=True,
                refusal_reason="RAG specialist output parsing failed (possible sensitive content).",
                preferred_source=None,
                used_sources=[],
            )
            out.score_breakdown = _light_score_heuristic(route, None, None, None, True)
            return out

        try:
            if web_agent:
                web_pack = web_agent.invoke(question)
        except Exception:
            text = ask_deepseek(question, context="Web specialist failed to produce structured output. Provide a polite refusal.")
            out = AggregatorOutput(
                final_answer=text,
                final_refuse=True,
                refusal_reason="Web specialist output parsing failed (possible sensitive content).",
                preferred_source=None,
                used_sources=[],
            )
            out.score_breakdown = _light_score_heuristic(route, None, None, None, True)
            return out

        # 2) Verifier (also safe)
        verdict = None
        if verifier:
            try:
                verdict = verifier.invoke(question, rag_pack=rag_pack, web_pack=web_pack)
            except Exception:
                verdict = None  # verifier failure shouldn't block final

        # 3) Aggregate
        return self.aggregate(
            question=question,
            rag_pack=rag_pack,
            web_pack=web_pack,
            verdict=verdict,
            route=route,
        )


def create_aggregator_judge_agent(
    *,
    use_deepseek_final: Optional[bool] = None,
    temperature: float = 0.2,
) -> AggregatorJudgeAgent:

    llm = build_llm(temperature=temperature)
    return AggregatorJudgeAgent(
        llm=llm,
        use_deepseek_final=use_deepseek_final,
    )
