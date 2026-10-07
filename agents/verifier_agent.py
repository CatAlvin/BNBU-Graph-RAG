from __future__ import annotations

import os
import re
from typing import List, Optional, Dict, Any, Literal, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field, confloat

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from .rag_specialist_agent import (
    EvidenceSpan,
    Claim,
    RAGSpecialistOutput,
)


# =========================================================
# 1) Verifier output schema
# =========================================================

IssueType = Literal[
    "NO_EVIDENCE",
    "WEAK_EVIDENCE",
    "POTENTIAL_CONFLICT",
    "UNRESOLVED_CONFLICT",
    "OVERCONFIDENT",
    "OUT_OF_SCOPE",
    "STALE_RISK",
]

PreferredSource = Literal["RAG", "WEB", "HYBRID", "NONE"]

class VerifierIssue(BaseModel):
    issue_type: IssueType
    description: str
    severity: int = Field(ge=1, le=5, default=3)
    related_claims: List[str] = Field(default_factory=list)
    related_sources: List[str] = Field(default_factory=list)

class VerifierOutput(BaseModel):
    preferred_source: PreferredSource = "HYBRID"
    recommend_refuse: bool = False
    refusal_reason: Optional[str] = None

    issues: List[VerifierIssue] = Field(default_factory=list)

    # 给 Aggregator 做“权重融合”用
    rag_trust: confloat(ge=0, le=1) = 0.6
    web_trust: confloat(ge=0, le=1) = 0.6

    # 简短给人看的总结
    summary: str = ""


def build_llm(model: Optional[str] = None, temperature: float = 0.0) -> ChatOllama:
    load_dotenv()
    model = model or os.getenv("VERIFIER_OLLAMA_CHAT_MODEL", "llama3.1:8b")
    return ChatOllama(model=model, temperature=temperature)


def _extract_urls(pack: Optional[RAGSpecialistOutput]) -> List[str]:
    if not pack:
        return []
    return list(dict.fromkeys(pack.used_sources or []))

def _claim_has_evidence(c: Claim) -> bool:
    if not c.evidence:
        return False
    for e in c.evidence:
        if e and (e.text or "").strip():
            return True
    return False

def _rule_issues_from_pack(
    pack: Optional[RAGSpecialistOutput],
    label: str
) -> List[VerifierIssue]:
    issues: List[VerifierIssue] = []
    if not pack:
        return issues

    # 1) claim-level evidence check
    for c in pack.claims:
        if not _claim_has_evidence(c):
            issues.append(VerifierIssue(
                issue_type="NO_EVIDENCE",
                description=f"{label} claim lacks explicit evidence.",
                severity=4,
                related_claims=[c.claim],
                related_sources=_extract_urls(pack),
            ))
        elif c.confidence < 0.4:
            issues.append(VerifierIssue(
                issue_type="WEAK_EVIDENCE",
                description=f"{label} claim confidence is low.",
                severity=3,
                related_claims=[c.claim],
                related_sources=_extract_urls(pack),
            ))

        # 2) overconfident style guard
        if c.confidence >= 0.9 and len((c.evidence or [])) <= 1:
            issues.append(VerifierIssue(
                issue_type="OVERCONFIDENT",
                description=f"{label} claim seems overconfident with thin evidence.",
                severity=2,
                related_claims=[c.claim],
                related_sources=_extract_urls(pack),
            ))

    # 3) pack-level refusal signal
    if pack.recommend_refuse:
        issues.append(VerifierIssue(
            issue_type="OUT_OF_SCOPE",
            description=f"{label} specialist recommends refusal.",
            severity=4,
            related_claims=[],
            related_sources=_extract_urls(pack),
        ))

    return issues


def _pack_to_brief(pack: Optional[RAGSpecialistOutput], label: str) -> str:
    if not pack:
        return f"{label}: <NONE>"

    lines = [f"{label}:"]
    if pack.answer_draft:
        lines.append(f"- answer_draft: {pack.answer_draft}")

    for i, c in enumerate(pack.claims[:8], start=1):
        ev = c.evidence[:2] if c.evidence else []
        ev_texts = []
        for e in ev:
            src = e.source
            pg = e.page
            t = (e.text or "")[:160]
            ev_texts.append(f"[{src} p={pg}] {t}")
        ev_join = " | ".join(ev_texts) if ev_texts else "NO_EVIDENCE"
        lines.append(f"- claim_{i} (conf={c.confidence:.2f}): {c.claim} :: {ev_join}")

    if pack.unknowns:
        lines.append(f"- unknowns: {pack.unknowns}")

    if pack.recommend_refuse:
        lines.append(f"- recommend_refuse: true ({pack.refusal_reason})")

    if pack.used_sources:
        lines.append(f"- used_sources: {pack.used_sources}")

    return "\n".join(lines)


class VerifierAgent:
    """
    Critic/Verifier:
    - Rule-based checks for missing/weak evidence
    - LLM-based detection of conflicts and source preference
    """

    def __init__(self, *, llm: ChatOllama):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=VerifierOutput)

        self.prompt = ChatPromptTemplate.from_template(
            """You are a strict Verifier/Critic for a multi-agent university QA system.

You receive:
- The user question
- A RAG specialist evidence pack
- A Web specialist evidence pack

Your tasks:
1) Detect conflicts between RAG and WEB claims if any.
2) Decide preferred_source: RAG / WEB / HYBRID / NONE.
3) Decide recommend_refuse when:
   - evidence is insufficient, or
   - conflicts cannot be resolved safely.
4) Estimate rag_trust and web_trust in [0,1].
5) Provide a short summary and a list of issues.

Policy:
- For stable campus facts (identity, structure, internal rules), prefer RAG.
- For time-sensitive facts (renaming, fees, admissions timeline, policy updates),
  prefer the source that appears more recent/official based on evidence text.
- Never invent missing evidence.
- A pack marked <NONE> means that source was not queried. Its absence is not a contradiction.
- Report a cross-source conflict only when two supplied claims state incompatible facts; quote both claims.

{format_instructions}

Question:
{question}

RAG Pack Brief:
{rag_brief}

WEB Pack Brief:
{web_brief}
"""
        ).partial(format_instructions=self.parser.get_format_instructions())

    def _normalize_pack(
        self,
        pack: Optional[Union[RAGSpecialistOutput, Dict[str, Any]]]
    ) -> Optional[RAGSpecialistOutput]:
        if pack is None:
            return None
        if isinstance(pack, RAGSpecialistOutput):
            return pack
        # dict
        try:
            return RAGSpecialistOutput.model_validate(pack)
        except Exception:
            return None

    def invoke(
        self,
        question: str,
        rag_pack: Optional[Union[RAGSpecialistOutput, Dict[str, Any]]] = None,
        web_pack: Optional[Union[RAGSpecialistOutput, Dict[str, Any]]] = None,
    ) -> VerifierOutput:

        rag_pack_n = self._normalize_pack(rag_pack)
        web_pack_n = self._normalize_pack(web_pack)

        # ---- rule-based issues
        issues = []
        issues += _rule_issues_from_pack(rag_pack_n, "RAG")
        issues += _rule_issues_from_pack(web_pack_n, "WEB")

        # ---- LLM judgement
        rag_brief = _pack_to_brief(rag_pack_n, "RAG")
        web_brief = _pack_to_brief(web_pack_n, "WEB")

        chain = (
            {
                "question": RunnablePassthrough(),
                "rag_brief": lambda _: rag_brief,
                "web_brief": lambda _: web_brief,
            }
            | self.prompt
            | self.llm
            | self.parser
        )

        verdict: VerifierOutput = chain.invoke(question)

        # ---- merge rule issues into LLM issues
        # avoid duplicates by (type + first 60 chars)
        dedup = {}
        for it in (verdict.issues or []):
            key = (it.issue_type, (it.description or "")[:60])
            dedup[key] = it
        for it in issues:
            key = (it.issue_type, (it.description or "")[:60])
            if key not in dedup:
                dedup[key] = it
        verdict.issues = list(dedup.values())

        # ---- conservative final guardrails
        # If both packs suggest refusal -> strong refuse
        if rag_pack_n and web_pack_n:
            if rag_pack_n.recommend_refuse and web_pack_n.recommend_refuse:
                verdict.recommend_refuse = True
                if not verdict.refusal_reason:
                    verdict.refusal_reason = "Both specialists report insufficient evidence."

        # If no claims from both -> refuse
        rag_claims = len(rag_pack_n.claims) if rag_pack_n else 0
        web_claims = len(web_pack_n.claims) if web_pack_n else 0
        if rag_claims == 0 and web_claims == 0:
            verdict.recommend_refuse = True
            verdict.preferred_source = "NONE"
            if not verdict.refusal_reason:
                verdict.refusal_reason = "No checkable claims provided by specialists."

        # If many NO_EVIDENCE issues -> raise refuse likelihood
        no_ev = [i for i in verdict.issues if i.issue_type == "NO_EVIDENCE"]
        if len(no_ev) >= 2 and not verdict.recommend_refuse:
            verdict.issues.append(VerifierIssue(
                issue_type="STALE_RISK",
                description="Multiple claims lack explicit evidence; consider more conservative wording.",
                severity=3,
            ))

        if verdict.recommend_refuse and not verdict.summary:
            verdict.summary = "Evidence is insufficient or conflicting; refusal is recommended."

        return verdict


def create_verifier_agent(
    *,
    chat_model: Optional[str] = None,
    temperature: float = 0.0,
) -> VerifierAgent:
    llm = build_llm(model=chat_model, temperature=temperature)
    return VerifierAgent(llm=llm)
