from __future__ import annotations

import os
from typing import Optional, Dict, Any, Literal, Union, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .router_agent import (
    create_router_agent,
    RouterAgent,
    RouterOutput,
)

from .rag_specialist_agent import (
    create_rag_specialist_agent,
    RAGSpecialistOutput,
)

from .web_specialist_agent import (
    create_web_specialist_agent,
)

from .verifier_agent import (
    create_verifier_agent,
    VerifierOutput,
)

from .aggregator_judge_agent import (
    create_aggregator_judge_agent,
    AggregatorJudgeAgent,
    AggregatorOutput,
)

# Stronger final fallback
import json
from openai import OpenAI


RouteLabel = Literal["RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE"]




class OrchestratorResult(BaseModel):
    question: str

    route_output: RouterOutput

    rag_pack: Optional[RAGSpecialistOutput] = None
    web_pack: Optional[RAGSpecialistOutput] = None
    verifier_output: Optional[VerifierOutput] = None

    final_output: AggregatorOutput

    # error tracking for logs
    error_stage: Optional[str] = None
    error_message: Optional[str] = None



def _env_bool(name: str, default: bool = False) -> bool:
    load_dotenv()
    v = os.getenv(name)
    if v is None:
        return default
    v = v.lower().strip()
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



class Orchestrator:

    def __init__(
        self,
        *,
        router_agent: Optional[RouterAgent] = None,
        aggregator: Optional[AggregatorJudgeAgent] = None,
        checkpoint_path: str = "./checkpoints/gpt2_124M_router_2stage.pth",
        enable_route_deepseek_correction: Optional[bool] = None,
        force_route_deepseek_correction: Optional[bool] = None,
    ):
        if enable_route_deepseek_correction is None:
            enable_route_deepseek_correction = _env_bool("ORCH_ENABLE_ROUTE_DEEPSEEK_CORRECTION", False)
        if force_route_deepseek_correction is None:
            force_route_deepseek_correction = _env_bool("ORCH_FORCE_ROUTE_DEEPSEEK_CORRECTION", False)

        self.router_agent = router_agent or create_router_agent(
            checkpoint_path=checkpoint_path,
            enable_deepseek_correction=enable_route_deepseek_correction,
            force_deepseek_correction=force_route_deepseek_correction,
        )

        self.aggregator = aggregator or create_aggregator_judge_agent()

        # Lazy-loaded specialists/verifier
        self._rag_agent = None
        self._web_agent = None
        self._verifier = None

    def _get_rag_agent(self):
        if self._rag_agent is None:
            self._rag_agent = create_rag_specialist_agent()
        return self._rag_agent

    def _get_web_agent(self):
        if self._web_agent is None:
            self._web_agent = create_web_specialist_agent()
        return self._web_agent

    def _get_verifier(self):
        if self._verifier is None:
            self._verifier = create_verifier_agent()
        return self._verifier


    def _deepseek_refusal(self, question: str, reason: str) -> AggregatorOutput:
        if ask_deepseek is not None:
            text = ask_deepseek(
                question=question,
                context=f"The system failed to produce safe structured evidence. Reason: {reason}. Please refuse politely."
            )
        else:
            text = "抱歉，我无法根据当前信息安全地回答这个问题。"

        # Build a minimal AggregatorOutput
        out = AggregatorOutput(
            final_answer=text,
            final_refuse=True,
            refusal_reason=reason,
            preferred_source=None,
            used_sources=[],
        )
        return out

    def run(
        self,
        question: str,
        *,
        route_override: Optional[RouteLabel] = None,
        use_route_deepseek_correction: Optional[bool] = None,
        force_route_correction: Optional[bool] = None,
        run_verifier: bool = True,
        return_trace: bool = True,
    ) -> OrchestratorResult:
        """
        Full pipeline with structured trace.
        """

        # 1) Route
        route_out = self.router_agent.route(
            question,
            use_deepseek_correction=use_route_deepseek_correction,
            force_correction=force_route_correction,
        )

        if route_override:
            # Respect override (useful for ablations)
            route_out.final_label = route_override  # type: ignore

        label: RouteLabel = route_out.final_label  # type: ignore

        # 2) If REFUSE, skip specialists and let aggregator handle
        if label == "REFUSE":
            final = self.aggregator.aggregate(
                question=question,
                rag_pack=None,
                web_pack=None,
                verdict=None,
                route="REFUSE",
            )
            result = OrchestratorResult(
                question=question,
                route_output=route_out,
                final_output=final,
            )
            return result

        # 3) Invoke specialists safely according to route
        rag_pack = None
        web_pack = None

        # RAG needed?
        if label in ("RAG_ONLY", "HYBRID"):
            try:
                rag_pack = self._get_rag_agent().invoke(question)
            except Exception as e:
                reason = "RAG specialist output parsing failed (possible sensitive content)."
                final = self._deepseek_refusal(question, reason)
                return OrchestratorResult(
                    question=question,
                    route_output=route_out,
                    rag_pack=None,
                    web_pack=None,
                    verifier_output=None,
                    final_output=final,
                    error_stage="RAG_SPECIALIST",
                    error_message=str(e),
                )

        # WEB needed?
        if label in ("WEB_ONLY", "HYBRID"):
            try:
                web_pack = self._get_web_agent().invoke(question)
            except Exception as e:
                reason = "Web specialist output parsing failed (possible sensitive content)."
                final = self._deepseek_refusal(question, reason)
                return OrchestratorResult(
                    question=question,
                    route_output=route_out,
                    rag_pack=rag_pack,
                    web_pack=None,
                    verifier_output=None,
                    final_output=final,
                    error_stage="WEB_SPECIALIST",
                    error_message=str(e),
                )

        # 4) Verifier (optional, safe)
        verdict = None
        if run_verifier:
            try:
                verdict = self._get_verifier().invoke(question, rag_pack=rag_pack, web_pack=web_pack)
            except Exception as e:
                verdict = None

        # 5) Aggregate final
        final = self.aggregator.aggregate(
            question=question,
            rag_pack=rag_pack,
            web_pack=web_pack,
            verdict=verdict,
            route=label,
        )

        result = OrchestratorResult(
            question=question,
            route_output=route_out,
            rag_pack=rag_pack,
            web_pack=web_pack,
            verifier_output=verdict,
            final_output=final,
            error_stage=None,
            error_message=None,
        )

        return result

    def answer(
        self,
        question: str,
        *,
        route_override: Optional[RouteLabel] = None,
        use_route_deepseek_correction: Optional[bool] = None,
        force_route_correction: Optional[bool] = None,
        run_verifier: bool = True,
    ) -> AggregatorOutput:
        """
        Minimal interface: returns only final output.
        """
        res = self.run(
            question,
            route_override=route_override,
            use_route_deepseek_correction=use_route_deepseek_correction,
            force_route_correction=force_route_correction,
            run_verifier=run_verifier,
            return_trace=False,
        )
        return res.final_output


def create_orchestrator(
    *,
    checkpoint_path: str = "./checkpoints/gpt2_124M_router_2stage.pth",
    enable_route_deepseek_correction: Optional[bool] = True,
    force_route_deepseek_correction: Optional[bool] = True,
) -> Orchestrator:
    """
    One-shot builder.
    """
    return Orchestrator(
        checkpoint_path=checkpoint_path,
        enable_route_deepseek_correction=enable_route_deepseek_correction,
        force_route_deepseek_correction=force_route_deepseek_correction,
    )
