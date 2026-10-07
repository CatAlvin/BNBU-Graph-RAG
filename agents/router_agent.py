from __future__ import annotations

import os
import re
from typing import Dict, Any, Optional, Literal, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, confloat

# Your local router implementation
from GPT2_router import LocalGPT2TwoStageRouter

import json
from openai import OpenAI



RouteLabel = Literal["RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE"]

class RouterOutput(BaseModel):
    question: str

    final_label: RouteLabel
    probs: Dict[str, Dict[str, confloat(ge=0, le=1)]] = Field(
        default_factory=dict,
        description="StageA/StageB probabilities from GPT2 router."
    )
    debug: Dict[str, Any] = Field(default_factory=dict)

    # DeepSeek correction (optional)
    corrected: bool = False
    corrected_label: Optional[RouteLabel] = None
    correction_reason: Optional[str] = None


def _env_bool(name: str, default: bool = False) -> bool:
    load_dotenv()
    v = os.getenv(name)
    if v is None:
        return default
    v = v.lower().strip()
    return v in ("1", "true", "yes", "y")


def _extract_label_from_text(text: str) -> Optional[RouteLabel]:
    if not text:
        return None

    # Normalize
    t = text.upper()

    # Direct match priority
    for lab in ["RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE"]:
        if re.search(rf"\b{lab}\b", t):
            return lab  # type: ignore

    # Allow loose match
    if "RAG" in t and "ONLY" in t:
        return "RAG_ONLY"
    if "WEB" in t and "ONLY" in t:
        return "WEB_ONLY"

    return None


def _should_attempt_correction(
    final_label: str,
    probs: Dict[str, Dict[str, float]],
    debug: Dict[str, Any],
    force: bool = False
) -> bool:
    if force:
        return True

    stageA = probs.get("stageA", {})
    stageB = probs.get("stageB", {})

    p_refuse = float(stageA.get("REFUSE", 0.0))
    if 0.45 <= p_refuse <= 0.65:
        return True

    if stageB:
        sorted_b = sorted(stageB.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_b) >= 2:
            margin = float(sorted_b[0][1]) - float(sorted_b[1][1])
            if margin <= 0.15:  
                return True

    reason = str(debug.get("reason", "")).lower()
    if "low_conf" in reason or "ambiguous" in reason:
        return True

    q = str(debug.get("question", "") or "").lower()
    time_keywords = ["latest", "this year", "今年", "最新", "now", "today", "最近", "排名", "rank", "news", "announcement"]
    if final_label == "RAG_ONLY" and any(k in q for k in time_keywords):
        return True

    return False

load_dotenv(".local.env")

def _get_env_var(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _ask_deepseek_router(
    question: str,
    final_label: str,
    probs: Dict[str, Dict[str, float]],
    debug: Dict[str, Any],
) -> str:
    deepseek_key = _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
    if not deepseek_key:
        return "REFUSE  # no_deepseek_key"

    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    system_prompt = (
        "You are a STRICT ROUTING JUDGE for a university QA system.\n"
        "Your ONLY job is to choose the BEST routing label for the user question.\n\n"
        "ALLOWED LABELS (exactly one):\n"
        "- RAG_ONLY\n"
        "- WEB_ONLY\n"
        "- HYBRID\n"
        "- REFUSE\n\n"
        "VERY IMPORTANT RULES:\n"
        "1) GPT2's label and probabilities are JUST NOISY PRIOR INFORMATION.\n"
        "   You MUST NOT blindly copy GPT2's label.\n"
        "2) You MUST override GPT2 if the question type clearly belongs to a different route.\n"
        "3) Think about the QUESTION FIRST, then use probabilities only as a soft hint.\n"
        "4) OUTPUT REQUIREMENT: return ONLY ONE of the four uppercase labels as plain text.\n"
        "   Do NOT output explanations or any other text.\n\n"
        "HEURISTICS:\n"
        "- Use REFUSE for harmful content, conspiracy theories, unverifiable accusations, or nonsense.\n"
        "- Use RAG_ONLY for stable internal campus facts likely in local PDFs/KB.\n"
        "- Use WEB_ONLY for time-sensitive public info: latest rankings, news, events, 'this year', 'today', etc.\n"
        "- Use HYBRID when both internal policy + latest public information are needed to answer well.\n"
    )

    payload = {
        "question": question,
        "gpt2_final_label": final_label,
        "probs": probs,
        "debug": debug,
    }
    user_prompt = (
        "Here is the routing context in JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Now, based on the QUESTION and the heuristics, choose the BEST label.\n"
        "Remember: you MUST NOT simply repeat GPT2's label if it doesn't match the question type.\n"
        "Return ONLY ONE of: RAG_ONLY, WEB_ONLY, HYBRID, REFUSE."
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.0,  # 路由建议温度拉到 0，减少随机性
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"REFUSE  # deepseek_error: {e}"


def _get_env_var(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def ask_deepseek(question: str, context: str | None = None) -> str:
    deepseek_key = _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
    if not deepseek_key:
        return "抱歉，我目前缺少必要的模型配置，暂时无法安全回答这个问题。"

    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )
    system_prompt = (
        "You are the final Aggregator/Judge for a multi-agent university QA system.\n"
        "You MUST base your answer strictly on the provided context when context is given.\n"
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
        # 再兜一层，确保异常时也能给出稳定输出
        return "抱歉，我无法基于现有证据安全回答这个问题。"



class RouterAgent:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        enable_deepseek_correction: bool = False,
        force_deepseek_correction: bool = False,
    ):
        self.router = LocalGPT2TwoStageRouter(checkpoint_path)
        self.enable_deepseek_correction = enable_deepseek_correction
        self.force_deepseek_correction = force_deepseek_correction

    def _deepseek_correct(
        self,
        question: str,
        final_label: RouteLabel,
        probs: Dict[str, Dict[str, float]],
        debug: Dict[str, Any],
    ) -> Tuple[Optional[RouteLabel], str]:
        raw = _ask_deepseek_router(
            question=question,
            final_label=final_label,
            probs=probs,
            debug=debug,
        )

        label = _extract_label_from_text(raw)
        return label, raw


    def route(
        self,
        question: str,
        *,
        use_deepseek_correction: Optional[bool] = None,
        force_correction: Optional[bool] = None,
    ) -> RouterOutput:
        final_label, probs, debug = self.router.route(question)

        # Normalize label type
        final_label = str(final_label).upper().strip()
        if final_label not in ("RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE"):
            # Fallback: safest default
            final_label = "REFUSE"

        out = RouterOutput(
            question=question,
            final_label=final_label,  # type: ignore
            probs=probs or {},
            debug=debug or {},
            corrected=False,
            corrected_label=None,
            correction_reason=None,
        )

        # Decide whether to run correction
        if use_deepseek_correction is None:
            use_deepseek_correction = self.enable_deepseek_correction

        if force_correction is None:
            force_correction = self.force_deepseek_correction

        if not use_deepseek_correction:
            return out

        # Gate correction for stability
        if not _should_attempt_correction(final_label, probs or {}, debug or {}, force=bool(force_correction)):
            return out

        corrected_label, raw = self._deepseek_correct(
            question=question,
            final_label=final_label,  # type: ignore
            probs=probs or {},
            debug=debug or {},
        )

        if corrected_label and corrected_label != out.final_label:
            out.corrected = True
            out.corrected_label = corrected_label
            out.correction_reason = "deepseek_override"
            out.final_label = corrected_label
        else:
            out.correction_reason = f"deepseek_no_change: {raw if raw else 'empty'}"

        return out


def create_router_agent(
    checkpoint_path: str = "./checkpoints/gpt2_124M_router_2stage.pth",
    *,
    enable_deepseek_correction: Optional[bool] = None,
    force_deepseek_correction: Optional[bool] = None,
) -> RouterAgent:
    if enable_deepseek_correction is None:
        enable_deepseek_correction = _env_bool("ROUTER_ENABLE_DEEPSEEK_CORRECTION", False)
    if force_deepseek_correction is None:
        force_deepseek_correction = _env_bool("ROUTER_FORCE_DEEPSEEK_CORRECTION", False)

    return RouterAgent(
        checkpoint_path,
        enable_deepseek_correction=enable_deepseek_correction,
        force_deepseek_correction=force_deepseek_correction,
    )
