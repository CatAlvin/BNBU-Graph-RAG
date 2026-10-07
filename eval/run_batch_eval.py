"""
Dataset item format:
{
  "id": "RAG_001",
  "question": "string",
  "category": "RAG_ONLY | WEB_ONLY | HYBRID | REFUSE",
  "difficulty": "L1 | L2 | L3",
  "gold_route": ["RAG"],
  "gold_answer": "string",
  "gold_evidence": [
    {
      "type": "rag_doc | web_page | rationale",
      "source_id": "BNBU Overview.pdf | BNBU Detail Infomation QA.pdf | WEB_###",
      "evidence_summary": "string"
    }
  ],
  "refusal_rule": "NONE | OUT_OF_DOMAIN | NON_EXISTENT | UNVERIFIABLE_RUMOR",
  "notes": "string"
}
"""

import os
import sys
import json
import glob
import time
import re
from typing import List, Dict, Any, Optional, Tuple, DefaultDict
from collections import defaultdict

from tqdm import tqdm
from dotenv import load_dotenv

# Ensure project root on path when running from ./eval
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.orchestrator import create_orchestrator
from agents.orchestrator import OrchestratorResult  # pydantic model

# Optional local LLM judge (only used if enabled)
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatOllama = None
    ChatPromptTemplate = None

import json
from openai import OpenAI


# -----------------------------------------
# Config
# -----------------------------------------

VAL_DIR_DEFAULT = os.path.join(THIS_DIR, "val_large.json")  # directory by default
OUT_DIR_DEFAULT = os.path.join(THIS_DIR, "outputs")

ROUTE_LABELS = ["RAG_ONLY", "WEB_ONLY", "HYBRID", "REFUSE"]
DIFFICULTIES = ["L1", "L2", "L3"]


# -----------------------------------------
# IO Utils
# -----------------------------------------

def load_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []

def load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def load_val_items(val_path: str) -> List[Dict[str, Any]]:
    """
    支持两种用法：
    1) val_path 是一个目录：读取下面所有 *.json / *.jsonl
    2) val_path 是一个文件：根据扩展名读取这一个文件
    """
    items: List[Dict[str, Any]] = []

    # --- 情况 1：val_path 是单个文件 ---
    if os.path.isfile(val_path):
        ext = os.path.splitext(val_path)[1].lower()
        if ext == ".json":
            items.extend(load_json_file(val_path))
        elif ext == ".jsonl":
            items.extend(load_jsonl_file(val_path))
        else:
            raise RuntimeError(f"Unsupported file extension for val file: {val_path}")
    else:
        # --- 情况 2：val_path 是目录 ---
        for p in sorted(glob.glob(os.path.join(val_path, "*.json"))):
            items.extend(load_json_file(p))

        for p in sorted(glob.glob(os.path.join(val_path, "*.jsonl"))):
            items.extend(load_jsonl_file(p))

    # minimal filtering: require question + category
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = it.get("question")
        cat = it.get("category")
        if isinstance(q, str) and q.strip() and isinstance(cat, str) and cat.strip():
            out.append(it)

    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# =========================================================
# DeepSeek local helper (no external import)
# =========================================================

# 参照你的 web_search 调用示例：优先加载 .local.env
load_dotenv(".local.env")

def _get_env_var(*names: str) -> str | None:
    """
    小工具：支持多个候选名字，比如：
    _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def ask_deepseek(question: str, context: str | None = None) -> str:
    deepseek_key = _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
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


# -----------------------------------------
# Metrics Helpers
# -----------------------------------------

def normalize_category(cat: str) -> str:
    c = (cat or "").upper().strip()
    if c in ROUTE_LABELS:
        return c
    if "RAG" in c:
        return "RAG_ONLY"
    if "WEB" in c:
        return "WEB_ONLY"
    if "HYBRID" in c:
        return "HYBRID"
    if "REFUSE" in c:
        return "REFUSE"
    return c or "UNKNOWN"

def normalize_difficulty(d: str) -> str:
    d = (d or "").upper().strip()
    return d if d in DIFFICULTIES else (d or "UNKNOWN")

def is_gold_refuse(item: Dict[str, Any]) -> bool:
    cat = normalize_category(item.get("category", ""))
    if cat == "REFUSE":
        return True
    rr = (item.get("refusal_rule") or "NONE").upper().strip()
    return rr != "NONE"

def count_total_claims(res: OrchestratorResult) -> int:
    n = 0
    if res.rag_pack and getattr(res.rag_pack, "claims", None):
        n += len(res.rag_pack.claims)
    if res.web_pack and getattr(res.web_pack, "claims", None):
        n += len(res.web_pack.claims)
    return n

def conflict_flag(res: OrchestratorResult) -> bool:
    v = res.verifier_output
    if not v or not getattr(v, "issues", None):
        return False
    for issue in v.issues:
        t = getattr(issue, "issue_type", "")
        if t in ("POTENTIAL_CONFLICT", "UNRESOLVED_CONFLICT"):
            return True
    return False

def evidence_overlap_flag(item: Dict[str, Any], used_sources: List[str]) -> bool:
    gold_evs = item.get("gold_evidence") or []
    if not isinstance(gold_evs, list) or not used_sources:
        return False

    src_join = " | ".join(used_sources)

    for ev in gold_evs:
        if not isinstance(ev, dict):
            continue
        ev_type = (ev.get("type") or "").lower().strip()
        sid = (ev.get("source_id") or "").strip()
        if not sid:
            continue

        if ev_type == "rag_doc" or sid.lower().endswith(".pdf"):
            if sid in src_join:
                return True

        if ev_type == "web_page":
            if sid.startswith("http"):
                for us in used_sources:
                    if sid in us or us in sid:
                        return True
            else:
                continue

    return False

def claims_coverage(item: Dict[str, Any], res: OrchestratorResult) -> Optional[float]:
    if is_gold_refuse(item):
        return None

    gold_evs = item.get("gold_evidence") or []
    gold_n = len(gold_evs) if isinstance(gold_evs, list) else 0
    gold_n = max(1, gold_n)

    claims_n = count_total_claims(res)
    cov = min(1.0, float(claims_n) / float(gold_n))
    return round(cov, 4)


# -----------------------------------------
# Confusion Matrix
# -----------------------------------------

def init_confusion() -> Dict[str, Dict[str, int]]:
    return {g: {p: 0 for p in ROUTE_LABELS} for g in ROUTE_LABELS}

def update_confusion(cm: Dict[str, Dict[str, int]], gold: str, pred: str):
    if gold in cm and pred in cm[gold]:
        cm[gold][pred] += 1


# -----------------------------------------
# Semantic Judge
# -----------------------------------------

def _extract_json_like(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to recover JSON from LLM text.
    """
    if not text:
        return None
    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

def _has_official_hint(answer: str, used_sources: List[str]) -> bool:
    """
    Simple heuristic to detect 'go check official site' suggestions.
    """
    a = (answer or "").lower()
    hints = [
        "official website", "official site", "university website",
        "官网", "官方", "academic registry", "ar.bnbu", "bnbu.edu",
        "please check", "建议", "以官网为准"
    ]
    if any(h in a for h in hints):
        return True
    s_join = " ".join(used_sources or []).lower()
    if "bnbu" in s_join or "ar.bnbu" in s_join or "bnbu.edu" in s_join:
        return True
    return False

def semantic_score_with_deepseek(
    *,
    item: Dict[str, Any],
    res: OrchestratorResult,
) -> Tuple[float, str, Dict[str, Any], str]:
    """
    Returns:
        (score 0-1, rationale, dimensions, judge_model)
    """

    gold_cat = normalize_category(item.get("category", ""))
    gold_answer = item.get("gold_answer", "") or ""
    refusal_rule = (item.get("refusal_rule") or "NONE").upper().strip()
    gold_evs = item.get("gold_evidence") or []

    pred_answer = res.final_output.final_answer or ""
    pred_refuse = bool(res.final_output.final_refuse)
    pred_route = res.route_output.final_label
    used_sources = res.final_output.used_sources or []

    # If deepseek is not available, fallback to heuristic
    if ask_deepseek is None:
        # Minimal heuristic scoring
        if gold_cat == "REFUSE" or refusal_rule != "NONE":
            score = 1.0 if pred_refuse else 0.0
            rationale = "heuristic_refuse_match"
            dims = {"refusal_match": score}
            return score, rationale, dims, "heuristic"

        # WEB-like gold answers often recommend official checking
        if gold_cat in ("WEB_ONLY", "HYBRID"):
            score = 0.8 if _has_official_hint(pred_answer, used_sources) else 0.4
            rationale = "heuristic_web_official_hint"
            dims = {"official_hint": 1.0 if _has_official_hint(pred_answer, used_sources) else 0.0}
            return score, rationale, dims, "heuristic"

        # RAG_ONLY basic heuristic
        score = 0.6 if pred_answer.strip() else 0.0
        rationale = "heuristic_nonempty_answer"
        dims = {"nonempty": 1.0 if pred_answer.strip() else 0.0}
        return score, rationale, dims, "heuristic"

    # Build a compact evidence summary to avoid huge prompts
    ev_briefs = []
    if isinstance(gold_evs, list):
        for ev in gold_evs[:4]:
            if not isinstance(ev, dict):
                continue
            ev_briefs.append({
                "type": ev.get("type"),
                "source_id": ev.get("source_id"),
                "evidence_summary": ev.get("evidence_summary"),
            })

    context = (
        "You are a strict semantic judge for a multi-agent university QA evaluation.\n"
        "Score the model output against the gold intent and rubric.\n\n"
        "Return JSON ONLY with the following schema:\n"
        "{\n"
        '  "semantic_score": 0.0-1.0,\n'
        '  "dimensions": {\n'
        '     "route_alignment": 0.0-1.0,\n'
        '     "answer_intent_match": 0.0-1.0,\n'
        '     "evidence_usefulness": 0.0-1.0,\n'
        '     "refusal_quality": 0.0-1.0,\n'
        '     "web_official_guidance": 0.0-1.0\n'
        "  },\n"
        '  "rationale": "short reason"\n'
        "}\n\n"
        "Rubric (very important):\n"
        "1) If gold category is REFUSE or refusal_rule != NONE:\n"
        "   - High score only if the model refuses politely and aligns with the rule.\n"
        "2) If gold category includes WEB (WEB_ONLY or HYBRID):\n"
        "   - Gold answer may intentionally advise users to check the official website.\n"
        "   - A high score is deserved if the model:\n"
        "       a) clearly suggests checking official/authoritative sources, AND/OR\n"
        "       b) provides a valid or plausible official link, OR\n"
        "       c) the used_sources contains an official BNBU/academic registry page.\n"
        "3) If gold category is RAG_ONLY:\n"
        "   - Prefer direct factual answer grounded in internal documents.\n"
        "   - Do not over-reward 'just go check the website' without giving the core fact.\n"
        "4) Route alignment matters but should not dominate correctness.\n\n"
        f"Gold category: {gold_cat}\n"
        f"Gold difficulty: {item.get('difficulty')}\n"
        f"Gold route hint: {item.get('gold_route')}\n"
        f"Gold refusal_rule: {refusal_rule}\n"
        f"Gold answer (may be brief/intention-focused): {gold_answer}\n"
        f"Gold evidence brief: {ev_briefs}\n\n"
        f"Predicted route: {pred_route}\n"
        f"Predicted refuse: {pred_refuse}\n"
        f"Predicted answer: {pred_answer}\n"
        f"Predicted used_sources: {used_sources[:6]}\n"
    )

    text = ask_deepseek(question=item["question"], context=context)
    obj = _extract_json_like(text) or {}

    score = obj.get("semantic_score", None)
    dims = obj.get("dimensions", {}) if isinstance(obj.get("dimensions", {}), dict) else {}
    rationale = obj.get("rationale", "") or ""

    # sanitize
    try:
        score_f = float(score)
    except Exception:
        # fallback if parsing fails
        score_f = 0.5 if pred_answer else 0.0

    score_f = max(0.0, min(1.0, score_f))
    # fill missing dims softly
    for k in ["route_alignment", "answer_intent_match", "evidence_usefulness", "refusal_quality", "web_official_guidance"]:
        if k not in dims:
            dims[k] = None

    return round(score_f, 4), rationale[:240], dims, "deepseek"


# -----------------------------------------
# Main Runner
# -----------------------------------------

def run_batch_eval(
    val_dir: str = VAL_DIR_DEFAULT,
    out_dir: str = OUT_DIR_DEFAULT,
    *,
    enable_route_ds_correction: Optional[bool] = None,
    force_route_ds_correction: Optional[bool] = None,
    enable_semantic_llm_score: bool = True,
) -> Tuple[str, Dict[str, Any]]:

    ensure_dir(out_dir)

    items = load_val_items(val_dir)
    if not items:
        raise RuntimeError(f"No valid items found under: {val_dir}")

    orch = create_orchestrator(
        enable_route_deepseek_correction=enable_route_ds_correction,
        force_route_deepseek_correction=force_route_ds_correction,
    )

    cm = init_confusion()

    total = 0
    route_correct = 0

    gold_refuse_n = 0
    pred_refuse_n = 0
    false_reject_n = 0
    missed_refuse_n = 0

    conflict_n = 0

    coverage_vals: List[float] = []
    overlap_hit_n = 0

    semantic_scores: List[float] = []
    semantic_by_cat: DefaultDict[str, List[float]] = defaultdict(list)
    semantic_by_diff: DefaultDict[str, List[float]] = defaultdict(list)

    rows: List[Dict[str, Any]] = []

    for it in tqdm(items, desc="Evaluating", ncols=90):
        total += 1  # ✅ FIXED: you missed this in your version

        qid = it.get("id", f"ID_{total:04d}")
        question = it["question"]
        gold_cat = normalize_category(it.get("category", ""))
        gold_diff = normalize_difficulty(it.get("difficulty", ""))
        gold_is_refuse = is_gold_refuse(it)

        # ---------- run orchestrator ----------
        start = time.time()
        res = orch.run(
            question,
            use_route_deepseek_correction=enable_route_ds_correction,
            force_route_correction=force_route_ds_correction,
            run_verifier=True,
            return_trace=True,
        )
        latency = round(time.time() - start, 4)

        pred_route = res.route_output.final_label
        pred_refuse = bool(res.final_output.final_refuse)

        # ---------- route confusion ----------
        if gold_cat in ROUTE_LABELS and pred_route in ROUTE_LABELS:
            update_confusion(cm, gold_cat, pred_route)
            if gold_cat == pred_route:
                route_correct += 1

        # ---------- refuse stats ----------
        if gold_is_refuse:
            gold_refuse_n += 1
        if pred_refuse:
            pred_refuse_n += 1

        if pred_refuse and not gold_is_refuse:
            false_reject_n += 1
        if (not pred_refuse) and gold_is_refuse:
            missed_refuse_n += 1

        # ---------- conflict frequency ----------
        if conflict_flag(res):
            conflict_n += 1

        # ---------- claims coverage ----------
        cov = claims_coverage(it, res)
        if cov is not None:
            coverage_vals.append(cov)

        # ---------- evidence overlap ----------
        used_sources = res.final_output.used_sources or []
        overlap_hit = evidence_overlap_flag(it, used_sources)
        if overlap_hit:
            overlap_hit_n += 1

        # ---------- semantic scoring ----------
        semantic_score = None
        semantic_rationale = None
        semantic_dims = None
        semantic_model = None

        if enable_semantic_llm_score:
            try:
                semantic_score, semantic_rationale, semantic_dims, semantic_model = semantic_score_with_deepseek(
                    item=it,
                    res=res,
                )
                if semantic_score is not None:
                    semantic_scores.append(semantic_score)
                    semantic_by_cat[gold_cat].append(semantic_score)
                    semantic_by_diff[gold_diff].append(semantic_score)
            except Exception as e:
                # don't break batch eval due to judge failure
                semantic_score = None
                semantic_rationale = f"judge_error: {str(e)[:120]}"
                semantic_dims = {}
                semantic_model = "judge_error"

        # ---------- build jsonl row ----------
        row = {
            "id": qid,
            "question": question,
            "category": it.get("category"),
            "difficulty": it.get("difficulty"),
            "gold_route": it.get("gold_route"),
            "gold_refuse": gold_is_refuse,
            "gold_answer": it.get("gold_answer"),
            "gold_evidence": it.get("gold_evidence"),
            "refusal_rule": it.get("refusal_rule"),
            "notes": it.get("notes"),

            "pred_route": pred_route,
            "pred_refuse": pred_refuse,
            "preferred_source": res.final_output.preferred_source,
            "used_sources": used_sources,

            "claims_count": count_total_claims(res),
            "claims_coverage": cov,
            "evidence_overlap_hit": overlap_hit,
            "conflict_flag": conflict_flag(res),

            "semantic_score": semantic_score,
            "semantic_rationale": semantic_rationale,
            "semantic_dimensions": semantic_dims,
            "semantic_judge_model": semantic_model,

            "latency_sec": latency,

            # full raw trace (for deep debugging / later re-scoring)
            "route_output": res.route_output.model_dump(),
            "rag_pack": res.rag_pack.model_dump() if res.rag_pack else None,
            "web_pack": res.web_pack.model_dump() if res.web_pack else None,
            "verifier_output": res.verifier_output.model_dump() if res.verifier_output else None,
            "final_output": res.final_output.model_dump(),

            # error fields
            "error_stage": res.error_stage,
            "error_message": res.error_message,
        }

        rows.append(row)

    # ---------- summary ----------
    route_acc = round(route_correct / max(1, total), 4)

    avg_cov = round(sum(coverage_vals) / max(1, len(coverage_vals)), 4) if coverage_vals else 0.0
    conflict_rate = round(conflict_n / max(1, total), 4)

    false_reject_rate = round(false_reject_n / max(1, (total - gold_refuse_n)), 4) if (total - gold_refuse_n) > 0 else 0.0
    missed_refuse_rate = round(missed_refuse_n / max(1, gold_refuse_n), 4) if gold_refuse_n > 0 else 0.0

    overlap_rate = round(overlap_hit_n / max(1, total), 4)

    # semantic summary
    avg_sem = round(sum(semantic_scores) / max(1, len(semantic_scores)), 4) if semantic_scores else 0.0

    avg_sem_by_cat = {}
    for c in ROUTE_LABELS:
        arr = semantic_by_cat.get(c, [])
        avg_sem_by_cat[c] = round(sum(arr) / max(1, len(arr)), 4) if arr else 0.0

    avg_sem_by_diff = {}
    for d in DIFFICULTIES:
        arr = semantic_by_diff.get(d, [])
        avg_sem_by_diff[d] = round(sum(arr) / max(1, len(arr)), 4) if arr else 0.0

    summary = {
        "total": total,
        "route_accuracy_overall": route_acc,
        "confusion_matrix": cm,

        "gold_refuse_n": gold_refuse_n,
        "pred_refuse_n": pred_refuse_n,
        "false_reject_n": false_reject_n,
        "missed_refuse_n": missed_refuse_n,
        "false_reject_rate": false_reject_rate,
        "missed_refuse_rate": missed_refuse_rate,

        "conflict_n": conflict_n,
        "conflict_rate": conflict_rate,

        "avg_claims_coverage": avg_cov,
        "claims_coverage_samples": len(coverage_vals),

        "evidence_overlap_hit_n": overlap_hit_n,
        "evidence_overlap_rate": overlap_rate,

        "semantic_scoring_enabled": enable_semantic_llm_score,
        "semantic_judge_available": bool(ask_deepseek is not None),
        "avg_semantic_score_overall": avg_sem,
        "avg_semantic_score_by_category": avg_sem_by_cat,
        "avg_semantic_score_by_difficulty": avg_sem_by_diff,
        "semantic_scored_samples": len(semantic_scores),
    }

    # ---------- write outputs ----------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(out_dir, f"batch_trace_{ts}.jsonl")
    out_summary = os.path.join(out_dir, f"batch_summary_{ts}.json")

    write_jsonl(out_jsonl, rows)

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_jsonl, summary


def print_confusion(cm: Dict[str, Dict[str, int]]):
    print("\n=== Routing Confusion Matrix (gold rows x pred cols) ===")
    header = ["G\\P"] + ROUTE_LABELS
    print("\t".join(header))
    for g in ROUTE_LABELS:
        row = [g] + [str(cm[g][p]) for p in ROUTE_LABELS]
        print("\t".join(row))


def main():
    val_dir = VAL_DIR_DEFAULT
    out_dir = OUT_DIR_DEFAULT

    # CLI:
    # python ./eval/run_batch_eval.py --val_dir ./eval/val_large --out_dir ./eval/outputs
    args = sys.argv[1:]
    if "--val_dir" in args:
        idx = args.index("--val_dir")
        if idx + 1 < len(args):
            val_dir = args[idx + 1]
    if "--out_dir" in args:
        idx = args.index("--out_dir")
        if idx + 1 < len(args):
            out_dir = args[idx + 1]

    # route correction flags
    enable_ds = False
    force_ds = False
    if "--enable_route_ds" in args:
        enable_ds = True
    if "--force_route_ds" in args:
        force_ds = True

    # semantic scoring toggle
    enable_sem = True
    if "--disable_semantic" in args:
        enable_sem = False

    print(f"Val path: {val_dir}")
    print(f"Out dir: {out_dir}")
    print(f"Route DeepSeek correction: enable={enable_ds}, force={force_ds}")
    print(f"Semantic LLM scoring: {enable_sem} (deepseek_available={ask_deepseek is not None})")

    out_jsonl, summary = run_batch_eval(
        val_dir=val_dir,
        out_dir=out_dir,
        enable_route_ds_correction=enable_ds,
        force_route_ds_correction=force_ds,
        enable_semantic_llm_score=enable_sem,
    )

    print(f"\n✅ JSONL trace saved to:\n{out_jsonl}")
    print(f"✅ Summary saved next to JSONL (same timestamp).")

    print_confusion(summary["confusion_matrix"])

    print("\n=== REFUSE Stats ===")
    print("gold_refuse_n:", summary["gold_refuse_n"])
    print("pred_refuse_n:", summary["pred_refuse_n"])
    print("false_reject_n:", summary["false_reject_n"], "rate:", summary["false_reject_rate"])
    print("missed_refuse_n:", summary["missed_refuse_n"], "rate:", summary["missed_refuse_rate"])

    print("\n=== Conflict Frequency ===")
    print("conflict_n:", summary["conflict_n"], "rate:", summary["conflict_rate"])

    print("\n=== Claims Coverage (heuristic) ===")
    print("avg_claims_coverage:", summary["avg_claims_coverage"])
    print("coverage_samples:", summary["claims_coverage_samples"])

    print("\n=== Evidence Overlap (heuristic) ===")
    print("overlap_hit_n:", summary["evidence_overlap_hit_n"], "rate:", summary["evidence_overlap_rate"])

    print("\n=== Semantic Score ===")
    print("avg_semantic_score_overall:", summary["avg_semantic_score_overall"])
    print("avg_semantic_score_by_category:", summary["avg_semantic_score_by_category"])
    print("avg_semantic_score_by_difficulty:", summary["avg_semantic_score_by_difficulty"])
    print("semantic_scored_samples:", summary["semantic_scored_samples"])


if __name__ == "__main__":
    main()
