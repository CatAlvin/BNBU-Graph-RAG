# web_search.py
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI


# === 1. 读取 .local.env，注入环境变量 ===
# 会把 .local.env 中的键值对写入 os.environ
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


# === 2. 初始化 Tavily Client（懒加载也行，这里简单写在模块级别） ===
def _get_tavily_client():
    key = _get_env_var("Tavily_API_KEY", "TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Web mode requires TAVILY_API_KEY in .local.env; local RAG works without it.")
    return TavilyClient(api_key=key)


def _deepseek_summarize_tavily(
    question: str,
    tavily_response: Dict[str, Any],
) -> str:
    """
    使用 DeepSeek 对 Tavily 的搜索结果进行“核查、整理和优化”，
    输出一段干净的 answer 文本（字符串）。
    然后我们会把这段文本写回 tavily_response["answer"]。
    """
    deepseek_key = _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError(
            "DeepSeek API key not found. Please set DeepSeek_API_KEY or DEEPSEEK_API_KEY in .local.env"
        )

    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    # 为了让 DeepSeek “对 Tavily 结果进行核查”，我们把原始 JSON 一并给它
    tavily_json_str = json.dumps(tavily_response, ensure_ascii=False)

    system_prompt = (
        "You are a helpful assistant that verifies, consolidates, and refines web "
        "search results from the Tavily Search API.\n\n"
        "You MUST base your answer strictly on the provided Tavily results and prove the URL of information inside '[]', do not "
        "invent new facts. If the information is insufficient or conflicting, you must "
        "explicitly say you are not sure.\n\n"
        "Your final output should be a concise and clear answer in the same language "
        "as the user's question."
    )

    user_prompt = f"""
User question:
{question}

Tavily search JSON response:
{tavily_json_str}

Now, based ONLY on this Tavily response, write the best possible answer to the user.
If the information is clearly insufficient, say so explicitly.
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    answer = resp.choices[0].message.content.strip()
    return answer


def web_search(
    question: str,
    require_ans : bool = True,
    require_raw : bool = False,
    use_deepseek: bool = False,
    **tavily_kwargs: Any,
) -> Dict[str, Any]:
    """
    使用 Tavily 进行网页搜索。

    参数：
        question: 用户自然语言问题
        use_deepseek: 是否使用 DeepSeek 对 Tavily 结果进行核查和优化
        **tavily_kwargs: 其他 Tavily search 的可选参数，例如：
            - max_results: int
            - search_depth: "basic" | "advanced"
            - topic: "general" | "news" | "finance"
            - time_range: "day" | "week" | "month" | "year" | "d" | "w" | "m" | "y"
            - include_raw_content: bool | "text" | "markdown"
            - include_images: bool
            - ...

    返回：
        一个 dict，对应 Tavily Search API 的返回格式：
        {
            "query": str,
            "results": [ { "title": ..., "url": ..., "content": ..., "score": ... }, ... ],
            "response_time": float,
            "request_id": str,
            # 如果 use_deepseek=True，则还会包含：
            "answer": str
        }
    """
    if not question:
        raise ValueError("question must be a non-empty string.")

    # --- 默认参数：你可以按需要改 ---
    default_params: Dict[str, Any] = {
        "query": question,
        "max_results": tavily_kwargs.pop("max_results", 5),
        "search_depth": tavily_kwargs.pop("search_depth", "basic"),
        "include_answer": tavily_kwargs.pop("include_answer", require_ans),
        # 让 Tavily 返回提取后的全文文本
        "include_raw_content": tavily_kwargs.pop("include_raw_content", require_raw),
    }


    default_params.update(tavily_kwargs)

    # --- 第一步：调用 Tavily 搜索 ---
    tavily_response: Dict[str, Any] = _get_tavily_client().search(**default_params)

    # 如果不需要 DeepSeek 调优，直接返回原始 Tavily 结果
    if not use_deepseek:
        return tavily_response

    refined_answer = _deepseek_summarize_tavily(question, tavily_response)

    # 按 Tavily 官方返回格式，在原始 dict 上增加/覆盖 "answer" 字段
    tavily_response["answer"] = refined_answer
    return tavily_response

def tavily_extract_top(tavily_response: dict, k: int = 1):
    """
    从 Tavily 搜索响应中提取按 score 排序后的第 k 个结果。
    返回：url, title, content, score
    
    参数:
        tavily_response: Tavily Search API 返回的 JSON 字典
        k: 要取第 k 个最高分结果（1-based index）
    """
    if "results" not in tavily_response:
        raise ValueError("Invalid Tavily response: missing 'results' field.")
    
    results = tavily_response.get("results", [])
    if not results:
        raise ValueError("No search results found in Tavily response.")

    if k < 1 or k > len(results):
        raise IndexError(f"k={k} is out of range (1 ~ {len(results)}).")

    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    item = sorted_results[k - 1]

    url = item.get("url")
    title = item.get("title")
    # 优先使用 raw_content，否则退回 content
    content = item.get("raw_content") or item.get("content")
    score = item.get("score")

    return url, title, content, score

def translate_with_deepseek(
    text: str,
    target_lang: str,
    source_lang: str | None = None,
) -> str:
    """
    使用 DeepSeek 将输入文本翻译为指定目标语言。

    参数：
        text: 原文文本
        target_lang: 目标语言，例如 "English", "Chinese", "简体中文", "Japanese" 等
        source_lang: 源语言，可选。如果为 None，则由模型自动检测。

    返回：
        翻译后的文本（字符串）
    """
    if not text:
        raise ValueError("text must be a non-empty string.")
    if not target_lang:
        raise ValueError("target_lang must be a non-empty string.")

    deepseek_key = _get_env_var("DeepSeek_API_KEY", "DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError(
            "DeepSeek API key not found. Please set DeepSeek_API_KEY or DEEPSEEK_API_KEY in .local.env"
        )

    client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    # 构造 system + user prompt，让模型只做翻译，不加解释
    if source_lang:
        system_prompt = (
            "You are a professional translator. "
            f"Translate the user's text from {source_lang} into {target_lang}. "
            "Only output the translated text, without any explanation or extra words. If the text is already in the target language, just return a character '.'"
        )
    else:
        system_prompt = (
            "You are a professional translator. "
            f"Translate the user's text into {target_lang}. "
            "Automatically detect the source language. "
            "Only output the translated text, without any explanation or extra words. If the text is already in the target language, just return a character '.'"
        )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        stream=False,
    )

    translated = resp.choices[0].message.content.strip()
    if translated == ".":
        translated = text  # 原文已经是目标语言
    return translated



if __name__ == "__main__":
    
    q = "What is Beijing Normal University-Hong Kong Baptist University United International College (UIC)?"
    res_raw = web_search(q, require_ans=True, require_raw=False,max_results=3, search_depth="basic")
    print("=== Raw Tavily response keys ===")
    print(res_raw)
    
    print("\n=== Raw answer ===")
    print(res_raw.get("answer", "No answer field in response."))
    
    print("\n=== Extracted top 1 result ===")
    url, title, content, score = tavily_extract_top(res_raw, k=1)
    print(f"URL: {url}\nTitle: {title}\nContent: {content}\nScore: {score}")


    print("\n=== Extracted top 2 result ===")
    url, title, content, score = tavily_extract_top(res_raw, k=2)
    print(f"URL: {url}\nTitle: {title}\nContent: {content}\nScore: {score}")
    
    
    
    # res_refined = web_search(q, use_deepseek=True, max_results=3, search_depth="advanced")
    # print("\n=== Refined answer by DeepSeek ===")
    # print(res_refined)
