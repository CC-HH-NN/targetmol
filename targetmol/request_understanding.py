"""自然语言请求理解与结构化补全。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from urllib import error, request

from targetmol.inputs import InputSpec
from targetmol.models import ModelsConfig
from targetmol.request_parser import extract_pdb_id


PLACEHOLDER_PREFIXES = ("YOUR_", "your_")
ALLOWED_TASK_TYPES = {"screen_only", "structure_based", "ligand_based", "hybrid"}


@dataclass
class RequestUnderstanding:
    """请求理解得到的结构化结果。"""

    target_name: str | None = None
    disease: str | None = None
    pdb_id: str | None = None
    task_type: str | None = None
    known_drug: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "RequestUnderstanding":
        """从 LLM 或其他结构化输入构造理解结果。"""
        pdb_id = _normalize_text(payload.get("pdb_id"))
        task_type = _normalize_text(payload.get("task_type"))
        if pdb_id:
            pdb_id = pdb_id.upper()
        if task_type and task_type not in ALLOWED_TASK_TYPES:
            task_type = None
        return cls(
            target_name=_normalize_text(payload.get("target_name")),
            disease=_normalize_text(payload.get("disease")),
            pdb_id=pdb_id,
            task_type=task_type,
            known_drug=_normalize_text(payload.get("known_drug")),
        )


def understand_request(
    *,
    request_text: str,
    models: ModelsConfig,
    llm_runner=None,
) -> RequestUnderstanding:
    """优先用 LLM 理解请求，失败时稳定回退到启发式规则。"""
    fallback = RequestUnderstanding(pdb_id=extract_pdb_id(request_text))
    if _is_placeholder(models.chat_api_key) or _is_placeholder(models.chat_base_url):
        return fallback

    runner = llm_runner or _call_request_understanding_llm
    try:
        payload = runner(request_text=request_text, models=models)
    except Exception:
        return fallback

    understanding = RequestUnderstanding.from_mapping(payload)
    if understanding.pdb_id is None:
        understanding.pdb_id = fallback.pdb_id
    return understanding


def enrich_input_spec_from_request(
    spec: InputSpec,
    models: ModelsConfig,
    *,
    llm_runner=None,
) -> InputSpec:
    """用请求理解结果补齐缺失字段，但不覆盖显式输入。"""
    if not spec.request_text:
        return spec
    understanding = understand_request(
        request_text=spec.request_text,
        models=models,
        llm_runner=llm_runner,
    )
    return replace(
        spec,
        pdb_id=spec.pdb_id or understanding.pdb_id,
        target_name=spec.target_name or understanding.target_name,
        disease=spec.disease or understanding.disease,
    )


def _normalize_text(value: object) -> str | None:
    """把可选文本值规范化为稳定字符串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_placeholder(value: str) -> bool:
    """判断配置值是否仍是占位内容。"""
    stripped = value.strip()
    return not stripped or stripped.startswith(PLACEHOLDER_PREFIXES)


def _build_openai_chat_url(base_url: str) -> str:
    """从 OpenAI 兼容 base url 拼出 chat completions 地址。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _call_request_understanding_llm(*, request_text: str, models: ModelsConfig) -> dict[str, object]:
    """调用 OpenAI 兼容聊天接口，返回结构化请求理解结果。"""
    url = _build_openai_chat_url(models.chat_base_url)
    prompt = (
        "You are a drug-discovery task parser. "
        "Return strict JSON with keys: target_name, disease, pdb_id, task_type, known_drug. "
        "Use null when unknown. "
        "Allowed task_type values: screen_only, structure_based, ligand_based, hybrid."
    )
    payload = {
        "model": models.chat_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": request_text},
        ],
    }
    raw = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=raw,
        headers={
            "Authorization": f"Bearer {models.chat_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"request understanding LLM 调用失败: {exc}") from exc

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("request understanding LLM 返回缺少 choices。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError("request understanding LLM 返回缺少 content。")
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        content = "".join(text_parts).strip()
    if not isinstance(content, str):
        raise RuntimeError("request understanding LLM content 不是字符串。")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("request understanding LLM 未返回 JSON object。")
    return parsed
