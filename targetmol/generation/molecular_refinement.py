"""围绕 shortlist 做单轮分子弱点修正，并保留 before/after 对照。"""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from targetmol.models import ModelsConfig
from targetmol.screening.properties import evaluate_candidate_properties
from targetmol.screening.types import ScreeningCandidate
from targetmol.target_context.grounding import GroundedTargetContext


PLACEHOLDER_PREFIXES = ("YOUR_", "your_")


def run_molecular_refinement(
    *,
    models: ModelsConfig,
    grounded_context: GroundedTargetContext | None,
    input_json_path: Path,
    output_json_path: Path,
    output_smiles_path: Path,
    llm_runner=None,
    max_candidates: int = 5,
) -> dict[str, object]:
    """读取 shortlist，做一轮小步分子修正并写出 before/after 记录。"""
    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    shortlist = payload.get("shortlist")
    if not isinstance(shortlist, list):
        raise ValueError(f"{input_json_path} 缺少有效的 shortlist 列表。")

    records: list[dict[str, object]] = []
    for item in shortlist[:max_candidates]:
        if not isinstance(item, dict):
            continue
        before = dict(item)
        dominant_issue = derive_dominant_issue(before)
        accepted = False
        fallback_reason = None
        after = None

        if _llm_available(models):
            runner = llm_runner or _call_molecular_refinement_llm
            try:
                update = runner(
                    grounded_context=grounded_context,
                    candidate=before,
                    dominant_issue=dominant_issue,
                    models=models,
                )
                updated_smiles = _normalize_text(update.get("updated_smiles"))
                rationale = _normalize_text(update.get("rationale"))
                if updated_smiles:
                    candidate_name = f"{before['name']}_r1"
                    after_properties = evaluate_candidate_properties(
                        ScreeningCandidate(name=candidate_name, smiles=updated_smiles)
                    )
                    if after_properties.get("is_valid", False):
                        after = {
                            "name": candidate_name,
                            "smiles": updated_smiles,
                            "rationale": rationale,
                            **after_properties,
                        }
                        after["name"] = candidate_name
                        accepted = True
                    else:
                        fallback_reason = "invalid_update"
                else:
                    fallback_reason = "empty_update"
            except Exception as exc:
                fallback_reason = f"llm_failed: {exc}"
        else:
            fallback_reason = "llm_not_configured"

        records.append(
            {
                "before": before,
                "after": after,
                "dominant_issue": dominant_issue,
                "accepted": accepted,
                "fallback_reason": fallback_reason,
            }
        )

    return _write_outputs(
        grounded_context=grounded_context,
        records=records,
        output_json_path=output_json_path,
        output_smiles_path=output_smiles_path,
    )


def derive_dominant_issue(candidate: dict[str, object]) -> str:
    """从 shortlist 候选里提炼当前最需要修正的一个弱点。"""
    if not candidate.get("is_valid", False):
        return "validity"
    if not candidate.get("lipinski_pass", False):
        return "lipinski"
    if int(candidate.get("pains_alert_count", 0) or 0) > 0:
        return "pains"
    sa_score = candidate.get("sa_score")
    if sa_score is not None and float(sa_score) >= 4.0:
        return "sa_score"
    return "minor_optimization"


def _write_outputs(
    *,
    grounded_context: GroundedTargetContext | None,
    records: list[dict[str, object]],
    output_json_path: Path,
    output_smiles_path: Path,
) -> dict[str, object]:
    """把 before/after refinement 结果写入运行目录。"""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_smiles_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grounded_context": {
            "target_name": grounded_context.target_name if grounded_context is not None else None,
            "disease": grounded_context.disease if grounded_context is not None else None,
            "known_drug": grounded_context.known_drug if grounded_context is not None else None,
            "anchor_smiles": grounded_context.anchor_smiles if grounded_context is not None else None,
        },
        "records": records,
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_smiles_path.open("w", encoding="utf-8") as handle:
        for record in records:
            after = record["after"]
            if not isinstance(after, dict):
                continue
            handle.write(f"{after['smiles']}\t{after['name']}\n")
    return payload


def _call_molecular_refinement_llm(
    *,
    grounded_context: GroundedTargetContext | None,
    candidate: dict[str, object],
    dominant_issue: str,
    models: ModelsConfig,
) -> dict[str, object]:
    """调用 OpenAI 兼容聊天接口，做单轮小步分子修正。"""
    url = _build_openai_chat_url(models.chat_base_url)
    body = {
        "model": models.chat_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medicinal chemistry molecular refinement assistant. "
                    "Return strict JSON with keys updated_smiles and rationale. "
                    "Make one small change that addresses the dominant issue while keeping the molecule close to the original. "
                    "If unsure, return the original smiles."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "grounded_context": {
                            "target_name": grounded_context.target_name if grounded_context is not None else None,
                            "disease": grounded_context.disease if grounded_context is not None else None,
                            "known_drug": grounded_context.known_drug if grounded_context is not None else None,
                            "anchor_smiles": grounded_context.anchor_smiles if grounded_context is not None else None,
                        },
                        "candidate": candidate,
                        "dominant_issue": dominant_issue,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {models.chat_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"molecular refinement LLM 调用失败: {exc}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("molecular refinement LLM 返回缺少 choices。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        content = "".join(text_parts).strip()
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("molecular refinement LLM 返回缺少 content。")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("molecular refinement LLM 未返回 JSON object。")
    return parsed


def _llm_available(models: ModelsConfig) -> bool:
    """判断当前是否可以调用聊天模型。"""
    return not _is_placeholder(models.chat_api_key) and not _is_placeholder(models.chat_base_url)


def _build_openai_chat_url(base_url: str) -> str:
    """从 OpenAI 兼容 base url 拼出 chat completions 地址。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _is_placeholder(value: str) -> bool:
    """判断配置值是否仍是占位内容。"""
    stripped = value.strip()
    return not stripped or stripped.startswith(PLACEHOLDER_PREFIXES)


def _normalize_text(value: object) -> str | None:
    """把可选文本值规范化为稳定字符串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
