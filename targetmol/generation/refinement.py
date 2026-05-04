"""围绕候选扩增结果做 refinement、排序与 shortlist 收口。"""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from targetmol.models import ModelsConfig
from targetmol.screening.properties import evaluate_candidate_properties
from targetmol.screening.types import ScreeningCandidate
from targetmol.target_context.grounding import GroundedTargetContext


PLACEHOLDER_PREFIXES = ("YOUR_", "your_")


def refine_candidate_pool(
    *,
    models: ModelsConfig,
    grounded_context: GroundedTargetContext | None,
    input_json_path: Path,
    output_json_path: Path,
    output_smiles_path: Path,
    shortlist_size: int = 12,
    llm_runner=None,
    expansion_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """读取候选扩增结果，做一轮轻量 refinement 并写出 shortlist。"""
    payload = expansion_payload
    if payload is None:
        payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"{input_json_path} 缺少有效的 candidates 列表。")

    evaluated = _evaluate_candidates(
        raw_candidates=raw_candidates,
        anchor_smiles=grounded_context.anchor_smiles if grounded_context is not None else None,
    )
    heuristic_ranked = sorted(evaluated, key=_refinement_sort_key)
    shortlist = [dict(item) for item in heuristic_ranked[:shortlist_size]]
    selection_strategy = "heuristic_only"
    degraded_reason = None

    if shortlist and _llm_available(models):
        runner = llm_runner or _call_refinement_llm
        try:
            selection = runner(
                grounded_context=grounded_context,
                ranked_candidates=shortlist,
                models=models,
            )
            shortlist = _apply_llm_selection(shortlist, selection)
            selection_strategy = "llm_plus_heuristic"
        except Exception as exc:
            degraded_reason = f"llm_failed: {exc}"

    return _write_refinement_outputs(
        grounded_context=grounded_context,
        evaluated_candidates=heuristic_ranked,
        shortlist=shortlist,
        selection_strategy=selection_strategy,
        degraded_reason=degraded_reason,
        output_json_path=output_json_path,
        output_smiles_path=output_smiles_path,
    )


def compute_anchor_similarity(smiles: str, anchor_smiles: str | None) -> tuple[float | None, bool]:
    """计算候选与锚点的相似度，并标记是否与锚点完全一致。"""
    if not anchor_smiles:
        return None, False
    if smiles.strip() == anchor_smiles.strip():
        return 1.0, True
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return None, False

    mol = Chem.MolFromSmiles(smiles)
    anchor_mol = Chem.MolFromSmiles(anchor_smiles)
    if mol is None or anchor_mol is None:
        return None, False

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    similarity = DataStructs.TanimotoSimilarity(
        generator.GetFingerprint(mol),
        generator.GetFingerprint(anchor_mol),
    )
    return round(float(similarity), 4), False


def _evaluate_candidates(
    *,
    raw_candidates: list[object],
    anchor_smiles: str | None,
) -> list[dict[str, object]]:
    """把扩增候选补齐性质与相似度信息。"""
    evaluated: list[dict[str, object]] = []
    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue
        smiles = _normalize_text(item.get("smiles"))
        name = _normalize_text(item.get("name")) or f"candidate_{index:04d}"
        if not smiles:
            continue
        candidate = ScreeningCandidate(
            name=name,
            smiles=smiles,
            source=_normalize_text(item.get("source")) or "candidate_expansion",
        )
        properties = evaluate_candidate_properties(candidate)
        similarity, is_anchor_identity = compute_anchor_similarity(smiles, anchor_smiles)
        rationale = _normalize_text(item.get("rationale"))
        record = {
            "name": name,
            "smiles": smiles,
            "source": candidate.source,
            "expansion_rationale": rationale,
            "anchor_similarity": similarity,
            "is_anchor_identity": is_anchor_identity,
            **properties,
        }
        record["refinement_score"] = round(_calculate_refinement_score(record), 4)
        evaluated.append(record)
    return evaluated


def _calculate_refinement_score(row: dict[str, object]) -> float:
    """把规则过滤、可合成性和锚点相似度汇成稳定分数。"""
    score = 0.0
    if row.get("is_valid", False):
        score += 5.0
    if row.get("lipinski_pass", False):
        score += 3.0
    score -= float(row.get("pains_alert_count", 0) or 0) * 1.5
    sa_score = row.get("sa_score")
    if sa_score is not None:
        score -= float(sa_score) * 0.4
    similarity = row.get("anchor_similarity")
    if similarity is not None:
        score += float(similarity) * 2.5
    if row.get("is_anchor_identity", False):
        score -= 0.75
    return score


def _refinement_sort_key(row: dict[str, object]) -> tuple[int, int, int, float, int, float, str]:
    """生成候选 refinement 的稳定排序键。"""
    return (
        0 if row.get("is_valid", False) else 1,
        0 if row.get("lipinski_pass", False) else 1,
        1 if row.get("is_anchor_identity", False) else 0,
        -float(row.get("refinement_score", -999.0)),
        int(row.get("pains_alert_count", 99) or 99),
        float(row.get("sa_score", 10.0) or 10.0),
        str(row.get("name", "")),
    )


def _apply_llm_selection(
    ranked_candidates: list[dict[str, object]],
    selection: dict[str, object],
) -> list[dict[str, object]]:
    """在启发式 shortlist 上应用 LLM 的显式选择。"""
    selected_names = selection.get("selected_names")
    rationales = selection.get("rationales")
    if not isinstance(selected_names, list):
        return ranked_candidates
    rationale_map = rationales if isinstance(rationales, dict) else {}
    candidate_map = {str(item["name"]): dict(item) for item in ranked_candidates}

    reordered: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in selected_names:
        name = str(value).strip()
        if not name or name not in candidate_map or name in seen:
            continue
        item = dict(candidate_map[name])
        note = _normalize_text(rationale_map.get(name))
        if note:
            item["selection_note"] = note
        reordered.append(item)
        seen.add(name)

    for item in ranked_candidates:
        name = str(item["name"])
        if name in seen:
            continue
        note = _normalize_text(rationale_map.get(name))
        copied = dict(item)
        if note:
            copied["selection_note"] = note
        reordered.append(copied)
    return reordered


def _write_refinement_outputs(
    *,
    grounded_context: GroundedTargetContext | None,
    evaluated_candidates: list[dict[str, object]],
    shortlist: list[dict[str, object]],
    selection_strategy: str,
    degraded_reason: str | None,
    output_json_path: Path,
    output_smiles_path: Path,
) -> dict[str, object]:
    """把 refinement 结果同时落成 JSON 和标准 smiles 文件。"""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_smiles_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grounded_context": {
            "target_name": grounded_context.target_name if grounded_context is not None else None,
            "disease": grounded_context.disease if grounded_context is not None else None,
            "known_drug": grounded_context.known_drug if grounded_context is not None else None,
            "anchor_smiles": grounded_context.anchor_smiles if grounded_context is not None else None,
            "uniprot_id": grounded_context.uniprot_id if grounded_context is not None else None,
            "rationale": grounded_context.rationale if grounded_context is not None else None,
            "degraded_reason": grounded_context.degraded_reason if grounded_context is not None else None,
        },
        "selection_strategy": selection_strategy,
        "degraded_reason": degraded_reason,
        "evaluated_candidates": evaluated_candidates,
        "shortlist": shortlist,
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_smiles_path.open("w", encoding="utf-8") as handle:
        for item in shortlist:
            handle.write(f"{item['smiles']}\t{item['name']}\n")
    return payload


def _call_refinement_llm(
    *,
    grounded_context: GroundedTargetContext | None,
    ranked_candidates: list[dict[str, object]],
    models: ModelsConfig,
) -> dict[str, object]:
    """调用 OpenAI 兼容聊天接口，从启发式候选里再选一层 shortlist。"""
    url = _build_openai_chat_url(models.chat_base_url)
    body = {
        "model": models.chat_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medicinal chemistry shortlist assistant. "
                    "Return strict JSON with keys selected_names and rationales. "
                    "selected_names must be a list of candidate names from the provided list. "
                    "Prefer valid, developable, anchor-related but not purely duplicated ideas."
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
                        "candidates": ranked_candidates,
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
        raise RuntimeError(f"candidate refinement LLM 调用失败: {exc}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("candidate refinement LLM 返回缺少 choices。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        content = "".join(text_parts).strip()
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("candidate refinement LLM 返回缺少 content。")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("candidate refinement LLM 未返回 JSON object。")
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
