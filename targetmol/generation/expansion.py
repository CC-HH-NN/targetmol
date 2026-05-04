"""Expand a candidate molecule pool around a trusted anchor."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from targetmol.models import ModelsConfig
from targetmol.target_context.grounding import GroundedTargetContext


PLACEHOLDER_PREFIXES = ("YOUR_", "your_")


def read_seed_smiles_file(path: Path) -> list[dict[str, str]]:
    """Read a seed SMILES file into normalized records."""
    records: list[dict[str, str]] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path} line {index} is not a valid seed SMILES record.")
        smiles = parts[0].strip()
        name = parts[1].strip()
        if not smiles or not name:
            raise ValueError(f"{path} line {index} is not a valid seed SMILES record.")
        records.append({"name": name, "smiles": smiles, "source": "seed_file"})
    return records


def expand_candidate_pool(
    *,
    models: ModelsConfig,
    grounded_context: GroundedTargetContext | None,
    seed_smiles_file: Path | None,
    output_json_path: Path,
    output_smiles_path: Path,
    llm_runner=None,
) -> dict[str, object]:
    """Build a candidate pool around seed or anchor molecules and write run artifacts."""
    if seed_smiles_file is not None:
        candidates = _deduplicate_candidates(read_seed_smiles_file(seed_smiles_file))
        return _write_expansion_outputs(
            grounded_context=grounded_context,
            anchor_smiles=grounded_context.anchor_smiles if grounded_context is not None else None,
            candidates=candidates,
            degraded_reason=None,
            output_json_path=output_json_path,
            output_smiles_path=output_smiles_path,
        )

    anchor_smiles = grounded_context.anchor_smiles if grounded_context is not None else None
    if not anchor_smiles:
        return _write_expansion_outputs(
            grounded_context=grounded_context,
            anchor_smiles=None,
            candidates=[],
            degraded_reason="missing_anchor_smiles",
            output_json_path=output_json_path,
            output_smiles_path=output_smiles_path,
        )

    if _is_placeholder(models.chat_api_key) or _is_placeholder(models.chat_base_url):
        return _write_expansion_outputs(
            grounded_context=grounded_context,
            anchor_smiles=anchor_smiles,
            candidates=[],
            degraded_reason="llm_not_configured",
            output_json_path=output_json_path,
            output_smiles_path=output_smiles_path,
        )

    runner = llm_runner or _call_expansion_llm
    try:
        payload = runner(grounded_context=grounded_context, models=models)
        generated = _parse_generated_candidates(payload)
        candidates = _deduplicate_candidates(_anchor_reference_candidate(anchor_smiles) + generated)
        degraded_reason = None
    except Exception as exc:
        candidates = []
        degraded_reason = f"llm_failed: {exc}"

    return _write_expansion_outputs(
        grounded_context=grounded_context,
        anchor_smiles=anchor_smiles,
        candidates=candidates,
        degraded_reason=degraded_reason,
        output_json_path=output_json_path,
        output_smiles_path=output_smiles_path,
    )


def _parse_generated_candidates(payload: dict[str, object]) -> list[dict[str, str]]:
    """Parse the candidate list returned by the LLM."""
    items = payload.get("candidates")
    if not isinstance(items, list):
        raise RuntimeError("Candidate expansion LLM response is missing the candidates list.")
    records: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        smiles = _normalize_text(item.get("smiles"))
        if not smiles:
            continue
        rationale = _normalize_text(item.get("rationale")) or "llm_candidate"
        records.append(
            {
                "name": f"expanded_{index:04d}",
                "smiles": smiles,
                "source": "llm_expansion",
                "rationale": rationale,
            }
        )
    return records


def _anchor_reference_candidate(anchor_smiles: str) -> list[dict[str, str]]:
    """Add the anchor as a traceable reference candidate when LLM generation succeeds."""
    return [
        {
            "name": "anchor_0001",
            "smiles": anchor_smiles,
            "source": "anchor",
            "rationale": "anchor_reference",
        }
    ]


def _deduplicate_candidates(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate candidates by SMILES while preserving first occurrence."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for record in records:
        smiles = record["smiles"].strip()
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        unique.append({**record, "smiles": smiles})
    return unique


def _write_expansion_outputs(
    *,
    grounded_context: GroundedTargetContext | None,
    anchor_smiles: str | None,
    candidates: list[dict[str, str]],
    degraded_reason: str | None,
    output_json_path: Path,
    output_smiles_path: Path,
) -> dict[str, object]:
    """Write candidate expansion results to JSON and SMILES files."""
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
        "anchor_smiles": anchor_smiles,
        "degraded_reason": degraded_reason,
        "candidates": candidates,
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_smiles_path.open("w", encoding="utf-8") as handle:
        for item in candidates:
            handle.write(f"{item['smiles']}\t{item['name']}\n")
    return payload


def _call_expansion_llm(*, grounded_context: GroundedTargetContext, models: ModelsConfig) -> dict[str, object]:
    """Call an OpenAI-compatible chat endpoint to generate candidates near the anchor."""
    url = _build_openai_chat_url(models.chat_base_url)
    system_prompt = (
        "You are a medicinal chemistry candidate expansion assistant. "
        "Return strict JSON with key candidates, whose value is a list of objects with keys smiles and rationale. "
        "Generate a small pool of plausible analog ideas around the provided anchor molecule."
    )
    body = {
        "model": models.chat_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_name": grounded_context.target_name,
                        "disease": grounded_context.disease,
                        "known_drug": grounded_context.known_drug,
                        "anchor_smiles": grounded_context.anchor_smiles,
                        "uniprot_id": grounded_context.uniprot_id,
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
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"candidate expansion LLM call failed: {exc}") from exc
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("Candidate expansion LLM response is missing choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        content = "".join(text_parts).strip()
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Candidate expansion LLM response is missing content.")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Candidate expansion LLM did not return a JSON object.")
    return parsed


def _build_openai_chat_url(base_url: str) -> str:
    """Build a chat completions URL from an OpenAI-compatible base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _normalize_text(value: object) -> str | None:
    """Normalize an optional text value into a stable string."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_placeholder(value: str) -> bool:
    """Return whether a configuration value is still a placeholder."""
    stripped = value.strip()
    return not stripped or stripped.startswith(PLACEHOLDER_PREFIXES)
