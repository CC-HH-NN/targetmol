"""把自然语言请求和显式靶点信息落成可追溯的锚点上下文。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from urllib import error, parse, request

from targetmol.inputs import InputSpec
from targetmol.models import ModelsConfig
from targetmol.target_context.embedding import rank_hits_by_embedding


PLACEHOLDER_PREFIXES = ("YOUR_", "your_")


@dataclass
class GroundedTargetContext:
    """靶点锚点落地结果。"""

    target_name: str | None = None
    disease: str | None = None
    known_drug: str | None = None
    anchor_smiles: str | None = None
    uniprot_id: str | None = None
    rationale: str | None = None
    search_queries: list[str] = field(default_factory=list)
    search_hits: list[dict[str, str]] = field(default_factory=list)
    degraded_reason: str | None = None
    evidence_ranker: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "GroundedTargetContext":
        """把结构化返回值规范化成上下文对象。"""
        return cls(
            target_name=_normalize_text(payload.get("target_name")),
            disease=_normalize_text(payload.get("disease")),
            known_drug=_normalize_text(payload.get("known_drug")),
            anchor_smiles=_normalize_text(payload.get("anchor_smiles")),
            uniprot_id=_normalize_text(payload.get("uniprot_id")),
            rationale=_normalize_text(payload.get("rationale")),
        )


def ground_input_spec_with_context(
    spec: InputSpec,
    models: ModelsConfig,
    *,
    serper_api_key: str,
    output_path: Path | None = None,
    llm_runner=None,
    search_runner=None,
    smiles_runner=None,
    embedding_runner=None,
) -> InputSpec:
    """把请求和显式字段落成锚点上下文，并只补齐缺失字段。"""
    enriched_spec, _ = ground_input_spec_with_context_data(
        spec,
        models,
        serper_api_key=serper_api_key,
        output_path=output_path,
        llm_runner=llm_runner,
        search_runner=search_runner,
        smiles_runner=smiles_runner,
        embedding_runner=embedding_runner,
    )
    return enriched_spec


def ground_input_spec_with_context_data(
    spec: InputSpec,
    models: ModelsConfig,
    *,
    serper_api_key: str,
    output_path: Path | None = None,
    llm_runner=None,
    search_runner=None,
    smiles_runner=None,
    embedding_runner=None,
) -> tuple[InputSpec, GroundedTargetContext | None]:
    """把请求和显式字段落成锚点上下文，并返回补齐后的输入与上下文。"""
    if not spec.request_text and not spec.target_name and not spec.disease:
        return spec, None

    context = ground_target_context(
        request_text=spec.request_text,
        target_name=spec.target_name,
        disease=spec.disease,
        models=models,
        serper_api_key=serper_api_key,
        llm_runner=llm_runner,
        search_runner=search_runner,
        smiles_runner=smiles_runner,
        embedding_runner=embedding_runner,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "request_text": spec.request_text,
                    "input_target_name": spec.target_name,
                    "input_disease": spec.disease,
                    "grounded": asdict(context),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    enriched_spec = replace(
        spec,
        target_name=spec.target_name or context.target_name,
        disease=spec.disease or context.disease,
    )
    return enriched_spec, context


def ground_target_context(
    *,
    request_text: str | None,
    target_name: str | None,
    disease: str | None,
    models: ModelsConfig,
    serper_api_key: str,
    llm_runner=None,
    search_runner=None,
    smiles_runner=None,
    embedding_runner=None,
) -> GroundedTargetContext:
    """优先用搜索、embedding 排序和 LLM 整理锚点上下文，失败时稳定回退。"""
    context = GroundedTargetContext(
        target_name=target_name,
        disease=disease,
    )
    queries = _build_search_queries(
        request_text=request_text,
        target_name=target_name,
        disease=disease,
    )
    context.search_queries = queries

    hits: list[dict[str, str]] = []
    if queries and serper_api_key and not _is_placeholder(serper_api_key):
        runner = search_runner or _search_with_serper
        try:
            for query in queries:
                hits.extend(runner(query=query, api_key=serper_api_key))
        except Exception as exc:
            context.degraded_reason = f"search_failed: {exc}"
    context.search_hits = _rank_search_hits(
        request_text=request_text,
        target_name=target_name,
        disease=disease,
        hits=hits,
        models=models,
        embedding_runner=embedding_runner,
        context=context,
    )[:6]

    if _is_placeholder(models.chat_api_key) or _is_placeholder(models.chat_base_url):
        if context.degraded_reason is None:
            context.degraded_reason = "llm_not_configured"
        return context

    runner = llm_runner or _call_grounding_llm
    try:
        payload = runner(
            request_text=request_text,
            target_name=target_name,
            disease=disease,
            search_hits=context.search_hits,
            models=models,
        )
        grounded = GroundedTargetContext.from_mapping(payload)
    except Exception as exc:
        if context.degraded_reason is None:
            context.degraded_reason = f"llm_failed: {exc}"
        return context

    grounded.search_queries = context.search_queries
    grounded.search_hits = context.search_hits
    grounded.degraded_reason = context.degraded_reason
    grounded.evidence_ranker = context.evidence_ranker
    grounded.target_name = grounded.target_name or target_name
    grounded.disease = grounded.disease or disease
    if not grounded.anchor_smiles and grounded.known_drug:
        resolved_smiles = _resolve_anchor_smiles_from_known_drug(
            grounded.known_drug,
            resolver=smiles_runner or _resolve_smiles_from_compound_name,
        )
        if resolved_smiles:
            grounded.anchor_smiles = resolved_smiles
    return grounded


def _rank_search_hits(
    *,
    request_text: str | None,
    target_name: str | None,
    disease: str | None,
    hits: list[dict[str, str]],
    models: ModelsConfig,
    embedding_runner,
    context: GroundedTargetContext,
) -> list[dict[str, str]]:
    """用 embedding 对搜索证据排序，失败时保留原顺序。"""
    if not hits:
        return []
    if _is_placeholder(models.embedding_api_key) or _is_placeholder(models.embedding_base_url):
        context.evidence_ranker = "serper_order"
        return hits
    try:
        ranked = rank_hits_by_embedding(
            request_text=request_text,
            target_name=target_name,
            disease=disease,
            hits=hits,
            models=models,
            embedding_runner=embedding_runner,
        )
    except Exception as exc:
        context.evidence_ranker = "serper_order"
        if context.degraded_reason is None:
            context.degraded_reason = f"embedding_failed: {exc}"
        return hits
    context.evidence_ranker = "embedding_similarity"
    return ranked


def _resolve_anchor_smiles_from_known_drug(
    known_drug: str,
    *,
    resolver,
) -> str | None:
    """把已知药名列表解析成第一个可用的锚点 SMILES。"""
    for drug_name in _split_known_drug_names(known_drug):
        try:
            resolved = resolver(drug_name)
        except Exception:
            continue
        normalized = _normalize_text(resolved)
        if normalized:
            return normalized
    return None


def _split_known_drug_names(value: str) -> list[str]:
    """把逗号或分号分隔的药名列表拆成稳定顺序。"""
    normalized = value.replace(";", ",").replace("，", ",")
    names: list[str] = []
    seen: set[str] = set()
    for chunk in normalized.split(","):
        name = chunk.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _build_search_queries(
    *,
    request_text: str | None,
    target_name: str | None,
    disease: str | None,
) -> list[str]:
    """基于目标和疾病构建最小搜索查询集。"""
    queries: list[str] = []
    if target_name and disease:
        queries.append(f"{target_name} {disease} known inhibitor")
        queries.append(f"{target_name} {disease} approved drug")
    elif target_name:
        queries.append(f"{target_name} known inhibitor")
        queries.append(f"{target_name} approved drug")
    elif request_text:
        queries.append(request_text.strip())
    seen = set()
    unique_queries = []
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_queries.append(normalized)
    return unique_queries[:3]


def _search_with_serper(*, query: str, api_key: str) -> list[dict[str, str]]:
    """调用 Serper 搜索接口，提取最小证据片段。"""
    req = request.Request(
        "https://google.serper.dev/search",
        data=json.dumps({"q": query}).encode("utf-8"),
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Serper 调用失败: {exc}") from exc

    hits = []
    for item in payload.get("organic", [])[:3]:
        title = _normalize_text(item.get("title")) or ""
        snippet = _normalize_text(item.get("snippet")) or ""
        link = _normalize_text(item.get("link")) or ""
        if title or snippet:
            hits.append({"title": title, "snippet": snippet, "link": link})
    return hits


def _resolve_smiles_from_compound_name(drug_name: str) -> str | None:
    """优先从 PubChem，再从 ChEMBL 解析药名对应的 canonical smiles。"""
    for resolver in (_resolve_smiles_with_pubchem, _resolve_smiles_with_chembl):
        try:
            resolved = resolver(drug_name)
        except Exception:
            continue
        normalized = _normalize_text(resolved)
        if normalized:
            return normalized
    return None


def _resolve_smiles_with_pubchem(drug_name: str) -> str | None:
    """调用 PubChem PUG REST 解析 canonical smiles。"""
    encoded_name = parse.quote(drug_name)
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded_name}/property/CanonicalSMILES/JSON"
    )
    payload = _read_json(url)
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not isinstance(properties, list):
        return None
    for item in properties:
        if isinstance(item, dict):
            smiles = _normalize_text(item.get("CanonicalSMILES"))
            if smiles:
                return smiles
    return None


def _resolve_smiles_with_chembl(drug_name: str) -> str | None:
    """调用 ChEMBL 搜索接口解析 canonical smiles。"""
    encoded_name = parse.quote(drug_name)
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={encoded_name}&format=json"
    payload = _read_json(url)
    molecules = payload.get("molecules", [])
    if not isinstance(molecules, list):
        return None
    for item in molecules:
        if not isinstance(item, dict):
            continue
        structures = item.get("molecule_structures")
        if not isinstance(structures, dict):
            continue
        smiles = _normalize_text(structures.get("canonical_smiles"))
        if smiles:
            return smiles
    return None


def _read_json(url: str) -> dict[str, object]:
    """读取公开 JSON 接口并返回字典结果。"""
    req = request.Request(url, headers={"User-Agent": "TargetMol/1.0"})
    try:
        with request.urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("接口没有返回 JSON object。")
    return payload


def _call_grounding_llm(
    *,
    request_text: str | None,
    target_name: str | None,
    disease: str | None,
    search_hits: list[dict[str, str]],
    models: ModelsConfig,
) -> dict[str, object]:
    """调用 OpenAI 兼容聊天接口做锚点落地。"""
    url = _build_openai_chat_url(models.chat_base_url)
    system_prompt = (
        "You are a medicinal chemistry task grounding assistant. "
        "Return strict JSON with keys: target_name, disease, known_drug, anchor_smiles, uniprot_id, rationale. "
        "Use null when unknown. Prefer concise factual grounding over speculation."
    )
    user_payload = {
        "request_text": request_text,
        "explicit_target_name": target_name,
        "explicit_disease": disease,
        "search_hits": search_hits,
    }
    body = {
        "model": models.chat_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
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
        raise RuntimeError(f"grounding LLM 调用失败: {exc}") from exc

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("grounding LLM 返回缺少 choices。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        content = "".join(text_parts).strip()
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("grounding LLM 返回缺少 content。")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("grounding LLM 未返回 JSON object。")
    return parsed


def _build_openai_chat_url(base_url: str) -> str:
    """从 OpenAI 兼容 base url 拼出 chat completions 地址。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


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
