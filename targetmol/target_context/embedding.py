"""为靶点 grounding 提供 embedding 证据排序能力。"""

from __future__ import annotations

import json
import math
from urllib import error, request

from targetmol.models import ModelsConfig


def rank_hits_by_embedding(
    *,
    request_text: str | None,
    target_name: str | None,
    disease: str | None,
    hits: list[dict[str, str]],
    models: ModelsConfig,
    embedding_runner=None,
) -> list[dict[str, str]]:
    """调用 embedding 模型并按语义相似度重排搜索证据。"""
    if not hits:
        return []
    query = _build_embedding_query(
        request_text=request_text,
        target_name=target_name,
        disease=disease,
    )
    texts = [_hit_to_text(hit) for hit in hits]
    runner = embedding_runner or call_embedding_model
    vectors = runner(texts=[query, *texts], models=models)
    return _rank_hits_by_vectors(hits=hits, query_vector=vectors[0], hit_vectors=vectors[1:])


def call_embedding_model(*, texts: list[str], models: ModelsConfig) -> list[list[float]]:
    """调用 OpenAI 兼容 embeddings 端点。"""
    url = _build_openai_embedding_url(models.embedding_base_url)
    req = request.Request(
        url,
        data=json.dumps({"input": texts, "model": models.embedding_model}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {models.embedding_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"embedding 调用失败: {exc}") from exc
    return _parse_embedding_payload(payload, expected_count=len(texts))


def _parse_embedding_payload(payload: dict[str, object], *, expected_count: int) -> list[list[float]]:
    """解析 embedding API 返回的向量列表。"""
    items = payload.get("data")
    if not isinstance(items, list) or len(items) != expected_count:
        raise RuntimeError("embedding 返回数量和输入文本数量不一致。")
    vectors: list[list[float]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise RuntimeError("embedding 返回格式无效。")
        vectors.append([float(value) for value in item["embedding"]])
    return vectors


def _build_embedding_query(*, request_text: str | None, target_name: str | None, disease: str | None) -> str:
    """构建证据排序查询文本。"""
    parts = [request_text, target_name, disease, "known inhibitor approved drug anchor molecule"]
    return " ".join(part.strip() for part in parts if part and part.strip())


def _hit_to_text(hit: dict[str, str]) -> str:
    """把搜索结果压成 embedding 输入文本。"""
    return " ".join(filter(None, [hit.get("title", ""), hit.get("snippet", ""), hit.get("link", "")])).strip()


def _rank_hits_by_vectors(
    *,
    hits: list[dict[str, str]],
    query_vector: list[float],
    hit_vectors: list[list[float]],
) -> list[dict[str, str]]:
    """按 query 与 hit 向量余弦相似度降序排序。"""
    scored = []
    for index, hit in enumerate(hits):
        score = _cosine_similarity(query_vector, hit_vectors[index])
        scored.append((score, index, hit))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [hit for _, _, hit in scored]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _build_openai_embedding_url(base_url: str) -> str:
    """从 OpenAI 兼容 base url 拼出 embeddings 地址。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/embeddings"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/embeddings"
    return f"{normalized}/v1/embeddings"
