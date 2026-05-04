"""clean-room screening 排序逻辑。"""

from __future__ import annotations


def _safe_float(value: object, default: float) -> float:
    """把可选数值安全转成 float。"""
    if value is None:
        return default
    return float(value)


def _sort_key(row: dict[str, object]) -> tuple[int, int, int, float, int, float, str]:
    """生成稳定排序键。"""
    is_valid_penalty = 0 if row.get("is_valid", True) else 1
    lipinski_penalty = 0 if row.get("lipinski_pass", False) else 1
    pains_status = str(row.get("pains_status", "unavailable"))
    pains_status_penalty = {
        "ok": 0,
        "degraded": 1,
        "unavailable": 2,
    }.get(pains_status, 2)
    docking_score = _safe_float(row.get("docking_score"), 999.0)
    pains_alert_count = int(row.get("pains_alert_count", 99))
    sa_score = _safe_float(row.get("sa_score"), 10.0)
    name = str(row.get("name", ""))
    return (
        is_valid_penalty,
        lipinski_penalty,
        pains_status_penalty,
        docking_score,
        pains_alert_count,
        sa_score,
        name,
    )


def rank_screening_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """按 docking、规则过滤和降级状态形成稳定排序。"""
    ranked = [dict(row) for row in sorted(rows, key=_sort_key)]
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked
