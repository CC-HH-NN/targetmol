"""clean-room screening 规则和性质评分。"""

from __future__ import annotations

from typing import Any

from targetmol.screening.types import ScreeningCandidate


def _load_rdkit_backend() -> dict[str, Any] | None:
    """按需加载 RDKit 相关模块。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski
    except ImportError:
        return None

    try:
        from rdkit.Chem import FilterCatalogParams
    except ImportError:
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalogParams
        except ImportError:
            return None

    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        sascorer = None

    return {
        "Chem": Chem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "FilterCatalog": FilterCatalog,
        "FilterCatalogParams": FilterCatalogParams,
        "Lipinski": Lipinski,
        "sascorer": sascorer,
    }


def _build_invalid_result(candidate: ScreeningCandidate, error: str) -> dict[str, object]:
    """生成稳定的失败结果。"""
    return {
        "name": candidate.name,
        "smiles": candidate.smiles,
        "is_valid": False,
        "mw": None,
        "logp": None,
        "hbd": None,
        "hba": None,
        "tpsa": None,
        "rotatable_bonds": None,
        "lipinski_violations": None,
        "lipinski_pass": False,
        "pains_alerts": [],
        "pains_alert_count": 0,
        "pains_status": "unavailable",
        "pains_error": None,
        "sa_score": None,
        "sa_score_source": None,
        "error": error,
    }


def _create_pains_catalog(backend: dict[str, Any]):
    """创建 PAINS 过滤目录，兼容不同 RDKit 接口。"""
    params_binding = backend["FilterCatalogParams"]
    catalog_module = backend["FilterCatalog"]
    params_class = getattr(params_binding, "FilterCatalogParams", params_binding)
    params = params_class()
    pains_enum = params_class.FilterCatalogs.PAINS
    if hasattr(params, "AddCatalog"):
        params.AddCatalog(pains_enum)
    elif hasattr(params, "AddCatalogs"):
        params.AddCatalogs(pains_enum)
    else:
        raise AttributeError("当前 RDKit 不支持 PAINS 目录初始化。")
    return catalog_module.FilterCatalog(params)


def _extract_pains_alerts(matches: list[object]) -> list[str]:
    """把 RDKit 告警对象转成稳定文本。"""
    alerts: list[str] = []
    for match in matches:
        if hasattr(match, "GetDescription"):
            alerts.append(str(match.GetDescription()))
            continue
        entry = getattr(match, "filterMatch", None)
        if entry is not None and hasattr(entry, "GetDescription"):
            alerts.append(str(entry.GetDescription()))
            continue
        alerts.append(str(match))
    return alerts


def _fallback_sa_like_score(*, mw: float, logp: float, rotatable_bonds: int, pains_alert_count: int) -> float:
    """在缺少 RDKit SA contrib 时给出第一阶段稳定近似分数。"""
    score = 1.0 + (mw / 250.0) + max(logp, 0.0) * 0.4 + rotatable_bonds * 0.35 + pains_alert_count * 0.5
    return max(1.0, min(10.0, score))


def _calculate_sa_like_score(
    backend: dict[str, Any],
    mol: object,
    *,
    mw: float,
    logp: float,
    rotatable_bonds: int,
    pains_alert_count: int,
) -> tuple[float, str]:
    """优先使用 RDKit SA Score，缺失时回退到稳定近似。"""
    sascorer = backend.get("sascorer")
    if sascorer is not None and hasattr(sascorer, "calculateScore"):
        return float(sascorer.calculateScore(mol)), "rdkit_sascorer"
    return (
        _fallback_sa_like_score(
        mw=mw,
        logp=logp,
        rotatable_bonds=rotatable_bonds,
        pains_alert_count=pains_alert_count,
        ),
        "fallback_approximation",
    )


def evaluate_candidate_properties(candidate: ScreeningCandidate) -> dict[str, object]:
    """计算第一阶段需要的基础性质和规则标记。"""
    backend = _load_rdkit_backend()
    if backend is None:
        return _build_invalid_result(candidate, "RDKit 不可用，无法计算性质。")

    mol = backend["Chem"].MolFromSmiles(candidate.smiles)
    if mol is None:
        return _build_invalid_result(candidate, "不是有效的 SMILES。")

    mw = float(backend["Descriptors"].MolWt(mol))
    logp = float(backend["Crippen"].MolLogP(mol))
    hbd = int(backend["Lipinski"].NumHDonors(mol))
    hba = int(backend["Lipinski"].NumHAcceptors(mol))
    tpsa = float(backend["Descriptors"].TPSA(mol))
    rotatable_bonds = int(backend["Lipinski"].NumRotatableBonds(mol))
    try:
        catalog = _create_pains_catalog(backend)
        alerts = _extract_pains_alerts(list(catalog.GetMatches(mol)))
        pains_status = "ok"
        pains_error = None
    except Exception as exc:
        alerts = []
        pains_status = "degraded"
        pains_error = f"PAINS 目录初始化失败：{exc}"

    lipinski_violations = sum(
        [
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ]
    )
    sa_score, sa_score_source = _calculate_sa_like_score(
        backend,
        mol,
        mw=mw,
        logp=logp,
        rotatable_bonds=rotatable_bonds,
        pains_alert_count=len(alerts),
    )

    return {
        "name": candidate.name,
        "smiles": candidate.smiles,
        "is_valid": True,
        "mw": round(mw, 4),
        "logp": round(logp, 4),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 4),
        "rotatable_bonds": rotatable_bonds,
        "lipinski_violations": lipinski_violations,
        "lipinski_pass": lipinski_violations == 0,
        "pains_alerts": alerts,
        "pains_alert_count": len(alerts),
        "pains_status": pains_status,
        "pains_error": pains_error,
        "sa_score": round(float(sa_score), 4),
        "sa_score_source": sa_score_source,
        "error": None,
    }
