"""Internal screening report output."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_FIELDS = [
    "rank",
    "name",
    "smiles",
    "source",
    "docking_score",
    "lipinski_pass",
    "lipinski_violations",
    "pains_status",
    "pains_alert_count",
    "sa_score",
    "sa_score_source",
    "is_valid",
    "error",
]


def write_final_reports(
    output_dir: Path,
    ranked_rows: list[dict[str, object]],
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, str]:
    """Write final_report files for the TargetMol workflow."""
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    json_path = final_dir / "final_report.json"
    csv_path = final_dir / "final_report.csv"
    smiles_path = final_dir / "ranked_candidates.smi"

    payload_metadata = metadata or {}
    json_payload = {
        "target": payload_metadata.get("target"),
        "total_input": payload_metadata.get("total_input"),
        "valid_ligands": payload_metadata.get("valid_ligands"),
        "docking_success": payload_metadata.get("docking_success"),
        "passed_filters": payload_metadata.get("passed_filters"),
        "execution_time_seconds": payload_metadata.get("execution_time_seconds"),
        "created_at": payload_metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "count": len(ranked_rows),
        "top_candidates": ranked_rows,
    }
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = list(DEFAULT_FIELDS)
    for row in ranked_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(row)

    with smiles_path.open("w", encoding="utf-8") as handle:
        for row in ranked_rows:
            handle.write(f"{row['smiles']}\t{row['name']}\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "smiles": str(smiles_path),
    }
