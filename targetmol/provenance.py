"""Record commands, timestamps, and paths for reproducibility."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


RUN_METADATA_FILENAME = "run_metadata.json"


@dataclass
class CommandRecord:
    """Single external command record."""

    step: str
    command: list[str]
    cwd: str
    created_at: str


class ProvenanceRecorder:
    """Write command records into the provenance directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.command_log = self.directory / "commands.jsonl"

    def record_command(self, *, step: str, command: list[str], cwd: Path) -> None:
        """Append one command record."""
        record = CommandRecord(
            step=step,
            command=command,
            cwd=str(cwd),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.command_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_run_metadata(directory: Path) -> dict[str, object]:
    """Read run-level metadata for summaries and provenance."""
    metadata_path = Path(directory) / RUN_METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def update_run_metadata(directory: Path, **fields: object) -> Path:
    """Merge run-level metadata without overwriting planning or execution fields."""
    metadata_path = Path(directory) / RUN_METADATA_FILENAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = read_run_metadata(metadata_path.parent)
    for key, value in fields.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path
