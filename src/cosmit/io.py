from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    records: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def write_json(path: str | Path, value: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
