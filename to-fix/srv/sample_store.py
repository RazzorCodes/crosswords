from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_LEGACY_DATASET_PATH = PROJECT_ROOT / "dataset.jsonl"
MAX_ITEMS_PER_FILE = 250

REGULAR = "regular"
HIGH_QUALITY = "high_quality"
PENDING_LLM = "pending_llm"
AUDIT = "audit"
TOMBSTONES = "tombstones"

SAMPLE_CATEGORIES = (REGULAR, HIGH_QUALITY)
EVENT_CATEGORIES = (PENDING_LLM, AUDIT, TOMBSTONES)
ALL_CATEGORIES = SAMPLE_CATEGORIES + EVENT_CATEGORIES


@dataclass
class Snapshot:
    legacy_samples: list[dict[str, Any]]
    regular_samples: dict[str, dict[str, Any]]
    high_quality_samples: dict[str, dict[str, Any]]
    pending_entries: dict[str, dict[str, Any]]
    tombstoned_ids: set[str]

    @property
    def usable_regular(self) -> list[dict[str, Any]]:
        return list(self.regular_samples.values())

    @property
    def usable_high_quality(self) -> list[dict[str, Any]]:
        return list(self.high_quality_samples.values())

    @property
    def usable_samples(self) -> list[dict[str, Any]]:
        return self.legacy_samples + self.usable_regular + self.usable_high_quality

    @property
    def total_usable_count(self) -> int:
        return len(self.usable_samples)

    @property
    def high_quality_share(self) -> float:
        if self.total_usable_count == 0:
            return 0.0
        return len(self.high_quality_samples) / self.total_usable_count

    @property
    def active_ids(self) -> set[str]:
        return set(self.regular_samples) | set(self.high_quality_samples)

    def pending_batch_candidates(self) -> list[dict[str, Any]]:
        queued: list[dict[str, Any]] = []
        for sample_id, pending in self.pending_entries.items():
            if pending.get("status") != "queued":
                continue
            if sample_id in self.tombstoned_ids:
                continue
            if sample_id in self.high_quality_samples:
                continue
            regular = self.regular_samples.get(sample_id)
            if regular is None:
                continue
            queued.append(
                {
                    "sample_id": sample_id,
                    "pending_at": pending.get("updated_at") or pending.get("created_at") or "",
                    "regular": regular,
                    "pending": pending,
                }
            )
        queued.sort(key=lambda item: (item["pending_at"], item["sample_id"]))
        return queued


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_data_root() -> Path:
    return Path(os.getenv("DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()


def get_legacy_dataset_path() -> Path:
    return Path(os.getenv("DATA_PATH", str(DEFAULT_LEGACY_DATASET_PATH))).expanduser()


def get_category_dir(category: str) -> Path:
    if category not in ALL_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    return get_data_root() / category


def ensure_data_dirs() -> None:
    for category in ALL_CATEGORIES:
        get_category_dir(category).mkdir(parents=True, exist_ok=True)


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_category_files(category: str) -> list[Path]:
    directory = get_category_dir(category)
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.jsonl") if path.is_file())


def iter_all_store_files(include_legacy: bool = False) -> list[Path]:
    files: list[Path] = []
    if include_legacy:
        legacy = get_legacy_dataset_path()
        if legacy.exists() and legacy.is_file():
            files.append(legacy)
    for category in ALL_CATEGORIES:
        files.extend(iter_category_files(category))
    return files


def get_active_chunk(category: str) -> Path:
    ensure_data_dirs()
    files = iter_category_files(category)
    if not files:
        return get_category_dir(category) / f"{category}-0001.jsonl"

    current = files[-1]
    if count_jsonl_lines(current) < MAX_ITEMS_PER_FILE:
        return current

    next_index = len(files) + 1
    return get_category_dir(category) / f"{category}-{next_index:04d}.jsonl"


def append_record(category: str, record: dict[str, Any]) -> None:
    path = get_active_chunk(category)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_audit_event(sample_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    record = {
        "sample_id": sample_id,
        "event": event,
        "created_at": utc_now_iso(),
    }
    if payload:
        record["payload"] = payload
    append_record(AUDIT, record)


def new_sample_id() -> str:
    return uuid.uuid4().hex


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        return None
    return label


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_latest_records(category: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in iter_category_files(category):
        for record in load_jsonl(path):
            sample_id = record.get("id")
            if isinstance(sample_id, str) and sample_id:
                latest[sample_id] = record
    return latest


def _load_pending_entries() -> dict[str, dict[str, Any]]:
    pending: dict[str, dict[str, Any]] = {}
    for path in iter_category_files(PENDING_LLM):
        for record in load_jsonl(path):
            sample_id = record.get("id")
            if isinstance(sample_id, str) and sample_id:
                pending[sample_id] = record
    return pending


def _load_tombstones() -> set[str]:
    tombstones: set[str] = set()
    for path in iter_category_files(TOMBSTONES):
        for record in load_jsonl(path):
            sample_id = record.get("id")
            if isinstance(sample_id, str) and sample_id:
                tombstones.add(sample_id)
    return tombstones


def load_legacy_samples() -> list[dict[str, Any]]:
    legacy_path = get_legacy_dataset_path()
    samples: list[dict[str, Any]] = []
    if not legacy_path.exists():
        return samples

    for index, record in enumerate(load_jsonl(legacy_path), start=1):
        label = normalize_label(record.get("label"))
        strokes = record.get("strokes")
        if label is None or not isinstance(strokes, list):
            continue
        samples.append(
            {
                "id": f"legacy-{index:08d}",
                "label": label,
                "strokes": strokes,
                "source": "legacy",
                "mode": "legacy",
                "created_at": "legacy",
                "updated_at": "legacy",
                "status": REGULAR,
                "legacy": True,
            }
        )
    return samples


def load_snapshot(include_legacy: bool = False) -> Snapshot:
    ensure_data_dirs()
    tombstoned_ids = _load_tombstones()
    regular_records = _load_latest_records(REGULAR)
    high_quality_records = _load_latest_records(HIGH_QUALITY)
    pending_entries = _load_pending_entries()

    usable_high_quality: dict[str, dict[str, Any]] = {}
    for sample_id, record in high_quality_records.items():
        if sample_id in tombstoned_ids:
            continue
        usable_high_quality[sample_id] = record

    usable_regular: dict[str, dict[str, Any]] = {}
    for sample_id, record in regular_records.items():
        if sample_id in tombstoned_ids:
            continue
        if sample_id in usable_high_quality:
            continue
        usable_regular[sample_id] = record

    legacy_samples = load_legacy_samples() if include_legacy else []
    return Snapshot(
        legacy_samples=legacy_samples,
        regular_samples=usable_regular,
        high_quality_samples=usable_high_quality,
        pending_entries=pending_entries,
        tombstoned_ids=tombstoned_ids,
    )


def create_sample(
    *,
    label: str,
    strokes: list[Any],
    stored_as: str,
    source: str,
    mode: str,
    queue_for_llm: bool,
    metadata: dict[str, Any] | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    if stored_as not in SAMPLE_CATEGORIES:
        raise ValueError(f"Invalid storage category: {stored_as}")

    sample_id = sample_id or new_sample_id()
    timestamp = utc_now_iso()
    sample = {
        "id": sample_id,
        "label": label,
        "strokes": strokes,
        "source": source,
        "mode": mode,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": stored_as,
        "queued_for_llm": bool(queue_for_llm),
        "tombstoned": False,
    }
    if metadata:
        sample["metadata"] = metadata

    append_record(stored_as, sample)
    append_audit_event(
        sample_id,
        "sample_created",
        {
            "stored_as": stored_as,
            "source": source,
            "mode": mode,
            "queued_for_llm": bool(queue_for_llm),
        },
    )

    if queue_for_llm:
        pending_entry = {
            "id": sample_id,
            "label": label,
            "status": "queued",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        append_record(PENDING_LLM, pending_entry)
        append_audit_event(sample_id, "queued_for_llm", {"label": label})

    return sample


def set_pending_status(sample_id: str, status: str, payload: dict[str, Any] | None = None) -> None:
    timestamp = utc_now_iso()
    record = {
        "id": sample_id,
        "status": status,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    if payload:
        record["payload"] = payload
    append_record(PENDING_LLM, record)
    append_audit_event(sample_id, f"pending_llm_{status}", payload)


def tombstone_sample(sample_id: str, reason: str = "deleted_by_user") -> bool:
    snapshot = load_snapshot(include_legacy=False)
    if sample_id.startswith("legacy-"):
        return False
    if sample_id not in snapshot.active_ids and sample_id not in snapshot.pending_entries:
        return False

    record = {
        "id": sample_id,
        "reason": reason,
        "created_at": utc_now_iso(),
    }
    append_record(TOMBSTONES, record)
    append_audit_event(sample_id, "sample_tombstoned", {"reason": reason})
    return True


def promote_sample_to_high_quality(
    sample_id: str,
    label: str,
    *,
    provider: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    snapshot = load_snapshot(include_legacy=False)
    if sample_id in snapshot.tombstoned_ids:
        return None

    regular = snapshot.regular_samples.get(sample_id)
    if regular is None:
        return None

    timestamp = utc_now_iso()
    promoted = {
        **regular,
        "label": label,
        "updated_at": timestamp,
        "promoted_at": timestamp,
        "status": HIGH_QUALITY,
        "queued_for_llm": False,
        "promoted_from_status": regular.get("status", REGULAR),
        "promotion_provider": provider,
        "tombstoned": False,
    }
    if metadata:
        promoted["promotion_metadata"] = metadata

    append_record(HIGH_QUALITY, promoted)
    set_pending_status(
        sample_id,
        "promoted",
        {
            "provider": provider,
            "label": label,
            "original_label": regular.get("label"),
        },
    )
    append_audit_event(
        sample_id,
        "sample_promoted_to_high_quality",
        {
            "provider": provider,
            "label": label,
            "original_label": regular.get("label"),
        },
    )
    return promoted


def get_label_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = {chr(code): 0 for code in range(65, 91)}
    for sample in samples:
        label = normalize_label(sample.get("label"))
        if label is not None:
            counts[label] += 1
    return counts
