from __future__ import annotations

import json
from pathlib import Path

from sample_store import (
    PENDING_LLM,
    REGULAR,
    append_audit_event,
    append_record,
    create_sample,
    ensure_data_dirs,
    load_jsonl,
    load_legacy_samples,
    load_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MIGRATION_STATE_PATH = PROJECT_ROOT / "data" / "audit" / "legacy-migration-state.json"


def migration_already_completed() -> bool:
    if not MIGRATION_STATE_PATH.exists():
        return False
    try:
        data = json.loads(MIGRATION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("completed"))


def write_migration_state(imported: int, queued: int) -> None:
    MIGRATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MIGRATION_STATE_PATH.write_text(
        json.dumps(
            {
                "completed": True,
                "imported": imported,
                "queued_for_llm": queued,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def pending_ids() -> set[str]:
    ids: set[str] = set()
    for record in load_snapshot(include_legacy=False).pending_entries.values():
        sample_id = record.get("id")
        if isinstance(sample_id, str) and sample_id:
            ids.add(sample_id)
    return ids


def main() -> None:
    ensure_data_dirs()

    if migration_already_completed():
        print("Legacy migration already completed.")
        return

    structured_snapshot = load_snapshot(include_legacy=False)
    existing_ids = structured_snapshot.active_ids | structured_snapshot.tombstoned_ids
    queued_ids = pending_ids()

    imported = 0
    queued = 0
    for sample in load_legacy_samples():
        sample_id = sample["id"]
        if sample_id in existing_ids:
            continue

        create_sample(
            sample_id=sample_id,
            label=sample["label"],
            strokes=sample["strokes"],
            stored_as=REGULAR,
            source="legacy-migrated",
            mode="legacy",
            queue_for_llm=False,
            metadata={"migrated_from": "dataset.jsonl"},
        )
        imported += 1
        existing_ids.add(sample_id)

        if sample_id not in queued_ids:
            append_record(
                PENDING_LLM,
                {
                    "id": sample_id,
                    "label": sample["label"],
                    "status": "queued",
                    "created_at": sample["created_at"],
                    "updated_at": sample["updated_at"],
                    "payload": {"source": "legacy-migration"},
                },
            )
            append_audit_event(sample_id, "queued_for_llm", {"source": "legacy-migration"})
            queued += 1
            queued_ids.add(sample_id)

    write_migration_state(imported, queued)
    print(f"Migrated {imported} legacy samples and queued {queued} for LiteLLM refinement.")


if __name__ == "__main__":
    main()
