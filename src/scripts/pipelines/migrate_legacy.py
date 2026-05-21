from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "data_old"
DEST_ROOT = PROJECT_ROOT / "data" / "migrated"
MIGRATION_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "data_miggrate" / "migrate_dataset_0.0.1.py"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(file_index: int, file_total: int, file_name: str, status: str | None = None) -> None:
    percent = max(0, min(100, int((file_index / file_total) * 100))) if file_total > 0 else 0
    parts = [f"[{timestamp()}]", "migrate", f"[{percent}%]", f"{file_index}/{file_total}", file_name]
    if status:
        parts.append(f"- {status}")
    print(" ".join(parts))


def source_files() -> list[Path]:
    files: list[Path] = []
    for directory_name in ("regular", "high_quality"):
        directory = SOURCE_ROOT / directory_name
        files.extend(sorted(directory.glob("*.jsonl")))
    return files


def main() -> int:
    files = source_files()
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    if not files:
        print(f"no legacy files found under {SOURCE_ROOT}", file=sys.stderr)
        return 1

    had_failure = False
    total_files = len(files)
    for index, source_path in enumerate(files, start=1):
        dest_path = DEST_ROOT / f"{source_path.stem}.json"
        log_progress(index, total_files, source_path.name)
        result = subprocess.run(
            [sys.executable, str(MIGRATION_SCRIPT), "--source", str(source_path), "--dest", str(dest_path)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            had_failure = True
            log_progress(index, total_files, source_path.name, "failed")
            print(f"migration failed for {source_path}", file=sys.stderr)
        else:
            log_progress(index, total_files, source_path.name, "ok")

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
