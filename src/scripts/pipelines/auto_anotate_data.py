from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "data" / "migrated"
DEST_ROOT = PROJECT_ROOT / "data" / "annotated"
ANNOTATION_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "data_annotation" / "auto_annotate.py"
DEFAULT_ADDRESS = os.getenv("AUTO_ANNOTATE_ADDRESS", "localhost:4000")
DEFAULT_MODEL = os.getenv("AUTO_ANNOTATE_MODEL", "fast")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(file_index: int, file_total: int, file_name: str, status: str | None = None) -> None:
    percent = max(0, min(100, int((file_index / file_total) * 100))) if file_total > 0 else 0
    parts = [f"[{timestamp()}]", "annotate", f"[{percent}%]", f"{file_index}/{file_total}", file_name]
    if status:
        parts.append(f"- {status}")
    print(" ".join(parts))


def main() -> int:
    files = sorted(SOURCE_ROOT.glob("*.json"))
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    if not files:
        print(f"no migrated files found under {SOURCE_ROOT}", file=sys.stderr)
        return 1

    had_failure = False
    total_files = len(files)
    for index, source_path in enumerate(files, start=1):
        dest_path = DEST_ROOT / source_path.name
        log_progress(index, total_files, source_path.name)
        result = subprocess.run(
            [
                sys.executable,
                str(ANNOTATION_SCRIPT),
                "--source",
                str(source_path),
                "--dest",
                str(dest_path),
                "--address",
                DEFAULT_ADDRESS,
                "--model",
                DEFAULT_MODEL,
                "--progress-phase",
                "annotate",
                "--file-index",
                str(index),
                "--file-total",
                str(total_files),
                "--file-name",
                source_path.name,
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            had_failure = True
            log_progress(index, total_files, source_path.name, "failed")
            print(f"annotation failed for {source_path}", file=sys.stderr)
        else:
            log_progress(index, total_files, source_path.name, "ok")

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
