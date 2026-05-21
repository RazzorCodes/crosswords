from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINES = {
    "migrate_legacy": PROJECT_ROOT / "src" / "scripts" / "pipelines" / "migrate_legacy.py",
    "auto_annotate_data": PROJECT_ROOT / "src" / "scripts" / "pipelines" / "auto_anotate_data.py",
    "auto_anotate_data": PROJECT_ROOT / "src" / "scripts" / "pipelines" / "auto_anotate_data.py",
}


def main(argv: list[str] | None = None) -> int:
    names = list(sys.argv[1:] if argv is None else argv)
    if not names:
        print(f"usage: {Path(__file__).name} <pipeline> [<pipeline> ...]", file=sys.stderr)
        return 1

    for name in names:
        script_path = PIPELINES.get(name)
        if script_path is None:
            print(f"unknown pipeline: {name}", file=sys.stderr)
            return 1

        result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
