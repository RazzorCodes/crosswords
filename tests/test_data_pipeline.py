from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Any
from unittest import TestCase, main
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATE_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "data_miggrate" / "migrate_dataset_0.0.1.py"
ANNOTATE_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "data_annotation" / "auto_annotate.py"
RUN_PIPELINES_SCRIPT = PROJECT_ROOT / "src" / "scripts" / "pipelines" / "run_pipelines.py"


def run_python(script: Path, *args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def sample_strokes() -> list[list[dict[str, float]]]:
    return [[{"x": 0.0, "y": 0.0, "t": 1.0}, {"x": 10.0, "y": 10.0, "t": 2.0}]]


class DataPipelineTests(TestCase):
    def load_auto_annotate_module(self) -> Any:
        spec = importlib.util.spec_from_file_location("auto_annotate_module", ANNOTATE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_migrates_valid_legacy_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "regular-0001.jsonl"
            dest = root / "regular-0001.json"
            source.write_text(
                json.dumps(
                    {
                        "id": "legacy-1",
                        "label": "A",
                        "strokes": sample_strokes(),
                        "source": "suggestion-bubble",
                        "mode": "play",
                        "status": "regular",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_python(MIGRATE_SCRIPT, "--source", str(source), "--dest", str(dest), cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(list(payload.keys()), ["regular-0001"])
            sample = payload["regular-0001"][0]
            self.assertEqual(sample["sample_id"], "legacy-1")
            self.assertEqual(sample["sample_label"], "A")
            self.assertEqual(sample["strokes"], sample_strokes())
            self.assertEqual(sample["extra_labels"], ["legacy", "suggestion-bubble", "play"])

    def test_migration_skips_malformed_and_nonstroke_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "regular-0002.jsonl"
            dest = root / "regular-0002.json"
            source.write_text(
                "\n".join(
                    [
                        '{"id":"bad-json"',
                        json.dumps({"id": "bad-strokes", "label": "A", "strokes": []}),
                        json.dumps({"label": "B", "strokes": sample_strokes()}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_python(MIGRATE_SCRIPT, "--source", str(source), "--dest", str(dest), cwd=PROJECT_ROOT)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("invalid json", result.stderr)
            self.assertIn("missing usable strokes", result.stderr)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            sample = payload["regular-0002"][0]
            self.assertEqual(sample["sample_id"], "regular-0002-000003")
            self.assertEqual(sample["sample_label"], "B")

    def test_auto_annotation_skips_annotated_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self.load_auto_annotate_module()
            root = Path(tmpdir)
            source = root / "dataset.json"
            dest = root / "annotated.json"
            source.write_text(
                json.dumps(
                    {
                        "dataset": [
                            {
                                "sample_id": "s1",
                                "sample_label": "A",
                                "extra_labels": ["legacy", "annotated"],
                                "strokes": sample_strokes(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(module.urllib.request, "urlopen") as urlopen:
                exit_code = module.main(
                    [
                        "--source",
                        str(source),
                        "--dest",
                        str(dest),
                        "--address",
                        "localhost:4000",
                        "--model",
                        "fast",
                    ]
                )

            self.assertEqual(exit_code, 0)
            urlopen.assert_not_called()
            payload = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"][0]["sample_label"], "A")

    def test_auto_annotation_marks_mismatches_unreliable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self.load_auto_annotate_module()
            root = Path(tmpdir)
            source = root / "dataset.json"
            dest = root / "annotated.json"
            source.write_text(
                json.dumps(
                    {
                        "dataset": [
                            {
                                "sample_id": "s1",
                                "sample_label": "A",
                                "extra_labels": ["legacy"],
                                "strokes": sample_strokes(),
                            },
                            {
                                "sample_id": "s2",
                                "sample_label": "B",
                                "extra_labels": ["legacy"],
                                "strokes": sample_strokes(),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            response = json.dumps({"choices": [{"message": {"content": ",".join(["Z", "?"] + ["?"] * 23)}}]}).encode("utf-8")
            response = json.dumps({"choices": [{"message": {"content": ",".join(["A", "Z"] + ["?"] * 23)}}]}).encode("utf-8")

            class FakeResponse:
                def __enter__(self) -> "FakeResponse":
                    return self

                def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                    return None

                def read(self) -> bytes:
                    return response

            with mock.patch.object(module.urllib.request, "urlopen", return_value=FakeResponse()) as urlopen:
                exit_code = module.main(
                    [
                        "--source",
                        str(source),
                        "--dest",
                        str(dest),
                        "--address",
                        "localhost:4000",
                        "--model",
                        "fast",
                    ]
                )

            self.assertEqual(exit_code, 0)
            request = urlopen.call_args.args[0]
            self.assertIn("/chat/completions", request.full_url)
            self.assertIn(b"data:image/png;base64,", request.data)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            first, second = payload["dataset"]
            self.assertEqual(first["sample_label"], "A")
            self.assertEqual(second["sample_label"], "B")
            self.assertIn("annotated", first["extra_labels"])
            self.assertIn("auto", first["extra_labels"])
            self.assertNotIn("unreliable", first["extra_labels"])
            self.assertIn("annotated", second["extra_labels"])
            self.assertIn("auto", second["extra_labels"])
            self.assertIn("unreliable", second["extra_labels"])

    def test_run_pipelines_migrate_creates_migrated_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "scripts" / "data_miggrate").mkdir(parents=True)
            (root / "src" / "scripts" / "data_annotation").mkdir(parents=True)
            (root / "src" / "scripts" / "pipelines").mkdir(parents=True)
            (root / "data_old" / "regular").mkdir(parents=True)
            (root / "data_old" / "high_quality").mkdir(parents=True)

            for relative in (
                Path("src/scripts/data_miggrate/migrate_dataset_0.0.1.py"),
                Path("src/scripts/data_annotation/auto_annotate.py"),
                Path("src/scripts/pipelines/migrate_legacy.py"),
                Path("src/scripts/pipelines/auto_anotate_data.py"),
                Path("src/scripts/pipelines/run_pipelines.py"),
            ):
                target = root / relative
                target.write_text((PROJECT_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")

            (root / "data_old" / "regular" / "regular-0001.jsonl").write_text(
                json.dumps({"id": "legacy-1", "label": "A", "strokes": sample_strokes()}) + "\n",
                encoding="utf-8",
            )

            result = run_python(
                root / "src" / "scripts" / "pipelines" / "run_pipelines.py",
                "migrate_legacy",
                cwd=root,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "data" / "migrated").is_dir())
            self.assertTrue((root / "data" / "migrated" / "regular-0001.json").exists())

    def test_run_pipelines_stops_before_annotation_on_migration_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "scripts" / "data_miggrate").mkdir(parents=True)
            (root / "src" / "scripts" / "data_annotation").mkdir(parents=True)
            (root / "src" / "scripts" / "pipelines").mkdir(parents=True)
            (root / "data_old" / "regular").mkdir(parents=True)
            (root / "data_old" / "high_quality").mkdir(parents=True)

            for relative in (
                Path("src/scripts/data_miggrate/migrate_dataset_0.0.1.py"),
                Path("src/scripts/data_annotation/auto_annotate.py"),
                Path("src/scripts/pipelines/migrate_legacy.py"),
                Path("src/scripts/pipelines/auto_anotate_data.py"),
                Path("src/scripts/pipelines/run_pipelines.py"),
            ):
                target = root / relative
                target.write_text((PROJECT_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")

            (root / "data_old" / "regular" / "regular-0001.jsonl").write_text('{"bad":true}\n', encoding="utf-8")

            env = os.environ.copy()
            env["AUTO_ANNOTATE_ADDRESS"] = "127.0.0.1:1"
            result = run_python(
                root / "src" / "scripts" / "pipelines" / "run_pipelines.py",
                "migrate_legacy",
                "auto_annotate_data",
                cwd=root,
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "data" / "annotated").exists())


if __name__ == "__main__":
    main()
