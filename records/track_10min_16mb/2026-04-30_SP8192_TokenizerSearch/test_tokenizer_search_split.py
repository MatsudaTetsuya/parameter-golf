from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenizer_search_split import (
    SEARCH_HOLDOUT_FILENAME,
    SEARCH_SPLIT_MANIFEST_FILENAME,
    SEARCH_TRAIN_FILENAME,
    SearchSplitConfig,
    load_search_split_manifest,
    write_search_split,
)


class TokenizerSearchSplitTest(unittest.TestCase):
    def _write_docs_jsonl(
        self,
        docs: list[str],
        *,
        sidecar: dict[str, object] | None = None,
    ) -> tuple[Path, Path]:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        docs_path = tmp_path / "docs_selected.jsonl"
        with docs_path.open("w", encoding="utf-8") as handle:
            for text in docs:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        sidecar_path = tmp_path / "docs_selected.source_manifest.json"
        if sidecar is not None:
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return docs_path, sidecar_path

    def test_write_search_split_skips_val_and_materializes_fixed_slices(self) -> None:
        docs_path, _ = self._write_docs_jsonl(
            [f"doc-{idx}" for idx in range(10)],
            sidecar={"docs_val": 2, "num_docs": 10, "docs_sha256": "abc123", "shuffle_seed": 7},
        )
        output_dir = docs_path.parent / "search_split"
        manifest = write_search_split(
            docs_path,
            output_dir=output_dir,
            config=SearchSplitConfig(
                search_train_docs=3,
                search_holdout_docs=2,
            ),
        )
        self.assertEqual(manifest.num_val_docs_skipped, 2)
        self.assertEqual(manifest.search_holdout_start_doc, 2)
        self.assertEqual(manifest.search_train_start_doc, 4)
        self.assertEqual(manifest.search_holdout_docs, 2)
        self.assertEqual(manifest.search_train_docs, 3)
        holdout_lines = (output_dir / SEARCH_HOLDOUT_FILENAME).read_text(encoding="utf-8").splitlines()
        train_lines = (output_dir / SEARCH_TRAIN_FILENAME).read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["text"] for line in holdout_lines], ["doc-2", "doc-3"])
        self.assertEqual([json.loads(line)["text"] for line in train_lines], ["doc-4", "doc-5", "doc-6"])
        roundtrip = load_search_split_manifest(output_dir / SEARCH_SPLIT_MANIFEST_FILENAME)
        self.assertEqual(roundtrip.docs_sha256, "abc123")
        self.assertEqual(roundtrip.shuffle_seed, 7)

    def test_explicit_num_val_docs_overrides_sidecar(self) -> None:
        docs_path, sidecar_path = self._write_docs_jsonl(
            [f"doc-{idx}" for idx in range(8)],
            sidecar={"docs_val": 3, "num_docs": 8},
        )
        output_dir = docs_path.parent / "search_split_override"
        manifest = write_search_split(
            docs_path,
            output_dir=output_dir,
            config=SearchSplitConfig(
                search_train_docs=2,
                search_holdout_docs=2,
                num_val_docs=1,
                sidecar_path=str(sidecar_path),
            ),
        )
        self.assertEqual(manifest.num_val_docs_skipped, 1)
        holdout_lines = (output_dir / SEARCH_HOLDOUT_FILENAME).read_text(encoding="utf-8").splitlines()
        train_lines = (output_dir / SEARCH_TRAIN_FILENAME).read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["text"] for line in holdout_lines], ["doc-1", "doc-2"])
        self.assertEqual([json.loads(line)["text"] for line in train_lines], ["doc-3", "doc-4"])


if __name__ == "__main__":
    unittest.main()
