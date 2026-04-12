from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_objective_sanity import TokenizerCandidateSpec, evaluate_candidate_specs, load_candidate_specs
from tokenizer_search_split import SearchSplitConfig, write_search_split


class TokenizerObjectiveSanityTest(unittest.TestCase):
    def _write_docs_jsonl(self, docs: list[str], *, docs_val: int) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        docs_path = tmp_path / "docs_selected.jsonl"
        with docs_path.open("w", encoding="utf-8") as handle:
            for text in docs:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        sidecar_path = tmp_path / "docs_selected.source_manifest.json"
        sidecar_path.write_text(
            json.dumps({"docs_val": docs_val, "num_docs": len(docs)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return docs_path

    def _train_bpe_model(self, corpus_text: str, *, vocab_size: int = 32) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text(corpus_text, encoding="utf-8")
        model_prefix = tmp_path / "toy_bpe"
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=1.0,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=-1,
            hard_vocab_limit=False,
            minloglevel=2,
        )
        return Path(str(model_prefix) + ".model")

    def _write_candidate_config(self, candidates: list[dict[str, object]]) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        config_path = tmp_path / "candidates.json"
        config_path.write_text(json.dumps({"candidates": candidates}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return config_path

    def test_load_candidate_specs_reads_json_config(self) -> None:
        config_path = self._write_candidate_config(
            [
                {
                    "name": "baseline",
                    "tokenizer_model_path": "/tmp/baseline.model",
                    "tokenizer_vocab_path": "/tmp/baseline.vocab",
                    "tokenizer_asset_paths": ["/tmp/meta.json"],
                }
            ]
        )
        specs = load_candidate_specs(config_path)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "baseline")
        self.assertEqual(specs[0].tokenizer_model_path, "/tmp/baseline.model")
        self.assertEqual(specs[0].tokenizer_vocab_path, "/tmp/baseline.vocab")
        self.assertEqual(specs[0].tokenizer_asset_paths, ("/tmp/meta.json",))

    def test_in_domain_candidate_ranks_ahead_of_bad_candidate(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "challenge val one",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
                "tokenizer search objective",
                "hello world tokenizer",
            ],
            docs_val=2,
        )
        split_dir = docs_path.parent / "search_split"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(
                search_train_docs=4,
                search_holdout_docs=2,
            ),
        )
        good_model = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\ntokenizer search objective\n",
        )
        bad_model = self._train_bpe_model(
            "zebra nebula quartz\nrare xenon phrases\nout of domain tokens\n",
        )
        report = evaluate_candidate_specs(
            [
                TokenizerCandidateSpec(name="good", tokenizer_model_path=str(good_model)),
                TokenizerCandidateSpec(name="bad", tokenizer_model_path=str(bad_model)),
            ],
            split_dir=split_dir,
            alpha=0.0,
            ngram_order=3,
            add_k=0.1,
        )
        self.assertEqual([candidate.name for candidate in report.ranked_candidates], ["good", "bad"])
        self.assertLess(
            report.ranked_candidates[0].evaluation.result.proxy_bpb_holdout,
            report.ranked_candidates[1].evaluation.result.proxy_bpb_holdout,
        )
        self.assertEqual(report.ranked_candidates[0].score_delta_vs_best, 0.0)
        self.assertGreater(report.ranked_candidates[1].score_delta_vs_best, 0.0)

    def test_alpha_penalizes_extra_asset_bytes_for_identical_model(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
            ],
            docs_val=1,
        )
        split_dir = docs_path.parent / "search_split_alpha"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(
                search_train_docs=2,
                search_holdout_docs=2,
            ),
        )
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\n",
        )
        extra_asset = model_path.parent / "tokenizer.meta.json"
        extra_asset.write_text(json.dumps({"note": "extra asset penalty"}, indent=2) + "\n", encoding="utf-8")
        report = evaluate_candidate_specs(
            [
                TokenizerCandidateSpec(name="plain", tokenizer_model_path=str(model_path)),
                TokenizerCandidateSpec(
                    name="penalized",
                    tokenizer_model_path=str(model_path),
                    tokenizer_asset_paths=(str(extra_asset),),
                ),
            ],
            split_dir=split_dir,
            alpha=1e-3,
            ngram_order=3,
            add_k=0.1,
        )
        self.assertEqual([candidate.name for candidate in report.ranked_candidates], ["plain", "penalized"])
        self.assertAlmostEqual(
            report.ranked_candidates[0].evaluation.result.proxy_bpb_holdout,
            report.ranked_candidates[1].evaluation.result.proxy_bpb_holdout,
        )
        self.assertGreater(report.ranked_candidates[1].tokenizer_asset_bytes_delta_vs_best, 0)
        self.assertGreater(report.ranked_candidates[1].score_delta_vs_best, 0.0)


if __name__ == "__main__":
    unittest.main()
