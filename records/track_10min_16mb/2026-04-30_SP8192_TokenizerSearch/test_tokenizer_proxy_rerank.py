from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import sentencepiece as spm

from tokenizer_objective_sanity import TokenizerCandidateSpec, evaluate_candidate_specs
from tokenizer_proxy_rerank import (
    PROXY_TRAIN_CACHE_SUBDIR,
    ProxyTrainingConfig,
    load_top_candidates_from_report,
    rerank_candidates_with_proxy_training,
    tokenize_docs_to_stream,
)
from tokenizer_search_split import SearchSplitConfig, write_search_split


class TokenizerProxyRerankTest(unittest.TestCase):
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

    def _write_report(self, report_payload: dict) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report_path

    def test_load_top_candidates_from_report_selects_top_k(self) -> None:
        report_path = self._write_report(
            {
                "ranked_candidates": [
                    {"name": "a", "rank": 1, "evaluation": {"tokenizer_model_path": "/tmp/a.model", "tokenizer_vocab_path": None, "split_manifest_path": "/tmp/split.json", "result": {"tokenizer_asset_bytes": 1, "holdout_tokens_per_byte": 1.0, "score": 1.0, "proxy_bpb_holdout": 1.0}}},
                    {"name": "b", "rank": 2, "evaluation": {"tokenizer_model_path": "/tmp/b.model", "tokenizer_vocab_path": None, "split_manifest_path": "/tmp/split.json", "result": {"tokenizer_asset_bytes": 1, "holdout_tokens_per_byte": 1.0, "score": 2.0, "proxy_bpb_holdout": 2.0}}},
                ]
            }
        )
        specs = load_top_candidates_from_report(report_path, top_k=1)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].label, "a")

    def test_tokenize_docs_to_stream_uses_bos_without_eos(self) -> None:
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\n",
        )
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        stream = tokenize_docs_to_stream(sp, ["hello world", "hello tokenizer"])
        bos_id = int(sp.bos_id())
        eos_id = int(sp.eos_id())
        self.assertEqual(int(stream[0].item()), bos_id)
        self.assertIn(bos_id, stream.tolist())
        self.assertNotIn(eos_id, stream.tolist())

    def test_proxy_rerank_penalizes_extra_asset_for_identical_tokenizer(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
                "hello world tokenizer",
            ],
            docs_val=1,
        )
        split_dir = docs_path.parent / "search_split"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(search_train_docs=3, search_holdout_docs=2),
        )
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\n",
        )
        extra_asset = model_path.parent / "extra.meta"
        extra_asset.write_text(json.dumps({"note": "penalty"}, indent=2) + "\n", encoding="utf-8")
        objective_report = evaluate_candidate_specs(
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
        report_path = self._write_report(asdict(objective_report))
        rerank_report = rerank_candidates_with_proxy_training(
            report_path=report_path,
            config=ProxyTrainingConfig(
                top_k=2,
                train_steps=8,
                batch_size=4,
                seq_len=8,
                model_dim=32,
                num_layers=1,
                num_heads=4,
                mlp_mult=2,
                learning_rate=5e-3,
                weight_decay=0.0,
                seed=123,
                device="cpu",
            ),
        )
        self.assertEqual([candidate.label for candidate in rerank_report.reranked_candidates], ["plain", "penalized"])
        self.assertAlmostEqual(
            rerank_report.reranked_candidates[0].trained_bits_per_token,
            rerank_report.reranked_candidates[1].trained_bits_per_token,
        )
        self.assertAlmostEqual(
            rerank_report.reranked_candidates[0].proxy_train_bpb_holdout,
            rerank_report.reranked_candidates[1].proxy_train_bpb_holdout,
        )
        self.assertLess(
            rerank_report.reranked_candidates[0].rerank_score,
            rerank_report.reranked_candidates[1].rerank_score,
        )

    def test_proxy_rerank_uses_persistent_cache(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
                "hello world tokenizer",
            ],
            docs_val=1,
        )
        split_dir = docs_path.parent / "search_split_cache"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(search_train_docs=3, search_holdout_docs=2),
        )
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\n",
        )
        objective_report = evaluate_candidate_specs(
            [TokenizerCandidateSpec(name="plain", tokenizer_model_path=str(model_path))],
            split_dir=split_dir,
            alpha=0.0,
            ngram_order=3,
            add_k=0.1,
        )
        report_path = self._write_report(asdict(objective_report))
        cache_dir = docs_path.parent / "proxy_cache"
        config = ProxyTrainingConfig(
            top_k=1,
            train_steps=6,
            batch_size=4,
            seq_len=8,
            model_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_mult=2,
            learning_rate=5e-3,
            weight_decay=0.0,
            seed=123,
            device="cpu",
        )
        first = rerank_candidates_with_proxy_training(
            report_path=report_path,
            config=config,
            cache_dir=cache_dir,
        )
        cache_files = list((cache_dir / PROXY_TRAIN_CACHE_SUBDIR).glob("*.json"))
        self.assertEqual(len(cache_files), 1)
        (split_dir / "search_train.jsonl").unlink()
        (split_dir / "search_holdout.jsonl").unlink()
        second = rerank_candidates_with_proxy_training(
            report_path=report_path,
            config=config,
            cache_dir=cache_dir,
        )
        self.assertFalse(first.reranked_candidates[0].cache_hit)
        self.assertTrue(second.reranked_candidates[0].cache_hit)
        self.assertAlmostEqual(
            first.reranked_candidates[0].trained_bits_per_token,
            second.reranked_candidates[0].trained_bits_per_token,
        )


if __name__ == "__main__":
    unittest.main()
