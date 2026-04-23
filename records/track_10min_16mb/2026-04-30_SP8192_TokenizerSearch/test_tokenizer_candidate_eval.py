from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_candidate_eval import evaluate_tokenizer_candidate
from tokenizer_eval_cache import TOKENIZER_EVAL_CACHE_SUBDIR
from tokenizer_scorer import NgramProxyConfig, TokenizerScorerConfig, score_tokenizer_paths
from tokenizer_search_split import SEARCH_SPLIT_MANIFEST_FILENAME, SearchSplitConfig, write_search_split


class TokenizerCandidateEvalTest(unittest.TestCase):
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

    def test_evaluate_tokenizer_candidate_uses_fixed_search_split(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "challenge val one",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
                "tokenizer search objective",
            ],
            docs_val=2,
        )
        split_dir = docs_path.parent / "search_split"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(
                search_train_docs=3,
                search_holdout_docs=2,
            ),
        )
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\ntokenizer search objective\n",
        )
        evaluation = evaluate_tokenizer_candidate(
            tokenizer_model_path=model_path,
            split_dir=split_dir,
            alpha=1e-6,
            ngram_order=3,
            add_k=0.1,
        )
        direct = score_tokenizer_paths(
            model_path=model_path,
            train_docs_path=split_dir / "search_train.jsonl",
            holdout_docs_path=split_dir / "search_holdout.jsonl",
            config=TokenizerScorerConfig(
                alpha=1e-6,
                ngram=NgramProxyConfig(order=3, add_k=0.1),
            ),
        )
        self.assertEqual(evaluation.split_manifest_path, str((split_dir / SEARCH_SPLIT_MANIFEST_FILENAME).resolve()))
        self.assertAlmostEqual(evaluation.result.score, direct.score)
        self.assertAlmostEqual(evaluation.result.proxy_bpb_holdout, direct.proxy_bpb_holdout)
        self.assertEqual(evaluation.result.holdout_token_count, direct.holdout_token_count)

    def test_evaluate_tokenizer_candidate_reuses_persistent_cache(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "challenge val one",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
                "tokenizer search objective",
            ],
            docs_val=2,
        )
        split_dir = docs_path.parent / "search_split_cache"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(
                search_train_docs=3,
                search_holdout_docs=2,
            ),
        )
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\ntokenizer search objective\n",
        )
        cache_dir = docs_path.parent / "eval_cache"
        first = evaluate_tokenizer_candidate(
            tokenizer_model_path=model_path,
            split_dir=split_dir,
            alpha=1e-6,
            ngram_order=3,
            add_k=0.1,
            cache_dir=cache_dir,
        )
        cache_files = list((cache_dir / TOKENIZER_EVAL_CACHE_SUBDIR).glob("*.json"))
        self.assertEqual(len(cache_files), 1)
        (split_dir / "search_train.jsonl").unlink()
        (split_dir / "search_holdout.jsonl").unlink()
        second = evaluate_tokenizer_candidate(
            tokenizer_model_path=model_path,
            split_dir=split_dir,
            alpha=1e-6,
            ngram_order=3,
            add_k=0.1,
            cache_dir=cache_dir,
        )
        self.assertAlmostEqual(first.result.score, second.result.score)
        self.assertAlmostEqual(first.result.proxy_bpb_holdout, second.result.proxy_bpb_holdout)

    def test_cache_key_changes_with_stream_mode(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "challenge val zero",
                "challenge val one",
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
                "world hello again",
                "tokenizer search objective",
            ],
            docs_val=2,
        )
        split_dir = docs_path.parent / "search_split_stream_mode"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(
                search_train_docs=3,
                search_holdout_docs=2,
            ),
        )
        model_path = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nworld hello again\ntokenizer search objective\n",
        )
        cache_dir = docs_path.parent / "eval_cache_stream_mode"
        evaluate_tokenizer_candidate(
            tokenizer_model_path=model_path,
            split_dir=split_dir,
            ngram_order=3,
            add_k=0.1,
            stream_mode="fullstack_bos",
            cache_dir=cache_dir,
        )
        evaluate_tokenizer_candidate(
            tokenizer_model_path=model_path,
            split_dir=split_dir,
            ngram_order=3,
            add_k=0.1,
            stream_mode="legacy_eos",
            cache_dir=cache_dir,
        )
        cache_files = list((cache_dir / TOKENIZER_EVAL_CACHE_SUBDIR).glob("*.json"))
        self.assertEqual(len(cache_files), 2)


if __name__ == "__main__":
    unittest.main()
