from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_scorer import (
    NgramProxyConfig,
    TokenizerScorerConfig,
    measure_tokenizer_asset_bytes,
    score_tokenizer_documents,
)


class TokenizerScorerTest(unittest.TestCase):
    def _train_bpe_model(self, corpus_text: str, *, vocab_size: int = 32) -> tuple[Path, spm.SentencePieceProcessor]:
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
        model_path = Path(str(model_prefix) + ".model")
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        return model_path, sp

    def test_measure_tokenizer_asset_bytes_counts_model_and_vocab(self) -> None:
        model_path, _ = self._train_bpe_model(
            "hello world\nsmall test corpus\nhello tokenizer search\n",
        )
        expected = model_path.stat().st_size + model_path.with_suffix(".vocab").stat().st_size
        self.assertEqual(measure_tokenizer_asset_bytes(model_path), expected)

    def test_measure_tokenizer_asset_bytes_rejects_missing_explicit_asset(self) -> None:
        model_path, _ = self._train_bpe_model(
            "hello world\nsmall test corpus\nhello tokenizer search\n",
        )
        missing_asset = model_path.parent / "missing.meta"
        with self.assertRaises(FileNotFoundError):
            measure_tokenizer_asset_bytes(
                model_path,
                extra_asset_paths=[missing_asset],
            )

    def test_score_breakdown_matches_formula(self) -> None:
        model_path, sp = self._train_bpe_model(
            "hello world\nhello tokenizer\nsmall tokenizer world\nsearch objective test\n",
        )
        tokenizer_asset_bytes = measure_tokenizer_asset_bytes(model_path)
        config = TokenizerScorerConfig(
            alpha=1e-6,
            ngram=NgramProxyConfig(order=3, add_k=0.1),
        )
        result = score_tokenizer_documents(
            sp,
            train_docs=[
                "hello world",
                "hello tokenizer",
                "small tokenizer world",
            ],
            holdout_docs=[
                "hello world",
                "tokenizer world",
            ],
            tokenizer_asset_bytes=tokenizer_asset_bytes,
            config=config,
        )
        self.assertGreater(result.proxy_bits_per_token, 0.0)
        self.assertGreater(result.holdout_tokens_per_byte, 0.0)
        self.assertGreater(result.proxy_bpb_holdout, 0.0)
        self.assertGreater(result.holdout_token_count, 0)
        self.assertGreater(result.holdout_byte_count, 0)
        self.assertGreater(result.train_token_count, 0)
        self.assertAlmostEqual(
            result.proxy_bpb_holdout,
            result.proxy_bits_per_token * result.holdout_tokens_per_byte,
        )
        self.assertAlmostEqual(
            result.score,
            result.proxy_bpb_holdout + config.alpha * result.tokenizer_asset_bytes,
        )

    def test_in_domain_holdout_scores_better_than_out_of_domain_holdout(self) -> None:
        model_path, sp = self._train_bpe_model(
            "hello world\nhello world again\nworld tokenizer hello\n",
        )
        tokenizer_asset_bytes = measure_tokenizer_asset_bytes(model_path)
        config = TokenizerScorerConfig(
            alpha=0.0,
            ngram=NgramProxyConfig(order=3, add_k=0.1),
        )
        in_domain = score_tokenizer_documents(
            sp,
            train_docs=[
                "hello world",
                "hello world again",
                "world tokenizer hello",
            ],
            holdout_docs=[
                "hello world",
                "world hello",
            ],
            tokenizer_asset_bytes=tokenizer_asset_bytes,
            config=config,
        )
        out_of_domain = score_tokenizer_documents(
            sp,
            train_docs=[
                "hello world",
                "hello world again",
                "world tokenizer hello",
            ],
            holdout_docs=[
                "rare zebra phrases",
                "out of domain text",
            ],
            tokenizer_asset_bytes=tokenizer_asset_bytes,
            config=config,
        )
        self.assertLess(in_domain.proxy_bits_per_token, out_of_domain.proxy_bits_per_token)
        self.assertLess(in_domain.proxy_bpb_holdout, out_of_domain.proxy_bpb_holdout)


if __name__ == "__main__":
    unittest.main()
