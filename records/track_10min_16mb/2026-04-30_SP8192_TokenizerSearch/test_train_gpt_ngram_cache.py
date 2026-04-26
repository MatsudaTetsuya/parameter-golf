import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


spec = importlib.util.spec_from_file_location("record_train_gpt", Path(__file__).with_name("train_gpt.py"))
assert spec is not None and spec.loader is not None
train_gpt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_gpt
spec.loader.exec_module(train_gpt)


class FastNgramSweepTest(unittest.TestCase):
    def test_sweep_reports_context_even_when_target_unseen(self) -> None:
        tokens = [5, 7, 5, 8]
        model_logp = np.zeros(len(tokens), dtype=np.float32)
        token_bytes = np.ones(len(tokens), dtype=np.int64)
        score_mask = np.array([False, True, False, True])

        loss, token_count, byte_count, stats = train_gpt.fast_ngram_sweep(
            tokens,
            model_logp,
            None,
            token_bytes,
            score_mask,
            vocab_size=8192,
            min_order=2,
            max_order=2,
            alpha_base=0.01,
            alpha_scale=0.0,
            alpha_slope=1.0,
            alpha_center=1.0,
            use_entropy_alpha=False,
            full_prefix=False,
        )

        self.assertEqual(token_count, 2)
        self.assertEqual(byte_count, 2)
        self.assertEqual(stats["context_hits"], 1)
        self.assertEqual(stats["target_hits"], 0)
        self.assertAlmostEqual(loss, -math.log(0.99))

    def test_full_prefix_updates_unscored_past_positions(self) -> None:
        tokens = [5, 7, 5, 8]
        model_logp = np.zeros(len(tokens), dtype=np.float32)
        token_bytes = np.ones(len(tokens), dtype=np.int64)
        score_mask = np.array([False, False, False, True])

        _loss, token_count, _byte_count, stats = train_gpt.fast_ngram_sweep(
            tokens,
            model_logp,
            None,
            token_bytes,
            score_mask,
            vocab_size=8192,
            min_order=2,
            max_order=2,
            alpha_base=0.01,
            alpha_scale=0.0,
            alpha_slope=1.0,
            alpha_center=1.0,
            use_entropy_alpha=False,
            full_prefix=True,
        )

        self.assertEqual(token_count, 1)
        self.assertEqual(stats["skipped_updates"], 2)
        self.assertEqual(stats["scored_updates"], 1)
        self.assertEqual(stats["context_hits"], 1)
        self.assertEqual(stats["target_hits"], 0)

    def test_entropy_alpha_is_opt_in(self) -> None:
        low = train_gpt.ngram_alpha(
            1.0,
            alpha_base=0.005,
            alpha_scale=0.05,
            alpha_slope=2.0,
            alpha_center=4.0,
            use_entropy_alpha=False,
        )
        high = train_gpt.ngram_alpha(
            7.0,
            alpha_base=0.005,
            alpha_scale=0.05,
            alpha_slope=2.0,
            alpha_center=4.0,
            use_entropy_alpha=False,
        )
        self.assertAlmostEqual(low, 0.005)
        self.assertAlmostEqual(high, 0.005)

        low_dynamic = train_gpt.ngram_alpha(
            1.0,
            alpha_base=0.005,
            alpha_scale=0.05,
            alpha_slope=2.0,
            alpha_center=4.0,
            use_entropy_alpha=True,
        )
        high_dynamic = train_gpt.ngram_alpha(
            7.0,
            alpha_base=0.005,
            alpha_scale=0.05,
            alpha_slope=2.0,
            alpha_center=4.0,
            use_entropy_alpha=True,
        )
        self.assertLess(low_dynamic, high_dynamic)

    def test_alpha_zero_matches_model_nll(self) -> None:
        tokens = np.array([1, 2, 1, 3, 1, 4], dtype=np.int64)
        model_logp = np.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
        token_bytes = np.ones(len(tokens), dtype=np.int64)
        score_mask = np.array([False, True, True, True, True, True])

        loss, token_count, byte_count, _stats = train_gpt.fast_ngram_sweep(
            tokens,
            model_logp,
            None,
            token_bytes,
            score_mask,
            vocab_size=8192,
            min_order=2,
            max_order=3,
            alpha_base=0.0,
            alpha_scale=0.0,
            alpha_slope=1.0,
            alpha_center=1.0,
            use_entropy_alpha=False,
            full_prefix=False,
        )

        expected = -float(model_logp[score_mask].sum())
        self.assertEqual(token_count, int(score_mask.sum()))
        self.assertEqual(byte_count, int(score_mask.sum()))
        self.assertAlmostEqual(loss, expected, places=6)

    def test_validate_ngram_score_mask_rejects_missing_full_eval_positions(self) -> None:
        score_mask = torch.tensor([0, 1, 0, 1], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "did not score every target"):
            train_gpt.validate_ngram_score_mask(score_mask, total_tokens=3, sample_chunk_indices=None)

    def test_validate_ngram_score_mask_allows_sampled_gaps(self) -> None:
        score_mask = torch.tensor([0, 1, 0, 1], dtype=torch.int32)

        total_scored = train_gpt.validate_ngram_score_mask(
            score_mask,
            total_tokens=3,
            sample_chunk_indices=(0,),
        )

        self.assertEqual(total_scored, 2)


if __name__ == "__main__":
    unittest.main()
