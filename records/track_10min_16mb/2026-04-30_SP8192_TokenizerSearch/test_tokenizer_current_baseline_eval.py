from __future__ import annotations

import unittest

from tokenizer_current_baseline_eval import rank_current_baseline_results
from tokenizer_fullstack_eval import FullStackCandidateResult, PreparedCandidateDataset


class TokenizerCurrentBaselineEvalTest(unittest.TestCase):
    def _dummy_dataset(self) -> PreparedCandidateDataset:
        return PreparedCandidateDataset(
            prepared_root="/tmp/prepared",
            dataset_dir="/tmp/dataset",
            dataset_manifest_path="/tmp/manifest.json",
            docs_jsonl_path="/tmp/docs.jsonl",
            tokenizer_model_path="/tmp/tokenizer.model",
            tokenizer_vocab_path=None,
            vocab_size=8192,
            num_docs=10,
            num_val_docs=2,
            shard_size=10**8,
            limit_docs=None,
            text_field="text",
            stats={},
            cache_key="cache",
        )

    def _result(
        self,
        *,
        label: str,
        primary_metric_val_bpb: float | None,
        submission_bytes_including_tokenizer: int | None,
    ) -> FullStackCandidateResult:
        return FullStackCandidateResult(
            rank=1,
            label=label,
            tokenizer_model_path="/tmp/tokenizer.model",
            tokenizer_vocab_path=None,
            tokenizer_asset_bytes=123,
            prepared_dataset=self._dummy_dataset(),
            work_dir="/tmp/run",
            stdout_path="/tmp/stdout.log",
            stderr_path="/tmp/stderr.log",
            train_log_path="/tmp/train.log",
            run_id="run",
            launcher="python",
            command=("python", "train_gpt.py"),
            exit_code=0,
            duration_seconds=1.0,
            code_bytes=100,
            raw_model_bytes=200,
            quantized_model_bytes=300,
            submission_bytes_excluding_tokenizer=400,
            submission_bytes_including_tokenizer=submission_bytes_including_tokenizer,
            last_logged_step=1,
            last_logged_val_loss=2.0,
            last_logged_val_bpb=1.3,
            final_roundtrip_val_loss=1.8,
            final_roundtrip_val_bpb=1.2,
            sliding_no_ttt_val_loss=1.7,
            sliding_no_ttt_val_bpb=primary_metric_val_bpb,
            legal_ttt_val_loss=None,
            legal_ttt_val_bpb=None,
            primary_metric_name="quantized_sliding_no_ttt_exact",
            primary_metric_val_loss=1.7 if primary_metric_val_bpb is not None else None,
            primary_metric_val_bpb=primary_metric_val_bpb,
        )

    def test_rank_current_baseline_results_prefers_within_limit(self) -> None:
        ranked = rank_current_baseline_results(
            [
                self._result(label="smaller_good", primary_metric_val_bpb=1.11, submission_bytes_including_tokenizer=15_999_999),
                self._result(label="slightly_better_but_oversize", primary_metric_val_bpb=1.10, submission_bytes_including_tokenizer=16_000_100),
            ]
        )
        self.assertEqual(ranked[0].label, "smaller_good")
        self.assertTrue(ranked[0].within_submission_limit)
        self.assertFalse(ranked[1].within_submission_limit)

    def test_rank_current_baseline_results_uses_metric_then_size(self) -> None:
        ranked = rank_current_baseline_results(
            [
                self._result(label="worse_metric", primary_metric_val_bpb=1.12, submission_bytes_including_tokenizer=15_900_000),
                self._result(label="better_metric", primary_metric_val_bpb=1.10, submission_bytes_including_tokenizer=15_950_000),
                self._result(label="same_metric_smaller", primary_metric_val_bpb=1.10, submission_bytes_including_tokenizer=15_940_000),
            ]
        )
        self.assertEqual(ranked[0].label, "same_metric_smaller")
        self.assertEqual(ranked[1].label, "better_metric")
        self.assertEqual(ranked[2].label, "worse_metric")


if __name__ == "__main__":
    unittest.main()
