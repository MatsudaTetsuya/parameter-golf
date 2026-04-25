from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_fullstack_eval import (
    BASELINE_PROFILE_LOCAL_4090,
    BASELINE_PROFILE_LOCAL_4090_MINIFULL,
    FullStackCandidateSpec,
    FullStackEvalConfig,
    baseline_profile_env,
    load_fullstack_candidates,
    parse_train_log_metrics,
    prepare_candidate_dataset,
    run_fullstack_evaluation,
    select_primary_metric,
)


class TokenizerFullStackEvalTest(unittest.TestCase):
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

    def _write_dummy_train_script(self) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        script_path = Path(tmp_dir.name) / "dummy_train.py"
        script_path.write_text(
            textwrap.dedent(
                """
                from __future__ import annotations

                import json
                import os
                from pathlib import Path

                data_path = Path(os.environ["DATA_PATH"])
                tokenizer_path = Path(os.environ["TOKENIZER_PATH"])
                vocab_size = int(os.environ["VOCAB_SIZE"])
                run_id = os.environ["RUN_ID"]
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)
                log_path = logs_dir / f"{run_id}.txt"

                payload = {
                    "data_path": str(data_path),
                    "tokenizer_path": str(tokenizer_path),
                    "vocab_size": vocab_size,
                    "train_batch_tokens": os.environ.get("TRAIN_BATCH_TOKENS"),
                    "sliding_eval_enabled": os.environ.get("SLIDING_EVAL_ENABLED"),
                    "sliding_sample_chunks": os.environ.get("SLIDING_SAMPLE_CHUNKS"),
                    "ttt_enabled": os.environ.get("TTT_ENABLED"),
                    "train_files": sorted(p.name for p in data_path.glob("fineweb_train_*.bin")),
                    "val_files": sorted(p.name for p in data_path.glob("fineweb_val_*.bin")),
                }
                Path("dataset_snapshot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                Path("final_model.pt").write_bytes(b"R" * 19)
                Path("final_model.int6.ptz").write_bytes(b"Q" * 11)

                lines = [
                    f"step:0/5 val_loss:2.0000 val_bpb:1.4000 train_time:0ms step_avg:0.00ms",
                    f"Serialized model: 19 bytes",
                    f"Serialized model quantized+brotli: 11 bytes",
                    f"Total submission size quantized+brotli: 111 bytes",
                    f"final_quantized_roundtrip val_loss:1.5000 val_bpb:1.3000 eval_time:1ms",
                    f"final_quantized_roundtrip_exact val_loss:1.50000000 val_bpb:1.30000000",
                    f"quantized_sampled_sliding_no_ttt_summary_exact seeds:1,2,3 mean_val_loss:1.45000000 mean_val_bpb:1.25000000 worst_val_bpb:1.27000000",
                    f"quantized_sliding_no_ttt val_loss:1.4000 val_bpb:1.2000 eval_time:1ms",
                    f"quantized_sliding_no_ttt_exact val_loss:1.40000000 val_bpb:1.20000000",
                ]
                log_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
                for line in lines:
                    print(line, flush=True)
                """
            ),
            encoding="utf-8",
        )
        return script_path

    def test_parse_train_log_metrics_extracts_logged_and_final_values(self) -> None:
        metrics = parse_train_log_metrics(
            "\n".join(
                [
                    "step:0/5 val_loss:2.1000 val_bpb:1.4100 train_time:0ms step_avg:0.00ms",
                    "step:5/5 val_loss:1.9000 val_bpb:1.2200 train_time:1ms step_avg:0.20ms",
                    "Serialized model: 19 bytes",
                    "Serialized model quantized+brotli: 11 bytes",
                    "Total submission size quantized+brotli: 111 bytes",
                    "final_quantized_roundtrip_exact val_loss:1.80000000 val_bpb:1.11111111",
                    "quantized_sampled_sliding_no_ttt_summary_exact seeds:1,2,3 mean_val_loss:1.75000000 mean_val_bpb:1.05000000 worst_val_bpb:1.07000000",
                    "quantized_sliding_no_ttt_exact val_loss:1.70000000 val_bpb:1.00000000",
                    "legal_ttt_exact val_loss:1.65000000 val_bpb:0.99000000",
                ]
            )
        )
        self.assertEqual(metrics.last_logged_step, 5)
        self.assertAlmostEqual(metrics.last_logged_val_bpb or 0.0, 1.22)
        self.assertAlmostEqual(metrics.final_roundtrip_val_bpb or 0.0, 1.11111111)
        self.assertAlmostEqual(metrics.sliding_no_ttt_val_bpb or 0.0, 1.0)
        self.assertAlmostEqual(metrics.sampled_sliding_no_ttt_mean_val_bpb or 0.0, 1.05)
        self.assertAlmostEqual(metrics.sampled_sliding_no_ttt_worst_val_bpb or 0.0, 1.07)
        self.assertAlmostEqual(metrics.legal_ttt_val_bpb or 0.0, 0.99)
        self.assertEqual(metrics.raw_model_bytes_logged, 19)
        self.assertEqual(metrics.quantized_model_bytes_logged, 11)
        self.assertEqual(metrics.total_submission_bytes_logged, 111)

    def test_baseline_profile_env_exposes_current_local_defaults(self) -> None:
        env = baseline_profile_env(BASELINE_PROFILE_LOCAL_4090)
        self.assertEqual(env["TRAIN_BATCH_TOKENS"], "131072")
        self.assertEqual(env["SLIDING_EVAL_ENABLED"], "1")
        self.assertEqual(env["TTT_ENABLED"], "0")

    def test_minifull_baseline_profile_uses_sampled_sliding(self) -> None:
        env = baseline_profile_env(BASELINE_PROFILE_LOCAL_4090_MINIFULL)
        self.assertEqual(env["TRAIN_BATCH_TOKENS"], "131072")
        self.assertEqual(env["SLIDING_EVAL_ENABLED"], "0")
        self.assertEqual(env["SLIDING_SAMPLE_CHUNKS"], "64")
        self.assertEqual(env["SLIDING_SAMPLE_SEEDS"], "1,2,3")
        self.assertEqual(env["SLIDING_SAMPLE_MODE"], "stratified_random")

    def test_sampled_sliding_is_primary_when_full_sliding_is_absent(self) -> None:
        metrics = parse_train_log_metrics(
            "\n".join(
                [
                    "final_quantized_roundtrip_exact val_loss:1.80000000 val_bpb:1.11111111",
                    "quantized_sampled_sliding_no_ttt_summary_exact seeds:1,2,3 mean_val_loss:1.75000000 mean_val_bpb:1.05000000 worst_val_bpb:1.07000000",
                ]
            )
        )
        name, loss, bpb = select_primary_metric(metrics)
        self.assertEqual(name, "quantized_sampled_sliding_no_ttt_worst_exact")
        self.assertAlmostEqual(loss or 0.0, 1.75)
        self.assertAlmostEqual(bpb or 0.0, 1.07)

    def test_load_fullstack_candidates_supports_ranked_and_reranked_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ranked_path = tmp_path / "ranked.json"
            ranked_path.write_text(
                json.dumps(
                    {
                        "ranked_candidates": [
                            {
                                "rank": 1,
                                "name": "good",
                                "evaluation": {
                                    "tokenizer_model_path": "/tmp/good.model",
                                    "tokenizer_vocab_path": "/tmp/good.vocab",
                                    "result": {"tokenizer_asset_bytes": 123},
                                },
                            }
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            reranked_path = tmp_path / "reranked.json"
            reranked_path.write_text(
                json.dumps(
                    {
                        "reranked_candidates": [
                            {
                                "rerank_rank": 1,
                                "label": "better",
                                "tokenizer_model_path": "/tmp/better.model",
                                "tokenizer_vocab_path": None,
                                "tokenizer_asset_bytes": 456,
                            }
                        ]
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            ranked = load_fullstack_candidates(ranked_path)
            reranked = load_fullstack_candidates(reranked_path)
            self.assertEqual(ranked[0].label, "good")
            self.assertEqual(ranked[0].tokenizer_asset_bytes, 123)
            self.assertEqual(reranked[0].label, "better")
            self.assertEqual(reranked[0].tokenizer_asset_bytes, 456)

    def test_prepare_candidate_dataset_exports_candidate_specific_bins(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "holdout zero",
                "holdout one",
                "hello tokenizer",
                "candidate dataset export",
            ],
            docs_val=2,
        )
        model_path = self._train_bpe_model(
            "holdout zero\nholdout one\nhello tokenizer\ncandidate dataset export\n",
            vocab_size=24,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = prepare_candidate_dataset(
                candidate=FullStackCandidateSpec(
                    label="baseline",
                    tokenizer_model_path=str(model_path),
                    tokenizer_vocab_path=str(model_path.with_suffix(".vocab")),
                ),
                docs_jsonl_path=docs_path,
                output_dir=tmp_dir,
            )
            dataset_dir = Path(dataset.dataset_dir)
            self.assertTrue((dataset_dir / "fineweb_val_000000.bin").exists())
            self.assertTrue((dataset_dir / "fineweb_train_000000.bin").exists())
            self.assertEqual(dataset.num_docs, 4)
            self.assertEqual(dataset.num_val_docs, 2)

    def test_run_fullstack_evaluation_builds_dataset_and_collects_metrics(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "holdout zero",
                "holdout one",
                "hello tokenizer",
                "candidate full stack",
                "another train example",
            ],
            docs_val=2,
        )
        model_path = self._train_bpe_model(
            "holdout zero\nholdout one\nhello tokenizer\ncandidate full stack\nanother train example\n",
            vocab_size=24,
        )
        train_script = self._write_dummy_train_script()
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = run_fullstack_evaluation(
                candidates=[
                    FullStackCandidateSpec(
                        label="baseline",
                        tokenizer_model_path=str(model_path),
                        tokenizer_vocab_path=str(model_path.with_suffix(".vocab")),
                    )
                ],
                config=FullStackEvalConfig(
                    train_script_path=str(train_script),
                    output_dir=tmp_dir,
                    docs_jsonl_path=str(docs_path),
                    baseline_profile=BASELINE_PROFILE_LOCAL_4090,
                ),
            )
            self.assertEqual(len(report.results), 1)
            result = report.results[0]
            self.assertEqual(result.exit_code, 0)
            self.assertAlmostEqual(result.final_roundtrip_val_bpb or 0.0, 1.3)
            self.assertAlmostEqual(result.sampled_sliding_no_ttt_mean_val_bpb or 0.0, 1.25)
            self.assertAlmostEqual(result.sampled_sliding_no_ttt_worst_val_bpb or 0.0, 1.27)
            self.assertAlmostEqual(result.sliding_no_ttt_val_bpb or 0.0, 1.2)
            self.assertEqual(result.primary_metric_name, "quantized_sliding_no_ttt_exact")
            self.assertAlmostEqual(result.primary_metric_val_bpb or 0.0, 1.2)
            self.assertEqual(result.quantized_model_bytes, 11)
            self.assertEqual(result.raw_model_bytes, 19)
            self.assertIsNotNone(result.submission_bytes_including_tokenizer)
            self.assertEqual(result.submission_bytes_excluding_tokenizer, 111)
            self.assertGreater(result.submission_bytes_including_tokenizer or 0, 111)
            dataset_snapshot = Path(result.work_dir) / "dataset_snapshot.json"
            snapshot = json.loads(dataset_snapshot.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["vocab_size"], 24)
            self.assertEqual(snapshot["val_files"], ["fineweb_val_000000.bin"])
            self.assertEqual(snapshot["train_files"], ["fineweb_train_000000.bin"])
            self.assertEqual(snapshot["train_batch_tokens"], "131072")
            self.assertEqual(snapshot["sliding_eval_enabled"], "1")
            self.assertEqual(snapshot["ttt_enabled"], "0")


if __name__ == "__main__":
    unittest.main()
