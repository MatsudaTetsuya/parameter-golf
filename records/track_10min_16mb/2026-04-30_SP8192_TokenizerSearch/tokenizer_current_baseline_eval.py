from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenizer_fullstack_eval import (
    BASELINE_PROFILE_ENVS,
    BASELINE_PROFILE_LOCAL_4090,
    FullStackCandidateResult,
    FullStackEvalConfig,
    PreparedDatasetConfig,
    load_fullstack_candidates,
    parse_key_value_pairs,
    run_fullstack_evaluation,
)


DEFAULT_SUBMISSION_LIMIT_BYTES = 16_000_000


@dataclass(frozen=True)
class RankedBaselineResult:
    rank: int
    label: str
    within_submission_limit: bool
    primary_metric_name: str
    primary_metric_val_bpb: float | None
    submission_bytes_including_tokenizer: int | None
    result: FullStackCandidateResult


@dataclass(frozen=True)
class CurrentBaselineEvalReport:
    candidates_path: str
    train_script_path: str
    baseline_profile: str
    submission_limit_bytes: int
    ranked_results: list[RankedBaselineResult]


def _sort_key(
    result: FullStackCandidateResult,
    *,
    submission_limit_bytes: int,
) -> tuple[bool, float, int, str]:
    submission_bytes = result.submission_bytes_including_tokenizer
    within_limit = submission_bytes is not None and submission_bytes <= submission_limit_bytes
    metric_bpb = float("inf") if result.primary_metric_val_bpb is None else float(result.primary_metric_val_bpb)
    size_key = int(2**63 - 1) if submission_bytes is None else int(submission_bytes)
    return (not within_limit, metric_bpb, size_key, result.label)


def rank_current_baseline_results(
    results: list[FullStackCandidateResult],
    *,
    submission_limit_bytes: int = DEFAULT_SUBMISSION_LIMIT_BYTES,
) -> list[RankedBaselineResult]:
    ranked: list[RankedBaselineResult] = []
    sorted_results = sorted(
        results,
        key=lambda result: _sort_key(result, submission_limit_bytes=submission_limit_bytes),
    )
    for rank, result in enumerate(sorted_results, start=1):
        submission_bytes = result.submission_bytes_including_tokenizer
        ranked.append(
            RankedBaselineResult(
                rank=rank,
                label=result.label,
                within_submission_limit=(
                    submission_bytes is not None and submission_bytes <= submission_limit_bytes
                ),
                primary_metric_name=result.primary_metric_name,
                primary_metric_val_bpb=result.primary_metric_val_bpb,
                submission_bytes_including_tokenizer=submission_bytes,
                result=result,
            )
        )
    return ranked


def run_current_baseline_evaluation(
    *,
    candidates_path: str | Path,
    docs_jsonl_path: str | Path,
    output_dir: str | Path,
    train_script_path: str | Path | None = None,
    top_k: int | None = None,
    num_val_docs: int | None = None,
    limit_docs: int | None = None,
    shard_size: int = 10**8,
    text_field: str = "text",
    launcher: str = "python",
    nproc_per_node: int = 1,
    python_executable: str | None = None,
    script_args: tuple[str, ...] = (),
    env_overrides: dict[str, str] | None = None,
    submission_limit_bytes: int = DEFAULT_SUBMISSION_LIMIT_BYTES,
    baseline_profile: str = BASELINE_PROFILE_LOCAL_4090,
) -> CurrentBaselineEvalReport:
    candidates_file = Path(candidates_path).expanduser().resolve()
    if train_script_path is None:
        train_script = Path(__file__).with_name("train_gpt.py")
    else:
        train_script = Path(train_script_path).expanduser().resolve()
    if python_executable is None:
        import sys

        python_executable = sys.executable
    fullstack_report = run_fullstack_evaluation(
        candidates=load_fullstack_candidates(candidates_file, top_k=top_k),
        config=FullStackEvalConfig(
            train_script_path=str(train_script),
            output_dir=str(Path(output_dir).expanduser().resolve()),
            docs_jsonl_path=str(Path(docs_jsonl_path).expanduser().resolve()),
            prepared_dataset=PreparedDatasetConfig(
                num_val_docs=num_val_docs,
                shard_size=shard_size,
                limit_docs=limit_docs,
                text_field=text_field,
            ),
            launcher=launcher,
            nproc_per_node=nproc_per_node,
            python_executable=python_executable,
            script_args=script_args,
            env_overrides=env_overrides,
            baseline_profile=baseline_profile,
        ),
    )
    return CurrentBaselineEvalReport(
        candidates_path=str(candidates_file),
        train_script_path=str(train_script),
        baseline_profile=baseline_profile,
        submission_limit_bytes=submission_limit_bytes,
        ranked_results=rank_current_baseline_results(
            fullstack_report.results,
            submission_limit_bytes=submission_limit_bytes,
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tokenizer candidates against the current SP8192 baseline and rank them by sliding no-TTT BPB"
    )
    parser.add_argument("--candidates", required=True, help="Path to a candidate config/report JSON")
    parser.add_argument("--docs-jsonl", required=True, help="Path to docs_selected.jsonl or another local docs JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared datasets, run logs, and reports")
    parser.add_argument("--train-script", default=None, help="Optional override for train_gpt.py path")
    parser.add_argument("--top-k", type=int, default=None, help="Evaluate only the top-K candidates from the input report")
    parser.add_argument("--num-val-docs", type=int, default=None, help="Validation doc count for dataset export")
    parser.add_argument("--limit-docs", type=int, default=None, help="Optional doc limit for smoke evaluations")
    parser.add_argument("--shard-size", type=int, default=10**8, help="Max tokens per exported .bin shard")
    parser.add_argument("--text-field", default="text", help="JSONL field containing document text")
    parser.add_argument("--launcher", choices=("python", "torchrun"), default="python", help="Process launcher")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun --nproc_per_node value")
    parser.add_argument("--python-executable", default=None, help="Python executable for launcher=python")
    parser.add_argument("--script-arg", action="append", default=[], help="Extra positional argument for the train script")
    parser.add_argument("--env", action="append", default=[], help="Environment override in KEY=VALUE form")
    parser.add_argument(
        "--baseline-profile",
        choices=sorted(BASELINE_PROFILE_ENVS),
        default=BASELINE_PROFILE_LOCAL_4090,
        help="Full-stack training/evaluation profile to use",
    )
    parser.add_argument(
        "--submission-limit-bytes",
        type=int,
        default=DEFAULT_SUBMISSION_LIMIT_BYTES,
        help="Submission size cap used when ranking candidates",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_current_baseline_evaluation(
        candidates_path=args.candidates,
        docs_jsonl_path=args.docs_jsonl,
        output_dir=args.output_dir,
        train_script_path=args.train_script,
        top_k=args.top_k,
        num_val_docs=args.num_val_docs,
        limit_docs=args.limit_docs,
        shard_size=args.shard_size,
        text_field=args.text_field,
        launcher=args.launcher,
        nproc_per_node=args.nproc_per_node,
        python_executable=args.python_executable,
        script_args=tuple(args.script_arg),
        env_overrides=parse_key_value_pairs(args.env),
        submission_limit_bytes=args.submission_limit_bytes,
        baseline_profile=args.baseline_profile,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
