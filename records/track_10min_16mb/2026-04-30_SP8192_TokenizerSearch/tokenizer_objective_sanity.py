from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenizer_candidate_eval import TokenizerCandidateEvaluation, evaluate_tokenizer_candidate


@dataclass(frozen=True)
class TokenizerCandidateSpec:
    name: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None = None
    tokenizer_asset_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedTokenizerCandidate:
    rank: int
    name: str
    score_delta_vs_best: float
    proxy_bpb_delta_vs_best: float
    tokenizer_asset_bytes_delta_vs_best: int
    evaluation: TokenizerCandidateEvaluation


@dataclass(frozen=True)
class ObjectiveSanityReport:
    split_manifest_path: str
    alpha: float
    ngram_order: int
    add_k: float
    stream_mode: str
    ranked_candidates: list[RankedTokenizerCandidate]


def load_candidate_specs(path: str | Path) -> list[TokenizerCandidateSpec]:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_specs = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("candidate config must define a non-empty list")
    specs: list[TokenizerCandidateSpec] = []
    seen_names: set[str] = set()
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("each candidate spec must be a JSON object")
        name = str(raw_spec["name"])
        if name in seen_names:
            raise ValueError(f"duplicate candidate name: {name}")
        seen_names.add(name)
        tokenizer_model_path = str(raw_spec["tokenizer_model_path"])
        tokenizer_vocab_path = raw_spec.get("tokenizer_vocab_path")
        asset_paths = raw_spec.get("tokenizer_asset_paths", ())
        if not isinstance(asset_paths, list | tuple):
            raise ValueError(f"tokenizer_asset_paths must be a list for candidate {name}")
        specs.append(
            TokenizerCandidateSpec(
                name=name,
                tokenizer_model_path=tokenizer_model_path,
                tokenizer_vocab_path=None if tokenizer_vocab_path is None else str(tokenizer_vocab_path),
                tokenizer_asset_paths=tuple(str(path) for path in asset_paths),
            )
        )
    return specs


def evaluate_candidate_specs(
    candidate_specs: list[TokenizerCandidateSpec],
    *,
    split_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    alpha: float = 0.0,
    ngram_order: int = 5,
    add_k: float = 0.1,
    stream_mode: str = "fullstack_bos",
    rare_token_freq_threshold: int = 64,
    rare_token_penalty_weight: float = 2.0,
    long_token_byte_threshold: int = 12,
    long_token_penalty_weight: float = 0.05,
    submission_base_bytes: int | None = None,
    submission_limit_bytes: int = 16_000_000,
    submission_guard_bytes: int = 15_950_000,
    submission_penalty_weight: float = 0.0,
) -> ObjectiveSanityReport:
    if not candidate_specs:
        raise ValueError("candidate_specs must not be empty")
    candidate_evaluations: list[tuple[TokenizerCandidateSpec, TokenizerCandidateEvaluation]] = []
    for spec in candidate_specs:
        evaluation = evaluate_tokenizer_candidate(
            tokenizer_model_path=spec.tokenizer_model_path,
            split_dir=split_dir,
            split_manifest_path=split_manifest_path,
            tokenizer_vocab_path=spec.tokenizer_vocab_path,
            tokenizer_asset_paths=spec.tokenizer_asset_paths,
            cache_dir=cache_dir,
            alpha=alpha,
            ngram_order=ngram_order,
            add_k=add_k,
            stream_mode=stream_mode,
            rare_token_freq_threshold=rare_token_freq_threshold,
            rare_token_penalty_weight=rare_token_penalty_weight,
            long_token_byte_threshold=long_token_byte_threshold,
            long_token_penalty_weight=long_token_penalty_weight,
            submission_base_bytes=submission_base_bytes,
            submission_limit_bytes=submission_limit_bytes,
            submission_guard_bytes=submission_guard_bytes,
            submission_penalty_weight=submission_penalty_weight,
        )
        candidate_evaluations.append((spec, evaluation))
    candidate_evaluations.sort(
        key=lambda item: (
            item[1].result.score,
            item[1].result.proxy_bpb_holdout,
            item[0].name,
        )
    )
    best_result = candidate_evaluations[0][1].result
    ranked_candidates: list[RankedTokenizerCandidate] = []
    for rank, (spec, evaluation) in enumerate(candidate_evaluations, start=1):
        ranked_candidates.append(
            RankedTokenizerCandidate(
                rank=rank,
                name=spec.name,
                score_delta_vs_best=evaluation.result.score - best_result.score,
                proxy_bpb_delta_vs_best=evaluation.result.proxy_bpb_holdout - best_result.proxy_bpb_holdout,
                tokenizer_asset_bytes_delta_vs_best=(
                    evaluation.result.tokenizer_asset_bytes - best_result.tokenizer_asset_bytes
                ),
                evaluation=evaluation,
            )
        )
    return ObjectiveSanityReport(
        split_manifest_path=ranked_candidates[0].evaluation.split_manifest_path,
        alpha=alpha,
        ngram_order=ngram_order,
        add_k=add_k,
        stream_mode=stream_mode,
        ranked_candidates=ranked_candidates,
    )


def run_objective_sanity_check(
    *,
    candidates_path: str | Path,
    split_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    alpha: float = 0.0,
    ngram_order: int = 5,
    add_k: float = 0.1,
    stream_mode: str = "fullstack_bos",
    rare_token_freq_threshold: int = 64,
    rare_token_penalty_weight: float = 2.0,
    long_token_byte_threshold: int = 12,
    long_token_penalty_weight: float = 0.05,
    submission_base_bytes: int | None = None,
    submission_limit_bytes: int = 16_000_000,
    submission_guard_bytes: int = 15_950_000,
    submission_penalty_weight: float = 0.0,
) -> ObjectiveSanityReport:
    return evaluate_candidate_specs(
        load_candidate_specs(candidates_path),
        split_dir=split_dir,
        split_manifest_path=split_manifest_path,
        cache_dir=cache_dir,
        alpha=alpha,
        ngram_order=ngram_order,
        add_k=add_k,
        stream_mode=stream_mode,
        rare_token_freq_threshold=rare_token_freq_threshold,
        rare_token_penalty_weight=rare_token_penalty_weight,
        long_token_byte_threshold=long_token_byte_threshold,
        long_token_penalty_weight=long_token_penalty_weight,
        submission_base_bytes=submission_base_bytes,
        submission_limit_bytes=submission_limit_bytes,
        submission_guard_bytes=submission_guard_bytes,
        submission_penalty_weight=submission_penalty_weight,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fixed-split tokenizer objective sanity check")
    parser.add_argument("--candidates", required=True, help="Path to candidate JSON config")
    parser.add_argument("--split-dir", default=None, help="Directory containing search_split.manifest.json")
    parser.add_argument("--split-manifest", default=None, help="Path to search_split.manifest.json")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for persistent evaluation cache")
    parser.add_argument("--ngram-order", type=int, default=5, help="Backoff n-gram order")
    parser.add_argument("--add-k", type=float, default=0.1, help="Additive smoothing constant")
    parser.add_argument("--alpha", type=float, default=0.0, help="Tokenizer asset byte penalty coefficient")
    parser.add_argument("--stream-mode", choices=("fullstack_bos", "legacy_eos"), default="fullstack_bos")
    parser.add_argument("--rare-token-freq-threshold", type=int, default=64)
    parser.add_argument("--rare-token-penalty-weight", type=float, default=2.0)
    parser.add_argument("--long-token-byte-threshold", type=int, default=12)
    parser.add_argument("--long-token-penalty-weight", type=float, default=0.05)
    parser.add_argument("--submission-base-bytes", type=int, default=None)
    parser.add_argument("--submission-limit-bytes", type=int, default=16_000_000)
    parser.add_argument("--submission-guard-bytes", type=int, default=15_950_000)
    parser.add_argument("--submission-penalty-weight", type=float, default=0.0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_objective_sanity_check(
        candidates_path=args.candidates,
        split_dir=args.split_dir,
        split_manifest_path=args.split_manifest,
        cache_dir=args.cache_dir,
        alpha=args.alpha,
        ngram_order=args.ngram_order,
        add_k=args.add_k,
        stream_mode=args.stream_mode,
        rare_token_freq_threshold=args.rare_token_freq_threshold,
        rare_token_penalty_weight=args.rare_token_penalty_weight,
        long_token_byte_threshold=args.long_token_byte_threshold,
        long_token_penalty_weight=args.long_token_penalty_weight,
        submission_base_bytes=args.submission_base_bytes,
        submission_limit_bytes=args.submission_limit_bytes,
        submission_guard_bytes=args.submission_guard_bytes,
        submission_penalty_weight=args.submission_penalty_weight,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
