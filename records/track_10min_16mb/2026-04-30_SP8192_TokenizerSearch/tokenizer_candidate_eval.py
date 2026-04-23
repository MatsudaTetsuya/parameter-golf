from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenizer_eval_cache import (
    build_tokenizer_eval_cache_key_payload,
    load_cached_tokenizer_evaluation,
    store_cached_tokenizer_evaluation,
    tokenizer_eval_cache_key_from_payload,
)
from tokenizer_scorer import NgramProxyConfig, TokenizerScorerConfig, TokenizerScoreBreakdown, score_tokenizer_paths
from tokenizer_search_split import SEARCH_SPLIT_MANIFEST_FILENAME, SearchSplitManifest, load_search_split_manifest


@dataclass(frozen=True)
class TokenizerCandidateEvaluation:
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None
    split_manifest_path: str
    search_train_path: str
    search_holdout_path: str
    alpha: float
    ngram_order: int
    add_k: float
    stream_mode: str
    result: TokenizerScoreBreakdown


def _evaluation_from_dict(payload: dict) -> TokenizerCandidateEvaluation:
    result_payload = dict(payload["result"])
    return TokenizerCandidateEvaluation(
        tokenizer_model_path=payload["tokenizer_model_path"],
        tokenizer_vocab_path=payload["tokenizer_vocab_path"],
        split_manifest_path=payload["split_manifest_path"],
        search_train_path=payload["search_train_path"],
        search_holdout_path=payload["search_holdout_path"],
        alpha=payload["alpha"],
        ngram_order=payload["ngram_order"],
        add_k=payload["add_k"],
        stream_mode=payload["stream_mode"],
        result=TokenizerScoreBreakdown(**result_payload),
    )


def resolve_search_split_manifest_path(
    *,
    split_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
) -> Path:
    if split_manifest_path is not None:
        return Path(split_manifest_path).expanduser().resolve()
    if split_dir is None:
        raise ValueError("either split_dir or split_manifest_path must be provided")
    return Path(split_dir).expanduser().resolve() / SEARCH_SPLIT_MANIFEST_FILENAME


def evaluate_tokenizer_candidate(
    *,
    tokenizer_model_path: str | Path,
    split_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    tokenizer_vocab_path: str | Path | None = None,
    tokenizer_asset_paths: tuple[str, ...] = (),
    alpha: float = 0.0,
    ngram_order: int = 5,
    add_k: float = 0.03,
    stream_mode: str = "fullstack_bos",
    rare_token_freq_threshold: int = 64,
    rare_token_penalty_weight: float = 0.0,
    long_token_byte_threshold: int = 12,
    long_token_penalty_weight: float = 0.05,
    submission_base_bytes: int | None = None,
    submission_limit_bytes: int = 16_000_000,
    submission_guard_bytes: int = 15_950_000,
    submission_penalty_weight: float = 0.0,
    cache_dir: str | Path | None = None,
) -> TokenizerCandidateEvaluation:
    manifest_path = resolve_search_split_manifest_path(
        split_dir=split_dir,
        split_manifest_path=split_manifest_path,
    )
    manifest: SearchSplitManifest = load_search_split_manifest(manifest_path)
    resolved_model_path = Path(tokenizer_model_path).expanduser().resolve()
    resolved_vocab_path = None if tokenizer_vocab_path is None else Path(tokenizer_vocab_path).expanduser().resolve()
    scorer_config = TokenizerScorerConfig(
        alpha=alpha,
        ngram=NgramProxyConfig(order=ngram_order, add_k=add_k),
        text_field=manifest.text_field,
        extra_asset_paths=tokenizer_asset_paths,
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
    if cache_dir is not None:
        key_payload = build_tokenizer_eval_cache_key_payload(
            tokenizer_model_path=resolved_model_path,
            tokenizer_vocab_path=resolved_vocab_path,
            tokenizer_extra_asset_paths=tokenizer_asset_paths,
            split_manifest_path=manifest_path,
            scorer_config=scorer_config,
        )
        cache_key = tokenizer_eval_cache_key_from_payload(key_payload)
        cached = load_cached_tokenizer_evaluation(cache_dir, cache_key)
        if cached is not None:
            return _evaluation_from_dict(cached["evaluation"])
    result = score_tokenizer_paths(
        model_path=resolved_model_path,
        train_docs_path=manifest.search_train_path,
        holdout_docs_path=manifest.search_holdout_path,
        vocab_path=resolved_vocab_path,
        config=scorer_config,
    )
    vocab_path_value = None if resolved_vocab_path is None else str(resolved_vocab_path)
    evaluation = TokenizerCandidateEvaluation(
        tokenizer_model_path=str(resolved_model_path),
        tokenizer_vocab_path=vocab_path_value,
        split_manifest_path=str(manifest_path),
        search_train_path=manifest.search_train_path,
        search_holdout_path=manifest.search_holdout_path,
        alpha=alpha,
        ngram_order=ngram_order,
        add_k=add_k,
        stream_mode=stream_mode,
        result=result,
    )
    if cache_dir is not None:
        store_cached_tokenizer_evaluation(
            cache_dir,
            cache_key=cache_key,
            key_payload=key_payload,
            evaluation_payload=asdict(evaluation),
        )
    return evaluation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one tokenizer candidate on a fixed search split")
    parser.add_argument("--tokenizer-model", required=True, help="Path to the SentencePiece .model file")
    parser.add_argument("--tokenizer-vocab", default=None, help="Optional path to the tokenizer .vocab file")
    parser.add_argument("--tokenizer-asset", action="append", default=[], help="Additional tokenizer asset path")
    parser.add_argument("--split-dir", default=None, help="Directory containing search_split.manifest.json")
    parser.add_argument("--split-manifest", default=None, help="Path to search_split.manifest.json")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for persistent evaluation cache")
    parser.add_argument("--ngram-order", type=int, default=5, help="Backoff n-gram order")
    parser.add_argument("--add-k", type=float, default=0.03, help="Additive smoothing constant")
    parser.add_argument("--alpha", type=float, default=0.0, help="Tokenizer asset byte penalty coefficient")
    parser.add_argument(
        "--stream-mode",
        choices=("fullstack_bos", "legacy_eos"),
        default="fullstack_bos",
        help="Token stream construction mode for the proxy objective",
    )
    parser.add_argument("--rare-token-freq-threshold", type=int, default=64, help="Train-frequency threshold for rare-token mass")
    parser.add_argument("--rare-token-penalty-weight", type=float, default=0.0, help="Penalty weight for rare-token mass")
    parser.add_argument("--long-token-byte-threshold", type=int, default=12, help="Byte-length threshold for long-token penalty")
    parser.add_argument("--long-token-penalty-weight", type=float, default=0.05, help="Penalty weight for long-token excess length")
    parser.add_argument("--submission-base-bytes", type=int, default=None, help="Base submission bytes before tokenizer assets")
    parser.add_argument("--submission-limit-bytes", type=int, default=16_000_000, help="Submission limit used by the budget penalty")
    parser.add_argument("--submission-guard-bytes", type=int, default=15_950_000, help="Guard-band threshold before submission penalty activates")
    parser.add_argument("--submission-penalty-weight", type=float, default=0.0, help="Quadratic penalty weight for submission overflow")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    evaluation = evaluate_tokenizer_candidate(
        tokenizer_model_path=args.tokenizer_model,
        split_dir=args.split_dir,
        split_manifest_path=args.split_manifest,
        tokenizer_vocab_path=args.tokenizer_vocab,
        tokenizer_asset_paths=tuple(args.tokenizer_asset),
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
        cache_dir=args.cache_dir,
    )
    print(json.dumps(asdict(evaluation), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
