from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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
    result: TokenizerScoreBreakdown


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
    ngram_order: int = 3,
    add_k: float = 0.1,
) -> TokenizerCandidateEvaluation:
    manifest_path = resolve_search_split_manifest_path(
        split_dir=split_dir,
        split_manifest_path=split_manifest_path,
    )
    manifest: SearchSplitManifest = load_search_split_manifest(manifest_path)
    result = score_tokenizer_paths(
        model_path=tokenizer_model_path,
        train_docs_path=manifest.search_train_path,
        holdout_docs_path=manifest.search_holdout_path,
        vocab_path=tokenizer_vocab_path,
        config=TokenizerScorerConfig(
            alpha=alpha,
            ngram=NgramProxyConfig(order=ngram_order, add_k=add_k),
            text_field=manifest.text_field,
            extra_asset_paths=tokenizer_asset_paths,
        ),
    )
    vocab_path_value = None if tokenizer_vocab_path is None else str(Path(tokenizer_vocab_path).expanduser().resolve())
    return TokenizerCandidateEvaluation(
        tokenizer_model_path=str(Path(tokenizer_model_path).expanduser().resolve()),
        tokenizer_vocab_path=vocab_path_value,
        split_manifest_path=str(manifest_path),
        search_train_path=manifest.search_train_path,
        search_holdout_path=manifest.search_holdout_path,
        alpha=alpha,
        ngram_order=ngram_order,
        add_k=add_k,
        result=result,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one tokenizer candidate on a fixed search split")
    parser.add_argument("--tokenizer-model", required=True, help="Path to the SentencePiece .model file")
    parser.add_argument("--tokenizer-vocab", default=None, help="Optional path to the tokenizer .vocab file")
    parser.add_argument("--tokenizer-asset", action="append", default=[], help="Additional tokenizer asset path")
    parser.add_argument("--split-dir", default=None, help="Directory containing search_split.manifest.json")
    parser.add_argument("--split-manifest", default=None, help="Path to search_split.manifest.json")
    parser.add_argument("--ngram-order", type=int, default=3, help="Backoff n-gram order")
    parser.add_argument("--add-k", type=float, default=0.1, help="Additive smoothing constant")
    parser.add_argument("--alpha", type=float, default=0.0, help="Tokenizer asset byte penalty coefficient")
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
    )
    print(json.dumps(asdict(evaluation), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
