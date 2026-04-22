from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sentencepiece as spm

from tokenizer_checks import (
    assert_has_standalone_space_token,
    build_sentencepiece_byte_lut,
    count_bytes_from_token_ids,
    load_sentencepiece_model,
)


@dataclass(frozen=True)
class NgramProxyConfig:
    order: int = 5
    add_k: float = 0.1

    def __post_init__(self) -> None:
        if self.order <= 0:
            raise ValueError(f"order must be positive, got {self.order}")
        if self.add_k <= 0.0:
            raise ValueError(f"add_k must be positive, got {self.add_k}")


@dataclass(frozen=True)
class TokenizerScorerConfig:
    alpha: float = 0.0
    ngram: NgramProxyConfig = field(default_factory=NgramProxyConfig)
    text_field: str = "text"
    extra_asset_paths: tuple[str, ...] = ()
    stream_mode: str = "fullstack_bos"
    rare_token_freq_threshold: int = 64
    rare_token_penalty_weight: float = 2.0
    long_token_byte_threshold: int = 12
    long_token_penalty_weight: float = 0.05
    submission_base_bytes: int | None = None
    submission_limit_bytes: int = 16_000_000
    submission_guard_bytes: int = 15_950_000
    submission_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.stream_mode not in {"fullstack_bos", "legacy_eos"}:
            raise ValueError(f"unsupported stream_mode {self.stream_mode!r}")
        if self.rare_token_freq_threshold < 0:
            raise ValueError(
                f"rare_token_freq_threshold must be non-negative, got {self.rare_token_freq_threshold}"
            )
        if self.rare_token_penalty_weight < 0.0:
            raise ValueError(
                f"rare_token_penalty_weight must be non-negative, got {self.rare_token_penalty_weight}"
            )
        if self.long_token_byte_threshold < 0:
            raise ValueError(
                f"long_token_byte_threshold must be non-negative, got {self.long_token_byte_threshold}"
            )
        if self.long_token_penalty_weight < 0.0:
            raise ValueError(
                f"long_token_penalty_weight must be non-negative, got {self.long_token_penalty_weight}"
            )
        if self.submission_base_bytes is not None and self.submission_base_bytes < 0:
            raise ValueError(f"submission_base_bytes must be non-negative, got {self.submission_base_bytes}")
        if self.submission_limit_bytes <= 0:
            raise ValueError(f"submission_limit_bytes must be positive, got {self.submission_limit_bytes}")
        if self.submission_guard_bytes <= 0:
            raise ValueError(f"submission_guard_bytes must be positive, got {self.submission_guard_bytes}")
        if self.submission_penalty_weight < 0.0:
            raise ValueError(
                f"submission_penalty_weight must be non-negative, got {self.submission_penalty_weight}"
            )


@dataclass(frozen=True)
class TokenizerScoreBreakdown:
    proxy_bits_per_token: float
    holdout_tokens_per_byte: float
    proxy_bpb_holdout: float
    tokenizer_asset_bytes: int
    score: float
    holdout_token_count: int
    holdout_byte_count: int
    train_token_count: int
    vocab_size: int
    ngram_order: int
    add_k: float
    stream_mode: str
    rare_token_mass: float
    long_token_excess_penalty: float
    submission_size_proxy_bytes: int | None
    submission_penalty: float


def load_docs(path: str | Path, *, text_field: str = "text") -> list[str]:
    docs_path = Path(path).expanduser().resolve()
    if not docs_path.is_file():
        raise FileNotFoundError(f"docs file not found: {docs_path}")
    if docs_path.suffix == ".jsonl":
        docs: list[str] = []
        with docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if text_field not in payload:
                    raise KeyError(f"missing text field {text_field!r} in {docs_path}")
                docs.append(str(payload[text_field]))
        return docs
    docs = []
    with docs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            doc = line.rstrip("\n")
            if not doc:
                continue
            docs.append(doc)
    return docs


def tokenizer_asset_paths(
    model_path: str | Path,
    *,
    vocab_path: str | Path | None = None,
    extra_asset_paths: Sequence[str | Path] = (),
) -> list[Path]:
    model_file = Path(model_path).expanduser().resolve()
    paths = [model_file]
    if vocab_path is None:
        inferred_vocab = model_file.with_suffix(".vocab")
        if inferred_vocab.exists():
            paths.append(inferred_vocab)
    else:
        paths.append(Path(vocab_path).expanduser().resolve())
    for extra_path in extra_asset_paths:
        paths.append(Path(extra_path).expanduser().resolve())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def measure_tokenizer_asset_bytes(
    model_path: str | Path,
    *,
    vocab_path: str | Path | None = None,
    extra_asset_paths: Sequence[str | Path] = (),
) -> int:
    total = 0
    for path in tokenizer_asset_paths(
        model_path,
        vocab_path=vocab_path,
        extra_asset_paths=extra_asset_paths,
    ):
        if not path.exists():
            raise FileNotFoundError(f"tokenizer asset not found: {path}")
        total += path.stat().st_size
    return total


def _bos_context_id(sp: spm.SentencePieceProcessor) -> int:
    bos_id = int(sp.bos_id())
    if bos_id >= 0:
        return bos_id
    unk_id = int(sp.unk_id())
    if unk_id >= 0:
        return unk_id
    return 0


def _tokenize_docs_legacy_eos(
    sp: spm.SentencePieceProcessor,
    docs: Iterable[str],
) -> list[list[int]]:
    eos_id = int(sp.eos_id())
    sequences: list[list[int]] = []
    for doc in docs:
        tokens = list(sp.encode(doc, out_type=int))
        if eos_id >= 0:
            tokens.append(eos_id)
        sequences.append(tokens)
    return sequences


def _build_fullstack_bos_stream(
    sp: spm.SentencePieceProcessor,
    docs: Iterable[str],
) -> list[int]:
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("SentencePiece model must define bos_id for fullstack_bos stream mode")
    stream: list[int] = []
    for doc in docs:
        stream.append(bos_id)
        stream.extend(sp.encode(doc, out_type=int))
    return stream


class AdditiveBackoffNgramLM:
    def __init__(
        self,
        *,
        vocab_size: int,
        order: int,
        add_k: float,
        bos_context_id: int,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if order <= 0:
            raise ValueError(f"order must be positive, got {order}")
        if add_k <= 0.0:
            raise ValueError(f"add_k must be positive, got {add_k}")
        self.vocab_size = vocab_size
        self.order = order
        self.add_k = add_k
        self.bos_context_id = bos_context_id
        self.context_totals = [Counter() for _ in range(order)]
        self.ngram_counts = [Counter() for _ in range(order)]
        self.trained_token_count = 0

    def fit_stream(self, stream: Sequence[int]) -> None:
        prefix = [self.bos_context_id] * (self.order - 1)
        padded = prefix + list(stream)
        for pos in range(self.order - 1, len(padded)):
            target = padded[pos]
            history = padded[max(0, pos - self.order + 1) : pos]
            for context_len in range(self.order):
                context = tuple(history[-context_len:]) if context_len else ()
                self.context_totals[context_len][context] += 1
                self.ngram_counts[context_len][context + (target,)] += 1
            self.trained_token_count += 1

    def token_probability(self, history: Sequence[int], target: int) -> float:
        if self.trained_token_count <= 0:
            raise ValueError("ngram model must be fit before scoring")
        usable_history = tuple(history[-(self.order - 1) :]) if self.order > 1 else ()
        for context_len in range(min(len(usable_history), self.order - 1), -1, -1):
            context = usable_history[-context_len:] if context_len else ()
            context_count = self.context_totals[context_len][context]
            if context_count > 0 or context_len == 0:
                numerator = self.ngram_counts[context_len][context + (target,)] + self.add_k
                denominator = context_count + self.add_k * self.vocab_size
                return float(numerator / denominator)
        raise RuntimeError("unreachable backoff path")

    def average_bits_per_token_stream(self, stream: Sequence[int]) -> float:
        prefix = [self.bos_context_id] * (self.order - 1)
        total_bits = 0.0
        total_tokens = 0
        padded = prefix + list(stream)
        for pos in range(self.order - 1, len(padded)):
            target = padded[pos]
            history = padded[max(0, pos - self.order + 1) : pos]
            total_bits -= math.log2(self.token_probability(history, target))
            total_tokens += 1
        if total_tokens <= 0:
            raise ValueError("holdout token count must be positive")
        return total_bits / total_tokens


def _count_total_bytes(tokenized_docs: Sequence[Sequence[int]], *, sp: spm.SentencePieceProcessor) -> int:
    lut = build_sentencepiece_byte_lut(sp)
    return sum(count_bytes_from_token_ids(tokens, lut=lut) for tokens in tokenized_docs)


def _token_streams_for_mode(
    sp: spm.SentencePieceProcessor,
    docs: Sequence[str],
    *,
    stream_mode: str,
) -> tuple[list[int], list[list[int]]]:
    if stream_mode == "legacy_eos":
        tokenized_docs = _tokenize_docs_legacy_eos(sp, docs)
        flat_stream = [token for doc in tokenized_docs for token in doc]
        return flat_stream, tokenized_docs
    if stream_mode == "fullstack_bos":
        flat_stream = _build_fullstack_bos_stream(sp, docs)
        return flat_stream, [flat_stream]
    raise ValueError(f"unsupported stream_mode {stream_mode!r}")


def _token_byte_lengths_by_id(sp: spm.SentencePieceProcessor) -> list[int]:
    lut = build_sentencepiece_byte_lut(sp)
    token_lengths: list[int] = []
    for token_id in range(int(sp.vocab_size())):
        token_lengths.append(int(lut.base_bytes[token_id] + int(lut.has_leading_space[token_id])))
    return token_lengths


def score_tokenizer_documents(
    sp: spm.SentencePieceProcessor,
    *,
    train_docs: Sequence[str],
    holdout_docs: Sequence[str],
    tokenizer_asset_bytes: int,
    config: TokenizerScorerConfig | None = None,
) -> TokenizerScoreBreakdown:
    scorer_config = TokenizerScorerConfig() if config is None else config
    assert_has_standalone_space_token(sp)
    train_stream, train_tokens = _token_streams_for_mode(
        sp,
        train_docs,
        stream_mode=scorer_config.stream_mode,
    )
    holdout_stream, holdout_tokens = _token_streams_for_mode(
        sp,
        holdout_docs,
        stream_mode=scorer_config.stream_mode,
    )
    train_token_count = len(train_stream)
    holdout_token_count = len(holdout_stream)
    if train_token_count <= 0:
        raise ValueError("train token count must be positive")
    if holdout_token_count <= 0:
        raise ValueError("holdout token count must be positive")
    holdout_byte_count = _count_total_bytes(holdout_tokens, sp=sp)
    if holdout_byte_count <= 0:
        raise ValueError("holdout byte count must be positive")
    ngram = AdditiveBackoffNgramLM(
        vocab_size=int(sp.vocab_size()),
        order=scorer_config.ngram.order,
        add_k=scorer_config.ngram.add_k,
        bos_context_id=_bos_context_id(sp),
    )
    ngram.fit_stream(train_stream)
    proxy_bits_per_token = ngram.average_bits_per_token_stream(holdout_stream)
    holdout_tokens_per_byte = holdout_token_count / holdout_byte_count
    proxy_bpb_holdout = proxy_bits_per_token * holdout_tokens_per_byte
    train_token_counts = Counter(train_stream)
    holdout_token_counts = Counter(holdout_stream)
    rare_token_mass = 0.0
    if scorer_config.rare_token_freq_threshold > 0 and holdout_token_count > 0:
        rare_token_mass = (
            sum(
                count
                for token_id, count in holdout_token_counts.items()
                if train_token_counts.get(token_id, 0) < scorer_config.rare_token_freq_threshold
            )
            / holdout_token_count
        )
    token_byte_lengths = _token_byte_lengths_by_id(sp)
    long_token_excess_penalty = 0.0
    if scorer_config.long_token_penalty_weight > 0.0 and holdout_token_count > 0:
        long_token_excess_penalty = sum(
            (count / holdout_token_count)
            * max(0, token_byte_lengths[token_id] - scorer_config.long_token_byte_threshold) ** 2
            for token_id, count in holdout_token_counts.items()
            if 0 <= token_id < len(token_byte_lengths)
        )
    submission_size_proxy_bytes = None
    submission_penalty = 0.0
    if scorer_config.submission_base_bytes is not None:
        submission_size_proxy_bytes = scorer_config.submission_base_bytes + tokenizer_asset_bytes
        overflow = max(0, submission_size_proxy_bytes - scorer_config.submission_guard_bytes)
        submission_penalty = scorer_config.submission_penalty_weight * float(overflow**2)
    score = (
        proxy_bpb_holdout
        + scorer_config.alpha * tokenizer_asset_bytes
        + scorer_config.rare_token_penalty_weight * rare_token_mass
        + scorer_config.long_token_penalty_weight * long_token_excess_penalty
        + submission_penalty
    )
    return TokenizerScoreBreakdown(
        proxy_bits_per_token=proxy_bits_per_token,
        holdout_tokens_per_byte=holdout_tokens_per_byte,
        proxy_bpb_holdout=proxy_bpb_holdout,
        tokenizer_asset_bytes=tokenizer_asset_bytes,
        score=score,
        holdout_token_count=holdout_token_count,
        holdout_byte_count=holdout_byte_count,
        train_token_count=train_token_count,
        vocab_size=int(sp.vocab_size()),
        ngram_order=scorer_config.ngram.order,
        add_k=scorer_config.ngram.add_k,
        stream_mode=scorer_config.stream_mode,
        rare_token_mass=rare_token_mass,
        long_token_excess_penalty=long_token_excess_penalty,
        submission_size_proxy_bytes=submission_size_proxy_bytes,
        submission_penalty=submission_penalty,
    )


def score_tokenizer_paths(
    *,
    model_path: str | Path,
    train_docs_path: str | Path,
    holdout_docs_path: str | Path,
    vocab_path: str | Path | None = None,
    config: TokenizerScorerConfig | None = None,
) -> TokenizerScoreBreakdown:
    scorer_config = TokenizerScorerConfig() if config is None else config
    sp = load_sentencepiece_model(model_path)
    tokenizer_bytes = measure_tokenizer_asset_bytes(
        model_path,
        vocab_path=vocab_path,
        extra_asset_paths=scorer_config.extra_asset_paths,
    )
    train_docs = load_docs(train_docs_path, text_field=scorer_config.text_field)
    holdout_docs = load_docs(holdout_docs_path, text_field=scorer_config.text_field)
    return score_tokenizer_documents(
        sp,
        train_docs=train_docs,
        holdout_docs=holdout_docs,
        tokenizer_asset_bytes=tokenizer_bytes,
        config=scorer_config,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a SentencePiece tokenizer on train/holdout docs")
    parser.add_argument("--tokenizer-model", required=True, help="Path to the SentencePiece .model file")
    parser.add_argument("--tokenizer-vocab", default=None, help="Optional path to the tokenizer .vocab file")
    parser.add_argument("--tokenizer-asset", action="append", default=[], help="Additional tokenizer asset path")
    parser.add_argument("--train-docs", required=True, help="Train docs path (.txt or .jsonl)")
    parser.add_argument("--holdout-docs", required=True, help="Holdout docs path (.txt or .jsonl)")
    parser.add_argument("--text-field", default="text", help="JSONL text field name")
    parser.add_argument("--ngram-order", type=int, default=5, help="Backoff n-gram order")
    parser.add_argument("--add-k", type=float, default=0.1, help="Additive smoothing constant")
    parser.add_argument("--alpha", type=float, default=0.0, help="Tokenizer asset byte penalty coefficient")
    parser.add_argument(
        "--stream-mode",
        choices=("fullstack_bos", "legacy_eos"),
        default="fullstack_bos",
        help="Token stream construction mode for proxy scoring",
    )
    parser.add_argument("--rare-token-freq-threshold", type=int, default=64, help="Train-frequency threshold for rare-token mass")
    parser.add_argument("--rare-token-penalty-weight", type=float, default=2.0, help="Penalty weight on holdout mass assigned to rare train tokens")
    parser.add_argument("--long-token-byte-threshold", type=int, default=12, help="Byte-length threshold above which tokens are penalized")
    parser.add_argument("--long-token-penalty-weight", type=float, default=0.05, help="Penalty weight on holdout mass assigned to overly long tokens")
    parser.add_argument("--submission-base-bytes", type=int, default=None, help="Optional fixed base submission bytes used for budget-aware penalty")
    parser.add_argument("--submission-limit-bytes", type=int, default=16_000_000, help="Submission byte cap")
    parser.add_argument("--submission-guard-bytes", type=int, default=15_950_000, help="Guard-band submission byte cap")
    parser.add_argument("--submission-penalty-weight", type=float, default=0.0, help="Quadratic penalty weight for submission budget overflow")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = score_tokenizer_paths(
        model_path=args.tokenizer_model,
        train_docs_path=args.train_docs,
        holdout_docs_path=args.holdout_docs,
        vocab_path=args.tokenizer_vocab,
        config=TokenizerScorerConfig(
            alpha=args.alpha,
            ngram=NgramProxyConfig(order=args.ngram_order, add_k=args.add_k),
            text_field=args.text_field,
            extra_asset_paths=tuple(args.tokenizer_asset),
            stream_mode=args.stream_mode,
            rare_token_freq_threshold=args.rare_token_freq_threshold,
            rare_token_penalty_weight=args.rare_token_penalty_weight,
            long_token_byte_threshold=args.long_token_byte_threshold,
            long_token_penalty_weight=args.long_token_penalty_weight,
            submission_base_bytes=args.submission_base_bytes,
            submission_limit_bytes=args.submission_limit_bytes,
            submission_guard_bytes=args.submission_guard_bytes,
            submission_penalty_weight=args.submission_penalty_weight,
        ),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
