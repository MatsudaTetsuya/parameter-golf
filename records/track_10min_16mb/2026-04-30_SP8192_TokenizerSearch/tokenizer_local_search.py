from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

from tokenizer_candidate_eval import TokenizerCandidateEvaluation, evaluate_tokenizer_candidate
from tokenizer_checks import STANDALONE_SPACE_PIECE
from tokenizer_scorer import load_docs
from tokenizer_search_split import SEARCH_SPLIT_MANIFEST_FILENAME, load_search_split_manifest


@dataclass(frozen=True)
class FixedVocabLocalSearchConfig:
    alpha: float = 0.0
    ngram_order: int = 3
    add_k: float = 0.1
    max_steps: int = 4
    beam_width: int = 1
    neighbors_per_candidate: int = 16
    regrow_pool_size: int = 64
    prune_pool_size: int = 64
    merge_score_bonus: float = 0.5
    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "ngram_order": self.ngram_order,
            "max_steps": self.max_steps,
            "beam_width": self.beam_width,
            "neighbors_per_candidate": self.neighbors_per_candidate,
            "regrow_pool_size": self.regrow_pool_size,
            "prune_pool_size": self.prune_pool_size,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.add_k <= 0.0:
            raise ValueError(f"add_k must be positive, got {self.add_k}")
        if self.merge_score_bonus <= 0.0:
            raise ValueError(f"merge_score_bonus must be positive, got {self.merge_score_bonus}")
        if self.min_improvement < 0.0:
            raise ValueError(f"min_improvement must be non-negative, got {self.min_improvement}")


@dataclass(frozen=True)
class MergeCandidate:
    merged_piece: str
    left_piece: str
    right_piece: str
    left_token_id: int
    right_token_id: int
    pair_count: int
    proposed_score: float


@dataclass(frozen=True)
class PrunablePiece:
    token_id: int
    piece: str
    count: int
    score: float


@dataclass(frozen=True)
class PieceSwapOperation:
    pruned_token_id: int
    pruned_piece: str
    pruned_count: int
    regrown_piece: str
    regrow_pair_count: int
    left_piece: str
    right_piece: str
    assigned_score: float


@dataclass(frozen=True)
class LocalSearchCandidate:
    candidate_id: str
    parent_candidate_id: str | None
    step: int
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None
    model_sha256: str
    evaluation: TokenizerCandidateEvaluation
    swaps: tuple[PieceSwapOperation, ...] = ()


@dataclass(frozen=True)
class FixedVocabLocalSearchReport:
    split_manifest_path: str
    output_dir: str
    base_tokenizer_model_path: str
    base_tokenizer_vocab_path: str | None
    config: FixedVocabLocalSearchConfig
    ranked_candidates: list[LocalSearchCandidate]


def load_model_proto(model_path: str | Path) -> sp_pb2.ModelProto:
    proto = sp_pb2.ModelProto()
    proto.ParseFromString(Path(model_path).expanduser().resolve().read_bytes())
    return proto


def exact_piece_exists(sp: spm.SentencePieceProcessor, piece: str) -> bool:
    token_id = int(sp.piece_to_id(piece))
    return 0 <= token_id < int(sp.vocab_size()) and sp.id_to_piece(token_id) == piece


def is_prunable_piece(sp: spm.SentencePieceProcessor, token_id: int) -> bool:
    if token_id < 0 or token_id >= int(sp.vocab_size()):
        return False
    if sp.is_unknown(token_id) or sp.is_control(token_id) or sp.is_unused(token_id) or sp.is_byte(token_id):
        return False
    return sp.id_to_piece(token_id) != STANDALONE_SPACE_PIECE


def can_regrow_from_pair(
    *,
    left_piece: str,
    right_piece: str,
    merged_piece: str,
    max_piece_length: int,
) -> bool:
    if right_piece.startswith(STANDALONE_SPACE_PIECE):
        return False
    if STANDALONE_SPACE_PIECE in merged_piece[1:]:
        return False
    return len(merged_piece) <= max_piece_length


def tokenize_docs_with_ids(
    sp: spm.SentencePieceProcessor,
    docs: list[str],
) -> list[list[int]]:
    return [list(sp.encode(doc, out_type=int)) for doc in docs]


def count_token_usage(tokenized_docs: list[list[int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for doc in tokenized_docs:
        counts.update(doc)
    return counts


def mine_merge_candidates(
    sp: spm.SentencePieceProcessor,
    *,
    tokenized_docs: list[list[int]],
    pool_size: int,
    merge_score_bonus: float,
) -> list[MergeCandidate]:
    max_piece_length = 16
    pair_counts: Counter[tuple[int, int]] = Counter()
    for doc in tokenized_docs:
        for left_token_id, right_token_id in zip(doc, doc[1:]):
            if not is_prunable_piece(sp, left_token_id) or not is_prunable_piece(sp, right_token_id):
                continue
            left_piece = sp.id_to_piece(left_token_id)
            right_piece = sp.id_to_piece(right_token_id)
            merged_piece = left_piece + right_piece
            if not can_regrow_from_pair(
                left_piece=left_piece,
                right_piece=right_piece,
                merged_piece=merged_piece,
                max_piece_length=max_piece_length,
            ):
                continue
            if exact_piece_exists(sp, merged_piece):
                continue
            pair_counts[(left_token_id, right_token_id)] += 1
    ranked_pairs = sorted(
        pair_counts.items(),
        key=lambda item: (
            -item[1],
            -len((sp.id_to_piece(item[0][0]) + sp.id_to_piece(item[0][1])).encode("utf-8")),
            sp.id_to_piece(item[0][0]) + sp.id_to_piece(item[0][1]),
        ),
    )
    candidates: list[MergeCandidate] = []
    for (left_token_id, right_token_id), pair_count in ranked_pairs[:pool_size]:
        left_piece = sp.id_to_piece(left_token_id)
        right_piece = sp.id_to_piece(right_token_id)
        candidates.append(
            MergeCandidate(
                merged_piece=left_piece + right_piece,
                left_piece=left_piece,
                right_piece=right_piece,
                left_token_id=left_token_id,
                right_token_id=right_token_id,
                pair_count=pair_count,
                proposed_score=max(float(sp.get_score(left_token_id)), float(sp.get_score(right_token_id))) + merge_score_bonus,
            )
        )
    return candidates


def mine_prunable_pieces(
    sp: spm.SentencePieceProcessor,
    *,
    token_counts: Counter[int],
    pool_size: int,
) -> list[PrunablePiece]:
    prunable: list[PrunablePiece] = []
    for token_id in range(int(sp.vocab_size())):
        if not is_prunable_piece(sp, token_id):
            continue
        prunable.append(
            PrunablePiece(
                token_id=token_id,
                piece=sp.id_to_piece(token_id),
                count=int(token_counts[token_id]),
                score=float(sp.get_score(token_id)),
            )
        )
    prunable.sort(key=lambda item: (item.count, item.score, item.piece))
    return prunable[:pool_size]


def swap_piece_in_proto(
    proto: sp_pb2.ModelProto,
    *,
    prune_token_id: int,
    regrow_candidate: MergeCandidate,
) -> sp_pb2.ModelProto:
    updated = sp_pb2.ModelProto()
    updated.CopyFrom(proto)
    pieces = [piece.piece for piece in updated.pieces]
    if regrow_candidate.merged_piece in pieces and pieces[prune_token_id] != regrow_candidate.merged_piece:
        raise ValueError(f"piece already exists in model: {regrow_candidate.merged_piece}")
    updated_piece = updated.pieces[prune_token_id]
    updated_piece.piece = regrow_candidate.merged_piece
    updated_piece.score = regrow_candidate.proposed_score
    updated_piece.type = sp_pb2.ModelProto.SentencePiece.NORMAL
    return updated


def write_model_artifacts(
    proto: sp_pb2.ModelProto,
    *,
    model_path: str | Path,
    vocab_path: str | Path,
) -> str:
    model_file = Path(model_path).expanduser().resolve()
    vocab_file = Path(vocab_path).expanduser().resolve()
    model_file.parent.mkdir(parents=True, exist_ok=True)
    vocab_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = proto.SerializeToString()
    model_file.write_bytes(serialized)
    with vocab_file.open("w", encoding="utf-8") as handle:
        for piece in proto.pieces:
            handle.write(f"{piece.piece}\t{piece.score}\n")
    return hashlib.sha256(serialized).hexdigest()


def _resolve_split_manifest_path(
    *,
    split_dir: str | Path | None,
    split_manifest_path: str | Path | None,
) -> Path:
    if split_manifest_path is not None:
        return Path(split_manifest_path).expanduser().resolve()
    if split_dir is None:
        raise ValueError("either split_dir or split_manifest_path must be provided")
    return Path(split_dir).expanduser().resolve() / SEARCH_SPLIT_MANIFEST_FILENAME


def _rank_candidates(candidates: list[LocalSearchCandidate]) -> list[LocalSearchCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            item.evaluation.result.score,
            item.evaluation.result.proxy_bpb_holdout,
            item.candidate_id,
        ),
    )


def run_fixed_vocab_local_search(
    *,
    base_tokenizer_model_path: str | Path,
    split_dir: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
    output_dir: str | Path,
    base_tokenizer_vocab_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    config: FixedVocabLocalSearchConfig | None = None,
) -> FixedVocabLocalSearchReport:
    search_config = FixedVocabLocalSearchConfig() if config is None else config
    manifest_path = _resolve_split_manifest_path(
        split_dir=split_dir,
        split_manifest_path=split_manifest_path,
    )
    manifest = load_search_split_manifest(manifest_path)
    search_train_docs = load_docs(manifest.search_train_path, text_field=manifest.text_field)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = Path(base_tokenizer_model_path).expanduser().resolve()
    base_vocab_path = None if base_tokenizer_vocab_path is None else Path(base_tokenizer_vocab_path).expanduser().resolve()
    base_evaluation = evaluate_tokenizer_candidate(
        tokenizer_model_path=base_model_path,
        split_manifest_path=manifest_path,
        tokenizer_vocab_path=base_vocab_path,
        cache_dir=cache_dir,
        alpha=search_config.alpha,
        ngram_order=search_config.ngram_order,
        add_k=search_config.add_k,
    )
    base_sha256 = hashlib.sha256(base_model_path.read_bytes()).hexdigest()
    base_candidate = LocalSearchCandidate(
        candidate_id="base",
        parent_candidate_id=None,
        step=0,
        tokenizer_model_path=str(base_model_path),
        tokenizer_vocab_path=None if base_vocab_path is None else str(base_vocab_path),
        model_sha256=base_sha256,
        evaluation=base_evaluation,
        swaps=(),
    )
    all_candidates: dict[str, LocalSearchCandidate] = {base_sha256: base_candidate}
    beam: list[LocalSearchCandidate] = [base_candidate]
    best_score = base_candidate.evaluation.result.score
    next_candidate_index = 1

    for step in range(1, search_config.max_steps + 1):
        step_candidates: list[LocalSearchCandidate] = []
        for beam_candidate in beam:
            sp = spm.SentencePieceProcessor(model_file=beam_candidate.tokenizer_model_path)
            tokenized_docs = tokenize_docs_with_ids(sp, search_train_docs)
            token_counts = count_token_usage(tokenized_docs)
            merge_candidates = mine_merge_candidates(
                sp,
                tokenized_docs=tokenized_docs,
                pool_size=search_config.regrow_pool_size,
                merge_score_bonus=search_config.merge_score_bonus,
            )
            prunable_pieces = mine_prunable_pieces(
                sp,
                token_counts=token_counts,
                pool_size=search_config.prune_pool_size,
            )
            if not merge_candidates or not prunable_pieces:
                continue
            base_proto = load_model_proto(beam_candidate.tokenizer_model_path)
            generated_for_parent = 0
            for merge_candidate in merge_candidates:
                for prunable_piece in prunable_pieces:
                    if prunable_piece.token_id in {merge_candidate.left_token_id, merge_candidate.right_token_id}:
                        continue
                    if prunable_piece.piece == merge_candidate.merged_piece:
                        continue
                    candidate_id = f"step{step:02d}_{next_candidate_index:04d}"
                    updated_proto = swap_piece_in_proto(
                        base_proto,
                        prune_token_id=prunable_piece.token_id,
                        regrow_candidate=merge_candidate,
                    )
                    model_path = candidate_dir / f"{candidate_id}.model"
                    vocab_path = candidate_dir / f"{candidate_id}.vocab"
                    model_sha256 = write_model_artifacts(
                        updated_proto,
                        model_path=model_path,
                        vocab_path=vocab_path,
                    )
                    if model_sha256 in all_candidates:
                        model_path.unlink(missing_ok=True)
                        vocab_path.unlink(missing_ok=True)
                        continue
                    evaluation = evaluate_tokenizer_candidate(
                        tokenizer_model_path=model_path,
                        split_manifest_path=manifest_path,
                        tokenizer_vocab_path=vocab_path,
                        cache_dir=cache_dir,
                        alpha=search_config.alpha,
                        ngram_order=search_config.ngram_order,
                        add_k=search_config.add_k,
                    )
                    candidate = LocalSearchCandidate(
                        candidate_id=candidate_id,
                        parent_candidate_id=beam_candidate.candidate_id,
                        step=step,
                        tokenizer_model_path=str(model_path),
                        tokenizer_vocab_path=str(vocab_path),
                        model_sha256=model_sha256,
                        evaluation=evaluation,
                        swaps=beam_candidate.swaps
                        + (
                            PieceSwapOperation(
                                pruned_token_id=prunable_piece.token_id,
                                pruned_piece=prunable_piece.piece,
                                pruned_count=prunable_piece.count,
                                regrown_piece=merge_candidate.merged_piece,
                                regrow_pair_count=merge_candidate.pair_count,
                                left_piece=merge_candidate.left_piece,
                                right_piece=merge_candidate.right_piece,
                                assigned_score=merge_candidate.proposed_score,
                            ),
                        ),
                    )
                    all_candidates[model_sha256] = candidate
                    step_candidates.append(candidate)
                    generated_for_parent += 1
                    next_candidate_index += 1
                    if generated_for_parent >= search_config.neighbors_per_candidate:
                        break
                if generated_for_parent >= search_config.neighbors_per_candidate:
                    break
        if not step_candidates:
            break
        ranked_all = _rank_candidates(list(all_candidates.values()))
        new_best_score = ranked_all[0].evaluation.result.score
        beam = ranked_all[: search_config.beam_width]
        if best_score - new_best_score <= search_config.min_improvement:
            best_score = new_best_score
            break
        best_score = new_best_score

    ranked_candidates = _rank_candidates(list(all_candidates.values()))
    return FixedVocabLocalSearchReport(
        split_manifest_path=str(manifest_path),
        output_dir=str(output_root),
        base_tokenizer_model_path=str(base_model_path),
        base_tokenizer_vocab_path=None if base_vocab_path is None else str(base_vocab_path),
        config=search_config,
        ranked_candidates=ranked_candidates,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-vocab local search for a SentencePiece BPE tokenizer")
    parser.add_argument("--base-tokenizer-model", required=True, help="Path to the base SentencePiece .model file")
    parser.add_argument("--base-tokenizer-vocab", default=None, help="Optional path to the base SentencePiece .vocab file")
    parser.add_argument("--split-dir", default=None, help="Directory containing search_split.manifest.json")
    parser.add_argument("--split-manifest", default=None, help="Path to search_split.manifest.json")
    parser.add_argument("--output-dir", required=True, help="Directory for generated candidate models and search reports")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for persistent evaluation cache")
    parser.add_argument("--alpha", type=float, default=0.0, help="Tokenizer asset byte penalty coefficient")
    parser.add_argument("--ngram-order", type=int, default=3, help="Backoff n-gram order")
    parser.add_argument("--add-k", type=float, default=0.1, help="Additive smoothing constant")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum local-search steps")
    parser.add_argument("--beam-width", type=int, default=1, help="Beam width")
    parser.add_argument("--neighbors-per-candidate", type=int, default=16, help="Neighbor budget per beam candidate")
    parser.add_argument("--regrow-pool-size", type=int, default=64, help="Number of regrow merge proposals to consider")
    parser.add_argument("--prune-pool-size", type=int, default=64, help="Number of low-frequency tail pieces to consider pruning")
    parser.add_argument("--merge-score-bonus", type=float, default=0.5, help="Score bonus assigned to regrown one-step merges")
    parser.add_argument("--min-improvement", type=float, default=0.0, help="Minimum score improvement required to continue")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_fixed_vocab_local_search(
        base_tokenizer_model_path=args.base_tokenizer_model,
        base_tokenizer_vocab_path=args.base_tokenizer_vocab,
        split_dir=args.split_dir,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        config=FixedVocabLocalSearchConfig(
            alpha=args.alpha,
            ngram_order=args.ngram_order,
            add_k=args.add_k,
            max_steps=args.max_steps,
            beam_width=args.beam_width,
            neighbors_per_candidate=args.neighbors_per_candidate,
            regrow_pool_size=args.regrow_pool_size,
            prune_pool_size=args.prune_pool_size,
            merge_score_bonus=args.merge_score_bonus,
            min_improvement=args.min_improvement,
        ),
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
