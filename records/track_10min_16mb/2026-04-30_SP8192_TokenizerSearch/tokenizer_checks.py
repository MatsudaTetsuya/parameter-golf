from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


STANDALONE_SPACE_PIECE = "▁"


@dataclass(frozen=True)
class StandaloneSpaceTokenInfo:
    token_id: int
    piece: str
    is_byte: bool
    is_control: bool
    is_unknown: bool
    is_unused: bool


@dataclass(frozen=True)
class SentencePieceByteLUT:
    base_bytes: tuple[int, ...]
    has_leading_space: tuple[bool, ...]
    is_boundary_token: tuple[bool, ...]


def load_sentencepiece_model(model_path: str | Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(Path(model_path).expanduser().resolve()))


def find_standalone_space_token(
    sp: spm.SentencePieceProcessor,
) -> StandaloneSpaceTokenInfo | None:
    token_id = int(sp.piece_to_id(STANDALONE_SPACE_PIECE))
    if token_id < 0 or token_id >= int(sp.vocab_size()):
        return None
    piece = sp.id_to_piece(token_id)
    if piece != STANDALONE_SPACE_PIECE:
        return None
    return StandaloneSpaceTokenInfo(
        token_id=token_id,
        piece=piece,
        is_byte=bool(sp.is_byte(token_id)),
        is_control=bool(sp.is_control(token_id)),
        is_unknown=bool(sp.is_unknown(token_id)),
        is_unused=bool(sp.is_unused(token_id)),
    )


def assert_has_standalone_space_token(
    sp: spm.SentencePieceProcessor,
) -> StandaloneSpaceTokenInfo:
    info = find_standalone_space_token(sp)
    if info is None:
        raise AssertionError(
            "SentencePiece model is missing a standalone '▁' token. "
            "This can break byte accounting for spaces and artificially lower BPB."
        )
    if info.is_unknown or info.is_control or info.is_unused or info.is_byte:
        raise AssertionError(
            "SentencePiece standalone '▁' token must be a normal piece, "
            f"got byte={info.is_byte} control={info.is_control} "
            f"unknown={info.is_unknown} unused={info.is_unused}."
        )
    return info


def build_sentencepiece_byte_lut(
    sp: spm.SentencePieceProcessor,
    *,
    vocab_size: int | None = None,
) -> SentencePieceByteLUT:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, 0 if vocab_size is None else int(vocab_size))
    base_bytes = [0] * table_size
    has_leading_space = [False] * table_size
    is_boundary_token = [True] * table_size
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith(STANDALONE_SPACE_PIECE):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return SentencePieceByteLUT(
        base_bytes=tuple(base_bytes),
        has_leading_space=tuple(has_leading_space),
        is_boundary_token=tuple(is_boundary_token),
    )


def token_byte_value(
    token_id: int,
    *,
    prev_token_id: int | None,
    lut: SentencePieceByteLUT,
) -> int:
    if 0 <= token_id < len(lut.base_bytes):
        base_bytes = lut.base_bytes[token_id]
        has_leading_space = lut.has_leading_space[token_id]
    else:
        base_bytes = 0
        has_leading_space = False
    prev_is_boundary = True
    if prev_token_id is not None and 0 <= prev_token_id < len(lut.is_boundary_token):
        prev_is_boundary = lut.is_boundary_token[prev_token_id]
    return base_bytes + int(has_leading_space and not prev_is_boundary)


def token_byte_values(
    token_ids: Sequence[int],
    *,
    lut: SentencePieceByteLUT,
) -> list[int]:
    values: list[int] = []
    prev_token_id: int | None = None
    for token_id in token_ids:
        values.append(token_byte_value(token_id, prev_token_id=prev_token_id, lut=lut))
        prev_token_id = token_id
    return values


def count_bytes_from_token_ids(
    token_ids: Sequence[int],
    *,
    lut: SentencePieceByteLUT,
) -> int:
    return sum(token_byte_values(token_ids, lut=lut))
