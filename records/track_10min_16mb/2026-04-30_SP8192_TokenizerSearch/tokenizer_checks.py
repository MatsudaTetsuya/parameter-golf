from __future__ import annotations

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
