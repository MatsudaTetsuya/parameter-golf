from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_checks import (
    STANDALONE_SPACE_PIECE,
    assert_has_standalone_space_token,
)


class FakeSentencePieceProcessor:
    def __init__(self) -> None:
        self._pieces = ["<unk>", "<s>", "</s>", "hello"]
        self._piece_to_id = {piece: idx for idx, piece in enumerate(self._pieces)}

    def piece_to_id(self, piece: str) -> int:
        return self._piece_to_id.get(piece, 0)

    def vocab_size(self) -> int:
        return len(self._pieces)

    def id_to_piece(self, token_id: int) -> str:
        return self._pieces[token_id]

    def is_byte(self, token_id: int) -> bool:
        return False

    def is_control(self, token_id: int) -> bool:
        return token_id in (1, 2)

    def is_unknown(self, token_id: int) -> bool:
        return token_id == 0

    def is_unused(self, token_id: int) -> bool:
        return False


class StandaloneSpaceTokenTest(unittest.TestCase):
    def test_assert_rejects_missing_standalone_space_piece(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing a standalone '▁' token"):
            assert_has_standalone_space_token(FakeSentencePieceProcessor())

    def test_toy_bpe_model_contains_standalone_space_piece(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corpus_path = tmp_path / "corpus.txt"
            corpus_path.write_text(
                "hello world\nsmall test corpus\nhello tokenizer search\n",
                encoding="utf-8",
            )
            model_prefix = tmp_path / "toy_bpe"
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                model_type="bpe",
                vocab_size=32,
                character_coverage=1.0,
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=-1,
            )
            sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

        info = assert_has_standalone_space_token(sp)
        self.assertEqual(info.piece, STANDALONE_SPACE_PIECE)
        self.assertFalse(info.is_unknown)
        self.assertFalse(info.is_control)
        self.assertFalse(info.is_byte)
        self.assertFalse(info.is_unused)


if __name__ == "__main__":
    unittest.main()
