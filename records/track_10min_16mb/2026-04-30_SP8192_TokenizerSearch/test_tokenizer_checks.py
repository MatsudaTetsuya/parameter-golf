from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_checks import (
    STANDALONE_SPACE_PIECE,
    assert_has_standalone_space_token,
    build_sentencepiece_byte_lut,
    count_bytes_from_token_ids,
    token_byte_values,
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
    def _train_bpe_model(self, corpus_text: str, *, vocab_size: int) -> spm.SentencePieceProcessor:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
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
            return spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

    def _train_byte_fallback_bpe_model(self, corpus_text: str) -> spm.SentencePieceProcessor:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            corpus_path = tmp_path / "corpus.txt"
            corpus_path.write_text(corpus_text, encoding="utf-8")
            model_prefix = tmp_path / "toy_byte_bpe"
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                model_type="bpe",
                vocab_size=300,
                character_coverage=1.0,
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=-1,
                byte_fallback=True,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            return spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

    def test_assert_rejects_missing_standalone_space_piece(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing a standalone '▁' token"):
            assert_has_standalone_space_token(FakeSentencePieceProcessor())

    def test_toy_bpe_model_contains_standalone_space_piece(self) -> None:
        sp = self._train_bpe_model(
            "hello world\nsmall test corpus\nhello tokenizer search\n",
            vocab_size=32,
        )
        info = assert_has_standalone_space_token(sp)
        self.assertEqual(info.piece, STANDALONE_SPACE_PIECE)
        self.assertFalse(info.is_unknown)
        self.assertFalse(info.is_control)
        self.assertFalse(info.is_byte)
        self.assertFalse(info.is_unused)

    def test_byte_accounting_handles_leading_space_tokens(self) -> None:
        sp = self._train_bpe_model("a b\na b\n", vocab_size=16)
        token_ids = sp.encode("a b", out_type=int)
        pieces = [sp.id_to_piece(token_id) for token_id in token_ids]
        self.assertTrue(any(piece.startswith(STANDALONE_SPACE_PIECE) for piece in pieces))
        lut = build_sentencepiece_byte_lut(sp)
        self.assertEqual(
            count_bytes_from_token_ids(token_ids, lut=lut),
            len(sp.decode(token_ids).encode("utf-8")),
        )

    def test_byte_accounting_counts_standalone_space_piece_in_middle(self) -> None:
        sp = self._train_bpe_model("a b\na b\n", vocab_size=16)
        a_id = int(sp.piece_to_id("a"))
        b_id = int(sp.piece_to_id("b"))
        space_id = int(sp.piece_to_id(STANDALONE_SPACE_PIECE))
        self.assertGreaterEqual(a_id, 0)
        self.assertGreaterEqual(b_id, 0)
        self.assertGreaterEqual(space_id, 0)
        token_ids = [a_id, space_id, b_id]
        lut = build_sentencepiece_byte_lut(sp)
        self.assertEqual(sp.decode(token_ids), "a b")
        self.assertEqual(token_byte_values(token_ids, lut=lut), [1, 1, 1])
        self.assertEqual(count_bytes_from_token_ids(token_ids, lut=lut), 3)

    def test_byte_accounting_ignores_bos_eos_control_bytes(self) -> None:
        sp = self._train_bpe_model("a b\na b\n", vocab_size=16)
        token_ids = [int(sp.bos_id()), *sp.encode("a b", out_type=int), int(sp.eos_id())]
        lut = build_sentencepiece_byte_lut(sp)
        self.assertEqual(count_bytes_from_token_ids(token_ids, lut=lut), len("a b".encode("utf-8")))

    def test_byte_accounting_counts_byte_fallback_tokens_as_single_bytes(self) -> None:
        sp = self._train_byte_fallback_bpe_model("plain ascii corpus only\nhello world\n")
        token_ids = sp.encode("hello é", out_type=int)
        byte_token_ids = [token_id for token_id in token_ids if sp.is_byte(token_id)]
        self.assertTrue(byte_token_ids)
        lut = build_sentencepiece_byte_lut(sp)
        self.assertTrue(all(lut.base_bytes[token_id] == 1 for token_id in byte_token_ids))
        self.assertEqual(
            count_bytes_from_token_ids(token_ids, lut=lut),
            len(sp.decode(token_ids).encode("utf-8")),
        )


if __name__ == "__main__":
    unittest.main()
