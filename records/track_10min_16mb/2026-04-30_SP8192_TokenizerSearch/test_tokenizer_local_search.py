from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm

from tokenizer_local_search import (
    FixedVocabLocalSearchConfig,
    MergeCandidate,
    PrunablePiece,
    candidate_search_score,
    class_mismatch_penalty_for_swap,
    count_token_usage,
    exact_piece_exists,
    load_model_proto,
    mine_merge_candidates,
    mine_prunable_pieces,
    run_fixed_vocab_local_search,
    select_diversified_swap_pairs,
    swap_piece_in_proto,
    tokenize_docs_with_ids,
    write_model_artifacts,
)
from tokenizer_search_split import SearchSplitConfig, write_search_split


class TokenizerLocalSearchTest(unittest.TestCase):
    def _write_docs_jsonl(self, docs: list[str], *, docs_val: int) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
        docs_path = tmp_path / "docs_selected.jsonl"
        with docs_path.open("w", encoding="utf-8") as handle:
            for text in docs:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        sidecar_path = tmp_path / "docs_selected.source_manifest.json"
        sidecar_path.write_text(
            json.dumps({"docs_val": docs_val, "num_docs": len(docs)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return docs_path

    def _train_bpe_model(self, corpus_text: str, *, vocab_size: int = 32) -> Path:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        tmp_path = Path(tmp_dir.name)
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
        return Path(str(model_prefix) + ".model")

    def test_mine_merge_candidates_finds_missing_one_step_merge(self) -> None:
        model_path = self._train_bpe_model(
            "hello there world\nhello there world\nhello there world\n",
            vocab_size=16,
        )
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        docs = ["hello there world", "hello there world"]
        tokenized_docs = tokenize_docs_with_ids(sp, docs)
        merge_candidates = mine_merge_candidates(
            sp,
            tokenized_docs=tokenized_docs,
            pool_size=32,
            merge_score_bonus=0.5,
        )
        merged_pieces = {candidate.merged_piece for candidate in merge_candidates}
        self.assertIn("hell", merged_pieces)

    def test_swap_piece_in_proto_makes_new_piece_encodable(self) -> None:
        model_path = self._train_bpe_model(
            "hello there world\nhello there world\nhello there world\n",
            vocab_size=16,
        )
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        docs = ["hello there world", "hello there world"]
        tokenized_docs = tokenize_docs_with_ids(sp, docs)
        merge_candidate = next(
            candidate
            for candidate in mine_merge_candidates(
                sp,
                tokenized_docs=tokenized_docs,
                pool_size=32,
                merge_score_bonus=0.5,
            )
            if candidate.merged_piece == "hell"
        )
        prunable_piece = next(
            piece
            for piece in mine_prunable_pieces(
                sp,
                token_counts=count_token_usage(tokenized_docs),
                pool_size=32,
            )
            if piece.piece != "he" and piece.piece != "ll"
        )
        updated_proto = swap_piece_in_proto(
            load_model_proto(model_path),
            prune_token_id=prunable_piece.token_id,
            regrow_candidate=merge_candidate,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_out = Path(tmp_dir) / "candidate.model"
            vocab_out = Path(tmp_dir) / "candidate.vocab"
            write_model_artifacts(updated_proto, model_path=model_out, vocab_path=vocab_out)
            candidate_sp = spm.SentencePieceProcessor(model_file=str(model_out))
            self.assertTrue(exact_piece_exists(candidate_sp, "hell"))
            self.assertIn("hell", candidate_sp.encode("hello there world", out_type=str))

    def test_select_diversified_swap_pairs_includes_multiple_merge_ranks(self) -> None:
        merge_candidates = [
            MergeCandidate("aa", "a", "a", 10, 11, 100, 1.0),
            MergeCandidate("bb", "b", "b", 20, 21, 90, 0.9),
            MergeCandidate("cc", "c", "c", 30, 31, 80, 0.8),
        ]
        prunable_pieces = [
            PrunablePiece(100, "x", 0, -1.0),
            PrunablePiece(101, "y", 1, -0.5),
            PrunablePiece(102, "z", 2, -0.1),
        ]
        selected = select_diversified_swap_pairs(
            merge_candidates,
            prunable_pieces,
            neighbors_per_candidate=4,
            max_pairs_per_prune=2,
            class_mismatch_penalty=0.0,
        )
        self.assertEqual(len(selected), 4)
        selected_merge_ranks = {pair.merge_rank for pair in selected}
        self.assertIn(0, selected_merge_ranks)
        self.assertIn(1, selected_merge_ranks)
        self.assertIn(2, selected_merge_ranks)
        prune_counts: dict[int, int] = {}
        for pair in selected:
            prune_counts[pair.prune_rank] = prune_counts.get(pair.prune_rank, 0) + 1
        self.assertLessEqual(max(prune_counts.values()), 2)

    def test_class_mismatch_penalty_marks_word_to_punct_or_digit_swap(self) -> None:
        self.assertGreater(
            class_mismatch_penalty_for_swap(
                pruned_piece="▁Ira",
                regrown_piece="20",
                penalty=0.05,
            ),
            0.0,
        )
        self.assertEqual(
            class_mismatch_penalty_for_swap(
                pruned_piece="▁COVID",
                regrown_piece="▁News",
                penalty=0.05,
            ),
            0.0,
        )

    def test_run_fixed_vocab_local_search_generates_and_ranks_candidates(self) -> None:
        docs_path = self._write_docs_jsonl(
            [
                "holdout example one",
                "hello there world",
                "hello there world",
                "hello there world",
                "hello there world",
                "hello there world",
                "hello there world",
            ],
            docs_val=1,
        )
        split_dir = docs_path.parent / "search_split"
        write_search_split(
            docs_path,
            output_dir=split_dir,
            config=SearchSplitConfig(
                search_train_docs=4,
                search_holdout_docs=2,
            ),
        )
        model_path = self._train_bpe_model(
            "hello there world\nhello there world\nhello there world\n",
            vocab_size=16,
        )
        output_dir = docs_path.parent / "local_search"
        report = run_fixed_vocab_local_search(
            base_tokenizer_model_path=model_path,
            base_tokenizer_vocab_path=model_path.with_suffix(".vocab"),
            split_dir=split_dir,
            output_dir=output_dir,
            config=FixedVocabLocalSearchConfig(
                alpha=0.0,
                ngram_order=3,
                add_k=0.1,
                max_steps=2,
                beam_width=1,
                neighbors_per_candidate=4,
                regrow_pool_size=8,
                prune_pool_size=8,
            ),
        )
        self.assertGreaterEqual(len(report.ranked_candidates), 1)
        self.assertTrue((output_dir / "candidates").exists())
        self.assertEqual(report.ranked_candidates, sorted(
            report.ranked_candidates,
            key=lambda item: (
                candidate_search_score(item),
                item.evaluation.result.score,
                item.evaluation.result.proxy_bpb_holdout,
                item.candidate_id,
            ),
        ))
        self.assertTrue(any(candidate.candidate_id != "base" for candidate in report.ranked_candidates))


if __name__ == "__main__":
    unittest.main()
