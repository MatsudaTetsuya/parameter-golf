# WIP: SP8192 Tokenizer Search

Offline tokenizer search for an intended SP8192 stack under the 10min / 16MB track.

This branch is keeping the tokenizer-search tooling and `train_gpt.py` branch-owned and readable. When ideas are adapted from prior records, the intent is to port only the needed mechanism and leave concise source comments in code rather than copying whole submission scripts.

## Key Techniques

1. **Fixed-vocab SentencePiece BPE search at 8192** — the initial search loop keeps vocab size fixed and explores prune/regrow style local edits.
2. **Proxy objective with size awareness** — current scorer uses `proxy_bpb_ngram_holdout(tau) + alpha * tokenizer_asset_bytes(tau)` on a train-only held-out slice.
3. **BPB-safe byte accounting** — explicit checks for standalone `▁`, byte fallback pieces, and BOS/EOS/control handling to avoid artificial BPB gains from tokenizer accounting bugs.
4. **Candidate reranking pipeline** — fixed split evaluation, local search, top-K short proxy training rerank, and a full-stack evaluation harness that exports candidate-specific tokenized datasets before running a `train_gpt.py`-style evaluator.
5. **Planned downstream stack** — SentencePiece BPE 8192, GPTQ including embeddings, SDClip, XSA-all, depth recurrence, QK-gain, and legal score-first TTT.

## Credits

- **@clarkkev** — SP8192 base direction, GPTQ embeddings, SDClip, MuonEq-R, and depth-recurrence lineage from the 2026-04-05 SP8192 record and PR #1394.
- **@dexhunter** — legal TTT on SP8192 and later recurrence extensions, especially the 2026-04-06 and 2026-04-09 SP8192 records.
- **@abaybektursun** — score-first TTT precedent and GPTQ/XSA lineage, especially PR #549 and PR #1019.
- **@Robby955** — parallel residuals on the SP8192 stack, especially the 2026-04-08 record and PR #1412.
- **@msisovic** — parallel residuals / recurrence ideas referenced by later SP8192 records, especially PR #1204 and PR #1260.
- **OpenAI Parameter Golf participants** — prior records and writeups in `records/track_10min_16mb/` that clarified the stack this branch is targeting.
