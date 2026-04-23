from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn

from tokenizer_eval_cache import file_fingerprint, sha256_bytes
from tokenizer_search_split import load_search_split_manifest
from tokenizer_scorer import load_docs


PROXY_TRAIN_CACHE_VERSION = 2
PROXY_TRAIN_CACHE_SUBDIR = "proxy_training"


@dataclass(frozen=True)
class ProxyTrainingConfig:
    alpha: float | None = None
    top_k: int = 8
    train_steps: int = 60
    batch_size: int = 16
    seq_len: int = 64
    model_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_mult: int = 2
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    seed: int = 1337
    device: str = "auto"

    def __post_init__(self) -> None:
        integer_fields = {
            "top_k": self.top_k,
            "train_steps": self.train_steps,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_mult": self.mlp_mult,
            "seed": self.seed,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.alpha is not None and self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative when set, got {self.alpha}")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim must be divisible by num_heads, got model_dim={self.model_dim}, num_heads={self.num_heads}"
            )


@dataclass(frozen=True)
class ProxyRerankCandidateSpec:
    label: str
    original_rank: int
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None
    split_manifest_path: str
    tokenizer_asset_bytes: int
    holdout_tokens_per_byte: float
    original_score: float
    original_proxy_bpb_holdout: float


@dataclass(frozen=True)
class ProxyTrainingCoreMetrics:
    trained_bits_per_token: float
    final_train_bits_per_token: float
    train_stream_token_count: int
    holdout_stream_token_count: int
    train_steps: int
    seq_len: int
    batch_size: int
    model_dim: int
    num_layers: int
    num_heads: int
    mlp_mult: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str
    cache_hit: bool


@dataclass(frozen=True)
class ProxyRerankedCandidate:
    rerank_rank: int
    label: str
    original_rank: int
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None
    tokenizer_asset_bytes: int
    original_score: float
    original_proxy_bpb_holdout: float
    trained_bits_per_token: float
    final_train_bits_per_token: float
    proxy_train_bpb_holdout: float
    rerank_score: float
    holdout_tokens_per_byte: float
    train_stream_token_count: int
    holdout_stream_token_count: int
    cache_hit: bool


@dataclass(frozen=True)
class ProxyRerankReport:
    source_report_path: str
    split_manifest_path: str
    alpha: float
    config: ProxyTrainingConfig
    reranked_candidates: list[ProxyRerankedCandidate]


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return self.proj(y)


class TinyTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_mult: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_dim)
        self.attn = TinyCausalSelfAttention(model_dim, num_heads)
        self.ln_2 = nn.LayerNorm(model_dim)
        hidden_dim = model_dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_mult: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(seq_len, model_dim)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(model_dim=model_dim, num_heads=num_heads, mlp_mult=mlp_mult) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        h = self.tok_emb(x) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_proxy_train_cache_key_payload(
    *,
    tokenizer_model_path: str | Path,
    split_manifest_path: str | Path,
    config: ProxyTrainingConfig,
    device: str,
) -> dict[str, Any]:
    return {
        "version": PROXY_TRAIN_CACHE_VERSION,
        "tokenizer_model": asdict(file_fingerprint(tokenizer_model_path)),
        "split_manifest": asdict(file_fingerprint(split_manifest_path)),
        "config": {
            "train_steps": config.train_steps,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "model_dim": config.model_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "mlp_mult": config.mlp_mult,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "device": device,
            "stream_mode": "fullstack_bos",
        },
    }


def proxy_train_cache_key(payload: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def resolve_proxy_train_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir).expanduser().resolve() / PROXY_TRAIN_CACHE_SUBDIR / f"{cache_key}.json"


def load_cached_proxy_training(cache_dir: str | Path, cache_key: str) -> dict[str, Any] | None:
    cache_path = resolve_proxy_train_cache_path(cache_dir, cache_key)
    if not cache_path.is_file():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def store_cached_proxy_training(
    cache_dir: str | Path,
    *,
    cache_key: str,
    key_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> Path:
    cache_path = resolve_proxy_train_cache_path(cache_dir, cache_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "key": key_payload,
        "metrics": metrics_payload,
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(cache_path)
    return cache_path


def _core_metrics_from_payload(payload: dict[str, Any], *, cache_hit: bool) -> ProxyTrainingCoreMetrics:
    return ProxyTrainingCoreMetrics(cache_hit=cache_hit, **payload)


def load_top_candidates_from_report(report_path: str | Path, *, top_k: int) -> list[ProxyRerankCandidateSpec]:
    report_file = Path(report_path).expanduser().resolve()
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    ranked_candidates = payload.get("ranked_candidates")
    if not isinstance(ranked_candidates, list) or not ranked_candidates:
        raise ValueError(f"report has no ranked_candidates: {report_file}")
    selected = ranked_candidates[:top_k]
    specs: list[ProxyRerankCandidateSpec] = []
    for index, candidate in enumerate(selected, start=1):
        evaluation = candidate["evaluation"]
        result = evaluation["result"]
        label = str(candidate.get("name", candidate.get("candidate_id", f"candidate_{index}")))
        specs.append(
            ProxyRerankCandidateSpec(
                label=label,
                original_rank=int(candidate.get("rank", index)),
                tokenizer_model_path=str(evaluation["tokenizer_model_path"]),
                tokenizer_vocab_path=evaluation.get("tokenizer_vocab_path"),
                split_manifest_path=str(evaluation["split_manifest_path"]),
                tokenizer_asset_bytes=int(result["tokenizer_asset_bytes"]),
                holdout_tokens_per_byte=float(result["holdout_tokens_per_byte"]),
                original_score=float(result["score"]),
                original_proxy_bpb_holdout=float(result["proxy_bpb_holdout"]),
            )
        )
    return specs


def infer_alpha_from_report_payload(payload: dict[str, Any]) -> float:
    if payload.get("alpha") is not None:
        return float(payload["alpha"])
    config = payload.get("config")
    if isinstance(config, dict) and config.get("alpha") is not None:
        return float(config["alpha"])
    return 0.0


def tokenize_docs_to_stream(
    sp: spm.SentencePieceProcessor,
    docs: list[str],
) -> torch.Tensor:
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("SentencePiece model must define bos_id")
    tokens: list[int] = []
    for doc in docs:
        tokens.append(bos_id)
        tokens.extend(sp.encode(doc, out_type=int))
    return torch.tensor(tokens, dtype=torch.long)


def sample_batch(
    token_stream: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_stream.numel() - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    x = torch.stack([token_stream[start : start + seq_len] for start in starts.tolist()], dim=0).to(device)
    y = torch.stack([token_stream[start + 1 : start + seq_len + 1] for start in starts.tolist()], dim=0).to(device)
    return x, y


def evaluate_bits_per_token(
    model: TinyTransformerLM,
    token_stream: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    usable_tokens = ((token_stream.numel() - 1) // seq_len) * seq_len
    if usable_tokens <= 0:
        raise ValueError(f"holdout stream must contain at least seq_len+1 tokens, got {token_stream.numel()}")
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for offset in range(0, usable_tokens, batch_size * seq_len):
            batch_x: list[torch.Tensor] = []
            batch_y: list[torch.Tensor] = []
            batch_stop = min(offset + batch_size * seq_len, usable_tokens)
            for start in range(offset, batch_stop, seq_len):
                batch_x.append(token_stream[start : start + seq_len])
                batch_y.append(token_stream[start + 1 : start + seq_len + 1])
            x = torch.stack(batch_x, dim=0).to(device)
            y = torch.stack(batch_y, dim=0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(y.numel())
    return total_loss / total_tokens / math.log(2.0)


def run_proxy_training(
    *,
    tokenizer_model_path: str | Path,
    split_manifest_path: str | Path,
    config: ProxyTrainingConfig,
    cache_dir: str | Path | None = None,
) -> ProxyTrainingCoreMetrics:
    resolved_device = resolve_device(config.device)
    cache_key = None
    key_payload = None
    if cache_dir is not None:
        key_payload = build_proxy_train_cache_key_payload(
            tokenizer_model_path=tokenizer_model_path,
            split_manifest_path=split_manifest_path,
            config=config,
            device=resolved_device,
        )
        cache_key = proxy_train_cache_key(key_payload)
        cached = load_cached_proxy_training(cache_dir, cache_key)
        if cached is not None:
            return _core_metrics_from_payload(cached["metrics"], cache_hit=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(resolved_device)
    manifest = load_search_split_manifest(split_manifest_path)
    train_docs = load_docs(manifest.search_train_path, text_field=manifest.text_field)
    holdout_docs = load_docs(manifest.search_holdout_path, text_field=manifest.text_field)
    sp = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_model_path).expanduser().resolve()))
    train_stream = tokenize_docs_to_stream(sp, train_docs)
    holdout_stream = tokenize_docs_to_stream(sp, holdout_docs)
    if train_stream.numel() <= config.seq_len:
        raise ValueError(
            f"train stream must contain at least seq_len+1 tokens, got {train_stream.numel()} for seq_len={config.seq_len}"
        )
    if holdout_stream.numel() <= config.seq_len:
        raise ValueError(
            f"holdout stream must contain at least seq_len+1 tokens, got {holdout_stream.numel()} for seq_len={config.seq_len}"
        )
    model = TinyTransformerLM(
        vocab_size=int(sp.vocab_size()),
        seq_len=config.seq_len,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_mult=config.mlp_mult,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    final_train_bits_per_token = 0.0
    for _ in range(config.train_steps):
        x, y = sample_batch(
            train_stream,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        final_train_bits_per_token = float(loss.item()) / math.log(2.0)
    model.eval()
    trained_bits_per_token = evaluate_bits_per_token(
        model,
        holdout_stream,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        device=device,
    )
    metrics = ProxyTrainingCoreMetrics(
        trained_bits_per_token=trained_bits_per_token,
        final_train_bits_per_token=final_train_bits_per_token,
        train_stream_token_count=int(train_stream.numel()),
        holdout_stream_token_count=int(holdout_stream.numel()),
        train_steps=config.train_steps,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        model_dim=config.model_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_mult=config.mlp_mult,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
        device=resolved_device,
        cache_hit=False,
    )
    if cache_dir is not None and cache_key is not None and key_payload is not None:
        store_cached_proxy_training(
            cache_dir,
            cache_key=cache_key,
            key_payload=key_payload,
            metrics_payload={k: v for k, v in asdict(metrics).items() if k != "cache_hit"},
        )
    return metrics


def rerank_candidates_with_proxy_training(
    *,
    report_path: str | Path,
    config: ProxyTrainingConfig | None = None,
    cache_dir: str | Path | None = None,
) -> ProxyRerankReport:
    rerank_config = ProxyTrainingConfig() if config is None else config
    report_file = Path(report_path).expanduser().resolve()
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    alpha = infer_alpha_from_report_payload(report_payload) if rerank_config.alpha is None else rerank_config.alpha
    candidate_specs = load_top_candidates_from_report(report_file, top_k=rerank_config.top_k)
    if not candidate_specs:
        raise ValueError("no candidates selected for proxy rerank")
    core_metrics_by_key: dict[tuple[str, str], ProxyTrainingCoreMetrics] = {}
    reranked: list[ProxyRerankedCandidate] = []
    for candidate in candidate_specs:
        key = (candidate.tokenizer_model_path, candidate.split_manifest_path)
        if key not in core_metrics_by_key:
            core_metrics_by_key[key] = run_proxy_training(
                tokenizer_model_path=candidate.tokenizer_model_path,
                split_manifest_path=candidate.split_manifest_path,
                config=rerank_config,
                cache_dir=cache_dir,
            )
        core_metrics = core_metrics_by_key[key]
        proxy_train_bpb_holdout = core_metrics.trained_bits_per_token * candidate.holdout_tokens_per_byte
        rerank_score = proxy_train_bpb_holdout + alpha * candidate.tokenizer_asset_bytes
        reranked.append(
            ProxyRerankedCandidate(
                rerank_rank=0,
                label=candidate.label,
                original_rank=candidate.original_rank,
                tokenizer_model_path=candidate.tokenizer_model_path,
                tokenizer_vocab_path=candidate.tokenizer_vocab_path,
                tokenizer_asset_bytes=candidate.tokenizer_asset_bytes,
                original_score=candidate.original_score,
                original_proxy_bpb_holdout=candidate.original_proxy_bpb_holdout,
                trained_bits_per_token=core_metrics.trained_bits_per_token,
                final_train_bits_per_token=core_metrics.final_train_bits_per_token,
                proxy_train_bpb_holdout=proxy_train_bpb_holdout,
                rerank_score=rerank_score,
                holdout_tokens_per_byte=candidate.holdout_tokens_per_byte,
                train_stream_token_count=core_metrics.train_stream_token_count,
                holdout_stream_token_count=core_metrics.holdout_stream_token_count,
                cache_hit=core_metrics.cache_hit,
            )
        )
    reranked.sort(key=lambda item: (item.rerank_score, item.proxy_train_bpb_holdout, item.label))
    finalized: list[ProxyRerankedCandidate] = []
    for rank, candidate in enumerate(reranked, start=1):
        finalized.append(
            ProxyRerankedCandidate(
                rerank_rank=rank,
                label=candidate.label,
                original_rank=candidate.original_rank,
                tokenizer_model_path=candidate.tokenizer_model_path,
                tokenizer_vocab_path=candidate.tokenizer_vocab_path,
                tokenizer_asset_bytes=candidate.tokenizer_asset_bytes,
                original_score=candidate.original_score,
                original_proxy_bpb_holdout=candidate.original_proxy_bpb_holdout,
                trained_bits_per_token=candidate.trained_bits_per_token,
                final_train_bits_per_token=candidate.final_train_bits_per_token,
                proxy_train_bpb_holdout=candidate.proxy_train_bpb_holdout,
                rerank_score=candidate.rerank_score,
                holdout_tokens_per_byte=candidate.holdout_tokens_per_byte,
                train_stream_token_count=candidate.train_stream_token_count,
                holdout_stream_token_count=candidate.holdout_stream_token_count,
                cache_hit=candidate.cache_hit,
            )
        )
    return ProxyRerankReport(
        source_report_path=str(report_file),
        split_manifest_path=candidate_specs[0].split_manifest_path,
        alpha=alpha,
        config=rerank_config,
        reranked_candidates=finalized,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run short proxy-training rerank on top-K tokenizer candidates")
    parser.add_argument("--report", required=True, help="Path to a local-search or objective-sanity report JSON")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for persistent proxy-training cache")
    parser.add_argument("--alpha", type=float, default=None, help="Optional asset-byte coefficient override; defaults to source report alpha")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top candidates from the source report to rerank")
    parser.add_argument("--train-steps", type=int, default=60, help="Short proxy-training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Training and eval batch size in sequences")
    parser.add_argument("--seq-len", type=int, default=64, help="Proxy-training sequence length")
    parser.add_argument("--model-dim", type=int, default=64, help="Tiny proxy model width")
    parser.add_argument("--num-layers", type=int, default=2, help="Tiny proxy model layer count")
    parser.add_argument("--num-heads", type=int, default=4, help="Tiny proxy model attention heads")
    parser.add_argument("--mlp-mult", type=int, default=2, help="Tiny proxy model MLP multiplier")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Proxy-training learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Proxy-training weight decay")
    parser.add_argument("--seed", type=int, default=1337, help="Proxy-training random seed")
    parser.add_argument("--device", default="auto", help="Device for proxy training: auto, cpu, or cuda")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = rerank_candidates_with_proxy_training(
        report_path=args.report,
        cache_dir=args.cache_dir,
        config=ProxyTrainingConfig(
            alpha=args.alpha,
            top_k=args.top_k,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_mult=args.mlp_mult,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
