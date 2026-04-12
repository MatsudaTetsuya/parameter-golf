"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default SP8192-oriented static stack:
# - 11 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 4x MLP expansion
# - vocab size 8192, sequence length 2048, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 4.0))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    negative_slope = float(os.environ.get("NEGATIVE_SLOPE", 0.5))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Quantization & compression.
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))
    quantized_model_path = os.environ.get("QUANTIZED_MODEL_PATH", "final_model.int6.ptz")

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    space_piece_id = int(sp.piece_to_id("▁"))
    if (
        space_piece_id < 0
        or space_piece_id == int(sp.unk_id())
        or sp.is_control(space_piece_id)
        or sp.is_unknown(space_piece_id)
        or sp.is_unused(space_piece_id)
    ):
        raise ValueError(
            "Tokenizer has no standalone ▁ piece. Current byte accounting rejects such tokenizers "
            "because BPB can be mis-counted."
        )
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


# Adapted from the manifest-based dataset/tokenizer guard in train_gpt_mlx.py:
# val_bpb is only meaningful if the tokenizer used for byte accounting matches
# the tokenizer that produced the dataset shards.
def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")

    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None and actual_train_files > int(expected_train_files):
        raise ValueError(
            f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
            f"manifest says {int(expected_train_files)}"
        )
    return dataset_dir.name, actual_train_files


def validate_batching_args(args: Hyperparameters, world_size: int, grad_accum_steps: int) -> None:
    train_denom = world_size * grad_accum_steps * args.train_seq_len
    if args.train_batch_tokens % train_denom != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS={args.train_batch_tokens} must be divisible by "
            f"WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN={train_denom}"
        )
    val_denom = world_size * grad_accum_steps * args.train_seq_len
    if args.val_batch_size % val_denom != 0:
        raise ValueError(
            f"VAL_BATCH_SIZE={args.val_batch_size} must be divisible by "
            f"WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN={val_denom}"
        )


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    if args.val_batch_size % (world_size * grad_accum_steps * args.train_seq_len) != 0:
        raise ValueError(
            f"VAL_BATCH_SIZE={args.val_batch_size} must be divisible by "
            "WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN for reshaping"
        )
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we quantize the trained weights into a submission artifact and validate
# the round-tripped model on the standard BPB evaluation.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates",
    ).split(",")
    if pattern
)
GPTQ_KEEP_FLOAT_MAX_NUMEL = 65_536
GPTQ_SCALE_DTYPE = torch.float16
_BSHF_MAGIC = b"BSHF"

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def _passthrough_quant_tensor(t: Tensor) -> Tensor:
    if t.is_floating_point() and t.dtype in {torch.float32, torch.bfloat16}:
        return t.to(dtype=torch.float16).contiguous()
    return t.contiguous()


# Adapted from the readable SP8192 GPTQ/SDClip implementation in the 2026-04-05 record:
# collect Hessian approximations from calibration activations, quantize large matrices,
# and leave small/control tensors in a compact passthrough format.
def collect_hessians(
    model: nn.Module,
    train_loader: "DistributedTokenLoader",
    global_tokens: int,
    seq_len: int,
    grad_accum_steps: int,
    n_calibration_batches: int,
    distributed: bool,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook_fn(_module, inputs, _output):
            x = inputs[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=x.device
                )
            hessians[name].addmm_(x.T, x)

        return hook_fn

    def make_output_hook(name: str):
        def hook_fn(_module, _inputs, output):
            x = output.detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=x.device
                )
            hessians[name].addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > GPTQ_KEEP_FLOAT_MAX_NUMEL:
            hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if getattr(model, "tie_embeddings", False):
        hooks.append(model.final_norm.register_forward_hook(make_output_hook("tok_emb.weight")))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _y = train_loader.next_batch(global_tokens, seq_len, grad_accum_steps)
            model.forward_logits(x)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    normalizer = float(n_calibration_batches)
    if distributed:
        for hessian in hessians.values():
            dist.all_reduce(hessian, op=dist.ReduceOp.SUM)
        normalizer *= float(dist.get_world_size())
    for name, hessian in hessians.items():
        hessians[name] = (hessian / normalizer).cpu().contiguous()
    return hessians


def gptq_quantize_weight(
    w: Tensor,
    hessian: Tensor,
    clip_sigmas: float,
    clip_range: int,
    block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    w_orig = w.float().clone()
    rows, cols = w_orig.shape
    hessian = hessian.float().clone()

    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    damp = 0.01 * hessian.diag().mean()
    hessian.diagonal().add_(damp)

    perm = torch.argsort(hessian.diag(), descending=True)
    invperm = torch.argsort(perm)
    w_perm = w_orig[:, perm].clone()
    w_perm[:, dead[perm]] = 0
    hessian = hessian[perm][:, perm]

    hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)

    row_std = w_orig.std(dim=1)
    scale = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(GPTQ_SCALE_DTYPE)
    scale_f = scale.float()

    quantized = torch.zeros(rows, cols, dtype=torch.int8)
    w_work = w_perm.clone()
    for block_start in range(0, cols, block_size):
        block_end = min(block_start + block_size, cols)
        w_block = w_work[:, block_start:block_end].clone()
        hinv_block = hessian_inv[block_start:block_end, block_start:block_end]
        err_block = torch.zeros(rows, block_end - block_start)
        for j in range(block_end - block_start):
            w_col = w_block[:, j]
            denom = hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / scale_f), -clip_range, clip_range)
            quantized[:, block_start + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * scale_f) / denom
            err_block[:, j] = err
            w_block[:, j:] -= err.unsqueeze(1) * hinv_block[j, j:].unsqueeze(0)
        if block_end < cols:
            w_work[:, block_end:] -= err_block @ hessian_inv[block_start:block_end, block_end:]

    return quantized[:, invperm].contiguous(), scale.contiguous()


def gptq_mixed_quantize(
    state_dict: dict[str, Tensor],
    hessians: dict[str, Tensor],
    args: Hyperparameters,
) -> tuple[dict[str, Tensor], dict[str, dict[str, object]], dict[str, int]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, dict[str, object]] = {}
    stats = {
        "baseline_tensor_bytes": 0,
        "quantized_payload_bytes": 0,
        "num_quantized_tensors": 0,
        "num_passthrough_tensors": 0,
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        should_quantize = (
            t.is_floating_point()
            and t.ndim == 2
            and t.numel() > GPTQ_KEEP_FLOAT_MAX_NUMEL
            and name in hessians
        )
        if not should_quantize:
            kept = _passthrough_quant_tensor(t)
            result[name] = kept
            meta[name] = {"scheme": "passthrough"}
            stats["quantized_payload_bytes"] += tensor_nbytes(kept)
            stats["num_passthrough_tensors"] += 1
            continue

        is_embedding = name == "tok_emb.weight"
        bits = args.embed_bits if is_embedding else args.matrix_bits
        clip_sigmas = args.embed_clip_sigmas if is_embedding else args.matrix_clip_sigmas
        q, scale = gptq_quantize_weight(
            t,
            hessians[name],
            clip_sigmas=clip_sigmas,
            clip_range=2 ** (bits - 1) - 1,
        )
        result[name + ".q"] = q
        result[name + ".scale"] = scale
        meta[name] = {"scheme": "gptq", "bits": bits}
        stats["quantized_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(scale)
        stats["num_quantized_tensors"] += 1

    return result, meta, stats


def dequantize_mixed(
    result: dict[str, Tensor],
    meta: dict[str, dict[str, object]],
    template_state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_state_dict.items():
        info = meta.get(name)
        if info is None:
            continue
        if info["scheme"] == "passthrough":
            t = result[name]
            if t.is_floating_point():
                out[name] = t.to(dtype=orig.dtype).contiguous()
            else:
                out[name] = t.contiguous()
            continue
        q = result[name + ".q"]
        scale = result[name + ".scale"].float()
        out[name] = (
            q.float() * scale.view(q.shape[0], *([1] * (q.ndim - 1)))
        ).to(dtype=orig.dtype).contiguous()
    return out


def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(src), dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    out = np.empty(len(payload), dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = len(payload) // stride + (1 if pos < len(payload) % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def _compress(data: bytes, compressor: str) -> bytes:
    data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    if compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data: bytes, compressor: str) -> bytes:
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli

        raw = brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    return _byte_unshuffle(raw)


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes one contiguous chunk shared across ranks, then each rank reads
    # a locally shifted window so no rank boundary drops a target token.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if global_tokens % (self.world_size * grad_accum_steps) != 0:
            raise ValueError(
                f"global_tokens={global_tokens} must be divisible by "
                f"world_size*grad_accum_steps={self.world_size * grad_accum_steps}"
            )
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens % seq_len != 0:
            raise ValueError(
                f"local_tokens={local_tokens} must be divisible by seq_len={seq_len}; "
                f"got global_tokens={global_tokens}, world_size={self.world_size}, "
                f"grad_accum_steps={grad_accum_steps}"
            )
        chunk = self.stream.take(local_tokens * self.world_size + 1)
        start = self.rank * local_tokens
        local = chunk[start : start + local_tokens + 1].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Adapted from the readable SP8192 stack in 2026-04-05:
    # partial RoPE plus train-sequence-aware base scaling for longer eval contexts.
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        effective_rope_dims = rope_dims if rope_dims > 0 else dim
        if effective_rope_dims > dim:
            raise ValueError(f"rope_dims={effective_rope_dims} exceeds head_dim={dim}")
        if effective_rope_dims % 2 != 0:
            raise ValueError(f"rope_dims must be even, got {effective_rope_dims}")
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = effective_rope_dims
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (self.rope_dims / (self.rope_dims - 2)))
                inv_freq = 1.0 / (
                    new_base
                    ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32, device=device) / self.rope_dims)
                )
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if 0 < rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
        rope_dims: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        self.rotary = Rotary(
            self.head_dim,
            base=rope_base,
            train_seq_len=train_seq_len,
            rope_dims=self.rope_dims,
        )
        self.use_xsa = False

    # Adapted from the readable SP8192 stack in 2026-04-05 and earlier efficient XSA implementations:
    # remove the self-value component without materializing repeated KV heads.
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = y.shape
        num_kv_heads = v.size(1)
        group_size = num_heads // num_kv_heads
        y_grouped = y.reshape(batch_size, num_kv_heads, group_size, seq_len, head_dim)
        v_normalized = F.normalize(v, dim=-1).unsqueeze(2)
        projection = (y_grouped * v_normalized).sum(dim=-1, keepdim=True) * v_normalized
        return (y_grouped - projection).reshape(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Adapted from the readable SP8192 stack in 2026-04-05:
    # LeakyReLU(0.5)^2 replaces relu^2 in the static architecture.
    def __init__(self, dim: int, mlp_mult: float, negative_slope: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=self.negative_slope)
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
        rope_dims: int,
        negative_slope: float,
        layer_idx: int,
        ln_scale: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            train_seq_len,
            rope_dims,
        )
        self.mlp = MLP(dim, mlp_mult, negative_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor
        )
        return x_out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        skip_gates_enabled: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        xsa_last_n: int,
        num_loops: int,
        loop_start: int,
        loop_end: int,
        train_seq_len: int,
        rope_dims: int,
        ln_scale: bool,
        negative_slope: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    train_seq_len,
                    rope_dims,
                    negative_slope,
                    i,
                    ln_scale,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # Adapted from the SP8192 recurrence path documented in the 2026-04-09 record:
        # loop layers 3-5 to realize a 17-layer virtual stack, activated partway through training.
        self.looping_active = False
        if num_loops > 0:
            if not (0 <= loop_start <= loop_end < num_layers):
                raise ValueError(
                    f"invalid loop range [{loop_start}, {loop_end}] for num_layers={num_layers}"
                )
            loop_segment = list(range(loop_start, loop_end + 1))
            all_indices = list(range(loop_start))
            for _ in range(num_loops + 1):
                all_indices.extend(loop_segment)
            all_indices.extend(range(loop_end + 1, num_layers))
            num_encoder_layers = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_encoder_layers]
            self.decoder_indices = all_indices[num_encoder_layers:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, num_layers))
        self.virtual_num_layers = len(self.encoder_indices) + len(self.decoder_indices)
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = (
            nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
            if skip_gates_enabled
            else None
        )
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward_features(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        encoder_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        decoder_iter = (
            self.decoder_indices
            if self.looping_active
            else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        )

        # First half stores skips; second half reuses them in reverse order.
        for i in encoder_iter:
            x = self.blocks[i](x, x0)
            skips.append(x)
        for skip_idx, i in enumerate(decoder_iter):
            if skips and skip_idx < self.num_skip_weights:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None and skip_idx < self.skip_gates.size(0):
                    gate = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, gate)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0)

        return self.final_norm(x)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.forward_features(input_ids).reshape(-1, self.tok_emb.weight.size(1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    validate_batching_args(args, world_size, grad_accum_steps)
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        skip_gates_enabled=args.skip_gates_enabled,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n,
        num_loops=args.num_loops,
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        train_seq_len=args.train_seq_len,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        negative_slope=args.negative_slope,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
        scalar_params.append(base_model.skip_gates)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    xsa_layers = [i for i, block in enumerate(base_model.blocks) if block.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    if args.num_loops > 0:
        log0(
            f"layer_loop:planned start:{args.loop_start} end:{args.loop_end} repeats:{args.num_loops} "
            f"enable_at:{args.enable_looping_at:.3f} virtual_layers:{base_model.virtual_num_layers} "
            f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
        )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= args.gptq_reserve_seconds * 1000.0
        log0(f"gptq:reserving {args.gptq_reserve_seconds:.0f}s effective_train_budget:{max_wallclock_ms:.0f}ms")

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        if args.num_loops > 0:
            base_model.looping_active = True
            log0(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(args.warmup_steps):
                zero_grad_all()
                for micro_step in range(grad_accum_steps):
                    if distributed:
                        model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(x, y)
                    (warmup_loss * grad_scale).backward()
                for opt in optimizers:
                    opt.step()
                zero_grad_all()
                if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                    log0(f"loop_warmup_step:{warmup_step + 1}/{args.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        train_frac = (
            step / max(args.iterations, 1)
            if max_wallclock_ms is None
            else elapsed_ms / max(max_wallclock_ms, 1e-9)
        )
        if args.num_loops > 0 and not base_model.looping_active and train_frac >= args.enable_looping_at:
            base_model.looping_active = True
            log0(
                f"layer_loop:enabled step:{step} frac:{train_frac:.3f} "
                f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then produce
    # the compressed GPTQ artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    sd_cpu = {name: tensor.detach().cpu().contiguous() for name, tensor in base_model.state_dict().items()}
    log0("gptq:collecting calibration Hessians...")
    t_gptq = time.perf_counter()
    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    hessians = collect_hessians(
        base_model,
        calib_loader,
        args.train_batch_tokens,
        args.train_seq_len,
        grad_accum_steps,
        args.gptq_calibration_batches,
        distributed,
    )
    log0(
        f"gptq:collected {len(hessians)} hessians in {time.perf_counter() - t_gptq:.1f}s "
        f"matrix_bits:{args.matrix_bits} embed_bits:{args.embed_bits}"
    )
    quant_result, quant_meta, quant_stats = gptq_mixed_quantize(sd_cpu, hessians, args)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, args.compressor)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open(args.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(args.quantized_model_path)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["quantized_payload_bytes"], 1)
        log0(
            f"Serialized model quantized+{args.compressor}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['quantized_payload_bytes']} raw_torch:{quant_raw_bytes} "
            f"payload_ratio:{ratio:.2f}x quantized_tensors:{quant_stats['num_quantized_tensors']})"
        )
        log0(f"Total submission size quantized+{args.compressor}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(args.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob_disk, args.compressor)), map_location="cpu")
    base_model.load_state_dict(dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_quantized_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_quantized_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
