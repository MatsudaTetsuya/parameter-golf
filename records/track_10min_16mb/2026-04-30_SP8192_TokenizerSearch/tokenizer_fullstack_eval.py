from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sentencepiece as spm

from tokenizer_eval_cache import file_fingerprint, sha256_bytes
from tokenizer_scorer import measure_tokenizer_asset_bytes


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_SHARD_SIZE = 10**8
DEFAULT_TEXT_FIELD = "text"
DEFAULT_TOKENIZER_THREADS = max(1, int(os.environ.get("MATCHED_FINEWEB_TOKENIZER_THREADS", str(os.cpu_count() or 8))))
PREPARED_DATASET_MANIFEST = "prepared_dataset.manifest.json"

STEP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+val_loss:(?P<val_loss>[0-9eE+.\-]+)\s+val_bpb:(?P<val_bpb>[0-9eE+.\-]+)"
)
FINAL_ROUNDTRIP_RE = re.compile(
    r"final_int8_zlib_roundtrip(?:_exact)?\s+val_loss:(?P<val_loss>[0-9eE+.\-]+)\s+val_bpb:(?P<val_bpb>[0-9eE+.\-]+)"
)
RAW_MODEL_BYTES_RE = re.compile(r"Serialized model:\s+(?P<bytes>\d+)\s+bytes")
QUANT_MODEL_BYTES_RE = re.compile(r"Serialized model int8\+zlib:\s+(?P<bytes>\d+)\s+bytes")


@dataclass(frozen=True)
class FullStackCandidateSpec:
    label: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None = None
    tokenizer_asset_paths: tuple[str, ...] = ()
    tokenizer_asset_bytes: int | None = None


@dataclass(frozen=True)
class PreparedDatasetConfig:
    num_val_docs: int | None = None
    shard_size: int = DEFAULT_SHARD_SIZE
    limit_docs: int | None = None
    text_field: str = DEFAULT_TEXT_FIELD

    def __post_init__(self) -> None:
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        if self.num_val_docs is not None and self.num_val_docs < 0:
            raise ValueError(f"num_val_docs must be non-negative when set, got {self.num_val_docs}")
        if self.limit_docs is not None and self.limit_docs <= 0:
            raise ValueError(f"limit_docs must be positive when set, got {self.limit_docs}")


@dataclass(frozen=True)
class FullStackEvalConfig:
    train_script_path: str
    output_dir: str
    docs_jsonl_path: str | None = None
    prepared_dataset: PreparedDatasetConfig = PreparedDatasetConfig()
    launcher: str = "python"
    nproc_per_node: int = 1
    python_executable: str = sys.executable
    script_args: tuple[str, ...] = ()
    env_overrides: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.launcher not in {"python", "torchrun"}:
            raise ValueError(f"unsupported launcher {self.launcher!r}")
        if self.nproc_per_node <= 0:
            raise ValueError(f"nproc_per_node must be positive, got {self.nproc_per_node}")


@dataclass(frozen=True)
class PreparedCandidateDataset:
    prepared_root: str
    dataset_dir: str
    dataset_manifest_path: str
    docs_jsonl_path: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None
    vocab_size: int
    num_docs: int
    num_val_docs: int
    shard_size: int
    limit_docs: int | None
    text_field: str
    stats: dict[str, int]
    cache_key: str


@dataclass(frozen=True)
class ParsedTrainLogMetrics:
    last_logged_step: int | None
    last_logged_val_loss: float | None
    last_logged_val_bpb: float | None
    final_roundtrip_val_loss: float | None
    final_roundtrip_val_bpb: float | None
    raw_model_bytes_logged: int | None
    quantized_model_bytes_logged: int | None


@dataclass(frozen=True)
class FullStackCandidateResult:
    rank: int
    label: str
    tokenizer_model_path: str
    tokenizer_vocab_path: str | None
    tokenizer_asset_bytes: int
    prepared_dataset: PreparedCandidateDataset
    work_dir: str
    stdout_path: str
    stderr_path: str
    train_log_path: str | None
    run_id: str
    launcher: str
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    code_bytes: int
    raw_model_bytes: int | None
    quantized_model_bytes: int | None
    submission_bytes_excluding_tokenizer: int | None
    submission_bytes_including_tokenizer: int | None
    last_logged_step: int | None
    last_logged_val_loss: float | None
    last_logged_val_bpb: float | None
    final_roundtrip_val_loss: float | None
    final_roundtrip_val_bpb: float | None


@dataclass(frozen=True)
class FullStackEvalReport:
    config: FullStackEvalConfig
    candidates: list[FullStackCandidateSpec]
    results: list[FullStackCandidateResult]


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")
    return cleaned or "candidate"


def parse_key_value_pairs(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid empty environment variable name in {value!r}")
        parsed[key] = raw
    return parsed


def maybe_load_docs_sidecar_meta(docs_jsonl: str | Path) -> dict[str, Any] | None:
    docs_path = Path(docs_jsonl).expanduser().resolve()
    sidecar = docs_path.with_name(f"{docs_path.stem}.source_manifest.json")
    if not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"docs sidecar must be a JSON object: {sidecar}")
    return payload


def count_docs_jsonl(docs_jsonl: str | Path) -> int:
    docs_path = Path(docs_jsonl).expanduser().resolve()
    with docs_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_docs_jsonl(docs_jsonl: str | Path, *, text_field: str, limit_docs: int | None = None):
    docs_path = Path(docs_jsonl).expanduser().resolve()
    with docs_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit_docs is not None and index >= limit_docs:
                break
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {docs_path}, got {type(payload).__name__}")
            yield str(payload[text_field])


def write_datafile(path: str | Path, toks: np.ndarray) -> None:
    data = np.asarray(toks)
    if data.ndim != 1:
        raise ValueError(f"expected 1D token array, got shape={data.shape}")
    if data.size >= 2**31:
        raise ValueError("token count too large for challenge datafile header")
    if data.dtype != np.uint16:
        if not ((0 <= data).all() and (data < 2**16).all()):
            raise ValueError("token ids must fit in uint16")
        data = data.astype("<u2", copy=False)
    else:
        data = data.astype("<u2", copy=False)
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(data.size)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(data.tobytes())


def export_sentencepiece_shards(
    *,
    docs_jsonl: str | Path,
    text_field: str,
    tokenizer_model_path: str | Path,
    output_dir: str | Path,
    num_docs: int,
    num_val_docs: int,
    shard_size: int,
) -> dict[str, int]:
    if not (0 <= num_val_docs <= num_docs):
        raise ValueError(f"num_val_docs must be in [0, {num_docs}], got {num_val_docs}")

    sp = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_model_path).expanduser().resolve()))
    vocab_size = int(sp.vocab_size())
    if vocab_size > 2**16:
        raise ValueError(f"vocab_size={vocab_size} is too large for uint16 shard storage")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_root.glob(pattern):
            stale.unlink()

    bos_id = int(sp.bos_id())
    stats = {
        "docs_total": 0,
        "docs_val": 0,
        "docs_train": 0,
        "files_total": 0,
        "files_val": 0,
        "files_train": 0,
        "tokens_total": 0,
        "tokens_val": 0,
        "tokens_train": 0,
    }
    buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}

    def flush_current_shard() -> None:
        nonlocal fill
        if fill == 0:
            return
        write_datafile(output_root / f"fineweb_{split}_{shards[split]:06d}.bin", buf[:fill])
        stats["files_total"] += 1
        stats[f"files_{split}"] += 1
        shards[split] += 1
        fill = 0

    def batched_docs() -> Iterable[list[str]]:
        batch: list[str] = []
        for text in iter_docs_jsonl(docs_jsonl, text_field=text_field, limit_docs=num_docs):
            batch.append(text)
            if len(batch) == DEFAULT_TOKENIZER_THREADS:
                yield batch
                batch = []
        if batch:
            yield batch

    for texts in batched_docs():
        encoded_docs = sp.encode(texts, out_type=int, num_threads=DEFAULT_TOKENIZER_THREADS)
        for encoded in encoded_docs:
            split_for_doc = "val" if stats["docs_total"] < num_val_docs else "train"
            if split_for_doc != split:
                flush_current_shard()
                split = split_for_doc

            encoded_arr = np.asarray(encoded, dtype=np.int32)
            toks = np.empty((encoded_arr.size + 1,), dtype=np.int32)
            toks[0] = bos_id
            toks[1:] = encoded_arr
            if not ((0 <= toks).all() and (toks < vocab_size).all()):
                bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
                raise ValueError(f"token id {bad} outside declared vocab_size={vocab_size}")
            toks = toks.astype("<u2", copy=False)

            stats["docs_total"] += 1
            stats[f"docs_{split}"] += 1
            stats["tokens_total"] += int(toks.size)
            stats[f"tokens_{split}"] += int(toks.size)

            pos = 0
            while pos < toks.size:
                take = min(shard_size - fill, toks.size - pos)
                buf[fill : fill + take] = toks[pos : pos + take]
                fill += take
                pos += take
                if fill == shard_size:
                    flush_current_shard()

    flush_current_shard()
    if stats["docs_total"] != num_docs:
        raise ValueError(f"expected {num_docs} docs, exported {stats['docs_total']}")
    return stats


def _copy_tokenizer_artifact(src: str | Path, dst: str | Path) -> None:
    source = Path(src).expanduser().resolve()
    destination = Path(dst).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)


def build_prepared_dataset_cache_key_payload(
    *,
    candidate: FullStackCandidateSpec,
    docs_jsonl_path: str | Path,
    config: PreparedDatasetConfig,
) -> dict[str, Any]:
    model_path = Path(candidate.tokenizer_model_path).expanduser().resolve()
    vocab_path = None if candidate.tokenizer_vocab_path is None else Path(candidate.tokenizer_vocab_path).expanduser().resolve()
    docs_path = Path(docs_jsonl_path).expanduser().resolve()
    return {
        "version": 1,
        "docs_jsonl": asdict(file_fingerprint(docs_path)),
        "tokenizer_model": asdict(file_fingerprint(model_path)),
        "tokenizer_vocab": None if vocab_path is None else asdict(file_fingerprint(vocab_path)),
        "prepared_dataset": {
            "num_val_docs": config.num_val_docs,
            "shard_size": config.shard_size,
            "limit_docs": config.limit_docs,
            "text_field": config.text_field,
        },
    }


def prepared_dataset_cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(serialized)


def _load_prepared_dataset_from_manifest(manifest_path: Path) -> PreparedCandidateDataset:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PreparedCandidateDataset(
        prepared_root=payload["prepared_root"],
        dataset_dir=payload["dataset_dir"],
        dataset_manifest_path=payload["dataset_manifest_path"],
        docs_jsonl_path=payload["docs_jsonl_path"],
        tokenizer_model_path=payload["tokenizer_model_path"],
        tokenizer_vocab_path=payload["tokenizer_vocab_path"],
        vocab_size=int(payload["vocab_size"]),
        num_docs=int(payload["num_docs"]),
        num_val_docs=int(payload["num_val_docs"]),
        shard_size=int(payload["shard_size"]),
        limit_docs=None if payload["limit_docs"] is None else int(payload["limit_docs"]),
        text_field=payload["text_field"],
        stats={key: int(value) for key, value in payload["stats"].items()},
        cache_key=payload["cache_key"],
    )


def prepare_candidate_dataset(
    *,
    candidate: FullStackCandidateSpec,
    docs_jsonl_path: str | Path,
    output_dir: str | Path,
    config: PreparedDatasetConfig | None = None,
) -> PreparedCandidateDataset:
    prepared_config = PreparedDatasetConfig() if config is None else config
    docs_path = Path(docs_jsonl_path).expanduser().resolve()
    if not docs_path.is_file():
        raise FileNotFoundError(docs_path)
    payload = build_prepared_dataset_cache_key_payload(
        candidate=candidate,
        docs_jsonl_path=docs_path,
        config=prepared_config,
    )
    cache_key = prepared_dataset_cache_key(payload)
    prepared_root = Path(output_dir).expanduser().resolve() / "prepared_datasets" / cache_key
    manifest_path = prepared_root / PREPARED_DATASET_MANIFEST
    if manifest_path.is_file():
        return _load_prepared_dataset_from_manifest(manifest_path)

    sidecar = maybe_load_docs_sidecar_meta(docs_path)
    docs_total = int(sidecar["num_docs"]) if sidecar is not None and sidecar.get("num_docs") is not None else count_docs_jsonl(docs_path)
    if prepared_config.limit_docs is not None:
        docs_total = min(docs_total, prepared_config.limit_docs)
    if docs_total <= 0:
        raise ValueError(f"no docs available in {docs_path}")
    if prepared_config.num_val_docs is not None:
        num_val_docs = int(prepared_config.num_val_docs)
    elif sidecar is not None and sidecar.get("docs_val") is not None:
        num_val_docs = int(sidecar["docs_val"])
    else:
        raise ValueError("num_val_docs is required when docs sidecar is missing docs_val")
    if num_val_docs > docs_total:
        raise ValueError(f"num_val_docs={num_val_docs} exceeds available docs_total={docs_total}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = prepared_root / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    model_src = Path(candidate.tokenizer_model_path).expanduser().resolve()
    model_dst = tokenizers_dir / model_src.name
    _copy_tokenizer_artifact(model_src, model_dst)
    vocab_dst: Path | None = None
    if candidate.tokenizer_vocab_path is not None:
        vocab_src = Path(candidate.tokenizer_vocab_path).expanduser().resolve()
        vocab_dst = tokenizers_dir / vocab_src.name
        _copy_tokenizer_artifact(vocab_src, vocab_dst)

    safe_label = sanitize_label(candidate.label)
    dataset_dir = prepared_root / "datasets" / f"fineweb10B_{safe_label}"
    stats = export_sentencepiece_shards(
        docs_jsonl=docs_path,
        text_field=prepared_config.text_field,
        tokenizer_model_path=model_dst,
        output_dir=dataset_dir,
        num_docs=docs_total,
        num_val_docs=num_val_docs,
        shard_size=prepared_config.shard_size,
    )
    sp = spm.SentencePieceProcessor(model_file=str(model_dst))
    dataset = PreparedCandidateDataset(
        prepared_root=str(prepared_root),
        dataset_dir=str(dataset_dir),
        dataset_manifest_path=str(manifest_path),
        docs_jsonl_path=str(docs_path),
        tokenizer_model_path=str(model_dst),
        tokenizer_vocab_path=None if vocab_dst is None else str(vocab_dst),
        vocab_size=int(sp.vocab_size()),
        num_docs=docs_total,
        num_val_docs=num_val_docs,
        shard_size=prepared_config.shard_size,
        limit_docs=prepared_config.limit_docs,
        text_field=prepared_config.text_field,
        stats=stats,
        cache_key=cache_key,
    )
    manifest_path.write_text(json.dumps(asdict(dataset), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def _candidate_spec_from_report_entry(candidate: dict[str, Any], *, fallback_label: str) -> FullStackCandidateSpec:
    evaluation = candidate.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("ranked candidate is missing evaluation payload")
    result = evaluation.get("result")
    if not isinstance(result, dict):
        raise ValueError("candidate evaluation is missing result payload")
    return FullStackCandidateSpec(
        label=str(candidate.get("label", candidate.get("name", candidate.get("candidate_id", fallback_label)))),
        tokenizer_model_path=str(evaluation["tokenizer_model_path"]),
        tokenizer_vocab_path=None if evaluation.get("tokenizer_vocab_path") is None else str(evaluation["tokenizer_vocab_path"]),
        tokenizer_asset_bytes=int(result["tokenizer_asset_bytes"]),
    )


def load_fullstack_candidates(path: str | Path, *, top_k: int | None = None) -> list[FullStackCandidateSpec]:
    report_path = Path(path).expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_candidates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        raw_candidates = payload["candidates"]
    elif isinstance(payload, dict) and isinstance(payload.get("ranked_candidates"), list):
        ranked_candidates = payload["ranked_candidates"]
        limit = len(ranked_candidates) if top_k is None else min(len(ranked_candidates), top_k)
        return [
            _candidate_spec_from_report_entry(candidate, fallback_label=f"candidate_{index}")
            for index, candidate in enumerate(ranked_candidates[:limit], start=1)
        ]
    elif isinstance(payload, dict) and isinstance(payload.get("reranked_candidates"), list):
        reranked_candidates = payload["reranked_candidates"]
        limit = len(reranked_candidates) if top_k is None else min(len(reranked_candidates), top_k)
        return [
            FullStackCandidateSpec(
                label=str(candidate.get("label", candidate.get("name", f"candidate_{index}"))),
                tokenizer_model_path=str(candidate["tokenizer_model_path"]),
                tokenizer_vocab_path=None
                if candidate.get("tokenizer_vocab_path") is None
                else str(candidate["tokenizer_vocab_path"]),
                tokenizer_asset_bytes=int(candidate["tokenizer_asset_bytes"]),
            )
            for index, candidate in enumerate(reranked_candidates[:limit], start=1)
        ]
    else:
        raise ValueError(f"unsupported candidate/report format: {report_path}")

    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"candidate config must define a non-empty list: {report_path}")
    if top_k is not None:
        raw_candidates = raw_candidates[:top_k]
    specs: list[FullStackCandidateSpec] = []
    for index, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("each candidate spec must be a JSON object")
        label = str(candidate.get("label", candidate.get("name", f"candidate_{index}")))
        asset_paths = candidate.get("tokenizer_asset_paths", ())
        if not isinstance(asset_paths, list | tuple):
            raise ValueError(f"tokenizer_asset_paths must be a list for candidate {label}")
        asset_bytes = candidate.get("tokenizer_asset_bytes")
        specs.append(
            FullStackCandidateSpec(
                label=label,
                tokenizer_model_path=str(candidate["tokenizer_model_path"]),
                tokenizer_vocab_path=None
                if candidate.get("tokenizer_vocab_path") is None
                else str(candidate["tokenizer_vocab_path"]),
                tokenizer_asset_paths=tuple(str(path) for path in asset_paths),
                tokenizer_asset_bytes=None if asset_bytes is None else int(asset_bytes),
            )
        )
    return specs


def build_fullstack_command(
    *,
    config: FullStackEvalConfig,
    train_script_path: str | Path,
) -> list[str]:
    script_path = str(Path(train_script_path).expanduser().resolve())
    if config.launcher == "python":
        return [config.python_executable, script_path, *config.script_args]
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={config.nproc_per_node}",
        script_path,
        *config.script_args,
    ]


def parse_train_log_metrics(text: str) -> ParsedTrainLogMetrics:
    last_step = None
    last_val_loss = None
    last_val_bpb = None
    for match in STEP_VAL_RE.finditer(text):
        last_step = int(match.group("step"))
        last_val_loss = float(match.group("val_loss"))
        last_val_bpb = float(match.group("val_bpb"))

    final_matches = list(FINAL_ROUNDTRIP_RE.finditer(text))
    final_val_loss = None
    final_val_bpb = None
    if final_matches:
        final_match = final_matches[-1]
        final_val_loss = float(final_match.group("val_loss"))
        final_val_bpb = float(final_match.group("val_bpb"))

    raw_matches = list(RAW_MODEL_BYTES_RE.finditer(text))
    quant_matches = list(QUANT_MODEL_BYTES_RE.finditer(text))
    return ParsedTrainLogMetrics(
        last_logged_step=last_step,
        last_logged_val_loss=last_val_loss,
        last_logged_val_bpb=last_val_bpb,
        final_roundtrip_val_loss=final_val_loss,
        final_roundtrip_val_bpb=final_val_bpb,
        raw_model_bytes_logged=None if not raw_matches else int(raw_matches[-1].group("bytes")),
        quantized_model_bytes_logged=None if not quant_matches else int(quant_matches[-1].group("bytes")),
    )


def _read_text_if_exists(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve()
    return file_path.read_text(encoding="utf-8") if file_path.is_file() else ""


def run_fullstack_evaluation(
    *,
    candidates: list[FullStackCandidateSpec],
    config: FullStackEvalConfig,
) -> FullStackEvalReport:
    if not candidates:
        raise ValueError("candidates must not be empty")
    output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_script_path = Path(config.train_script_path).expanduser().resolve()
    if not train_script_path.is_file():
        raise FileNotFoundError(train_script_path)
    if config.docs_jsonl_path is None:
        raise ValueError("docs_jsonl_path is required for full-stack evaluation")

    results: list[FullStackCandidateResult] = []
    for rank, candidate in enumerate(candidates, start=1):
        prepared_dataset = prepare_candidate_dataset(
            candidate=candidate,
            docs_jsonl_path=config.docs_jsonl_path,
            output_dir=output_root,
            config=config.prepared_dataset,
        )
        run_id = f"fullstack_{rank:02d}_{sanitize_label(candidate.label)}"
        work_dir = output_root / "runs" / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = work_dir / "stdout.log"
        stderr_path = work_dir / "stderr.log"
        train_log_path = work_dir / "logs" / f"{run_id}.txt"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["RUN_ID"] = run_id
        env["DATA_PATH"] = prepared_dataset.dataset_dir
        env["TOKENIZER_PATH"] = prepared_dataset.tokenizer_model_path
        env["VOCAB_SIZE"] = str(prepared_dataset.vocab_size)
        if config.env_overrides:
            env.update(config.env_overrides)
        command = build_fullstack_command(config=config, train_script_path=train_script_path)
        t0 = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(work_dir),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        duration_seconds = time.perf_counter() - t0

        log_text = _read_text_if_exists(train_log_path) + "\n" + _read_text_if_exists(stdout_path) + "\n" + _read_text_if_exists(stderr_path)
        parsed = parse_train_log_metrics(log_text)

        tokenizer_asset_bytes = (
            int(candidate.tokenizer_asset_bytes)
            if candidate.tokenizer_asset_bytes is not None
            else measure_tokenizer_asset_bytes(
                candidate.tokenizer_model_path,
                vocab_path=candidate.tokenizer_vocab_path,
                extra_asset_paths=candidate.tokenizer_asset_paths,
            )
        )
        code_bytes = len(train_script_path.read_text(encoding="utf-8").encode("utf-8"))
        raw_model_path = work_dir / "final_model.pt"
        quantized_model_path = work_dir / "final_model.int8.ptz"
        raw_model_bytes = raw_model_path.stat().st_size if raw_model_path.is_file() else parsed.raw_model_bytes_logged
        quantized_model_bytes = (
            quantized_model_path.stat().st_size if quantized_model_path.is_file() else parsed.quantized_model_bytes_logged
        )
        submission_bytes_excluding_tokenizer = None
        submission_bytes_including_tokenizer = None
        if quantized_model_bytes is not None:
            submission_bytes_excluding_tokenizer = int(quantized_model_bytes) + code_bytes
            submission_bytes_including_tokenizer = submission_bytes_excluding_tokenizer + tokenizer_asset_bytes

        results.append(
            FullStackCandidateResult(
                rank=rank,
                label=candidate.label,
                tokenizer_model_path=candidate.tokenizer_model_path,
                tokenizer_vocab_path=candidate.tokenizer_vocab_path,
                tokenizer_asset_bytes=tokenizer_asset_bytes,
                prepared_dataset=prepared_dataset,
                work_dir=str(work_dir),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                train_log_path=str(train_log_path) if train_log_path.is_file() else None,
                run_id=run_id,
                launcher=config.launcher,
                command=tuple(command),
                exit_code=int(completed.returncode),
                duration_seconds=duration_seconds,
                code_bytes=code_bytes,
                raw_model_bytes=raw_model_bytes,
                quantized_model_bytes=quantized_model_bytes,
                submission_bytes_excluding_tokenizer=submission_bytes_excluding_tokenizer,
                submission_bytes_including_tokenizer=submission_bytes_including_tokenizer,
                last_logged_step=parsed.last_logged_step,
                last_logged_val_loss=parsed.last_logged_val_loss,
                last_logged_val_bpb=parsed.last_logged_val_bpb,
                final_roundtrip_val_loss=parsed.final_roundtrip_val_loss,
                final_roundtrip_val_bpb=parsed.final_roundtrip_val_bpb,
            )
        )
    return FullStackEvalReport(config=config, candidates=candidates, results=results)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run candidate-specific full-stack tokenizer evaluations")
    parser.add_argument("--candidates", required=True, help="Path to a candidate config/report JSON")
    parser.add_argument("--top-k", type=int, default=None, help="Evaluate only the top-K candidates from the input report")
    parser.add_argument("--train-script", required=True, help="Path to the train_gpt.py-style evaluation script")
    parser.add_argument("--docs-jsonl", required=True, help="Path to docs_selected.jsonl or another local docs JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared datasets, run logs, and reports")
    parser.add_argument("--num-val-docs", type=int, default=None, help="Validation doc count for dataset export")
    parser.add_argument("--limit-docs", type=int, default=None, help="Optional doc limit for smoke evaluations")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Max tokens per exported .bin shard")
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD, help="JSONL field containing document text")
    parser.add_argument("--launcher", choices=("python", "torchrun"), default="python", help="Process launcher")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun --nproc_per_node value")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for launcher=python")
    parser.add_argument("--script-arg", action="append", default=[], help="Extra positional argument for the train script")
    parser.add_argument("--env", action="append", default=[], help="Environment override in KEY=VALUE form")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    candidates = load_fullstack_candidates(args.candidates, top_k=args.top_k)
    report = run_fullstack_evaluation(
        candidates=candidates,
        config=FullStackEvalConfig(
            train_script_path=args.train_script,
            output_dir=args.output_dir,
            docs_jsonl_path=args.docs_jsonl,
            prepared_dataset=PreparedDatasetConfig(
                num_val_docs=args.num_val_docs,
                shard_size=args.shard_size,
                limit_docs=args.limit_docs,
                text_field=args.text_field,
            ),
            launcher=args.launcher,
            nproc_per_node=args.nproc_per_node,
            python_executable=args.python_executable,
            script_args=tuple(args.script_arg),
            env_overrides=parse_key_value_pairs(args.env),
        ),
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
