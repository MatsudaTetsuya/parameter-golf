from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenizer_scorer import tokenizer_asset_paths


TOKENIZER_EVAL_CACHE_VERSION = 1
TOKENIZER_EVAL_CACHE_SUBDIR = "evaluations"


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size_bytes: int
    sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> FileFingerprint:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"cache fingerprint target not found: {file_path}")
    return FileFingerprint(
        path=str(file_path),
        size_bytes=file_path.stat().st_size,
        sha256=sha256_file(file_path),
    )


def build_tokenizer_eval_cache_key_payload(
    *,
    tokenizer_model_path: str | Path,
    tokenizer_vocab_path: str | Path | None,
    tokenizer_extra_asset_paths: tuple[str, ...],
    split_manifest_path: str | Path,
    alpha: float,
    ngram_order: int,
    add_k: float,
) -> dict[str, Any]:
    model_path = Path(tokenizer_model_path).expanduser().resolve()
    vocab_path = None if tokenizer_vocab_path is None else Path(tokenizer_vocab_path).expanduser().resolve()
    manifest_path = Path(split_manifest_path).expanduser().resolve()
    asset_fingerprints = [
        asdict(file_fingerprint(path))
        for path in tokenizer_asset_paths(
            model_path,
            vocab_path=vocab_path,
            extra_asset_paths=tokenizer_extra_asset_paths,
        )
    ]
    return {
        "version": TOKENIZER_EVAL_CACHE_VERSION,
        "split_manifest": asdict(file_fingerprint(manifest_path)),
        "tokenizer_model_path": str(model_path),
        "tokenizer_vocab_path": None if vocab_path is None else str(vocab_path),
        "tokenizer_extra_asset_paths": [str(Path(path).expanduser().resolve()) for path in tokenizer_extra_asset_paths],
        "asset_fingerprints": asset_fingerprints,
        "alpha": alpha,
        "ngram_order": ngram_order,
        "add_k": add_k,
    }


def tokenizer_eval_cache_key_from_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(serialized)


def resolve_tokenizer_eval_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    base_dir = Path(cache_dir).expanduser().resolve()
    return base_dir / TOKENIZER_EVAL_CACHE_SUBDIR / f"{cache_key}.json"


def load_cached_tokenizer_evaluation(cache_dir: str | Path, cache_key: str) -> dict[str, Any] | None:
    cache_path = resolve_tokenizer_eval_cache_path(cache_dir, cache_key)
    if not cache_path.is_file():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def store_cached_tokenizer_evaluation(
    cache_dir: str | Path,
    *,
    cache_key: str,
    key_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
) -> Path:
    cache_path = resolve_tokenizer_eval_cache_path(cache_dir, cache_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "key": key_payload,
        "evaluation": evaluation_payload,
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(cache_path)
    return cache_path
