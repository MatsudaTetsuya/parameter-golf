from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_NUM_VAL_DOCS = 50_000
DEFAULT_TEXT_FIELD = "text"
SEARCH_HOLDOUT_FILENAME = "search_holdout.jsonl"
SEARCH_TRAIN_FILENAME = "search_train.jsonl"
SEARCH_SPLIT_MANIFEST_FILENAME = "search_split.manifest.json"
SEARCH_SPLIT_VERSION = 1


@dataclass(frozen=True)
class SearchSplitConfig:
    search_train_docs: int
    search_holdout_docs: int
    num_val_docs: int | None = None
    text_field: str = DEFAULT_TEXT_FIELD
    sidecar_path: str | None = None

    def __post_init__(self) -> None:
        if self.search_train_docs <= 0:
            raise ValueError(f"search_train_docs must be positive, got {self.search_train_docs}")
        if self.search_holdout_docs <= 0:
            raise ValueError(f"search_holdout_docs must be positive, got {self.search_holdout_docs}")
        if self.num_val_docs is not None and self.num_val_docs < 0:
            raise ValueError(f"num_val_docs must be non-negative, got {self.num_val_docs}")


@dataclass(frozen=True)
class SearchSplitManifest:
    version: int
    docs_path: str
    docs_format: str
    text_field: str
    source_sidecar_path: str | None
    docs_sha256: str | None
    shuffle_seed: int | None
    num_docs_total: int
    num_val_docs_skipped: int
    search_holdout_start_doc: int
    search_holdout_docs: int
    search_train_start_doc: int
    search_train_docs: int
    search_holdout_path: str
    search_train_path: str


def default_sidecar_path(docs_path: str | Path) -> Path:
    docs_file = Path(docs_path).expanduser().resolve()
    return docs_file.with_name(f"{docs_file.stem}.source_manifest.json")


def load_split_sidecar(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    sidecar_file = Path(path).expanduser().resolve()
    if not sidecar_file.is_file():
        return None
    payload = json.loads(sidecar_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"split sidecar must be a JSON object: {sidecar_file}")
    return payload


def resolve_num_val_docs(
    *,
    explicit_num_val_docs: int | None,
    sidecar: dict[str, Any] | None,
) -> int:
    if explicit_num_val_docs is not None:
        return explicit_num_val_docs
    if sidecar is not None and sidecar.get("docs_val") is not None:
        return int(sidecar["docs_val"])
    return DEFAULT_NUM_VAL_DOCS


def iter_docs(path: str | Path, *, text_field: str = DEFAULT_TEXT_FIELD):
    docs_file = Path(path).expanduser().resolve()
    suffix = docs_file.suffix.lower()
    if suffix == ".jsonl":
        with docs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if text_field not in payload:
                    raise KeyError(f"missing text field {text_field!r} in {docs_file}")
                yield str(payload[text_field])
        return
    if suffix == ".txt":
        with docs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.rstrip("\n")
                if text:
                    yield text
        return
    raise ValueError(f"unsupported docs format for search split: {docs_file}")


def docs_format_name(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".txt":
        return "txt"
    raise ValueError(f"unsupported docs format for search split: {path}")


def write_search_split(
    docs_path: str | Path,
    *,
    output_dir: str | Path,
    config: SearchSplitConfig,
) -> SearchSplitManifest:
    docs_file = Path(docs_path).expanduser().resolve()
    if not docs_file.is_file():
        raise FileNotFoundError(f"docs file not found: {docs_file}")
    split_dir = Path(output_dir).expanduser().resolve()
    split_dir.mkdir(parents=True, exist_ok=True)
    sidecar_file = (
        None if config.sidecar_path is None else Path(config.sidecar_path).expanduser().resolve()
    )
    sidecar = load_split_sidecar(default_sidecar_path(docs_file) if sidecar_file is None else sidecar_file)
    num_docs_total_hint = None if sidecar is None or sidecar.get("num_docs") is None else int(sidecar["num_docs"])
    num_val_docs = resolve_num_val_docs(explicit_num_val_docs=config.num_val_docs, sidecar=sidecar)
    if num_docs_total_hint is not None and num_val_docs > num_docs_total_hint:
        raise ValueError(f"num_val_docs must be in [0, {num_docs_total_hint}], got {num_val_docs}")
    holdout_start_doc = num_val_docs
    train_start_doc = holdout_start_doc + config.search_holdout_docs
    stop_after_doc = train_start_doc + config.search_train_docs

    holdout_path = split_dir / SEARCH_HOLDOUT_FILENAME
    train_path = split_dir / SEARCH_TRAIN_FILENAME
    tmp_holdout_path = holdout_path.with_suffix(f"{holdout_path.suffix}.tmp")
    tmp_train_path = train_path.with_suffix(f"{train_path.suffix}.tmp")
    manifest_path = split_dir / SEARCH_SPLIT_MANIFEST_FILENAME
    holdout_count = 0
    train_count = 0
    docs_total = 0

    try:
        with (
            tmp_holdout_path.open("w", encoding="utf-8") as holdout_handle,
            tmp_train_path.open("w", encoding="utf-8") as train_handle,
        ):
            for doc_index, text in enumerate(iter_docs(docs_file, text_field=config.text_field)):
                docs_total = doc_index + 1
                if doc_index < holdout_start_doc:
                    continue
                if doc_index < train_start_doc:
                    holdout_handle.write(json.dumps({config.text_field: text}, ensure_ascii=False) + "\n")
                    holdout_count += 1
                    continue
                if doc_index < stop_after_doc:
                    train_handle.write(json.dumps({config.text_field: text}, ensure_ascii=False) + "\n")
                    train_count += 1
                    continue
                if num_docs_total_hint is not None:
                    docs_total = num_docs_total_hint
                    break
        if holdout_count != config.search_holdout_docs:
            raise ValueError(
                f"requested {config.search_holdout_docs} holdout docs, found {holdout_count} after skipping {num_val_docs}"
            )
        if train_count != config.search_train_docs:
            raise ValueError(
                f"requested {config.search_train_docs} train docs, found {train_count} after holdout selection"
            )
        tmp_holdout_path.replace(holdout_path)
        tmp_train_path.replace(train_path)
    except Exception:
        tmp_holdout_path.unlink(missing_ok=True)
        tmp_train_path.unlink(missing_ok=True)
        raise

    manifest = SearchSplitManifest(
        version=SEARCH_SPLIT_VERSION,
        docs_path=str(docs_file),
        docs_format=docs_format_name(docs_file),
        text_field=config.text_field,
        source_sidecar_path=None if sidecar_file is None and sidecar is None else str((default_sidecar_path(docs_file) if sidecar_file is None else sidecar_file)),
        docs_sha256=None if sidecar is None else sidecar.get("docs_sha256"),
        shuffle_seed=None if sidecar is None or sidecar.get("shuffle_seed") is None else int(sidecar["shuffle_seed"]),
        num_docs_total=num_docs_total_hint if num_docs_total_hint is not None else docs_total,
        num_val_docs_skipped=num_val_docs,
        search_holdout_start_doc=holdout_start_doc,
        search_holdout_docs=holdout_count,
        search_train_start_doc=train_start_doc,
        search_train_docs=train_count,
        search_holdout_path=str(holdout_path),
        search_train_path=str(train_path),
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_search_split_manifest(path: str | Path) -> SearchSplitManifest:
    manifest_file = Path(path).expanduser().resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    return SearchSplitManifest(**payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize fixed search_train/search_holdout docs")
    parser.add_argument("--docs", required=True, help="Path to docs_selected.jsonl or a plain text docs file")
    parser.add_argument("--output-dir", required=True, help="Directory for search_train/search_holdout outputs")
    parser.add_argument("--sidecar", default=None, help="Optional docs sidecar path")
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD, help="JSONL text field name")
    parser.add_argument("--num-val-docs", type=int, default=None, help="Override the number of challenge val docs to skip")
    parser.add_argument("--search-train-docs", type=int, required=True, help="Number of train docs to materialize")
    parser.add_argument("--search-holdout-docs", type=int, required=True, help="Number of holdout docs to materialize")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    manifest = write_search_split(
        args.docs,
        output_dir=args.output_dir,
        config=SearchSplitConfig(
            search_train_docs=args.search_train_docs,
            search_holdout_docs=args.search_holdout_docs,
            num_val_docs=args.num_val_docs,
            text_field=args.text_field,
            sidecar_path=args.sidecar,
        ),
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
