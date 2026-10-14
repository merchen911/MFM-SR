"""Shared helpers for the preprocessing scripts."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "preprocess" / "datasets.yaml"
OFFICIAL_URL = "https://nijianmo.github.io/amazon/index.html"
DIR_KEYS = ("raw_dir", "interim_dir", "processed_dir", "image_cache_dir")


def load_config(path=None) -> dict:
    """Load datasets.yaml; relative directories are resolved against the repository root."""
    with open(Path(path) if path else DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in DIR_KEYS:
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else REPO_ROOT / p
    return cfg


def require_file(path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"Missing raw file: {path}\n"
            f"Download '{path.name}' from {OFFICIAL_URL} and place it in {path.parent}/"
        )
    return path


def iter_json_gz(path):
    with gzip.open(require_file(path), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_meta(path) -> pd.DataFrame:
    rows = [
        (d.get("asin"), d.get("category"), d.get("imageURLHighRes"), d.get("description"))
        for d in iter_json_gz(path)
    ]
    return pd.DataFrame(rows, columns=["asin", "category", "imageURLHighRes", "description"])


def read_reviews(path) -> pd.DataFrame:
    rows = [(d["reviewerID"], d["asin"], d["reviewTime"]) for d in iter_json_gz(path)]
    return pd.DataFrame(rows, columns=["reviewerID", "asin", "reviewTime"])


def read_atomic(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_atomic_table(df: pd.DataFrame, path) -> None:
    """Write a RecBole atomic file (tab-separated, header, LF) via a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def save_features(path, ids, feats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, ids=np.asarray(ids, dtype=str), feats=np.asarray(feats, dtype=np.float32))
    os.replace(tmp, path)


def load_features(path):
    with np.load(path) as z:
        return [str(x) for x in z["ids"]], z["feats"]


def load_item_meta(interim_dir) -> dict:
    out = {}
    with open(Path(interim_dir) / "item_meta.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["asin"]] = r
    return out


def dataset_dirs(cfg: dict, name: str):
    return cfg["interim_dir"] / name, cfg["processed_dir"] / name


def should_skip(outputs, force: bool) -> bool:
    return (not force) and all(Path(p).exists() for p in outputs)


def resolve_device(device: str) -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {device} was requested but CUDA is not available. Use --device cpu.")
    return device
