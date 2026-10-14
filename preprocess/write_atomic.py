"""Stage 5: write RecBole atomic feature files (.itememb_text and .itememb_image)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from common import dataset_dirs, load_config, load_features, read_atomic, should_skip

FEATURE_FILES = {"text": "text_features.npz", "image": "image_features.npz"}


def format_vector(vec) -> str:
    return " ".join(f"{x:.8g}" for x in vec)


def write_embedding_file(path, field: str, ids, feats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"itemid_{field}:token\titem_{field}_emb:float_seq\n")
        for i, v in zip(ids, feats):
            f.write(f"{i}\t{format_vector(v)}\n")
    os.replace(tmp, path)


def read_embedding_file(path):
    ids, rows = [], []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            item, _, values = line.rstrip("\n").partition("\t")
            ids.append(item)
            rows.append(np.array(values.split(), dtype=np.float32))
    return ids, np.stack(rows)


def run(cfg: dict, names, force: bool = False) -> None:
    for n in names:
        interim, processed = dataset_dirs(cfg, n)
        item_ids = read_atomic(processed / f"{n}.item")["item_id:token"].tolist()
        for field, fname in FEATURE_FILES.items():
            out = processed / f"{n}.itememb_{field}"
            if should_skip([out], force):
                print(f"[skip] {out.name} exists (use --force to rebuild)")
                continue
            ids, feats = load_features(interim / fname)
            if ids != item_ids:
                raise SystemExit(
                    f"{interim / fname} does not list the items of {n}.item in the same order "
                    f"({len(ids)} vs {len(item_ids)} items). Re-run the {field} stage without --limit."
                )
            write_embedding_file(out, field, ids, feats)
            print(f"[done] {out.name}: {feats.shape}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    run(cfg, a.datasets or list(cfg["datasets"]), a.force)


if __name__ == "__main__":
    main()
