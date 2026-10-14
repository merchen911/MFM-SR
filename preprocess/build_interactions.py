"""Stage 1: raw Amazon files -> interactions (.inter) and raw item lists.

Items keep only metadata with a category, a high-resolution image URL and a description,
restricted to items that appear in the review file. Clothing is split into the
Boys_Girls / Women / Men datasets by category position. Interactions are kept for the
configured years. The 5-core filter is applied later by RecBole when the data are loaded.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import dataset_dirs, load_config, read_meta, read_reviews, should_skip, write_atomic_table

INTER_COLUMNS = {"reviewerID": "user_id:token", "asin": "item_id:token", "timestamp": "timestamp:float"}


def _non_empty(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return len(value) != 0


def filter_meta(meta: pd.DataFrame) -> pd.DataFrame:
    keep = (
        meta["category"].map(_non_empty)
        & meta["imageURLHighRes"].map(_non_empty)
        & meta["description"].map(_non_empty)
    )
    return meta[keep].reset_index(drop=True)


def intersect(meta: pd.DataFrame, reviews: pd.DataFrame):
    asins = set(reviews["asin"]) & set(meta["asin"])
    reviews = reviews[reviews["asin"].isin(asins)].reset_index(drop=True)
    meta = meta[meta["asin"].isin(asins)].drop_duplicates(["asin"]).reset_index(drop=True)
    return meta, reviews


def _column(table: pd.DataFrame, i: int) -> pd.Series:
    return table[i] if i in table.columns else pd.Series([None] * len(table), index=table.index)


def split_clothing(meta: pd.DataFrame) -> dict:
    """Split Clothing items by the category name at list positions 2/3 of str(category)."""
    temp = (
        meta["category"].map(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.split(", ", expand=True)
    )
    c2, c3 = _column(temp, 2), _column(temp, 3)
    women = list(temp.index[c2 == "'Women'"]) + list(temp.index[c3 == "'Women'"])
    men = list(temp.index[c2 == "'Men'"]) + list(temp.index[c3 == "'Men'"])
    boys_girls = list(temp.index[c2 == "'Boys'"]) + list(temp.index[c2 == "'Girls'"])
    return {"women": meta.iloc[women], "men": meta.iloc[men], "boys_girls": meta.iloc[boys_girls]}


def make_inter(reviews: pd.DataFrame) -> pd.DataFrame:
    out = reviews[["reviewerID", "asin"]].copy()
    out["timestamp"] = pd.to_datetime(reviews["reviewTime"], format="%m %d, %Y").values.astype(np.int64) // 10**9
    return out.rename(columns=INTER_COLUMNS).reset_index(drop=True)


def filter_years(inter: pd.DataFrame, years) -> pd.DataFrame:
    year = pd.to_datetime(inter["timestamp:float"], unit="s").dt.year
    return inter[year.isin(list(years))].reset_index(drop=True)


def build_dataset(meta_subset: pd.DataFrame, reviews: pd.DataFrame, years):
    d = reviews[reviews["asin"].isin(set(meta_subset["asin"]))].reset_index(drop=True)
    inter_all = make_inter(d)
    items = meta_subset[meta_subset["asin"].isin(set(d["asin"]))].drop_duplicates(["asin"]).reset_index(drop=True)
    return inter_all, items, filter_years(inter_all, years)


def outputs_for(cfg: dict, name: str):
    interim, processed = dataset_dirs(cfg, name)
    return [interim / "items_raw.tsv", interim / "item_meta.jsonl", processed / f"{name}.inter"]


def _write(cfg: dict, name: str, items: pd.DataFrame, inter: pd.DataFrame) -> None:
    interim, processed = dataset_dirs(cfg, name)
    raw = pd.DataFrame({"item_id:token": items["asin"].values, "categories:token_seq": items["category"].map(str).values})
    write_atomic_table(raw, interim / "items_raw.tsv")
    with open(interim / "item_meta.jsonl", "w", encoding="utf-8") as f:
        for r in items.itertuples(index=False):
            f.write(json.dumps({"asin": r.asin, "description": r.description, "imageURLHighRes": r.imageURLHighRes}, ensure_ascii=False) + "\n")
    write_atomic_table(inter, processed / f"{name}.inter")


def run(cfg: dict, names, force: bool = False) -> None:
    by_source: dict = {}
    for n in names:
        by_source.setdefault(cfg["datasets"][n]["source"], []).append(n)
    for source, members in by_source.items():
        todo = [n for n in members if not should_skip(outputs_for(cfg, n), force)]
        for n in sorted(set(members) - set(todo)):
            print(f"[skip] {n}: stage 1 outputs exist (use --force to rebuild)")
        if not todo:
            continue
        src = cfg["sources"][source]
        meta = filter_meta(read_meta(cfg["raw_dir"] / src["meta"]))
        meta, reviews = intersect(meta, read_reviews(cfg["raw_dir"] / src["reviews"]))
        splits = split_clothing(meta) if any(cfg["datasets"][n]["split"] for n in todo) else {}
        for n in todo:
            split = cfg["datasets"][n]["split"]
            _, items, inter = build_dataset(splits[split] if split else meta, reviews, cfg["years"])
            _write(cfg, n, items, inter)
            print(f"[done] {n}: {len(inter)} interactions in {cfg['years']}, {len(items)} candidate items")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="path to datasets.yaml (default: preprocess/datasets.yaml)")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    run(cfg, a.datasets or list(cfg["datasets"]), a.force)


if __name__ == "__main__":
    main()
