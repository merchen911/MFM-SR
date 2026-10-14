"""Stage 2: category tokens of the .item file.

Category strings are normalised per dataset family, rare category tokens are removed,
and the result is restricted to the items that occur in the (year-filtered) .inter file.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import pandas as pd

from common import dataset_dirs, load_config, read_atomic, should_skip, write_atomic_table

ITEM, CAT = "item_id:token", "categories:token_seq"
VG_SEPARATOR = ", '</span></span></span>'"


def clean_clothing(items: pd.DataFrame) -> pd.DataFrame:
    """Remove brackets, quotes and spaces; tokens from position 3 seen fewer than 3 times become None."""
    s = (
        items[CAT].str.replace("[", "", regex=False).str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False).str.replace(" ", "", regex=False)
    )
    temp = s.str.split(",", expand=True)
    for i in range(3, temp.shape[1]):
        counts = temp[i].value_counts()
        temp.loc[temp[i].isin(counts.index[counts < 3]), i] = None
    joined = temp.apply(lambda row: ",".join(row.values.astype(str)), axis=1).str.replace("None,", "", regex=False)
    out = items.copy()
    out[CAT] = joined.values
    return out


def clean_video_games(items: pd.DataFrame) -> pd.DataFrame:
    """Cut the HTML tail of Video Games categories and remove the list brackets."""
    out = items.copy()
    out[CAT] = items[CAT].map(lambda s: re.sub(r"\[|\]", "", s.split(VG_SEPARATOR, 1)[0])).values
    return out


def clothing_tokens(cats):
    return [c.split(",") for c in cats]


def video_games_tokens(cats):
    return [['"' + x.strip("'").replace(" ", "") + '"' for x in c.split(", ")] for c in cats]


def drop_rare_tokens(token_lists, min_count: int):
    counts = Counter(t for tokens in token_lists for t in tokens)
    drop = {t for t, c in counts.items() if c < min_count} | {"None"}
    return [str([t for t in tokens if t not in drop and t != ""]) for tokens in token_lists]


RULES = {"clothing": (clean_clothing, clothing_tokens), "video_games": (clean_video_games, video_games_tokens)}


def build_items(items_raw: pd.DataFrame, inter_item_ids, rule: str, min_count: int) -> pd.DataFrame:
    clean, tokenize = RULES[rule]
    cleaned = clean(items_raw)  # position-wise counts use the full, pre-year-filter item list
    kept = cleaned[cleaned[ITEM].isin(set(inter_item_ids))].reset_index(drop=True)
    return pd.DataFrame({ITEM: kept[ITEM].values, CAT: drop_rare_tokens(tokenize(kept[CAT].tolist()), min_count)})


def run(cfg: dict, names, force: bool = False) -> None:
    for n in names:
        interim, processed = dataset_dirs(cfg, n)
        out = processed / f"{n}.item"
        if should_skip([out], force):
            print(f"[skip] {n}.item exists (use --force to rebuild)")
            continue
        inter = read_atomic(processed / f"{n}.inter")
        item = build_items(read_atomic(interim / "items_raw.tsv"), inter[ITEM], cfg["datasets"][n]["category_rule"], cfg["rare_category_min_count"])
        write_atomic_table(item, out)
        print(f"[done] {n}.item: {len(item)} items")


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
