from datetime import datetime, timezone

import pandas as pd

import build_interactions as bi
import common
from fixtures import make_tiny_cfg


def ts(y, m, d):
    return str(int(datetime(y, m, d, tzinfo=timezone.utc).timestamp()))


def test_filter_meta_requires_category_image_and_description():
    meta = pd.DataFrame({
        "asin": ["a", "b", "c", "d"],
        "category": [["x"], [], ["x"], ["x"]],
        "imageURLHighRes": [["u"], ["u"], None, ["u"]],
        "description": [["t"], ["t"], ["t"], []],
    })
    assert bi.filter_meta(meta)["asin"].tolist() == ["a"]


def test_split_clothing_uses_category_positions_and_order():
    meta = pd.DataFrame({"asin": ["W1", "N1", "M1", "B1", "G1"], "category": [
        ["Clothing, Shoes & Jewelry", "Women", "Clothing"],
        ["Clothing, Shoes & Jewelry", "Novelty & More", "Women"],
        ["Clothing, Shoes & Jewelry", "Men"],
        ["Clothing, Shoes & Jewelry", "Boys"],
        ["Clothing, Shoes & Jewelry", "Girls"],
    ]})
    s = bi.split_clothing(meta)
    assert s["women"]["asin"].tolist() == ["W1", "N1"]
    assert s["men"]["asin"].tolist() == ["M1"]
    assert s["boys_girls"]["asin"].tolist() == ["B1", "G1"]


def test_run_writes_expected_files(tmp_path):
    cfg = make_tiny_cfg(tmp_path)
    bi.run(cfg, list(cfg["datasets"]))
    women = common.read_atomic(cfg["processed_dir"] / "Women" / "Women.inter")
    assert list(women.columns) == ["user_id:token", "item_id:token", "timestamp:float"]
    assert women.values.tolist() == [["u1", "W1", ts(2016, 1, 5)], ["u2", "N1", ts(2017, 3, 3)]]
    raw_items = common.read_atomic(cfg["interim_dir"] / "Women" / "items_raw.tsv")
    assert raw_items["item_id:token"].tolist() == ["W1", "N1"]
    assert raw_items.loc[0, "categories:token_seq"] == "['Clothing, Shoes & Jewelry', 'Women', 'Clothing', 'Tops']"
    bg_items = common.read_atomic(cfg["interim_dir"] / "Boys_Girls" / "items_raw.tsv")["item_id:token"].tolist()
    assert bg_items == ["B1", "G1"]  # pre-time-filter list keeps B1 (only reviewed in 2018)
    assert common.read_atomic(cfg["processed_dir"] / "Boys_Girls" / "Boys_Girls.inter")["item_id:token"].tolist() == ["G1"]
    vg = common.read_atomic(cfg["processed_dir"] / "Video_Games" / "Video_Games.inter")
    assert vg["item_id:token"].tolist() == ["V1", "V2", "V3"]
    meta = common.load_item_meta(cfg["interim_dir"] / "Women")
    assert meta["N1"]["description"] == ["Novelty shirt", "Second paragraph"]


def test_run_skips_existing_and_force_rewrites(tmp_path):
    cfg = make_tiny_cfg(tmp_path)
    bi.run(cfg, ["Men"])
    out = cfg["processed_dir"] / "Men" / "Men.inter"
    out.write_text("sentinel")
    bi.run(cfg, ["Men"])
    assert out.read_text() == "sentinel"
    bi.run(cfg, ["Men"], force=True)
    assert out.read_text().startswith("user_id:token")
