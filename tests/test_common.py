import gzip
import json

import numpy as np
import pandas as pd
import pytest

import common


def test_require_file_message_names_file_and_url(tmp_path):
    with pytest.raises(SystemExit) as e:
        common.require_file(tmp_path / "meta_Video_Games.json.gz")
    assert "meta_Video_Games.json.gz" in str(e.value)
    assert common.OFFICIAL_URL in str(e.value)


def test_read_meta_and_reviews(tmp_path):
    with gzip.open(tmp_path / "m.json.gz", "wt") as f:
        f.write(json.dumps({"asin": "A", "category": ["X"], "imageURLHighRes": ["u"], "description": ["d"]}) + "\n")
        f.write(json.dumps({"asin": "B"}) + "\n")
    meta = common.read_meta(tmp_path / "m.json.gz")
    assert list(meta.columns) == ["asin", "category", "imageURLHighRes", "description"]
    assert meta.loc[1, "category"] is None
    with gzip.open(tmp_path / "r.json.gz", "wt") as f:
        f.write(json.dumps({"reviewerID": "U", "asin": "A", "reviewTime": "01 5, 2016", "overall": 5.0}) + "\n")
    assert common.read_reviews(tmp_path / "r.json.gz").values.tolist() == [["U", "A", "01 5, 2016"]]


def test_load_config_resolves_relative_dirs_against_repo_root():
    cfg = common.load_config()
    assert cfg["raw_dir"] == common.REPO_ROOT / "data" / "raw"
    assert set(cfg["datasets"]) == {"Video_Games", "Boys_Girls", "Women", "Men"}
    assert cfg["years"] == [2016, 2017]
    assert cfg["rare_category_min_count"] == 5


def test_write_atomic_table_keeps_none_strings_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "x" / "a.item"
    common.write_atomic_table(pd.DataFrame({"item_id:token": ["0700099867"], "categories:token_seq": ["None"]}), p)
    assert p.read_text() == "item_id:token\tcategories:token_seq\n0700099867\tNone\n"
    assert common.read_atomic(p).loc[0, "categories:token_seq"] == "None"
    assert list(p.parent.iterdir()) == [p]


def test_features_roundtrip(tmp_path):
    p = tmp_path / "f.npz"
    common.save_features(p, ["a", "b"], np.ones((2, 3)))
    ids, feats = common.load_features(p)
    assert ids == ["a", "b"] and feats.dtype == np.float32 and feats.shape == (2, 3)


def test_should_skip(tmp_path):
    a = tmp_path / "a"
    assert not common.should_skip([a], force=False)
    a.write_text("x")
    assert common.should_skip([a], force=False)
    assert not common.should_skip([a], force=True)


def test_resolve_device_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert common.resolve_device("cpu") == "cpu"
    with pytest.raises(SystemExit) as e:
        common.resolve_device("cuda:1")
    assert "cuda:1" in str(e.value) and "--device cpu" in str(e.value)
