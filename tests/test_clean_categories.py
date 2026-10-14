import pandas as pd

import clean_categories as cc
import common
from fixtures import make_tiny_cfg

ITEM, CAT = "item_id:token", "categories:token_seq"


def raw(ids, cats):
    return pd.DataFrame({ITEM: ids, CAT: cats})


CLOTHING = raw(list("ABCDE"), [
    "['Clothing, Shoes & Jewelry', 'Women', 'Tops']",
    "['Clothing, Shoes & Jewelry', 'Women', 'Tops']",
    "['Clothing, Shoes & Jewelry', 'Women', 'Tops']",
    "['Clothing, Shoes & Jewelry', 'Men', 'Socks']",
    "['Clothing, Shoes & Jewelry', 'Girls']",
])


def test_clean_clothing_counts_rare_positions_over_full_item_list():
    out = cc.clean_clothing(CLOTHING)[CAT].tolist()
    assert out[0] == "Clothing,Shoes&Jewelry,Women,Tops"
    assert out[3] == "Clothing,Shoes&Jewelry,Men,None"
    assert out[4] == "Clothing,Shoes&Jewelry,Girls,None"


def test_build_items_counts_rare_tokens_over_kept_items_only():
    keep1 = cc.build_items(CLOTHING, ["A", "D", "E"], "clothing", 1)
    assert keep1[ITEM].tolist() == ["A", "D", "E"]
    assert keep1[CAT].tolist() == [
        "['Clothing', 'Shoes&Jewelry', 'Women', 'Tops']",
        "['Clothing', 'Shoes&Jewelry', 'Men']",
        "['Clothing', 'Shoes&Jewelry', 'Girls']",
    ]
    keep2 = cc.build_items(CLOTHING, ["A", "D", "E"], "clothing", 2)
    assert keep2[CAT].tolist() == ["['Clothing', 'Shoes&Jewelry']"] * 3


def test_video_games_rules():
    cleaned = cc.clean_video_games(raw(["V3"], ["['Video Games', 'PC', 'Games', '</span></span></span>', 'junk']"]))
    assert cleaned[CAT].tolist() == ["'Video Games', 'PC', 'Games'"]
    tokens = cc.video_games_tokens(["'Video Games', 'Xbox 360', 'Accessories', 'Faceplates, Protectors & Skins'"])
    assert tokens == [['"VideoGames"', '"Xbox360"', '"Accessories"', '"Faceplates"', '"Protectors&Skins"']]


def test_video_games_item_file_is_quoted_like_reference(tmp_path):
    p = tmp_path / "x.item"
    common.write_atomic_table(raw(["0700099867"], ["['\"VideoGames\"', '\"PC\"']"]), p)
    assert p.read_text() == "item_id:token\tcategories:token_seq\n0700099867\t\"['\"\"VideoGames\"\"', '\"\"PC\"\"']\"\n"


def test_run_on_tiny_data(tmp_path):
    import build_interactions

    cfg = make_tiny_cfg(tmp_path)
    build_interactions.run(cfg, list(cfg["datasets"]))
    cc.run(cfg, list(cfg["datasets"]))
    vg = common.read_atomic(cfg["processed_dir"] / "Video_Games" / "Video_Games.item")
    assert vg[ITEM].tolist() == ["V1", "V2", "V3"]
    assert vg.loc[2, CAT] == vg.loc[0, CAT] == "['\"VideoGames\"', '\"PC\"', '\"Games\"']"
    bg = common.read_atomic(cfg["processed_dir"] / "Boys_Girls" / "Boys_Girls.item")
    assert bg[ITEM].tolist() == ["G1"]
