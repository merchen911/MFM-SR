"""Tiny synthetic Amazon-style raw files for end-to-end tests."""
import copy
import gzip
import json
from pathlib import Path

from PIL import Image

CLOTHING_META = [
    {"asin": "W1", "category": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Tops"], "imageURLHighRes": ["IMG/W1.png"], "description": ["Soft cotton top"]},
    {"asin": "N1", "category": ["Clothing, Shoes & Jewelry", "Novelty & More", "Women", "Shirts"], "imageURLHighRes": ["IMG/missing.png", "IMG/N1.png"], "description": ["Novelty shirt", "Second paragraph"]},
    {"asin": "M1", "category": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Socks"], "imageURLHighRes": ["IMG/missing.png"], "description": ["Wool socks"]},
    {"asin": "B1", "category": ["Clothing, Shoes & Jewelry", "Boys", "Shoes"], "imageURLHighRes": ["IMG/B1.png"], "description": ["Kids sneakers"]},
    {"asin": "G1", "category": ["Clothing, Shoes & Jewelry", "Girls", "Clothing", "Dresses"], "imageURLHighRes": ["IMG/G1.png"], "description": ["Summer dress"]},
    {"asin": "X1", "category": ["Clothing, Shoes & Jewelry", "Women"], "imageURLHighRes": ["IMG/W1.png"], "description": []},
    {"asin": "W1", "category": ["Duplicate"], "imageURLHighRes": ["IMG/W1.png"], "description": ["dup"]},
    {"asin": "Z1", "category": ["Clothing, Shoes & Jewelry", "Women", "Bags"], "imageURLHighRes": ["IMG/W1.png"], "description": ["never reviewed"]},
]
CLOTHING_REVIEWS = [
    ("u1", "W1", "01 5, 2016"), ("u1", "M1", "02 10, 2016"), ("u2", "W1", "12 31, 2015"),
    ("u2", "N1", "03 3, 2017"), ("u3", "B1", "07 4, 2018"), ("u3", "G1", "08 8, 2017"),
    ("u4", "X1", "01 1, 2016"), ("u5", "Q9", "01 1, 2016"),
]
VG_META = [
    {"asin": "V1", "category": ["Video Games", "PC", "Games"], "imageURLHighRes": ["IMG/V1.png"], "description": ["A game"]},
    {"asin": "V2", "category": ["Video Games", "Xbox 360", "Accessories", "Faceplates, Protectors & Skins"], "imageURLHighRes": ["IMG/V2.png"], "description": ["Skin"]},
    {"asin": "V3", "category": ["Video Games", "PC", "Games", "</span></span></span>", "junk"], "imageURLHighRes": ["IMG/V3.png"], "description": ["Another"]},
]
VG_REVIEWS = [("a", "V1", "05 1, 2016"), ("b", "V2", "06 2, 2017"), ("c", "V3", "07 3, 2016"), ("d", "V1", "08 4, 2015")]
COLORS = {"W1": (255, 0, 0), "N1": (0, 255, 0), "B1": (0, 0, 255), "G1": (10, 20, 30), "V1": (200, 100, 50), "V2": (5, 5, 5), "V3": (90, 90, 90)}


def _write_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_tiny_cfg(tmp_path: Path) -> dict:
    raw, img = tmp_path / "raw", tmp_path / "img"
    raw.mkdir()
    img.mkdir()
    for asin, color in COLORS.items():
        Image.new("RGB", (32, 24), color).save(img / f"{asin}.png")
    url = lambda u: u.replace("IMG/", img.as_uri() + "/")
    clothing_meta, vg_meta = copy.deepcopy(CLOTHING_META), copy.deepcopy(VG_META)  # keep module constants unchanged
    for rows in (clothing_meta, vg_meta):
        for r in rows:
            r["imageURLHighRes"] = [url(u) for u in r["imageURLHighRes"]]
    _write_gz(raw / "meta_Clothing_Shoes_and_Jewelry.json.gz", clothing_meta)
    _write_gz(raw / "Clothing_Shoes_and_Jewelry_5.json.gz", [{"reviewerID": u, "asin": a, "reviewTime": t} for u, a, t in CLOTHING_REVIEWS])
    _write_gz(raw / "meta_Video_Games.json.gz", vg_meta)
    _write_gz(raw / "Video_Games_5.json.gz", [{"reviewerID": u, "asin": a, "reviewTime": t} for u, a, t in VG_REVIEWS])
    return {
        "raw_dir": raw, "interim_dir": tmp_path / "interim", "processed_dir": tmp_path / "processed",
        "image_cache_dir": tmp_path / "cache", "years": [2016, 2017], "rare_category_min_count": 1,
        "sources": {
            "clothing": {"reviews": "Clothing_Shoes_and_Jewelry_5.json.gz", "meta": "meta_Clothing_Shoes_and_Jewelry.json.gz"},
            "video_games": {"reviews": "Video_Games_5.json.gz", "meta": "meta_Video_Games.json.gz"},
        },
        "datasets": {
            "Video_Games": {"source": "video_games", "split": None, "category_rule": "video_games"},
            "Boys_Girls": {"source": "clothing", "split": "boys_girls", "category_rule": "clothing"},
            "Women": {"source": "clothing", "split": "women", "category_rule": "clothing"},
            "Men": {"source": "clothing", "split": "men", "category_rule": "clothing"},
        },
    }
