import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest

import extract_text as et


def test_description_text_joins_paragraphs():
    assert et.description_text(["First.", "Second."]) == "First. Second."
    assert et.description_text("plain") == "plain"
    assert et.description_text(None) == ""


@pytest.fixture(scope="module")
def encoder():
    try:
        return et.TextEncoder("cpu")
    except Exception as e:  # weights not cached and offline
        pytest.skip(f"bert-base-uncased is not available locally: {e}")


def test_encoder_is_in_eval_mode(encoder):
    assert encoder.model.training is False


def test_encode_shape_and_determinism(encoder):
    texts = ["A soft cotton shirt.", "Wireless controller for PC.", "word " * 1000]
    a = encoder.encode(texts, batch_size=2)
    b = encoder.encode(texts, batch_size=2)
    assert a.shape == (3, 768) and a.dtype == np.float32
    assert np.array_equal(a, b)


def test_batched_equals_single(encoder):
    texts = ["short", "a much longer description with many more tokens in it"]
    batched = encoder.encode(texts, batch_size=2)
    single = np.vstack([encoder.encode([t], batch_size=1) for t in texts])
    assert np.allclose(batched, single, atol=1e-5)


def test_empty_description_gives_zero_vector(encoder):
    out = encoder.encode(["", "   ", "real text"])
    assert not out[0].any() and not out[1].any() and out[2].any()


class FakeEncoder:
    def encode(self, texts, batch_size=32):
        return np.array([[len(t)] * 768 for t in texts], dtype=np.float32)


def test_run_writes_features_in_item_order(tmp_path):
    import build_interactions
    import clean_categories
    import common
    from fixtures import make_tiny_cfg

    cfg = make_tiny_cfg(tmp_path)
    build_interactions.run(cfg, ["Women"])
    clean_categories.run(cfg, ["Women"])
    et.run(cfg, ["Women"], encoder=FakeEncoder())
    ids, feats = common.load_features(cfg["interim_dir"] / "Women" / "text_features.npz")
    assert ids == ["W1", "N1"]
    assert feats[1, 0] == len("Novelty shirt Second paragraph")
    et.run(cfg, ["Women"], limit=1, encoder=FakeEncoder())
    ids, _ = common.load_features(cfg["interim_dir"] / "Women" / "text_features.sample.npz")
    assert ids == ["W1"]
