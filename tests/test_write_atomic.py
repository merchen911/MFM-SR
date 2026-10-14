import numpy as np
import pytest

import common
import write_atomic as wa


def test_embedding_file_format_and_roundtrip(tmp_path):
    p = tmp_path / "x.itememb_text"
    feats = np.array([[0.5, -1.25, 3.0], [1e-8, 2.0, -0.0]], dtype=np.float32)
    wa.write_embedding_file(p, "text", ["a", "b"], feats)
    lines = p.read_text().splitlines()
    assert lines[0] == "itemid_text:token\titem_text_emb:float_seq"
    assert lines[1] == "a\t0.5 -1.25 3"
    ids, back = wa.read_embedding_file(p)
    assert ids == ["a", "b"] and np.allclose(back, feats)


def _prepare(tmp_path):
    import build_interactions
    import clean_categories
    import extract_image
    import extract_text
    from fixtures import make_tiny_cfg

    class FakeText:
        def encode(self, texts, batch_size=32):
            return np.ones((len(texts), 768), dtype=np.float32)

    cfg = make_tiny_cfg(tmp_path)
    build_interactions.run(cfg, ["Video_Games"])
    clean_categories.run(cfg, ["Video_Games"])
    extract_text.run(cfg, ["Video_Games"], encoder=FakeText())
    extract_image.run(cfg, ["Video_Games"], encoder=extract_image.ImageEncoder("cpu", pretrained=False), retries=1, backoff=0, workers=2)
    return cfg


def test_run_writes_both_files_and_skips_existing(tmp_path):
    cfg = _prepare(tmp_path)
    wa.run(cfg, ["Video_Games"])
    d = cfg["processed_dir"] / "Video_Games"
    for field, dim in (("text", 768), ("image", 4096)):
        ids, feats = wa.read_embedding_file(d / f"Video_Games.itememb_{field}")
        assert ids == ["V1", "V2", "V3"] and feats.shape == (3, dim)
    (d / "Video_Games.itememb_text").write_text("sentinel")
    wa.run(cfg, ["Video_Games"])
    assert (d / "Video_Games.itememb_text").read_text() == "sentinel"


def test_mismatched_ids_stop_with_message(tmp_path):
    cfg = _prepare(tmp_path)
    common.save_features(cfg["interim_dir"] / "Video_Games" / "text_features.npz", ["V1"], np.ones((1, 768)))
    with pytest.raises(SystemExit) as e:
        wa.run(cfg, ["Video_Games"])
    assert "text_features.npz" in str(e.value) and "--limit" in str(e.value)
