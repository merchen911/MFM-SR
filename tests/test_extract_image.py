import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

import extract_image as ei


def png(path, color):
    Image.new("RGB", (40, 30), color).save(path)
    return path.as_uri()


def test_candidate_urls():
    assert ei.candidate_urls(["a", "b", "c"]) == ["a", "c"]
    assert ei.candidate_urls(["a"]) == ["a"]
    assert ei.candidate_urls([]) == [] and ei.candidate_urls(None) == []


def test_fallback_to_last_url_and_rgb_order(tmp_path):
    good = png(tmp_path / "red.png", (255, 0, 0))
    img, err = ei.load_image("A1", [(tmp_path / "nope.png").as_uri(), good], tmp_path / "cache", retries=1, backoff=0)
    assert err == "" and img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 0, 0)  # red stays red (no BGR swap)
    assert (tmp_path / "cache" / "A1.img").exists()


def test_cache_is_reused_without_network(tmp_path):
    src = tmp_path / "blue.png"
    ei.load_image("B1", [png(src, (0, 0, 255))], tmp_path / "cache", retries=1, backoff=0)
    src.unlink()
    img, err = ei.load_image("B1", [src.as_uri()], tmp_path / "cache", retries=1, backoff=0)
    assert err == "" and img.getpixel((0, 0)) == (0, 0, 255)


def test_non_image_content_is_a_failure(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<html>Not found</html>")
    img, err = ei.load_image("C1", [page.as_uri()], tmp_path / "cache", retries=1, backoff=0)
    assert img is None and "page.html" in err
    assert not (tmp_path / "cache" / "C1.img").exists()


def test_no_url_is_a_failure(tmp_path):
    img, err = ei.load_image("D1", [], tmp_path / "cache", retries=1, backoff=0)
    assert img is None and err == "no image URL"


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "err", {}, None)


def _patch_urlopen(monkeypatch, code):
    calls, sleeps = [], []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise _http_error(req.full_url, code)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ei.time, "sleep", lambda s: sleeps.append(s))
    return calls, sleeps


def test_http_404_is_not_retried(monkeypatch):
    calls, sleeps = _patch_urlopen(monkeypatch, 404)
    with pytest.raises(urllib.error.HTTPError):
        ei.fetch_bytes("http://x/a.jpg", retries=3, backoff=1)
    assert len(calls) == 1 and sleeps == []


def test_http_503_is_retried_without_final_sleep(monkeypatch):
    calls, sleeps = _patch_urlopen(monkeypatch, 503)
    with pytest.raises(urllib.error.HTTPError):
        ei.fetch_bytes("http://x/a.jpg", retries=3, backoff=1)
    assert len(calls) == 3 and len(sleeps) == 2


def test_retries_zero_means_one_attempt(monkeypatch):
    calls, _ = _patch_urlopen(monkeypatch, 503)
    with pytest.raises(urllib.error.HTTPError):
        ei.fetch_bytes("http://x/a.jpg", retries=0, backoff=0)
    assert len(calls) == 1


def test_failure_message_contains_http_status(monkeypatch, tmp_path):
    _patch_urlopen(monkeypatch, 404)
    img, err = ei.load_image("E1", ["http://x/a.jpg"], tmp_path / "cache", retries=3, backoff=0)
    assert img is None and "http://x/a.jpg: HTTPError 404" in err


@pytest.fixture(scope="module")
def encoder():
    return ei.ImageEncoder("cpu", pretrained=False)


def test_encoder_eval_mode_shape_and_determinism(encoder, tmp_path):
    assert encoder.model.training is False
    assert all(not m.training for m in encoder.model.modules())
    images = [Image.new("RGB", (50, 40), (i * 40, 100, 200)) for i in range(3)]
    a, b = encoder.encode(images), encoder.encode(images)
    assert a.shape == (3, 4096) and a.dtype == np.float32
    assert np.array_equal(a, b)


def test_run_zero_vector_and_failure_log(tmp_path, encoder):
    import build_interactions
    import clean_categories
    import common
    from fixtures import make_tiny_cfg

    cfg = make_tiny_cfg(tmp_path)
    build_interactions.run(cfg, ["Women", "Men"])
    clean_categories.run(cfg, ["Women", "Men"])
    ei.run(cfg, ["Women", "Men"], encoder=encoder, retries=1, backoff=0, workers=2)
    ids, feats = common.load_features(cfg["interim_dir"] / "Women" / "image_features.npz")
    assert ids == ["W1", "N1"] and feats.shape == (2, 4096)  # N1 falls back to its last URL
    men_ids, men_feats = common.load_features(cfg["interim_dir"] / "Men" / "image_features.npz")
    assert men_ids == ["M1"] and not men_feats.any()
    log = (cfg["interim_dir"] / "Men" / "image_failures.csv").read_text().splitlines()
    assert log[0] == "asin,error" and log[1].startswith("M1,")
