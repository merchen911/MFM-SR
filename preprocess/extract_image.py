"""Stage 4: image features from the item images.

For each item the first URL of imageURLHighRes is downloaded; if it fails, the last URL is
used instead. Images are converted to RGB, resized to 224x224 and converted to a tensor
(no further normalisation), and passed through VGG16 pretrained on ImageNet-1K with the
final classification layer removed (4096-dimensional output of the second fully connected
layer). Items whose image cannot be obtained get a zero vector and are listed in
image_failures.csv. Downloaded images are cached so that an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import dataset_dirs, load_config, load_item_meta, read_atomic, resolve_device, save_features, should_skip

DIM = 4096
USER_AGENT = "Mozilla/5.0 (compatible; mfm-sr-preprocessing)"


def candidate_urls(urls) -> list:
    if not urls:
        return []
    return list(dict.fromkeys([urls[0], urls[-1]]))


def fetch_bytes(url: str, timeout: float = 20, retries: int = 3, backoff: float = 1.0) -> bytes:
    retries = max(1, retries)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise  # permanent client error: do not retry
            last = e
        except Exception as e:  # network errors are retried, then reported
            last = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise last


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_image(asin: str, urls, cache_dir: Path, retries: int = 3, backoff: float = 1.0):
    """Return (RGB image, "") or (None, error message)."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{asin}.img"
    if path.exists():
        try:
            return _decode(path.read_bytes()), ""
        except Exception:
            path.unlink()  # unreadable cache entry: download again
    errors = []
    for url in candidate_urls(urls):
        try:
            data = fetch_bytes(url, retries=retries, backoff=backoff)
            img = _decode(data)
        except Exception as e:
            status = f" {e.code}" if isinstance(e, urllib.error.HTTPError) else ""
            errors.append(f"{url}: {type(e).__name__}{status}")
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return img, ""
    return None, "; ".join(errors) or "no image URL"


class ImageEncoder:
    def __init__(self, device: str = "cpu", pretrained: bool = True):
        from torchvision import models, transforms

        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = model.classifier[:-1]  # drop the 1000-way layer -> 4096-dim features
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    @torch.no_grad()
    def encode(self, images) -> np.ndarray:
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        return self.model(batch).float().cpu().numpy()


def run(cfg: dict, names, device: str = "cpu", batch_size: int = 64, workers: int = 16, limit=None,
        force: bool = False, encoder=None, retries: int = 3, backoff: float = 1.0) -> None:
    for n in names:
        interim, processed = dataset_dirs(cfg, n)
        suffix = ".sample" if limit else ""
        out, fail_log = interim / f"image_features{suffix}.npz", interim / f"image_failures{suffix}.csv"
        if should_skip([out, fail_log], force):
            print(f"[skip] {out} exists (use --force to rebuild)")
            continue
        ids = read_atomic(processed / f"{n}.item")["item_id:token"].tolist()[: limit or None]
        meta = load_item_meta(interim)
        urls = [meta.get(i, {}).get("imageURLHighRes") or [] for i in ids]
        if encoder is None:
            encoder = ImageEncoder(resolve_device(device))
        feats = np.zeros((len(ids), DIM), dtype=np.float32)
        failures = []
        load = lambda i: load_image(ids[i], urls[i], cfg["image_cache_dir"], retries, backoff)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(ids), batch_size):
                chunk = list(range(start, min(start + batch_size, len(ids))))
                results = list(pool.map(load, chunk))
                ok = [(i, img) for i, (img, _) in zip(chunk, results) if img is not None]
                failures += [(ids[i], err) for i, (img, err) in zip(chunk, results) if img is None]
                if ok:
                    feats[[i for i, _ in ok]] = encoder.encode([img for _, img in ok])
                if (start // batch_size) % 50 == 0:
                    print(f"  {n}: {min(start + batch_size, len(ids))}/{len(ids)} items, {len(failures)} failures", flush=True)
        save_features(out, ids, feats)
        with open(fail_log, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["asin", "error"])
            w.writerows(failures)
        print(f"[done] {n}: image features {feats.shape}, failures {len(failures)} (zero vectors, see {fail_log.name})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--device", default="cpu", help="e.g. cpu, cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=16, help="parallel downloads")
    p.add_argument("--limit", type=int, default=None, help="only the first N items (writes *.sample.* files)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    run(cfg, a.datasets or list(cfg["datasets"]), a.device, a.batch_size, a.workers, a.limit, a.force)


if __name__ == "__main__":
    main()
