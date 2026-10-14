"""Stage 3: text features from item descriptions.

All description paragraphs are joined with a space, tokenized with the bert-base-uncased
tokenizer (truncated to 512 tokens), and the pooled output ([CLS] -> linear -> tanh,
768 dimensions) is used as the text feature. Items without a description get a zero vector.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common import dataset_dirs, load_config, load_item_meta, read_atomic, resolve_device, save_features, should_skip

MODEL_NAME = "bert-base-uncased"
DIM = 768


def description_text(desc) -> str:
    if desc is None:
        return ""
    if isinstance(desc, (list, tuple)):
        return " ".join(str(x) for x in desc)
    return str(desc)


class TextEncoder:
    def __init__(self, device: str = "cpu", model_name: str = MODEL_NAME, max_length: int = 512):
        from transformers import BertModel, BertTokenizer

        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        idx = [i for i, t in enumerate(texts) if t.strip()]
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            enc = self.tokenizer(
                [texts[i] for i in chunk], truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            out[chunk] = self.model(**enc).pooler_output.float().cpu().numpy()
        return out


def run(cfg: dict, names, device: str = "cpu", batch_size: int = 32, limit=None, force: bool = False, encoder=None) -> None:
    for n in names:
        interim, processed = dataset_dirs(cfg, n)
        out = interim / ("text_features.sample.npz" if limit else "text_features.npz")
        if should_skip([out], force):
            print(f"[skip] {out} exists (use --force to rebuild)")
            continue
        ids = read_atomic(processed / f"{n}.item")["item_id:token"].tolist()[: limit or None]
        meta = load_item_meta(interim)
        texts = [description_text(meta[i]["description"]) if i in meta else "" for i in ids]
        if encoder is None:
            encoder = TextEncoder(resolve_device(device))
        feats = encoder.encode(texts, batch_size)
        save_features(out, ids, feats)
        print(f"[done] {n}: text features {feats.shape}, items without description: {sum(not t.strip() for t in texts)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--device", default="cpu", help="e.g. cpu, cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="only the first N items (writes text_features.sample.npz)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    run(cfg, a.datasets or list(cfg["datasets"]), a.device, a.batch_size, a.limit, a.force)


if __name__ == "__main__":
    main()
