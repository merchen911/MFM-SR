# mfm-sr

Code for **"Multimodal attention fusion-based sequential recommendation system"**
(H. Oh, W. Jo, D. Kim; under review at *Heliyon*).

The repository contains
- `preprocess/`: a pipeline that turns the raw Amazon Review Data (2018) into RecBole atomic files,
- `recbole/`: a modified copy of DIF-SR / RecBole that includes the proposed model (`SASRecDM_MTL`) and the baselines.

| Name in the paper | Model name in this repository |
|---|---|
| GRU4Rec, Caser, BERT4Rec, SASRec, SASRecF, FDSA | same |
| DIF-SR | `SASRecD` |
| Proposed | `SASRecDM_MTL` |

## 1. Environment

Tested with Python 3.10 and the versions in `requirements.txt` (PyTorch 2.2.1 with CUDA 12.1).

```bash
pip install -r requirements.txt        # add -r requirements-dev.txt to run the tests
python -m pytest tests                 # optional
```

## 2. Raw data

Download the following four files of the Amazon Review Data (2018) from
https://nijianmo.github.io/amazon/index.html and place them in `data/raw/`:

```
data/raw/Clothing_Shoes_and_Jewelry_5.json.gz
data/raw/meta_Clothing_Shoes_and_Jewelry.json.gz
data/raw/Video_Games_5.json.gz
data/raw/meta_Video_Games.json.gz
```

## 3. Preprocessing

```bash
DEVICE=cuda:0 bash preprocess/run_all.sh          # or DEVICE=cpu
```

| Stage | Script | Output |
|---|---|---|
| 1 | `build_interactions.py`: keep items with a category, a high-resolution image URL and a description that appear in the review file; split Clothing into Boys_Girls / Women / Men by category; keep interactions from 2016-2017 | `data/processed/<dataset>/<dataset>.inter` |
| 2 | `clean_categories.py`: normalise category tokens and remove tokens seen fewer than five times | `data/processed/<dataset>/<dataset>.item` |
| 3 | `extract_text.py`: join all description paragraphs, encode with `bert-base-uncased` (truncated to 512 tokens), use the pooled output (768-d) | `data/interim/<dataset>/text_features.npz` |
| 4 | `extract_image.py`: download the first `imageURLHighRes` URL (fall back to the last one), convert to RGB, resize to 224x224, convert to a tensor, encode with VGG16 pretrained on ImageNet-1K without its last layer (4096-d) | `data/interim/<dataset>/image_features.npz` |
| 5 | `write_atomic.py`: write the features in RecBole format | `data/processed/<dataset>/<dataset>.itememb_{text,image}` |

- Each stage skips outputs that already exist; `bash preprocess/run_all.sh --force` rebuilds everything.
- Items whose image cannot be downloaded receive a zero image vector and are listed in `data/interim/<dataset>/image_failures.csv`. Downloaded images are cached in `data/raw/images/`, so an interrupted image stage resumes where it stopped. Downloading all images of Women and Men takes several hours.
- Users and items with fewer than five interactions are removed iteratively (5-core filtering) by RecBole when the data are loaded (`user_inter_num_interval` and `item_inter_num_interval` set to `"[5,inf)"`).
- Every script accepts `--datasets` and `--help`.

## 4. Models

The proposed model is implemented in `recbole/model/sequential_recommender/sasrecdm_mtl.py`.
The files written by the preprocessing pipeline are standard RecBole atomic files; the feature files
are linked to the item IDs through `alias_of_item_id: [itemid_image, itemid_text]` and
`preload_weight: {itemid_image: item_image_emb, itemid_text: item_text_emb}` in the RecBole configuration.

## License and acknowledgements

The model code is a modified copy of [DIF-SR](https://github.com/AIM-SE/DIF-SR) (Apache-2.0),
which is built on [RecBole](https://github.com/RUCAIBox/RecBole) (MIT). See `LICENSE`, `NOTICE`
and `licenses/RecBole-LICENSE.txt`. The Amazon Review Data are provided by J. Ni, J. Li and
J. McAuley (EMNLP 2019); please follow the terms on the dataset page.
