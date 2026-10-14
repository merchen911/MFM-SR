# mfm-sr

Code for **"Multimodal attention fusion-based sequential recommendation system"**
(H. Oh, W. Jo, D. Kim; under review at *Heliyon*).

The repository contains
- `data/processed/`: the interaction and item files used in the experiments (gzip-compressed),
- `preprocess/`: a pipeline that turns the raw Amazon Review Data (2018) into RecBole atomic files,
- `recbole/`: a modified copy of DIF-SR / RecBole that includes the proposed model (`SASRecDM_MTL`) and the baselines,
- `table3/`: the per-seed test results behind Table 3 of the paper.

| Name in the paper | Model name in this repository |
|---|---|
| GRU4Rec, Caser, BERT4Rec, SASRec, SASRecF, FDSA | same |
| DIF-SR | `SASRecD` |
| Proposed | `SASRecDM_MTL` |

## 1. Environment

Requires Python >= 3.10 (tested with 3.10) and the versions in `requirements.txt` (PyTorch 2.2.1 with CUDA 12.1).

```bash
pip install -r requirements.txt        # add -r requirements-dev.txt to run the tests
python -m pytest tests                 # optional
```

## 2. Data

### Interaction and item files (included)

The interaction (`.inter`) and item (`.item`) files used in the experiments are included as gzip files,
for `<dataset>` = `Video_Games`, `Boys_Girls`, `Women` and `Men`:

```
data/processed/<dataset>/<dataset>.inter.gz
data/processed/<dataset>/<dataset>.item.gz
```

Decompress them before use:

```bash
gunzip -k data/processed/*/*.gz
```

Running stages 1-2 of the pipeline (Section 3) on our copy of the raw data reproduces these files
byte for byte. Raw files downloaded from the dataset page today may differ if the dataset has been
updated since then. The image and text feature files (`.itememb_image`, `.itememb_text`) are not
included; stages 3-5 build them from the raw data.

### Raw data

To rebuild the files or to build the image and text features, download the following four files of the
Amazon Review Data (2018) from https://nijianmo.github.io/amazon/index.html (the metadata files are
provided through a request form on that page) and place them in `data/raw/`:

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
- Items whose image cannot be downloaded receive a zero image vector and are listed in `data/interim/<dataset>/image_failures.csv`. Downloaded images are cached in `data/raw/images/`; after an interruption, cached images are reused, but encoding restarts and previously failed items are retried. Downloading all images of Women and Men can take many hours.
- Disk use: the image cache holds about 123k images for Women alone, and the `.itememb_image` files are several GB.
- The `.inter` files keep all interactions of the selected years. RecBole applies iterative 5-core filtering at load time when `user_inter_num_interval` and `item_inter_num_interval` are set to `"[5,inf)"` (see Section 4).
- Each Python stage script accepts `--datasets`, `--force` and `--help`; `run_all.sh` takes `--force` and the `DATASETS`, `DEVICE` and `PYTHON` environment variables.

## 4. Models

The proposed model is implemented in `recbole/model/sequential_recommender/sasrecdm_mtl.py`.
The files written by the preprocessing pipeline are standard RecBole atomic files. Training
configuration files are not included in this repository. The minimum RecBole configuration needed
to use the produced files is:

```yaml
data_path: data/processed/
load_col:
  inter: [user_id, item_id, timestamp]
  item: [item_id, categories]
  itememb_image: [itemid_image, item_image_emb]
  itememb_text: [itemid_text, item_text_emb]
additional_feat_suffix: [itememb_image, itememb_text]
alias_of_item_id: [itemid_image, itemid_text]
preload_weight:
  itemid_image: item_image_emb
  itemid_text: item_text_emb
user_inter_num_interval: "[5,inf)"   # iterative 5-core filtering at load time
item_inter_num_interval: "[5,inf)"
```

## 5. Per-seed results

`table3/per_seed_results.csv` lists the test results of every model on every dataset for each of the
five random seeds (101, 202, 303, 404 and 505) used for Table 3 of the paper, as recorded in the
training logs (four decimal places). Columns: `dataset`, `model` (names as in the paper), `seed`,
`Recall@10`, `Recall@20`, `NDCG@10` and `NDCG@20`.

Table 3 reports the mean over the five seeds ± the half-width of the 95% confidence interval,
t(0.975, 4) · s / √5, where s is the sample standard deviation. Where the proposed model is the best,
an asterisk marks a significant improvement over the second-best model (one-sided Wilcoxon
signed-rank test paired by seed, p < 0.05).

## License and acknowledgements

The model code is a modified copy of [DIF-SR](https://github.com/AIM-SE/DIF-SR) (Apache-2.0),
which is built on [RecBole](https://github.com/RUCAIBox/RecBole) (MIT). See `LICENSE`, `NOTICE`
and `licenses/RecBole-LICENSE.txt`. The Amazon Review Data are provided by J. Ni, J. Li and
J. McAuley (EMNLP 2019); please follow the terms on the dataset page.
