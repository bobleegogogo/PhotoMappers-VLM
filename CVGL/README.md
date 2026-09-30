# PhotoMappers — cross-view geo-localization

Code and weights for the cross-view geo-localization part of *"A Decade of PhotoMappers: A Longitudinal
Study of Crowdsourced Disaster Photos Geolocalization with Vision-Language Models and Geospatial
Reasoning"*. It contains everything needed to train the models, evaluate them, and generate the
prediction CSVs used in the paper.

Each **group** is one crowdsourced disaster photo (**VGI**) plus the satellite tile (**RSI**, 512×512)
and the street-view panorama (**SVI**, 1024×512) at its coordinate. The task is to localize the VGI photo
by retrieving the correct group from the 2,347 test groups.

## Results (test split, 2,347 groups, gallery = the same 2,347 groups)

| System | R@1 | R@5 | R@10 | R@1% |
|---|---|---|---|---|
| VGI–RSI | 27.82 | 47.59 | 57.95 | 68.51 |
| VGI–SVI | 25.56 | 45.85 | 54.67 | 64.89 |
| VGI–SVI–RSI | 36.17 | 57.82 | 65.83 | 74.52 |
| **VGI–SVI–RSI + Sinkhorn** | **40.90** | **61.10** | **69.45** | **77.46** |

`python predict.py` with the bundled checkpoints reproduces this table exactly, and `make_csvs.py`
reproduces the CSV files byte for byte.

## Method

[Sample4Geo](https://github.com/Skyy93/Sample4Geo) contrastive training (symmetric InfoNCE, shared
encoder for query and reference), with the ConvNeXt backbone replaced by the self-supervised
**DINOv3 ViT-L/16** (`timm: vit_large_patch16_dinov3.lvd1689m`). Two branches are trained separately:

| Branch | Query | Reference | LR | Checkpoint |
|---|---|---|---|---|
| `vgi_rsi` | VGI 384×384 | RSI 384×384 | 1e-4 | `checkpoints/vgi_rsi/weights_e36_27.8228.pth` |
| `svi_vgi` | SVI 336×672 | VGI 336×336 | 1e-5 | `checkpoints/svi_vgi/weights_e28_25.4793.pth` |

The `svi_vgi` branch uses a dynamic position embedding, because the shared ViT sees two input shapes.
Both branches share one recipe. They start from the DINOv3 SSL weights, then train with batch 32 for
40 epochs, using AdamW, a cosine schedule with 1 warm-up epoch, label smoothing 0.1, AMP, gradient
checkpointing and random batch sampling.

The third modality is combined post hoc, without extra training:

- **VGI–SVI–RSI fusion**: `S = 0.6·z(S_VGI→SVI) + 0.4·z(S_VGI→RSI)`, where z is a z-score over the
  whole similarity matrix.
- **+ Sinkhorn**: Sinkhorn normalization of `S` to a doubly-stochastic matrix (temperature 0.3,
  50 iterations). This uses the fact that each query has exactly one matching group in the gallery.

## Layout

```
build_dataset.py      spreadsheet + image folders -> CVUSA-format symlink trees + mapping_11737.csv
train.py              train one branch (--task vgi_rsi | svi_vgi)
predict.py            features -> 4 systems -> recall_<tag>.csv + <tag>_<system>_top10.csv
make_csvs.py          join onto spreadsheet keys -> predictions_11737_{top1,top10_long}.csv, breakdown
run_all.sh            all of the above
sample4geo/           the parts of Sample4Geo used here (see "Changes vs. upstream Sample4Geo")
checkpoints/          the two trained branches (+ their training logs, SHA256SUMS)
```

## Setup

```bash
conda create -n photomapper python=3.13 && conda activate photomapper
pip install -r requirements.txt
```

The original runs used 8× V100-32GB, 4 GPUs per branch. Inference needs one GPU. `train.py` downloads
the DINOv3 backbone from the Hugging Face hub on first use. `predict.py` needs no download.

Put the data under `data/` (or pass the paths with `--xlsx/--old-dir/--new-dir`):

```
data/split_80%_11737.xlsx                      sheets '80%train' (9,390) and '20%test' (2,347)
data/Dataset20260630/{RSI,SVI,VGI}/            original 8,780 groups, "<lat>, <lon>.jpg"
data/Dataset_new2957/{RSI,SVI,VGI}_2957/       2,957 newly added groups
```

## Run

```bash
bash run_all.sh               # build + evaluate the bundled checkpoints + CSVs   (~10 min, 1 GPU)
bash run_all.sh --train       # build + retrain both branches + evaluate + CSVs   (~5 h, 8 GPUs)
```

or step by step:

```bash
python build_dataset.py                    # -> data/cvformat/{disaster_vgi,disaster_svi2vgi}, output/mapping_11737.csv

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --task vgi_rsi --data data/cvformat/disaster_vgi    --out runs/vgi_rsi
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task svi_vgi --data data/cvformat/disaster_svi2vgi --out runs/svi_vgi

python predict.py                                        # bundled checkpoints
python predict.py --vgi-rsi runs/vgi_rsi --svi-vgi runs/svi_vgi --tag my_run   # your own run dirs
python make_csvs.py [--tag my_run --suffix _my_run]
```

## Output files (`output/`)

| File | Rows | Content |
|---|---|---|
| `mapping_11737.csv` | 11,737 | **Join table.** `gid` ↔ `excel_sheet`, `excel_row_in_file` (row in the .xlsx, header = row 1), `GlobalId`, `AttachmentId`, …, `batch` (old/new), `image_name`, true `latitude`/`longitude`, VGI/RSI/SVI paths |
| `recall_dinov3l_11737.csv` | 4 | The results table above |
| `predictions_11737_top1.csv` | 2,347 | One row per test group. For both fusion systems: the top-1 retrieved group, its coordinates and score, `correct`, `error_km` (great-circle distance to the true location) and `rank_of_true` (blank if not in the top 10) |
| `predictions_11737_top10_long.csv` | 46,940 | 10 ranks per query per fusion system, with coordinates, score and `error_km` |
| `breakdown_11737.csv` | 6 | Top-1 accuracy and km error for all test groups, the original ones and the newly added ones |
| `dinov3l_11737_{rsi_only,svi_only,fusion,fusion_sinkhorn}_top10.csv` | 23,470 each | Raw top-10 per system |

To join the predictions onto the spreadsheet, use `excel_row_in_file`. `GlobalId` is not unique: 18
rows share one with another row.

```python
test = pd.read_excel("data/split_80%_11737.xlsx", sheet_name="20%test")
test["excel_row_in_file"] = test.index + 2
merged = test.merge(pd.read_csv("output/predictions_11737_top1.csv"), on="excel_row_in_file", validate="1:1")
```

## Data notes

- **X/Y are swapped in the 2,957 new rows.** In the original 8,780 rows, `Y` is the latitude and the file
  is `f"{Y}, {X}.jpg"`. In the new rows, `X` is the latitude and the file is `f"{X}, {Y}.jpg"`.
  `build_dataset.py` resolves each row by checking which of the two file names exists in that row's
  batch folder. Every row resolves exactly once. The `latitude`/`longitude` columns in all outputs are
  the true values.
- **gids** are assigned train-first in sheet order: train = 0…9389, test = 9390…11736.
- The 11,737 split is a strict superset of the earlier 8,780-group split. No earlier test group is in
  train, and no earlier training group is in test.
- Five coordinates appear in both batches. They are kept as separate groups because they have
  different VGI photos, even though they share the same RSI and SVI.

