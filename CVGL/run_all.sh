#!/usr/bin/env bash
# End-to-end: build the dataset trees, (optionally) retrain both branches, evaluate, write the CSVs.
#
#   bash run_all.sh            # evaluate the bundled checkpoints (minutes, 1 GPU)
#   bash run_all.sh --train    # retrain both branches first (~5 h on 8x V100-32GB, 4 GPUs per branch)
set -euo pipefail
cd "$(dirname "$0")"

python build_dataset.py

if [[ "${1:-}" == "--train" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --task vgi_rsi \
      --data data/cvformat/disaster_vgi --out runs/vgi_rsi > runs_vgi_rsi.log 2>&1 &
  sleep 5
  CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task svi_vgi \
      --data data/cvformat/disaster_svi2vgi --out runs/svi_vgi > runs_svi_vgi.log 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=0 python predict.py --vgi-rsi runs/vgi_rsi --svi-vgi runs/svi_vgi --tag dinov3l_11737_myrun
  python make_csvs.py --tag dinov3l_11737_myrun --suffix _myrun
else
  CUDA_VISIBLE_DEVICES=0 python predict.py
  python make_csvs.py
fi
