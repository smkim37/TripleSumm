#!/bin/bash
set -e

python main.py \
    --exp_name test-exp \
    --mode test \
    --dataset mrhisum \
    --model triplesumm \
    --data_dir ./data \
    --model_ckpt checkpoints/best_model_ckpt_mrhisum.pth