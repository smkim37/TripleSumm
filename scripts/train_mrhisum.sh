#!/bin/bash
set -e

python main.py \
    --exp_name train-exp \
    --mode train \
    --dataset mrhisum \
    --model triplesumm \
    --data_dir ./data