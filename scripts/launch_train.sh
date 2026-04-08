#!/bin/bash

CONFIG=${1:-training/configs/qwen_train.yaml}

NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Launching DDP training with $NUM_GPUS GPUs"
echo "Using config: $CONFIG"

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=29501 \
  training/train.py \
  --config $CONFIG