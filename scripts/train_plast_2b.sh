#!/bin/bash

export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XTUNER_DATASET_TIMEOUT=120

torchrun \
  --nproc_per_node=4 \
  -m xtuner.tools.train \
  ../workspace/plast_2b_tinyllama_chat.py \
  --deepspeed deepspeed_zero2 \
  --launcher pytorch