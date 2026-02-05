#!/bin/bash

export HF_EVALUATE_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

xtuner test ./workspace/plast_2b_tinyllama_chat.py \
    --checkpoint "./work_dirs/plast_2b_tinyllama_chat/epoch_1.pth" \
    --launcher none