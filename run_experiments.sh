#!/bin/bash

for seed in {0..3}; do
    python3 test_control.py \
        --policy TD3 \
        --hidden_size 64 \
        --batch_size 128 \
        --seed $seed
done


: << 'COMMENT'

To comment 

COMMENT