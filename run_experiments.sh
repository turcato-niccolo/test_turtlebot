#!/bin/bash

for seed in {0..3}; do
    python3 test_control.py \
        --policy SAC \
        --hidden_size 64 \
        --batch_size 128 \
        --seed $seed \
        --expl_noise 0.3
done


: << 'COMMENT'

To comment 

COMMENT