ò#!/bin/bash

for seed in {0..3}; do
    python3 test_control.py \
        --policy TD3 \
        --hidden_size 256 \ò
        --batch_size 256 \
        --seed $seed
done
ò

: << 'COMMENT'

To comment 

COMMENT