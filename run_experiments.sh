#!/bin/bash

algorithms=("TD3" "DDPG" "ExpD3" "SAC")

for algo in "${algorithms[@]}"; do
    for seed in {0..3}; do
        if [ "$algo" == "SAC" ]; then
            expl_noise=0.0
        else
            expl_noise=0.3
        fi

        python3 train.py \
            --policy "$algo" \
            --hidden_size 64 \
            --batch_size 128 \
            --seed "$seed" \
            --expl_noise "$expl_noise" \
            --load_model ""
    done
done

: << 'COMMENT'

python3 test_13.py \
    --policy DDPG \
    --hidden_size 64 \
    --batch_size 128 \
    --seed 1 \
    --expl_noise 0.1 \
    --load_model "default"
    

python3 train_14.py \
    --policy SAC \
    --hidden_size 64 \
    --batch_size 128 \
    --seed 0 \
    --expl_noise 0.1 \
    --load_model "default"

------------------------------------------------------------------------------
algorithms=("SAC" "DDPG" "TD3")

for algo in "${algorithms[@]}"; do
    for seed in {0..3}; do
        if [ "$algo" == "SAC" ]; then
            expl_noise=0.0
        else
            expl_noise=0.3
        fi

        python3 test_14.py \
            --policy "$algo" \
            --hidden_size 64 \
            --batch_size 128 \
            --seed "$seed" \
            --expl_noise "$expl_noise" \
            --load_model "default"
    done
done

------------------------------------------------------------------------------
algorithms=("DDPG" "TD3" "SAC")

for algo in "${algorithms[@]}"; do
    for seed in {0..3}; do
        if [ "$algo" == "SAC" ]; then
            expl_noise=0.0
        else
            expl_noise=0.3
        fi

        python3 test_13.py \
            --policy "$algo" \
            --hidden_size 64 \
            --batch_size 128 \
            --seed "$seed" \
            --expl_noise "$expl_noise"
    done
done

------------------------------------------------------------------------------
algorithms=("DDPG" "ExpD3" "TD3")

for algo in "${algorithms[@]}"; do
    for seed in {0..3}; do
        for model in {0..19}; do
            if [ "$algo" == "SAC" ]; then
                expl_noise=0.0
            else
                expl_noise=0.3
            fi

            python3 test.py \
                --policy "$algo" \
                --hidden_size 64 \
                --batch_size 128 \
                --seed "$seed" \
                --expl_noise "$expl_noise" \
                --load_model "./runs/models/${algo}/seed${seed}/${model}"
        done
    done
done
------------------------------------------------------------------------------
COMMENT