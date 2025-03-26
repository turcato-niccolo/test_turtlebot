#!/bin/bash


algorithms=("DDPG" "TD3")

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
            --load_model "default"
    done
done

: << 'COMMENT'

algorithms=("DDPG" "TD3")

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