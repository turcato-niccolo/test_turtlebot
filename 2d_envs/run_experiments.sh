#!/bin/bash

# Scripts to reproduce results

for ((i=0;i<10;i+=1))
do
    #python3 main.py \
    #--policy "OurDDPG" \
    #--env "MR-env" \
    #--seed $i &
    
    #python3 main.py \
    #--policy "ExpD3" \
    #--env "MR-env" \
    #--seed $i &

    #python3 main.py \
    #--policy "TD3" \
    #--env "MR-env" \
    #--seed $i &

    python3 main.py \
    --policy "SAC" \
    --env "MR-env" \
    --seed $i &

done