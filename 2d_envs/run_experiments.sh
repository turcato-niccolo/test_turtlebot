#!/bin/bash

# Scripts to reproduce results

for ((i=0;i<2;i+=1))
do
    python3 main.py \
    --policy "OurDDPG" \
    --env "MR-env" \
    --seed $i

    #python3 main.py \
    #--policy "OurDDPG" \
    #--env "MR-corridor-env" \
    #--seed $i
    
    python3 main.py \
    --policy "ExpD3" \
    --env "MR-env" \
    --seed $i

    #python3 main.py \
    #--policy "ExpD3" \
    #--env "MR-corridor-env" \
    #--seed $i
done