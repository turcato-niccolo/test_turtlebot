#!/bin/bash

# Scripts to reproduce results
: << 'COMMENT'
for ((i=0;i<10;i+=1))
do
    #python3 main.py \
    #--policy "OurDDPG" \
    #--env "MR-env" \
    #--seed $i &
    
    python3 main.py \
    --policy "ExpD3" \
    --env "MR-env" \
    --seed $i &

    #python3 main.py \
    #--policy "TD3" \
    #--env "MR-env" \
    #--seed $i &

    #python3 main.py \
    #--policy "SAC" \
    #--env "MR-env" \
    #--seed $i &

done
COMMENT

# Script to reproduce results

max_parallel=3  # Maximum number of parallel processes allowed
current_jobs=0  # Counter for current number of jobs

for ((i=0;i<10;i+=1))
do
    python3 main.py \
    --policy "ExpD3" \
    --env "MR-env" \
    --seed $i &  # Run in background

    ((current_jobs+=1))

    # Check if we've reached the max parallel jobs
    if ((current_jobs >= max_parallel)); then
        wait -n  # Wait for any one job to finish
        ((current_jobs-=1))  # Decrement job counter when one finishes
    fi
done

# Wait for any remaining jobs to finish
wait
