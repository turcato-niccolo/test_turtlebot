#!/bin/bash

# Define the batch sizes and hidden sizes to test
batch_sizes=(32 64 128)
hidden_sizes=(32 64)

# Define the maximum number of parallel jobs
max_jobs=10

# Function to manage parallel jobs
function wait_for_jobs {
    while [ $(jobs -rp | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

# Loop through the seeds
for ((i=1; i<2; i+=1))
do
    # Loop through the batch sizes
    for batch_size in "${batch_sizes[@]}"
    do
        # Loop through the hidden sizes
        for hidden_size in "${hidden_sizes[@]}"
        do
            # Run the different policies
            for policy in "OurDDPG" "ExpD3" "TD3"
            do
                # Ensure no more than max_jobs are running
                wait_for_jobs

                # Start the job
                python3 main.py \
                --policy $policy \
                --batch_size $batch_size \
                --hidden_size $hidden_size \
                --env "MR-env" \
                --seed $i &

            done
        done
    done
done

# Wait for all remaining jobs to finish
wait

: << 'COMMENT'
for ((i=0;i<2;i+=1))
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



# Script to reproduce results

max_parallel=3  # Maximum number of parallel processes allowed
current_jobs=0  # Counter for current number of jobs

for ((i=0;i<10;i+=1))
do
    python3 main.py \
    --policy "TD3" \
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
COMMENT