#!/bin/bash



# Initial run of test_control.py
python3 test_control.py --policy ExpD3 --hidden_size 64 --batch_size 128

# Loop for 10 iterations
for ((i=0; i<10; i++))
do
    # Evaluate
    python3 evaluate.py

    # Test with loaded model
    python3 test_control.py --policy ExpD3 --hidden_size 64 --batch_size 128 --load_model default
done

# Evaluate
python3 evaluate.py


: << 'COMMENT'

To comment 

COMMENT