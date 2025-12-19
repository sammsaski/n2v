#!/bin/bash

instances=(
    "onnx/ACASXU_run2a_1_1_batch_2000.onnx vnnlib/prop_1.vnnlib"
    "onnx/ACASXU_run2a_2_3_batch_2000.onnx vnnlib/prop_2.vnnlib"
    "onnx/ACASXU_run2a_3_4_batch_2000.onnx vnnlib/prop_3.vnnlib"
    "onnx/ACASXU_run2a_2_5_batch_2000.onnx vnnlib/prop_4.vnnlib"
    "onnx/ACASXU_run2a_1_1_batch_2000.onnx vnnlib/prop_5.vnnlib"
    "onnx/ACASXU_run2a_1_1_batch_2000.onnx vnnlib/prop_6.vnnlib"
    "onnx/ACASXU_run2a_1_9_batch_2000.onnx vnnlib/prop_7.vnnlib"
    "onnx/ACASXU_run2a_2_9_batch_2000.onnx vnnlib/prop_8.vnnlib"
    "onnx/ACASXU_run2a_3_3_batch_2000.onnx vnnlib/prop_9.vnnlib"
    "onnx/ACASXU_run2a_4_5_batch_2000.onnx vnnlib/prop_10.vnnlib"
)

# Initialize CSV file with headers
echo "ONNX file,VNNLIB property,method,result,time" > results/results_exact.csv

# run with exact method
> results/results_exact.log

count=1
for pair in "${instances[@]}"; do
    set -- $pair  # splits pair into $1 (onnx) and $2 (vnnlib)
    onnx_file="$1"
    vnnlib_file="$2"

    echo "Sample $count: running $onnx_file with $vnnlib_file"

    # Run verification and capture output (with 600 second timeout)
    output=$(timeout 600 python verify_acasxu_star.py "$onnx_file" "$vnnlib_file" --method exact --parallel --workers 112 2>&1)
    exit_code=$?
    echo "$output" >> results/results_exact.log

    # Extract result and time from output
    if [ $exit_code -eq 124 ]; then
        # Timeout occurred
        result="TIMEOUT"
        time="600.0"
    else
        result=$(echo "$output" | grep "^Result:" | awk '{print $2}')
        time=$(echo "$output" | grep "^Time:" | sed 's/Time: \([0-9.]*\)s/\1/')
    fi

    # Append to CSV
    echo "$onnx_file,$vnnlib_file,exact,$result,$time" >> results/results_exact.csv

    # increase count
    count=$((count + 1))
done

# run with approx method
> results/results_approx.log

count=1
for pair in "${instances[@]}"; do
    set -- $pair  # splits pair into $1 (onnx) and $2 (vnnlib)
    onnx_file="$1"
    vnnlib_file="$2"

    echo "Sample $count: running $onnx_file with $vnnlib_file"

    # Run verification and capture output (with 600 second timeout)
    output=$(timeout 600 python verify_acasxu_star.py "$onnx_file" "$vnnlib_file" --method approx --parallel --workers 112 2>&1)
    exit_code=$?
    echo "$output" >> results/results_approx.log

    # Extract result and time from output
    if [ $exit_code -eq 124 ]; then
        # Timeout occurred
        result="TIMEOUT"
        time="600.0"
    else
        result=$(echo "$output" | grep "^Result:" | awk '{print $2}')
        time=$(echo "$output" | grep "^Time:" | sed 's/Time: \([0-9.]*\)s/\1/')
    fi

    # Append to CSV
    echo "$onnx_file,$vnnlib_file,approx,$result,$time" >> results/results_approx.csv

    # increase count
    count=$((count + 1))
done



