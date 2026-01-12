#!/bin/bash

os=$(uname -s)
gol_executable="./gol"
gol_executable_baseline="./gol-baseline"
gol_executable_cuda="./gol-cuda"
generations=250
input_dir="./step1000_in_250generations"
output_dir="./output_files"
mkdir -p $output_dir

rm -f ai25m063_cpu_time.csv ai25m063_openmp_time.csv ai25m063_cuda_time.csv ai25m063_baseline_time.csv

for rows in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
    echo "Benchmarking baseline single-threaded for ${rows}..."
    $gol_executable_baseline --load $input_dir/random${rows}_in.gol  --save $output_dir/ai25m063_${rows}_baseline_out.gol --generations $generations --measure >> ai25m063_baseline_time.csv
    echo "Benchmarking single-threaded for ${rows}..."
    $gol_executable --mode seq --load $input_dir/random${rows}_in.gol  --save $output_dir/ai25m063_${rows}_cpu_out.gol --generations $generations --measure >> ai25m063_cpu_time.csv
    echo "Benchmarking multi-threaded for ${rows}..."
    $gol_executable --mode omp --threads 4 --load $input_dir/random${rows}_in.gol  --save $output_dir/ai25m063_${rows}_openmp_out.gol --generations $generations --measure >> ai25m063_openmp_time.csv
    if [ "$os" = "Linux" ]; then
      echo "Benchmarking CUDA for ${rows}..."
      $gol_executable_cuda --load $input_dir/random${rows}_in.gol  --save $output_dir/ai25m063_${rows}_cuda_out.gol --generations $generations --measure >> ai25m063_cuda_time.csv
    fi
done
echo "Benchmarking completed."
