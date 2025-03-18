#!/bin/bash

# Set experiment parameters
seeds=(1 2 3)
env_id="HalfCheetah-v4"
gpu_id=5 # GPU ID to use

# Experiment configurations
exp_names=("baseline" "redo" "grad_redo")
activations=("relu" "leaky_relu" "tanh" "gelu" "sigmoid" "elu" "silu")
width_multipliers=(0.5 1 2)
depth_multipliers=(0.5 1 2)

# Ensure using specified GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# Create log directory
mkdir -p logs

# Run experiments for all seeds
for seed in "${seeds[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        for activation in "${activations[@]}"; do
            for width_multiplier in "${width_multipliers[@]}"; do
                for depth_multiplier in "${depth_multipliers[@]}"; do
                    log_file="logs/${exp_name}_${activation}_w${width_multiplier}_d${depth_multiplier}_s${seed}.log"
                    python main.py \
                        --env-id $env_id \
                        --exp-name $exp_name \
                        --activation $activation \
                        --width-multiplier $width_multiplier \
                        --depth-multiplier $depth_multiplier \
                        --seed $seed > "$log_file" 2>&1 &
                done
            done
        done
    done
done

# Wait for all background processes to complete
wait

echo "All experiments completed!" 