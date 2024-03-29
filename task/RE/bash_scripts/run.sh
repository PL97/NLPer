#!/usr/bin/env bash

workspace="workspace_$2/$3/baseline/"

rm -rf $workspace/$1 && mkdir -p $workspace/$1 || mkdir -p $workspace/$1

TF_CPP_MIN_LOG_LEVEL="2" CUDA_VISIBLE_DEVICES=0 python main.py \
    --ds $2 \
    --split $1 \
    --workspace $workspace \
    --model $3 \
    --epochs $4 > $workspace/$1/log.txt


## use case example
# sbatch simulate_single.slurm site-0 euadr bluebert 50
# ./bash_scripts/run.sh site-0 euadr bluebert 50