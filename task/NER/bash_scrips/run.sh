#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 python main.py --ds site-$1 > site-2/log.txt
#python main.py --ds site-$1 > site-$1/log.txt
rm -rf workspace/site-$1
mkdir workspace/site-$1
CUDA_VISIBLE_DEVICES=$2 python main.py \
    --ds site-$1 \
    > workspace/site-$1/log.txt