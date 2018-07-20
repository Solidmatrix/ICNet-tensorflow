#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python quantize_evaluate.py \
--model icnet \
--dataset cityscapes \
--measure-time \
--use-quantize
