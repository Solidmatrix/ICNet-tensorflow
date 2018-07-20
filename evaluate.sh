#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python evaluate.py \
--model icnet \
--dataset cityscapes \
--measure-time