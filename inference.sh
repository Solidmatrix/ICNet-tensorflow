#!/bin/bash

CUDA_VISIBLE_DEVICES=2 \
python inference.py \
--img-path ./input/outdoor_2.png \
--model icnet \
--dataset cityscapes \
--filter-scale 1 \
--save-dir ./result/


