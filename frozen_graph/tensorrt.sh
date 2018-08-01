#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python tensorrt.py \
--frozen_graph icnet_model.pb \
--output_node conv6_cls/BiasAdd \
--input_node Placeholder \
--batch_size 1 \
--fp32 \
--num_loops 500 \
--workspace_size 2048
