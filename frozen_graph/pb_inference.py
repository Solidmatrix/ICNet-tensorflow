#!/usr/bin/env python
# coding=utf-8

# add CUDA_VISIBLE_DEVICES=0

import tensorflow as tf 
import tensorflow.contrib.tensorrt as trt 
import os
import numpy as np
import time
from scipy import misc

from image_reader import read_labeled_image_list
from tqdm import trange

GPU_MEM_FRACTION = 0.5
#output_node = "ExpandDims"
output_node = "conv6_cls/BiasAdd"
output_name = output_node.split("/")[0]
frozen_graph = "icnet_model_" + output_name +".pb"
mode = "FP16"

evaluate = True
description = "evaluate" if evaluate else "inference"

def get_frozen_graph(graph_file):
	"""Read Frozen Graph file from disk."""
	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def

def get_gpu_config():
	"""Share GPU memory between image preprocessing and inference."""
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = GPU_MEM_FRACTION)
	return tf.ConfigProto(gpu_options=gpu_options)

def preprocess(img):
	# Convert RGB to BGR
	img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
	img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
	# Extract mean.
	IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
	img -= IMG_MEAN

	shape = [1024, 2048]
	#img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
	img.set_shape([shape[0], shape[1], 3])
	img = tf.expand_dims(img, axis=0)

	return img

if __name__ == "__main__":
	#tf.logging.set_verbosity(tf.logging.INFO)

	print(description + " start")

	frozen_graph_def = get_frozen_graph(frozen_graph)
	graph_name = os.path.basename(frozen_graph)

	trt_graph = trt.create_inference_graph(	
				frozen_graph_def, 
				[output_node], 
				max_batch_size = 1,
				max_workspace_size_bytes = 2048<<20,
				precision_mode = mode)

	tf.reset_default_graph()
	g = tf.Graph()
	with g.as_default():
		input_image = tf.placeholder(tf.float32, shape=(None, 1024, 2048, 3))
		return_tensors = tf.import_graph_def(
						graph_def = trt_graph,
						input_map = {"Placeholder": input_image},
						return_elements = [output_node])
		output = return_tensors[0].outputs[0]

		if evaluate:
			input_label = tf.placeholder(tf.float32, shape=(1024, 2048, 1))
			pred_flatten = tf.reshape(output, [-1,])
			raw_gt = tf.reshape(input_label, [-1,])
			mask = tf.not_equal(raw_gt, 255)
			indices = tf.squeeze(tf.where(mask), 1)
			gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
			pred = tf.gather(pred_flatten, indices)
			mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes = 19)

		# prepare image
		img_files, anno_files = read_labeled_image_list('../../cityscapes', '../list/cityscapes_val_list.txt')

		image_filename = tf.placeholder(dtype=tf.string)
		img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
		img.set_shape([None, None, 3])
		img = preprocess(img)

		if evaluate:
			anno_filename = tf.placeholder(dtype=tf.string)
			anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
			anno.set_shape([None, None, 1])

	with tf.Session(graph = g, config=get_gpu_config()) as sess:
		
		if evaluate:
			init = tf.global_variables_initializer()
			local_init = tf.local_variables_initializer()
			sess.run([local_init, init])

		time_list = []
		input_img = sess.run(img, feed_dict = {image_filename: img_files[0]})
		for i in trange(500, desc = description, leave = True):
			#input_img = sess.run(img, feed_dict = {image_filename: img_files[i]})
			
			time1 = time.time()
			out = sess.run(output, feed_dict = {input_image: input_img})
			time2 = time.time()

			time_list.append(time2 - time1)
			print('average inference time: {0} \t fps: {1}'.format(np.mean(time_list[-100:]), 1/np.mean(time_list[-100:])))
			#print('average inference time: {0} \t fps: {1}'.format(np.mean(time_list), 1/np.mean(time_list)))

			if evaluate:
				input_lbl = sess.run(anno, feed_dict = {anno_filename: anno_files[i]})
				_ = sess.run(update_op, feed_dict = {input_image: input_img, input_label: input_lbl})

		if evaluate:
			print('mIoU: {}'.format(sess.run(mIoU)))

	print(description + " finish")

