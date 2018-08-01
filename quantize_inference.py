from __future__ import print_function
import argparse
import os
import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from model import ICNet, ICNet_BN
from image_reader import read_labeled_image_list

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# define setting & model configuration
ADE20k_param = {'name': 'ade20k',
				'input_size': [480, 480],
				'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
				'ignore_label': 0,
				'num_steps': 2000,
				'data_dir': '../../ADEChallengeData2016/', 
				'data_list': './list/ade20k_val_list.txt'}
				
cityscapes_param = {'name': 'cityscapes',
					'input_size': [1024, 2048],
					'num_classes': 19,
					'ignore_label': 255,
					'num_steps': 500,
					'data_dir': '../cityscapes', 
					'data_list': './list/cityscapes_val_list.txt'}

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy', 
			  'trainval': './model/icnet_cityscapes_trainval_90k.npy',
			  'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
			  'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
			  'others': './model/',
			  'icnet': './model/icnet_model.npy'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN, 'icnet': ICNet}


def get_arguments():
	parser = argparse.ArgumentParser(description="Reproduced PSPNet")

	parser.add_argument("--measure-time", action="store_false",
						help="whether to measure inference time")
	parser.add_argument("--model", type=str, default='train_bn',
						help="Model to use.",
						choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others', 'icnet'])
	parser.add_argument("--flipped-eval", action="store_true",
						help="whether to evaluate with flipped img.")
	parser.add_argument("--dataset", type=str, default='cityscapes',
						choices=['ade20k', 'cityscapes'])
	parser.add_argument("--filter-scale", type=int, default=1,
						help="1 for using pruned model, while 2 for using non-pruned model.",
						choices=[1, 2])
	parser.add_argument("--quantize", action="store_true",
						help="whether to use quantization")
	return parser.parse_args()

def load(saver, sess, ckpt_path):
	saver.restore(sess, ckpt_path)
	print("Restored model parameters from {}".format(ckpt_path))

def preprocess(img, param):
	# Convert RGB to BGR
	img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
	img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
	# Extract mean.
	img -= IMG_MEAN

	shape = param['input_size']

	if param['name'] == 'cityscapes':
		#img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
		img.set_shape([shape[0], shape[1], 3])
		img = tf.expand_dims(img, axis=0)
	elif param['name'] == 'ade20k':
		img = tf.expand_dims(img, axis=0)
		img = tf.image.resize_bilinear(img, shape, align_corners=True)
		
	return img

def main():
	args = get_arguments()

	print("=========================================================")
	print("================== construct network ====================")
	print("=========================================================")
	
	if args.dataset == 'ade20k':
		param = ADE20k_param
	elif args.dataset == 'cityscapes':
		param = cityscapes_param
	
	input_image = tf.placeholder(tf.float32, shape=(None, 1024, 2048, 3))

	model = model_config[args.model]
	net = model({'data': input_image}, num_classes=param['num_classes'], 
					filter_scale=args.filter_scale, evaluation=True)
	
	# Predictions.
	# raw_output.name = conv6_cls/BiasAdd:0
	raw_output = net.layers['conv6_cls']  # (?, 255, 512, 19)

	
	raw_output_up = tf.image.resize_bilinear(raw_output, size=(1024,2048), align_corners=True)  # (?, 1024, 2048, 19)
	raw_output_up = tf.argmax(raw_output_up, axis=3)  # (?, 1024, 2048)
	raw_pred = tf.expand_dims(raw_output_up, dim=3)  # (?, 1024, 2048, 1)
	
	# Set up tf session and initialize variables.
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	local_init = tf.local_variables_initializer()
	
	sess.run(init)
	sess.run(local_init)

	#model_path = model_paths[args.model]
	model_path = "./icnet_finetuned_model"
	if True:
		ckpt = tf.train.get_checkpoint_state(model_path)
		if ckpt and ckpt.model_checkpoint_path:
			loader = tf.train.Saver(var_list=tf.global_variables())
			load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
			load(loader, sess, ckpt.model_checkpoint_path)
		else:
			print('No checkpoint file found.')
			exit(1)
	else:
		net.load(model_path, sess)
		print('Restore from {}'.format(model_path))

	if args.quantize:
		print("=========================================================")
		print("===================== use quantize ======================")
		print("=========================================================")
		tf.contrib.quantize.create_eval_graph()
		description = "quantize_evaluation"
	else:
		description = "evaluation"

	print("=========================================================")
	print("===================== save pb files =====================")
	print("=========================================================")

	if True:
		if not os.path.exists("frozen_graph"):
			os.makedirs("fr_graph")
		graph_def = tf.get_default_graph().as_graph_def()

		#output_node = "ExpandDims"
		output_node = "conv6_cls/BiasAdd"
		output_name = output_node.split("/")[0]
		
		output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, [output_node])
		model_f = tf.gfile.GFile("./frozen_graph/icnet_model_" + output_name +".pb","wb")
		model_f.write(output_graph_def.SerializeToString())
		exit(0)

	print("=========================================================")
	print("==================== start inference ====================")
	print("=========================================================")

	# prepare image and label
	image_filename = tf.placeholder(dtype=tf.string)
	img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
	img.set_shape([None, None, 3])
	img = preprocess(img, param)

	img_files, anno_files = read_labeled_image_list(param['data_dir'], param['data_list'])
	time_list = []
	# start inference
	for i in trange(param['num_steps'], desc=description, leave=True):

		input_img = sess.run(img, feed_dict = {image_filename: img_files[i]})

		time1 = time.time()
		_ = sess.run(raw_pred, feed_dict = {input_image: input_img})
		time2 = time.time()
		inference_time = time2 - time1
		time_list.append(inference_time)
		print('average inference time: {0} \t fps: {1}'.format(np.mean(time_list[-100:]), 1/np.mean(time_list[-100:])))
		print('average inference time: {0} \t fps: {1}'.format(np.mean(time_list), 1/np.mean(time_list)))


if __name__ == '__main__':
	main()













'''
from __future__ import print_function

import argparse
import os
import glob
import sys
import timeit
from tqdm import trange
import tensorflow as tf
import numpy as np
from scipy import misc

from model import ICNet, ICNet_BN
from tools import decode_labels

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
# define setting & model configuration
ADE20k_class = 150 # predict: [0~149] corresponding to label [1~150], ignore class 0 (background)
cityscapes_class = 19

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy', 
			  'trainval': './model/icnet_cityscapes_trainval_90k.npy',
			  'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
			  'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
			  'others': './model/',
			  'icnet': './model/icnet_model.npy'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 
				'trainval_bn': ICNet_BN, 'others': ICNet_BN, 'icnet': ICNet}

snapshot_dir = './snapshots'
SAVE_DIR = './output/'

def get_arguments():
	parser = argparse.ArgumentParser(description="Reproduced PSPNet")
	parser.add_argument("--img-path", type=str, default='',
						help="Path to the RGB image file or input directory.",
						required=True)
	parser.add_argument("--model", type=str, default='',
						help="Model to use.",
						choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others', 'icnet'],
						required=True)
	parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
						help="Path to save output.")
	parser.add_argument("--flipped-eval", action="store_true",
						help="whether to evaluate with flipped img.")
	parser.add_argument("--filter-scale", type=int, default=1,
						help="1 for using pruned model, while 2 for using non-pruned model.",
						choices=[1, 2])
	parser.add_argument("--dataset", type=str, default='',
						choices=['ade20k', 'cityscapes'],
						required=True)

	return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
	  os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
	saver.restore(sess, ckpt_path)
	print("Restored model parameters from {}".format(ckpt_path))

def load_img(img_path):
	if os.path.isfile(img_path):
		print('successful load img: {0}'.format(img_path))
	else:
		print('not found file: {0}'.format(img_path))
		sys.exit(0)

	filename = img_path.split('/')[-1]
	img = misc.imread(img_path, mode='RGB')
	print('input image shape: ', img.shape)

	return img, filename

def preprocess(img):
	# Convert RGB to BGR
	img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
	img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
	# Extract mean.
	img -= IMG_MEAN
	
	img = tf.expand_dims(img, dim=0)

	return img

def check_input(img):
	ori_h, ori_w = img.get_shape().as_list()[1:3]

	if ori_h % 32 != 0 or ori_w % 32 != 0:
		new_h = (int(ori_h/32) + 1) * 32
		new_w = (int(ori_w/32) + 1) * 32
		shape = [new_h, new_w]

		img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)
		
		print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
	else:
		shape = [ori_h, ori_w]

	return img, shape

def main():
	args = get_arguments()
	
	if args.dataset == 'cityscapes':
		num_classes = cityscapes_class
	else:
		num_classes = ADE20k_class

	# Read images from directory (size must be the same) or single input file
	imgs = []
	filenames = []
	if os.path.isdir(args.img_path):
		file_paths = glob.glob(os.path.join(args.img_path, '*'))
		for file_path in file_paths:
			ext = file_path.split('.')[-1].lower()

			if ext == 'png' or ext == 'jpg':
				img, filename = load_img(file_path)
				imgs.append(img)
				filenames.append(filename)
	else:
		img, filename = load_img(args.img_path)
		imgs.append(img)
		filenames.append(filename)

	shape = imgs[0].shape[0:2]


	x = tf.placeholder(dtype=tf.float32, shape=img.shape)
	img_tf = preprocess(x)
	img_tf, n_shape = check_input(img_tf)

	model = model_config[args.model]
	net = model({'data': img_tf}, num_classes=num_classes, filter_scale=args.filter_scale)
	
	raw_output = net.layers['conv6_cls']

	# Predictions.
	raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)
	raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
	raw_output_up = tf.argmax(raw_output_up, axis=3)
	# convert trainLabelID to color
	#pred = decode_labels(raw_output_up, shape, num_classes)

	# Init tf Session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()

	sess.run(init)

	restore_var = tf.global_variables()
	
	model_path = model_paths[args.model]
	if args.model == 'others':
		ckpt = tf.train.get_checkpoint_state(model_path)
		print(ckpt)
		print(ckpt.model_checkpoint_path)
		if ckpt and ckpt.model_checkpoint_path:
			loader = tf.train.Saver(var_list=tf.global_variables())
			load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
			load(loader, sess, ckpt.model_checkpoint_path)
		else:
			print('No checkpoint file found.')
	else:
		net.load(model_path, sess)
		print('Restore from {}'.format(model_path))

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)




	# |-----------------------|
	# |try quantizing to uint8|
	# |-----------------------|
	print("evaluation with quantizing to int8...")
	# Save the checkpoint and eval graph proto to disk for freezing and providing to TFLite.
	if not os.path.exists("quantize_graph"):
		os.makedirs("quantize_graph")

	g = tf.get_default_graph()

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	
	# float graph
	tf.train.write_graph(g.as_graph_def(), './quantize_graph', 'icnet_graph.pb', False)
	saver.save(sess, "./quantize_graph/icnet_weights")

	# int8 graph
	tf.contrib.quantize.create_eval_graph()
	tf.train.write_graph(g.as_graph_def(), './quantize_graph', 'icnet_quantize_graph.pb', False)
	saver.save(sess, "./quantize_graph/icnet_quantize_weights")
	
	exit(0)

	# |--------------|
	# |end quantizing|
	# |--------------|


	for i in trange(len(imgs), desc='Inference', leave=True):
		start_time = timeit.default_timer() 
		preds = sess.run(pred, feed_dict={x: imgs[i]})
		elapsed = timeit.default_timer() - start_time

		print('inference time: {}'.format(elapsed))
		misc.imsave(args.save_dir + filenames[i], preds[0])

if __name__ == '__main__':
	main()
'''