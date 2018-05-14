"""
Written by Matteo Dunnhofer - 2017

Helper functions and procedures
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from easydict import EasyDict as edict
import os
import random
import tensorflow as tf 
import numpy as np
from scipy.io import loadmat
from PIL import Image

WEIGHT_DECAY = 0.00002
def cabs(x):
    return tf.clip_by_value(x,-1,1)
    # max_x = tf.stop_gradient(tf.reduce_max(x))
    # return x/max_x
    # return tf.minimum(1.0, tf.abs(x), name = 'cabs')
    # return tf.minimum(1.0, tf.abs(x), name = 'cabs')
    # return tf.nn.tanh(x)


# def _fw(x, bitwidth):
#   x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)
#   return 2 * quantize(x, bitwidth) - 1
#   # x = quantize(x, bitwidth)
#   # return x

# def _fa(x, bitwidth):
#   x = quantize(x, bitwidth)
#   return x


def quantize(x, k):
    G = tf.get_default_graph()
    n = float(2**k)
    with G.gradient_override_map({"Round": "Identity"}):
      return tf.round(x * n) / n

def quantize_plus(x):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Round": "Identity"}):
      return tf.round(x) 

def _fw(x, bitW,force_quantization=False):
  G = tf.get_default_graph()
  if bitW == 32 and not force_quantization:
    return x
  if bitW == 1:   # BWN
    with G.gradient_override_map({"Sign": "Identity"}):
      E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
      return tf.sign(x / E) * E
        # x = tf.tanh(x)
        # x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
  x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0) # it seems as though most weights are within -1 to 1 region anyways
  return 2 * quantize(x, bitW) - 1

def _fa(x, bitA):
  # return quantize_plus(x)
  return quantize(x, bitA)

################ TensorFlow standard operations wrappers #####################

def msra_initializer():
    """ [ref] K. He et.al 2015: arxiv:1502.01852 
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        k = shape[0] * shape[1]        # kernel size
        d = shape[3]                   # filter number

        stddev = math.sqrt(2. / (k**2 * d))
        return tf.truncated_normal(shape, stddev=stddev, dtype=dtype)

    return _initializer

def lecun_initializer():
    """ [ref] K. He et.al 2015: arxiv:1502.01852 
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        k = shape[0] * shape[1]        # kernel size
        d = shape[3] *shape[2]                 # filter number

        stddev = math.sqrt(2. / (k**2 * d))
        return tf.truncated_normal(shape, stddev=stddev, dtype=dtype)

    return _initializer

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def weight(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.01)
	w = tf.Variable(initial, name=name)
	tf.add_to_collection('weights', w)
	return w

def bias(value, shape, name):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, stride, padding):
	return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool2d(x, kernel, stride, padding):
	return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padding)


def lrn(x, depth_radius, bias, alpha, beta):
	return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)

def relu(x):
	return tf.nn.relu(x)

def batch_norm(x):
	epsilon = 1e-3
	batch_mean, batch_var = tf.nn.moments(x, [0])
	return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, epsilon)



def fire_module(input, in_channel, s1x1, e1x1, e3x3, name):
	with tf.name_scope(name) as scope:
		s1x1_weight = weight([1,1,in_channel,s1x1], 's1x1_weight')
		s1x1_bias = bias(0.0, [s1x1], 's1x1_bias')
		s1x1_out = tf.add(conv2d(input, s1x1_weight, padding='SAME'), s1x1_bias)
		s1x1_out = relu(s1x1_out)

		e1x1_weight = weight([1,1,s1x1,e1x1], 'e1x1_weight')
		e1x1_bias = bias(0.0, [e1x1], 'e1x1_bias')
		e1x1_out = tf.add(conv2d(s1x1_out, e1x1_weight, padding = 'SAME'),e1x1_bias)
		e1x1_out = relu(e1x1_out)

		e3x3_weight = weight([3,3,s1x1,e3x3], 'e3x3_weight')
		e3x3_bias = bias(0.0, [e3x3], 'e3x3_bias')
		e3x3_out = tf.add(conv2d(s1x1_out, e3x3_weight, padding = 'SAME'), e3x3_bias)
		e3x3_out = relu(e3x3_out)

		out = tf.concat(3, [e1x1_out, e3x3_out])

	return out



def conv_layer(layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.5):
	with tf.variable_scope(layer_name) as scope:
		channels = inputs.get_shape()[3]

      	# re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      	# shape [h, w, in, out]
      	# kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
      	# bias_init = tf.constant_initializer(0.0)
      	# kernel_init = tf.constant_initializer(0.005)
      	kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
      	kernel = _variable_with_weight_decay(layer_name + 'kernels', shape=[size, size, int(channels), filters],
          wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      	# biases = _variable_on_device('biases', [filters], bias_init,
      	#                           trainable=(not freeze))
      	# inputs_quant = _fa(cabs(inputs))
      	# kernel_quant = _fw(kernel,7)
      	# biases_quant = _fw(biases,7)
      	conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding,name='convolution')
      	# conv_bias = tf.nn.bias_add(conv, biases_quant, name='bias_add')

      	if relu:
      		out = tf.nn.elu(conv, 'relu')
      	else:
      		out = conv
      	return out


def _conv_compress_layer(layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001):
	with tf.variable_scope(layer_name) as scope:
		channels = inputs.get_shape()[3]

      	# re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      	# shape [h, w, in, out]
      	# kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
      	# bias_init = tf.constant_initializer(0.0)
      	kernel_init = tf.contrib.layers.xavier_initializer_conv2d()

      	kernel = _variable_with_weight_decay(layer_name + 'kernels', shape=[size, size, int(channels), filters],
          wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      	# tf.add_to_collection('kernel', kernel)/
      	# biases = _variable_on_device('biases', [filters], bias_init,
      	#                           trainable=(not freeze))
      	inputs_quant = _fa(cabs(inputs), 4)
      	kernel_quant = _fw(kernel,1)
      	# biases_quant = _fw(biases,7)
      	conv = tf.nn.conv2d(inputs_quant, kernel_quant, [1, stride, stride, 1], padding=padding,name='convolution')
      	# conv_bias = tf.nn.bias_add(conv, biases_quant, name='bias_add')

      	if relu:
      		out = tf.nn.relu(conv, 'relu')
      	else:
      		out = conv
      	return out

def conv_weightonly_layer(layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001):
	with tf.variable_scope(layer_name) as scope:
		channels = inputs.get_shape()[3]

      	# re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      	# shape [h, w, in, out]
      	# kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
      	# bias_init = tf.constant_initializer(0.0)
      	kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
      	kernel = _variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
      	# tf.add_to_collection('kernel', kernel)
      	# biases = _variable_on_device('biases', [filters], bias_init,
      	#                           trainable=(not freeze))
      	# inputs_quant = _fa(cabs(inputs))
      	kernel_quant = _fw(kernel,1)
      	# biases_quant = _fw(biases,7)
      	conv = tf.nn.conv2d(inputs, kernel_quant, [1, stride, stride, 1], padding=padding,name='convolution')
      	# conv_bias = tf.nn.bias_add(conv, biases_quant, name='bias_add')

      	if relu:
      		out = tf.nn.relu(conv, 'relu')
      	else:
      		out = conv
      	return out

def pooling_layer(layer_name, inputs, size, stride, padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs,
                            ksize=[1, size, size, 1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
      return out

def avg_layer(layer_name, inputs, size, stride, padding='SAME'):
    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.avg_pool(inputs,
                            ksize=[1, size, size, 1],
                            strides=[1, stride, stride, 1],
                            padding=padding)
      return out

def fc_layer(layer_name, inputs, hiddens, flatten=True, relu=True
	,xavier=False, stddev=0.001):
	with tf.variable_scope(layer_name) as scope:
		input_shape = inputs.get_shape().as_list()
		print(input_shape)
		if flatten:
			dim = input_shape[1]*input_shape[2]*input_shape[3]
			inputs = tf.reshape(inputs, [-1, dim])
		else:
			dim = input_shape[1]
        kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
        # bias_init = tf.constant_initializer(0.0)
        weights = _variable_with_weight_decay(layer_name+'kernels', shape=[dim, hiddens], wd=WEIGHT_DECAY,initializer=kernel_init)
      	# tf.add_to_collection('kernel', weights)
      	# biases = _variable_on_device('biases', [hiddens], bias_init)
      	outputs = tf.matmul(inputs, weights)
      	if relu:
        	outputs = tf.nn.relu(outputs, 'relu')
      	return outputs

def compressed_fire_layer(layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,freeze=False):
	sq1x1 = _conv_compress_layer(layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,padding='SAME', stddev=stddev, freeze=freeze)
	ex1x1 = _conv_compress_layer(layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,padding='SAME', stddev=stddev, freeze=freeze)
	ex3x3 = _conv_compress_layer(layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,padding='SAME', stddev=stddev, freeze=freeze)
	return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

def fire_layer(layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01,freeze=False):
	sq1x1 = conv_layer(layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,padding='SAME', stddev=stddev, freeze=freeze)
	ex1x1 = conv_layer(layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,padding='SAME', stddev=stddev, freeze=freeze)
	ex3x3 = conv_layer(layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,padding='SAME', stddev=stddev, freeze=freeze)
	return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')


################ batch creation functions #####################

def onehot(index):
	""" It creates a one-hot vector with a 1.0 in
		position represented by index 
	"""
	onehot = np.zeros(1000)
	onehot[index] = 1.0
	return onehot

def read_batch(batch_size, images_source, wnid_labels):
	""" It returns a batch of single images (no data-augmentation)

		ILSVRC 2012 training set folder should be srtuctured like this: 
		ILSVRC2012_img_train
			|_n01440764
			|_n01443537
			|_n01484850
			|_n01491361
			|_ ... 

		Args:
			batch_size: need explanation? :)
			images_sources: path to ILSVRC 2012 training set folder
			wnid_labels: list of ImageNet wnid lexicographically ordered

		Returns:
			batch_images: a tensor (numpy array of images) of shape [batch_size, width, height, channels] 
			batch_labels: a tensor (numpy array of onehot vectors) of shape [batch_size, 1000]
	"""
	batch_images = []
	batch_labels = []

	for i in range(batch_size):
		# random class choice 
		# (randomly choose a folder of image of the same class from a list of previously sorted wnids)
		class_index = random.randint(0, 999)

		folder = wnid_labels[class_index]
		# print(os.path.join(images_source, folder))
		batch_images.append(read_image(os.path.join(images_source, folder)))
		batch_labels.append(onehot(class_index))

	np.vstack(batch_images)
	np.vstack(batch_labels)
	return batch_images, batch_labels

def read_image(images_folder):
	""" It reads a single image file into a numpy array and preprocess it

		Args:
			images_folder: path where to random choose an image

		Returns:
			im_array: the numpy array of the image [width, height, channels]
	"""
	# random image choice inside the folder 
	# (randomly choose an image inside the folder)
	image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder)))
	
	# load and normalize image
	im_array = preprocess_image(image_path)
	#im_array = read_k_patches(image_path, 1)[0]
		
	return im_array

def preprocess_image(image_path):
	""" It reads an image, it resize it to have the lowest dimesnion of 256px,
		it randomly choose a 224x224 crop inside the resized image and normilize the numpy 
		array subtracting the ImageNet training set mean

		Args:
			images_path: path of the image

		Returns:
			cropped_im_array: the numpy array of the image normalized [width, height, channels]
	"""
	IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

	img = Image.open(image_path).convert('RGB')

	# resize of the image (setting lowest dimension to 256px)
	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	# random 244x224 patch
	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_cropped = img.crop((x, y, x + 224, y + 224))

	cropped_im_array = np.array(img_cropped, dtype=np.float32)

	for i in range(3):
		cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

	#for i in range(3):
	#	mean = np.mean(img_c1_np[:,:,i])
	#	stddev = np.std(img_c1_np[:,:,i])
	#	img_c1_np[:,:,i] -= mean
	#	img_c1_np[:,:,i] /= stddev

	return cropped_im_array/255.


""" it reads a batch of images performing some data augmentation 
def read_batch_da(batch_size, im_source, labels):
	batch_im = []
	batch_cls = []

	for i in range(int(float(batch_size) / 4)):
		rand = random.randint(0, 999)

		folder = labels[rand]
		batch_im += read_image_da(os.path.join(im_source, folder))

		batch_l = []
		for j in range(4):
			batch_l.append(onehot(rand))
		batch_cls += batch_l

	np.vstack(batch_im)
	np.vstack(batch_cls)
	return batch_im, batch_cls
"""

""" it reads an image and performs some data augmentation on it
	resize the smallest edge to 256 px and take two random 224x224 patches 
	(with their vertical flip) from it
	so, from one image it will create four
def read_image_da(im_folder):
	batch = []

	im_path = os.path.join(im_folder, random.choice(os.listdir(im_folder)))
	img = Image.open(im_path).convert('RGB')

	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_c1 = img.crop((x, y, x + 224, y + 224))
	img_c1_np = np.array(img_c1, dtype=np.float32)
	img_c1_np[:,:,0] -= VGG_MEAN[2]
	img_c1_np[:,:,1] -= VGG_MEAN[1]
	img_c1_np[:,:,2] -= VGG_MEAN[0]
	
	img_f1 = img_c1.transpose(Image.FLIP_LEFT_RIGHT)
	img_f1_np = np.array(img_f1, dtype=np.float32)
	img_f1_np[:,:,0] -= VGG_MEAN[2]
	img_f1_np[:,:,1] -= VGG_MEAN[1]
	img_f1_np[:,:,2] -= VGG_MEAN[0]
	batch.append(img_c1_np)
	batch.append(img_f1_np)

	x = random.randint(0, img.size[0] - 224)
	y = random.randint(0, img.size[1] - 224)
	img_c2 = img.crop((x, y, x + 224, y + 224))
	img_c2_np = np.array(img_c2, dtype=np.float32)
	img_c2_np[:,:,0] -= VGG_MEAN[2]
	img_c2_np[:,:,1] -= VGG_MEAN[1]
	img_c2_np[:,:,2] -= VGG_MEAN[0]
	
	img_f2 = img_c2.transpose(Image.FLIP_LEFT_RIGHT)
	img_f2_np = np.array(img_f2, dtype=np.float32)
	img_f2_np[:,:,0] -= VGG_MEAN[2]
	img_f2_np[:,:,1] -= VGG_MEAN[1]
	img_f2_np[:,:,2] -= VGG_MEAN[0]
	batch.append(img_c2_np)
	batch.append(img_f2_np)

	return batch
"""

def read_k_patches(image_path, k):
	""" It reads k random crops from an image

		Args:
			images_path: path of the image
			k: number of random crops to take

		Returns:
			patches: a tensor (numpy array of images) of shape [k, 224, 224, 3]

	"""
	IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format

	img = Image.open(image_path).convert('RGB')

	# resize of the image (setting largest border to 256px)
	if img.size[0] < img.size[1]:
		h = int(float(256 * img.size[1]) / img.size[0])
		img = img.resize((256, h), Image.ANTIALIAS)
	else:
		w = int(float(256 * img.size[0]) / img.size[1])
		img = img.resize((w, 256), Image.ANTIALIAS)

	patches = []
	for i in range(k):
		# random 244x224 patch
		x = random.randint(0, img.size[0] - 224)
		y = random.randint(0, img.size[1] - 224)
		img_cropped = img.crop((x, y, x + 224, y + 224))

		cropped_im_array = np.array(img_cropped, dtype=np.float32)

		for i in range(3):
			cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]

		patches.append(cropped_im_array)

	np.vstack(patches)
	return patches

""" reading a batch of validation images from the validation set, 
	groundthruths label are inside an annotations file """
def read_validation_batch(batch_size, validation_source, annotations):
	batch_images_val = []
	batch_labels_val = []

	images_val = sorted(os.listdir(validation_source))

	# reading groundthruths labels
	with open(annotations) as f:
		gt_idxs = f.readlines()
		gt_idxs = [(int(x.strip()) - 1) for x in gt_idxs]

	for i in range(batch_size):
		# random image choice
		idx = random.randint(0, len(images_val) - 1)

		image = images_val[idx]
		batch_images_val.append(preprocess_image(os.path.join(validation_source, image)))
		batch_labels_val.append(onehot(gt_idxs[idx]))

	np.vstack(batch_images_val)
	np.vstack(batch_labels_val)
	return batch_images_val, batch_labels_val

################ Other helper procedures #####################

def load_imagenet_meta(meta_path):
	""" It reads ImageNet metadata from ILSVRC 2012 dev tool file

		Args:
			meta_path: path to ImageNet metadata file

		Returns:
			wnids: list of ImageNet wnids labels (as strings)
			words: list of words (as strings) referring to wnids labels and describing the classes 

	"""
	metadata = loadmat(meta_path, struct_as_record=False)
	
	# ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 'wordnet_height', 'num_train_images']
	synsets = np.squeeze(metadata['synsets'])
	ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
	wnids = np.squeeze(np.array([s.WNID for s in synsets]))
	words = np.squeeze(np.array([s.words for s in synsets]))
	return wnids, words

def read_test_labels(annotations_path):
	""" It reads groundthruth labels from ILSRVC 2012 annotations file

		Args:
			annotations_path: path to the annotations file

		Returns:
			gt_labels: a numpy vector of onehot labels
	"""
	gt_labels = []

	# reading groundthruths labels from ilsvrc12 annotations file
	with open(annotations_path) as f:
		gt_idxs = f.readlines()
		gt_idxs = [(int(x.strip()) - 1) for x in gt_idxs]

	for gt in gt_idxs:
		gt_labels.append(onehot(gt))

	np.vstack(gt_labels)

	return gt_labels

def format_time(time):
	""" It formats a datetime to print it

		Args:
			time: datetime

		Returns:
			a formatted string representing time
	"""
	m, s = divmod(time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))

def imagenet_size(im_source):
	""" It calculates the number of examples in ImageNet training-set

		Args:
			im_source: path to ILSVRC 2012 training set folder

		Returns:
			n: the number of training examples

	"""
	n = 0
	for d in os.listdir(im_source):
		for f in os.listdir(os.path.join(im_source, d)):
			n += 1
	return n
