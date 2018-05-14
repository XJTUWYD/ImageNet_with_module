"""
Written by Matteo Dunnhofer - 2017

Definition of AlexNet architecture
"""

import tensorflow as tf
import train_util as tu
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorpack.utils.argtools import graph_memoized



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('is_training', True,
							"""Train the model or test""")
@graph_memoized
def get_dorefa(bitW, bitA, bitG):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x, force_quantization=False):
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

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    @tf.RegisterGradient("FGGrad")
    def grad_fg(op, x):
        rank = x.get_shape().ndims
        assert rank is not None
        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        n = float(2**bitG - 1)
        x = x * 0.5 + 0.5 + tf.random_uniform(
            tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = quantize(x, bitG) - 0.5
        return x * maxx * 2

    def fg(x):
        if bitG == 32:
            return x
        with G.gradient_override_map({"Identity": "FGGrad"}):
            return tf.identity(x)
    return fw, fa, fg



BITW = 32
BITA = 32
BITG = 32
fw, fa, fg = get_dorefa(BITW, BITA, BITG)

def cabs(x):
    return tf.minimum(1.0, tf.abs(x), name = 'cabs')

def selu(x,name = "selu"):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)

def compute_threshold(x):
    # x_max=tf.reduce_max(x,reduce_indices= None, keep_dims= False, name= None)
    x_sum = tf.reduce_sum(tf.abs(x),reduction_indices= None, keep_dims =False ,name= None)
    threshold = tf.div(x_sum,tf.cast(tf.size(x), tf.float32),name= None)
    threshold = tf.multiply(0.7,threshold,name= None)
    return threshold


def compute_alpha(x):
    threshold = compute_threshold(x)
    alpha1_temp1 = tf.where(tf.greater(x,threshold), x, tf.zeros_like(x, tf.float32))
    alpha1_temp2 = tf.where(tf.less(x,-threshold), x, tf.zeros_like(x, tf.float32))
    alpha_array = tf.add(alpha1_temp1,alpha1_temp2,name = None)
    alpha_array_abs = tf.abs(alpha_array)
    alpha_array_abs1 = tf.where(tf.greater(alpha_array_abs,0),tf.ones_like(alpha_array_abs,tf.float32), tf.zeros_like(alpha_array_abs, tf.float32))
    alpha_sum = tf.reduce_sum(alpha_array_abs)
    n = tf.reduce_sum(alpha_array_abs1)
    alpha = tf.div(alpha_sum,n)
    return alpha


def tenary_opration(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("tenarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            threshold =compute_threshold(x)
            x=tf.sign(tf.add(tf.sign(tf.add(x,threshold)),tf.sign(tf.add(x,-threshold))))
            return x

def cnn(x):
	"""
	AlexNet convolutional layers definition

	Args:
		x: tensor of shape [batch_size, width, height, channels]

	Returns:
		pool5: tensor with all convolutions, pooling and lrn operations applied

	"""
	with tf.name_scope('alexnet_cnn') as scope:
		with tf.name_scope('alexnet_cnn_conv1') as inner_scope:
			wcnn1 = tu.weight([11, 11, 3, 96], name='wcnn1')
			# bcnn1 = tu.bias(0.0, [96], name='bcnn1')
			# wcnn1_t = fw(wcnn1)
			# x_t =fa(cabs(x))
			conv1 = tu.conv2d(x, wcnn1, stride=(4, 4), padding='SAME')
			#conv1 = tu.batch_norm(conv1)
			conv1 = tf.nn.relu(conv1)
			norm1 = tu.lrn(conv1, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
			pool1 = tu.max_pool2d(norm1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

		with tf.name_scope('alexnet_cnn_conv2') as inner_scope:
			wcnn2 = tu.weight([5, 5, 96, 256], name='wcnn2')
			# bcnn2 = tu.bias(1.0, [256], name='bcnn2')
			pool1_t = fa(cabs(pool1))
			wcnn2_t = fw(wcnn2)
			conv2 = tu.conv2d(pool1_t, wcnn2_t, stride=(1, 1), padding='SAME')
			#conv2 = tu.batch_norm(conv2)
			conv2 = tf.nn.relu(conv2)
			norm2 = tu.lrn(conv2, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
			pool2 = tu.max_pool2d(norm2, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

		with tf.name_scope('alexnet_cnn_conv3') as inner_scope:
			wcnn3 = tu.weight([3, 3, 256, 384], name='wcnn3')
			# bcnn3 = tu.bias(0.0, [384], name='bcnn3')
			pool2_t = fa(cabs(pool2))
			wcnn3_t = fw(wcnn3)
			conv3 = tu.conv2d(pool2_t, wcnn3_t, stride=(1, 1), padding='SAME')
			#conv3 = tu.batch_norm(conv3)
			conv3 = tf.nn.relu(conv3)

		with tf.name_scope('alexnet_cnn_conv4') as inner_scope:
			wcnn4 = tu.weight([3, 3, 384, 384], name='wcnn4')
			# bcnn4 = tu.bias(1.0, [384], name='bcnn4')
			conv3_t = fa(cabs(conv3))
			wcnn4_t = fw(wcnn4)
			conv4 = tu.conv2d(conv3_t, wcnn4_t, stride=(1, 1), padding='SAME')
			#conv4 = tu.batch_norm(conv4)
			conv4 = tf.nn.relu(conv4)

		with tf.name_scope('alexnet_cnn_conv5') as inner_scope:
			wcnn5 = tu.weight([3, 3, 384, 256], name='wcnn5')
			# bcnn5 = tu.bias(1.0, [256], name='bcnn5')
			conv4_t = fa(cabs(conv4))
			wcnn5_t = fw(wcnn5)
			conv5 = tu.conv2d(conv4_t, wcnn5_t, stride=(1, 1), padding='SAME')
			#conv5 = tu.batch_norm(conv5)
			conv5 = tf.nn.relu(conv5)
			pool5 = tu.max_pool2d(conv5, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

		return pool5

def classifier(x, dropout):
	"""
	AlexNet fully connected layers definition

	Args:
		x: tensor of shape [batch_size, width, height, channels]
		dropout: probability of non dropping out units

	Returns:
		fc3: 1000 linear tensor taken just before applying the softmax operation
			it is needed to feed it to tf.softmax_cross_entropy_with_logits()
		softmax: 1000 linear tensor representing the output probabilities of the image to classify

	"""
	pool5 = cnn(x)

	dim = pool5.get_shape().as_list()
	flat_dim = dim[1] * dim[2] * dim[3] # 6 * 6 * 256
	flat = tf.reshape(pool5, [-1, flat_dim])

	with tf.name_scope('alexnet_classifier') as scope:
		with tf.name_scope('alexnet_classifier_fc1') as inner_scope:
			wfc1 = tu.weight([flat_dim, 4096], name='wfc1')
			# bfc1 = tu.bias(0.0, [4096], name='bfc1')
			flat_t = fa(cabs(flat))
			wfc1_t = fw(wfc1)
			fc1 = tf.matmul(flat_t, wfc1_t)
			#fc1 = tu.batch_norm(fc1)
			fc1 = tf.nn.relu(fc1)
			fc1 = tf.nn.dropout(fc1, dropout)

		with tf.name_scope('alexnet_classifier_fc2') as inner_scope:
			wfc2 = tu.weight([4096, 4096], name='wfc2')
			# bfc2 = tu.bias(0.0, [4096], name='bfc2')
			fc1_t = fa(cabs(fc1))
			wfc2_t = fw(wfc2)
			fc2 = tf.matmul(fc1_t, wfc2_t)
			#fc2 = tu.batch_norm(fc2)
			fc2 = tf.nn.relu(fc2)
			fc2 = tf.nn.dropout(fc2, dropout)

		with tf.name_scope('alexnet_classifier_output') as inner_scope:
			wfc3 = tu.weight([4096, 1000], name='wfc3')
			# bfc3 = tu.bias(0.0, [1000], name='bfc3')
			# wfc3 = tenary_opration(wfc3)
			fc3 = tf.matmul(fc2, wfc3)
			softmax = tf.nn.softmax(fc3)

	return fc3, softmax

