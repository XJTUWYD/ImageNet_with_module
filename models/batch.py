from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import pdb
import tarfile
import ops

from six.moves import urllib
import tensorflow as tf
def batch_norm(inputs, is_training=True):
  """Construct the batch normalization layer.

  Args:
    inputs: Tensors from last layer.

  Returns:
    Tensor after batch normalization.
  """
  params_shape = inputs.get_shape()[-1:]
  gamma = tf.get_variable('gamma',
          shape=params_shape,
          initializer=tf.constant_initializer(1.0, tf.float32),
          trainable=True)

  beta = tf.get_variable('offset',
          shape=params_shape,
          initializer=tf.constant_initializer(0.0, tf.float32),
          trainable=True)

  moving_mean = tf.get_variable('moving_mean',
                shape=params_shape,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)

  moving_var = tf.get_variable('moving_var',
                shape=params_shape,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)

  if is_training:
    axes = range(len(inputs.get_shape())-1)
    batch_mean, batch_var = tf.nn.moments(inputs, axes, name='moments')

    train_mean = tf.assign(moving_mean,
                  moving_mean*BN_DECAY + batch_mean*(1 - BN_DECAY))
    train_var = tf.assign(moving_var,
                  moving_var*BN_DECAY + batch_var*(1 - BN_DECAY))
    with tf.control_dependencies([train_mean, train_var]):
      return tf.nn.batch_normalization(inputs, batch_mean,
                  batch_var, beta, gamma, BN_EPSILON)

  else:
    tf.summary.histogram(moving_mean.op.name, moving_mean)
    tf.summary.histogram(moving_var.op.name, moving_var)

    return tf.nn.batch_normalization(inputs, moving_mean,
                moving_var, beta, gamma, BN_EPSILON)