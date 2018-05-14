
import tensorflow as tf
import train_util as tu

def cnn(x):
	"""
	AlexNet convolutional layers definition

	Args:
		x: tensor of shape [batch_size, width, height, channels]

	Returns:
		pool5: tensor with all convolutions, pooling and lrn operations applied

	"""
	with tf.name_scope('vgg_cnn') as scope:
		with tf.name_scope('vgg_cnn_conv1') as inner_scope:
			wcnn1 = tu.weight([3, 3, 3, 64], name='wcnn1')
			bcnn1 = tu.bias(0.0, [64], name='bcnn1')
			conv1 = tf.add(tu.conv2d(x, wcnn1, stride=(1,1), padding='SAME'), bcnn1)
			conv1 = tu.relu(conv1)
			
		with tf.name_scope('vgg_cnn_conv2') as inner_scope:
			wcnn2 = tu.weight([3, 3, 64, 64], name='wcnn2')
			bcnn2 = tu.bias(0.0, [64], name='bcnn2')
			conv2 = tf.add(tu.conv2d(conv1, wcnn2, stride=(1,1),padding='SAME'), bcnn2)
			conv2 = tu.relu(conv2)

		with tf.name_scope('vgg_cnn_pool1') as inner_scope:
			pool1 = tu.max_pool2d(conv2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')

		with tf.name_scope('vgg_cnn_conv3') as inner_scope:
			wcnn3 = tu.weight([3, 3, 64, 128], name = 'wcnn3')
			bcnn3 = tu.bias(0.0, [128], name = 'bcnn3')
			conv3 = tf.add(tu.conv2d(pool1, wcnn3, stride = (1,1), padding = 'SAME'), bcnn3)
			conv3 = tu.relu(conv3)

		with tf.name_scope('vgg_cnn_conv4') as inner_scope:
			wcnn4 = tu.weight([3, 3, 128, 128], name = 'wcnn4')
			bcnn4 = tu.bias(0.0, [128], name = 'bcnn4')
			conv4 = tf.add(tu.conv2d(conv3, wcnn4, stride = (1,1), padding = 'SAME'), bcnn4)
			conv4 = tu.relu(conv4)

		with tf.name_scope('vgg_cnn_pool2') as inner_scope:
			pool2 = tu.max_pool2d(conv4, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')

		with tf.name_scope('vgg_cnn_conv5') as inner_scope:
			wcnn5 = tu.weight([3, 3, 128, 256 ], name='wcnn5')
			bcnn5 = tu.bias(0.0, [256], name='bcnn5')
			conv5 = tf.add(tu.conv2d(pool2, wcnn5, stride=(1,1), padding='SAME'), bcnn5)
			conv5 = tu.relu(conv5)

		with tf.name_scope('vgg_cnn_conv6') as inner_scope:
			wcnn6 = tu.weight([3, 3, 256, 256], name='wcnn6')
			bcnn6 = tu.bias(0.0, [256], name='bcnn6')
			conv6 = tf.add(tu.conv2d(conv5, wcnn6, stride=(1,1), padding='SAME'), bcnn5)
			conv6 = tu.relu(conv6)

		with tf.name_scope('vgg_cnn_conv7') as inner_scope:
			wcnn7 = tu.weight([3, 3, 256, 256], name='wcnn7')
			bcnn7 = tu.bias(0.0, [256], name='bcnn7')
			conv7 = tf.add(tu.conv2d(conv6, wcnn7, stride=(1,1), padding='SAME'), bcnn7)
			conv7 = tu.relu(conv7)

		with tf.name_scope('vgg_cnn_conv8') as inner_scope:
			wcnn8 = tu.weight([3, 3, 256, 256], name='wcnn8')
			bcnn8 = tu.bias(0.0, [256], name='bcnn8')
			conv8 = tf.add(tu.conv2d(conv7, wcnn8, stride=(1,1), padding='SAME'), bcnn8)
			conv8 = tu.relu(conv8)


		with tf.name_scope('vgg_cnn_pool3') as inner_scope:
			pool3 = tu.max_pool2d(conv8, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')

		with tf.name_scope('vgg_cnn_conv9') as inner_scope:
			wcnn9 = tu.weight([3, 3, 256, 512], name='wcnn9')
			bcnn9 = tu.bias(0.0, [512], name='bcnn9')
			conv9 = tf.add(tu.conv2d(pool3, wcnn9, stride=(1,1), padding='SAME'), bcnn9)
			conv9 = tu.relu(conv9)

		with tf.name_scope('vgg_cnn_conv10') as inner_scope:
			wcnn10 = tu.weight([3, 3, 512, 512], name='wcnn10')
			bcnn10 = tu.bias(0.0, [512], name='bcnn10')
			conv10 = tf.add(tu.conv2d(conv9, wcnn10, stride=(1,1), padding='SAME'), bcnn10)
			conv10 = tu.relu(conv10)

		with tf.name_scope('vgg_cnn_conv11') as inner_scope:
			wcnn11 = tu.weight([3, 3, 512, 512], name='wcnn11')
			bcnn11 = tu.bias(0.0, [512], name='bcnn11')
			conv11 = tf.add(tu.conv2d(conv10, wcnn11, stride=(1,1), padding='SAME'), bcnn11)
			conv11 = tu.relu(conv11)

		with tf.name_scope('vgg_cnn_conv12') as inner_scope:
			wcnn12 = tu.weight([3, 3, 512, 512], name='wcnn12')
			bcnn12 = tu.bias(0.0, [512], name='bcnn12')
			conv12 = tf.add(tu.conv2d(conv11, wcnn12, stride=(1,1), padding='SAME'), bcnn12)
			conv12 = tu.relu(conv12)

		with tf.name_scope('vgg_cnn_pool4') as inner_scope:
			pool4 = tu.max_pool2d(conv12, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')

		with tf.name_scope('vgg_cnn_conv13') as inner_scope:
			wcnn13 = tu.weight([3, 3, 512, 512], name='wcnn13')
			bcnn13 = tu.bias(0.0, [512], name='bcnn13')
			conv13 = tf.add(tu.conv2d(pool4, wcnn13, stride=(1,1), padding='SAME'), bcnn13)
			conv13 = tu.relu(conv13)

		with tf.name_scope('vgg_cnn_conv14') as inner_scope:
			wcnn14 = tu.weight([3, 3, 512, 512], name='wcnn14')
			bcnn14 = tu.bias(0.0, [512], name='bcnn14')
			conv14 = tf.add(tu.conv2d(conv13, wcnn14, stride=(1,1), padding='SAME'), bcnn14)
			conv14 = tu.relu(conv14)

		with tf.name_scope('vgg_cnn_conv15') as inner_scope:
			wcnn15 = tu.weight([3, 3, 512, 512], name='wcnn15')
			bcnn15 = tu.bias(0.0, [512], name='bcnn15')
			conv15 = tf.add(tu.conv2d(conv14, wcnn15, stride=(1,1), padding='SAME'), bcnn15)
			conv15 = tu.relu(conv15)

		with tf.name_scope('vgg_cnn_conv16') as inner_scope:
			wcnn16 = tu.weight([3, 3, 512, 512], name='wcnn16')
			bcnn16 = tu.bias(0.0, [512], name='bcnn16')
			conv16 = tf.add(tu.conv2d(conv15, wcnn16, stride=(1,1), padding='SAME'), bcnn16)
			conv16 = tu.relu(conv16)

		with tf.name_scope('vgg_cnn_pool5') as inner_scope:
			pool5 = tu.max_pool2d(conv16, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')

		return pool5


def classifier(x, keep_prob):
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
	flat_dim = dim[1] * dim[2] * dim[3] # 7 x 7 x 512
	flat = tf.reshape(pool5, [-1, flat_dim])

	with tf.name_scope('vgg_classifier') as scope:
		with tf.name_scope('vgg_classifier_fc1') as inner_scope:
			wfc1 = tu.weight([flat_dim, 4096], name='wfc1')
			bfc1 = tu.bias(0.0, [4096], name='bfc1')
			fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)
			#fc1 = tu.batch_norm(fc1)
			fc1 = tu.relu(fc1)
			fc1 = tf.nn.dropout(fc1, keep_prob)

		with tf.name_scope('alexnet_classifier_fc2') as inner_scope:
			wfc2 = tu.weight([4096, 4096], name='wfc2')
			bfc2 = tu.bias(0.0, [4096], name='bfc2')
			fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
			#fc2 = tu.batch_norm(fc2)
			fc2 = tu.relu(fc2)
			fc2 = tf.nn.dropout(fc2, keep_prob)

		with tf.name_scope('alexnet_classifier_output') as inner_scope:
			wfc3 = tu.weight([4096, 1000], name='wfc3')
			bfc3 = tu.bias(0.0, [1000], name='bfc3')
			fc3 = tf.add(tf.matmul(fc2, wfc3), bfc3)
			softmax = tf.nn.softmax(fc3)

	return fc3, softmax

