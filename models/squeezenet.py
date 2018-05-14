import tensorflow as tf
import train_util as tu

def _cnn(x):
	conv1 = tu.conv_layer(
	    'conv1', x, filters=96, size=7, stride=2,padding='SAME', freeze=False)
	pool1 = tu.pooling_layer(
	    'pool1', conv1 , size=3, stride=2, padding='SAME')
	# pool1_1 = tu.pooling_layer(
	#     'pool1_1', pool1, size=2, stride=2, padding='SAME')
	fire2 = tu.fire_layer(
	    'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
	fire3 = tu.fire_layer(
	    'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
	fire4 = tu.fire_layer(
	    'fire4', fire3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
	pool4 = tu.pooling_layer(
	    'pool4', fire4, size=3, stride=2, padding='SAME')
	fire5 = tu.fire_layer(
	    'fire5', pool4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
	fire6 = tu.fire_layer(
	    'fire6', fire5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
	fire7 = tu.fire_layer(
	    'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
	fire8 = tu.fire_layer(
	    'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
	pool8 = tu.pooling_layer(
	    'pool8', fire8, size=3, stride=2, padding='SAME')
	fire9 = tu.fire_layer(
	    'fire9', pool8, s1x1=64, e1x1=256, e3x3=256, freeze=False)
	return fire9

def classifier(x, dropout):
	fire = _cnn(x)
	# fire = tf.nn.dropout(fire, dropout)
	conv10 = tu.conv_layer(
	    'conv10', fire, filters=1000, size=1, stride=1,padding='SAME', freeze=False)
	global_avg_pooling = tu.avg_layer('global_avg_pool', conv10, 13, 1, padding = 'SAME')
	logits = tf.squeeze(global_avg_pooling)
	return logits


	# pool5 = cnn(x)

	# dim = pool5.get_shape().as_list()
	# flat_dim = dim[1] * dim[2] * dim[3] # 6 * 6 * 256
	# flat = tf.reshape(pool5, [-1, flat_dim])

	# with tf.name_scope('alexnet_classifier') as scope:
	# 	with tf.name_scope('alexnet_classifier_fc1') as inner_scope:
	# 		wfc1 = tu.weight([flat_dim, 4096], name='wfc1')
	# 		bfc1 = tu.bias(0.0, [4096], name='bfc1')
	# 		fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)
	# 		fc1 = tu.batch_norm(fc1)
	# 		fc1 = tu.relu(fc1)
	# 		fc1 = tf.nn.dropout(fc1, dropout)

	# 	with tf.name_scope('alexnet_classifier_fc2') as inner_scope:
	# 		wfc2 = tu.weight([4096, 4096], name='wfc2')
	# 		bfc2 = tu.bias(0.0, [4096], name='bfc2')
	# 		fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
	# 		fc2 = tu.batch_norm(fc2)
	# 		fc2 = tu.relu(fc2)
	# 		fc2 = tf.nn.dropout(fc2, dropout)

	# 	with tf.name_scope('alexnet_classifier_output') as inner_scope:
	# 		wfc3 = tu.weight([4096, 1000], name='wfc3')
	# 		bfc3 = tu.bias(0.0, [1000], name='bfc3')
	# 		fc3 = tf.add(tf.matmul(fc2, wfc3), bfc3)
	# 		softmax = tf.nn.softmax(fc3)

	# return fc3, softmax


