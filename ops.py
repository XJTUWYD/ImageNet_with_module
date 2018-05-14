import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x,name = "selu"):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
    


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(input, output_shape, is_train, activation_fn,
           k_h=5, k_w=5, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))
        activation = activation_fn(conv + biases)
        bn = tf.contrib.layers.batch_norm(activation, center=True, scale=True,
                                          decay=0.9, is_training=is_train,
                                          updates_collections=None)
    return bn


def fc(input, output_shape, is_train, activation_fn, name="fc"):
    output = slim.fully_connected(input, output_shape, activation_fn=activation_fn)
    return output

def tanh_apx(input_d):
	output = tf.zeros_like(input_d,tf.float32)  # 生成与输入大小相同、元素全为0的输出tensor
	# 将大于tanh(1)的值都置为1
	temp_1 = tf.nn.relu(3.875) 
	output_1 = tf.where(tf.less(input_d,temp_1),tf.zeros_like(input_d,tf.float32),tf.ones_like(input_d,tf.float32))
	alpha = 10*output_1
	input_d = tf.add(input_d,alpha)
	output = tf.add(output,output_1)
	# 将小于tanh(-1)的值都置为了-1
	temp_1 = tf.nn.relu(-3.875.00) 
	output_1 = tf.where(tf.less(input_d,temp_1),tf.ones_like(input_d,tf.float32),tf.zeros_like(input_d,tf.float32))
	alpha = 10*output_1
	input_d = tf.add(input_d,alpha)
	output = tf.add(output,-1*output_1)
	# 将介于tanh(-1)到tanh(1)之间的值分段处理
	for t in range(-16,17,1):
		i = t/8
		# 与第i+1个值对比时，比第i+1个值小的部分都置为第i个值的值的大小
		temp_1 = tf.nn.relu(i+0.125) 
		temp_2 = tf.nn.relu(i)
		# 获取标记矩阵，即获得小于第i个值的部分的位置
		output_1 = tf.where(tf.less(input_d,temp_1),tf.ones_like(input_d,tf.float32),tf.zeros_like(input_d,tf.float32))
		# alpha系数用于将输入中已经过变换的部分置高位，防止被后面再次变换
		alpha = 10*output_1
		input_d = tf.add(input_d,alpha)
		# 将标记矩阵x对应的第i个的值添加到输出tensor中
		output = tf.add(temp_2*output_1,output)
	return output

def selu_apx(input_d):
	def selu(x,name = "selu"):
		scale = 1.0507009873554804934193349852946
		alpha = 1.6732632423543772848170429916717
		return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)

	output = tf.zeros_like(input_d,tf.float32)  # 生成与输入大小相同、元素全为0的输出tensor
	# 将大于selu(0)的值都保持不变
	temp_1 = selu(0.00) 
	output_1 = tf.where(tf.less(input_d,temp_1),tf.zeros_like(input_d,tf.float32),input_d)
	output_temp = tf.where(tf.less(input_d,temp_1),tf.zeros_like(input_d,tf.float32),tf.ones_like(input_d,tf.float32))
	alpha = 10*output_temp
	input_d = tf.add(input_d,alpha)
	output = tf.add(output,output_1)
	# 将小于selu(-3.875)的值都置为了-1.6733
	temp_1 = selu(-3.875) 
	output_1 = tf.where(tf.less(input_d,temp_1),tf.ones_like(input_d,tf.float32),tf.zeros_like(input_d,tf.float32))
	alpha = 10*output_1
	input_d = tf.add(input_d,alpha)
	output = tf.add(output,-1.6733*output_1)
	# 将介于tanh(-1)到tanh(1)之间的值分段处理
	for t in range(-31,0,1):
		i = t/8
		# 与第i+1个值对比时，比第i+1个值小的部分都置为第i个值的值的大小
		temp_1 = selu(i+0.125) 
		temp_2 = selu(i)
		# 获取标记矩阵，即获得小于第i个值的部分的位置
		output_1 = tf.where(tf.less(input_d,temp_1),tf.ones_like(input_d,tf.float32),tf.zeros_like(input_d,tf.float32))
		# alpha系数用于将输入中已经过变换的部分置高位，防止被后面再次变换
		alpha = 10*output_1
		input_d = tf.add(input_d,alpha)
		# 将标记矩阵x对应的第i个的值添加到输出tensor中
		output = tf.add(temp_2*output_1,output)
	return output


def output_simulation(input_d):
	output = tf.zeros_like(input_d,tf.float32)  # 生成与输入大小相同、元素全为0的输出tensor
	# 将大于tanh(1)的值都置为1
	temp_1 = tf.nn.relu(3.875) 
	output_1 = tf.where(tf.less(input_d,temp_1),tf.zeros_like(input_d,tf.float32),tf.ones_like(input_d,tf.float32))
	alpha = 10*output_1
	input_d = tf.add(input_d,alpha)
	output = tf.add(output,output_1)
	# 将小于tanh(-1)的值都置为了-1
	temp_1 = tf.nn.relu(-3.875.00) 
	output_1 = tf.where(tf.less(input_d,temp_1),tf.ones_like(input_d,tf.float32),tf.zeros_like(input_d,tf.float32))
	alpha = 10*output_1
	input_d = tf.add(input_d,alpha)
	output = tf.add(output,-1*output_1)
	# 将介于tanh(-1)到tanh(1)之间的值分段处理
	for t in range(-16,17,1):
		i = t/8
		# 与第i+1个值对比时，比第i+1个值小的部分都置为第i个值的值的大小
		temp_1 = tf.nn.relu(i+0.125) 
		temp_2 = tf.nn.relu(i)
		# 获取标记矩阵，即获得小于第i个值的部分的位置
		output_1 = tf.where(tf.less(input_d,temp_1),tf.ones_like(input_d,tf.float32),tf.zeros_like(input_d,tf.float32))
		# alpha系数用于将输入中已经过变换的部分置高位，防止被后面再次变换
		alpha = 10*output_1
		input_d = tf.add(input_d,alpha)
		# 将标记矩阵x对应的第i个的值添加到输出tensor中
		output = tf.add(temp_2*output_1,output)
	return output