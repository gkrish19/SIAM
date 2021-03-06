#VGG19_testing  and NeuroSim code
#Author: Gokul Krishnan
import tensorflow as tf
from data_utility import *
from hardware_estimation import hardware_estimation
from pathlib import Path


iterations      = 200
batch_size      = 250
total_epoch     = 164
weight_decay    = 0.0003
dropout_rate    = 0.5
momentum_rate   = 0.9
log_save_path   = './vgg_logs'
model_save_path = './model_vgg16_cifar10/'



def _random_crop(batch, crop_shape, padding=None):
		oshape = np.shape(batch[0])

		if padding:
			oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
		new_batch = []
		npad = ((padding, padding), (padding, padding), (0, 0))
		for i in range(len(batch)):
			new_batch.append(batch[i])
			if padding:
				new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
										  mode='constant', constant_values=0)
			nh = random.randint(0, oshape[0] - crop_shape[0])
			nw = random.randint(0, oshape[1] - crop_shape[1])
			new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
										nw:nw + crop_shape[1]]
		return new_batch

def _random_flip_leftright(batch):
		for i in range(len(batch)):
			if bool(random.getrandbits(1)):
				batch[i] = np.fliplr(batch[i])
		return batch

def data_preprocessing(x_train,x_test):

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
	x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
	x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

	x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
	x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
	x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])

	return x_train, x_test

def learning_rate_schedule(epoch_num):
	  if epoch_num < 81:
		  return 0.1
	  elif epoch_num < 121:
		  return 0.01
	  else:
		  return 0.001

def data_augmentation(batch):
	batch = _random_flip_leftright(batch)
	batch = _random_crop(batch, [32,32], 4)
	return batch


def _count_trainable_params():
	total_parameters = 0
	for variable in tf.trainable_variables():
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	print("Total training params: %.1fM" % (total_parameters / 1e6))


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32 )
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(input, k_size=1, stride=1, name=None):
	return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME',name=name)

def batch_norm(input):
	return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=train_flag, updates_collections=None)


def run_testing(sess,ep):
	acc = 0.0
	loss = 0.0
	pre_index = 0
	add = 1000
	for it in range(10):
		batch_x = test_x[pre_index:pre_index+add]
		batch_y = test_y[pre_index:pre_index+add]
		pre_index = pre_index + add
		loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False})
		loss += loss_ / 10.0
		acc += acc_ / 10.0
	summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
							tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
	return acc, loss, summary



if __name__ == '__main__':

	train_x, train_y, test_x, test_y = prepare_data()
	train_x, test_x = data_preprocessing(train_x, test_x)

	# define placeholder x, y_ , keep_prob, learning_rate
	x  = tf.placeholder(tf.float32,[None, image_size, image_size, 3])
	y_ = tf.placeholder(tf.float32, [None, class_num])
	keep_prob = tf.placeholder(tf.float32)
	learning_rate = tf.placeholder(tf.float32)
	train_flag = tf.placeholder(tf.bool)

	_input_array = []
	_weights = []

	# build_network

	_input_array.append(tf.transpose(x, (0,3,1,2)))
	W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 6], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv1_1)
	b_conv1_1 = bias_variable([6])
	output  = tf.nn.relu( batch_norm(conv2d(x,W_conv1_1) + b_conv1_1))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 6, 6], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv1_2)
	b_conv1_2 = bias_variable([6])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv1_2) + b_conv1_2))
	output  = max_pool(output, 2, 2, "pool1")

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 6, 12], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv2_1)
	b_conv2_1 = bias_variable([12])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv2_1) + b_conv2_1))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 12, 12], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv2_2)
	b_conv2_2 = bias_variable([12])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv2_2) + b_conv2_2))
	output  = max_pool(output, 2, 2, "pool2")

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 12, 25], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv3_1)
	b_conv3_1 = bias_variable([25])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv3_1) + b_conv3_1))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 25, 25], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv3_2)
	b_conv3_2 = bias_variable([25])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv3_2) + b_conv3_2))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 25, 25], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv3_3)
	b_conv3_3 = bias_variable([25])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv3_3) + b_conv3_3))

	#_input_array.append(tf.transpose(output, (0,3,1,2)))
	#W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
	#_weights.append(W_conv3_4)
	#b_conv3_4 = bias_variable([256])
	#output  = tf.nn.relu( batch_norm(conv2d(output,W_conv3_4) + b_conv3_4))
	output  = max_pool(output, 2, 2, "pool3")

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 25, 50], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv4_1)
	b_conv4_1 = bias_variable([50])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv4_1) + b_conv4_1))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 50, 50], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv4_2)
	b_conv4_2 = bias_variable([50])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv4_2) + b_conv4_2))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 50, 50], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv4_3)
	b_conv4_3 = bias_variable([50])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv4_3) + b_conv4_3))

	#_input_array.append(tf.transpose(output, (0,3,1,2)))
	#W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
	#_weights.append(W_conv4_4)
	#b_conv4_4 = bias_variable([512])
	#output  = tf.nn.relu( batch_norm(conv2d(output,W_conv4_4)) + b_conv4_4)
	output  = max_pool(output, 2, 2)

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 50, 50], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv5_1)
	b_conv5_1 = bias_variable([50])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv5_1) + b_conv5_1))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 50, 50], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv5_2)
	b_conv5_2 = bias_variable([50])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv5_2) + b_conv5_2))

	_input_array.append(tf.transpose(output, (0,3,1,2)))
	W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 50, 50], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_conv5_3)
	b_conv5_3 = bias_variable([50])
	output  = tf.nn.relu( batch_norm(conv2d(output,W_conv5_3) + b_conv5_3))

	#_input_array.append(tf.transpose(output, (0,3,1,2)))
	#W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
	#_weights.append(W_conv5_4)
	#b_conv5_4 = bias_variable([512])
	#output  = tf.nn.relu( batch_norm(conv2d(output,W_conv5_4) + b_conv5_4))

	# output = tf.contrib.layers.flatten(output)
	output = tf.reshape(output,[-1,2*2*50])

	_input_array.append(output)
	W_fc1 = tf.get_variable('fc1', shape=[200,400], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_fc1)
	b_fc1 = bias_variable([400])
	output = tf.nn.relu( batch_norm(tf.matmul(output,W_fc1) + b_fc1) )
	output  = tf.nn.dropout(output,keep_prob)

	_input_array.append(output)
	W_fc2 = tf.get_variable('fc7', shape=[400,400], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_fc2)
	b_fc2 = bias_variable([400])
	output = tf.nn.relu( batch_norm(tf.matmul(output,W_fc2) + b_fc2) )
	output  = tf.nn.dropout(output,keep_prob)

	_input_array.append(output)
	W_fc3 = tf.get_variable('fc3', shape=[400,10], initializer=tf.contrib.keras.initializers.he_normal())
	_weights.append(W_fc3)
	b_fc3 = bias_variable([10])
	output = tf.nn.relu( batch_norm(tf.matmul(output,W_fc3) + b_fc3) )
	# output  = tf.reshape(output,[-1,10])


	# loss function: cross_entropy
	# train_step: training operation
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
	l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate,use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	# initial an saver to save model
	saver = tf.train.Saver()


	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		_count_trainable_params()
		summary_writer = tf.summary.FileWriter(log_save_path,sess.graph)

		# Restore variables from disk.
		saver.restore(sess, model_save_path)
		print("Model restored.")

		acc = 0.0
		loss = 0.0
		pre_index = 0
		add = 1000
		for it in range(10):
			batch_x = test_x[pre_index:pre_index+add]
			batch_y = test_y[pre_index:pre_index+add]
			pre_index = pre_index + add
			loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False})
			loss += loss_ / 10.0
			acc += acc_ / 10.0
		summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
								tf.Summary.Value(tag="test_accuracy", simple_value=acc)])




		#summary_writer.add_summary(test_summary, ep)
		#summary_writer.flush()

		print("Test_loss: %.4f, Test_acc: %.4f" %(loss, acc))



		print("Invoking SIAM now")

		pre_index = 0
		add = 1
		for it in range(1):
			batch_x = test_x[pre_index:pre_index+add]
			batch_y = test_y[pre_index:pre_index+add]
			pre_index = pre_index + add
			feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False}

		H, W = sess.run([_input_array, _weights], feed_dict=feed_dict)
		my_file = Path("./SIAM/to_interconnect/ip_activation.csv")
		if my_file.is_file():
			os.remove("./SIAM/to_interconnect/ip_activation.csv")
		my_file_1 = Path("./SIAM/to_interconnect/num_tiles_per_layer.csv")
		if my_file_1.is_file():
			os.remove("./SIAM/to_interconnect/num_tiles_per_layer.csv")
		my_file_2 = Path("./SIAM/to_interconnect/fps.csv")
		if my_file_2.is_file():
			os.remove("./SIAM/to_interconnect/fps.csv")
		my_file_3 = Path("./SIAM/Final_Results/area.csv")
		if my_file_3.is_file():
			os.remove("./SIAM/Final_Results/area.csv")
		my_file_4 = Path("./SIAM/Final_Results/Latency.csv")
		if my_file_4.is_file():
			os.remove("./SIAM/Final_Results/Latency.csv")
		my_file_5 = Path("./SIAM/Final_Results/Energy.csv")
		if my_file_5.is_file():
			os.remove("./SIAM/Final_Results/Energy.csv")

		hardware_estimation(H,W,32,32)
