from config import *
import tensorflow as tf
import numpy as np
from pact_dorefa import *
import math

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32 )
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(input, k_size=1, stride=1, name=None):
	return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME',name=name)

def batch_norm(input, is_training):
	return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=is_training, updates_collections=None)

def addDeviceVariation_SA(sess,stddevVar=0.5):
	"""
	This function adds variation	"""
	rramTensors = [v for v in tf.trainable_variables()]
	if stddevVar != 0.0:
		allParameters = [sess.run(v) for v in rramTensors]
		allShapes = [v.get_shape().as_list() for v in rramTensors]
		for i in range(len(allParameters)):
			if ((len(rramTensors[i].get_shape().as_list()) == 4) or (len(rramTensors[i].get_shape().as_list()) == 2)):
				# param = allParameters[i] * np.exp(np.random.normal(0, stddevVar, allShapes[i]))
				param = allParameters[i] * np.exp(np.random.normal(allParameters[i], stddevVar, allShapes[i]))
				signMat = np.ones(param.shape, dtype=np.float32)
				signMat[np.where(param < 0.0)] = -1.0
				param = np.absolute(param)
				param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
				param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
				param = param*signMat
				# rramTensors[i].load(param)
				assign_wgts = tf.assign(rramTensors[i], param)
				if sess.run(tf.reduce_all(tf.equal(rramTensors[i], assign_wgts))):
					pass
				else:
					print("Warning Unequal Tensors after Applying Variation")
			else:
				pass
				# print('Not adding write variation for ', rramTensors[i].name)
	addSA0(sess,rramTensors, cfg.RRAM.SA0)
	addSA1(sess,rramTensors, cfg.RRAM.SA1)

def addSA1(sess,rramTensors, percentSA1):
	"""
	This function adds the SAF low defects into the crossbar.
	"""
	allParameters = [sess.run(v) for v in rramTensors]
	shapes = [v.get_shape().as_list() for v in rramTensors]
	minValues = []
	for i in range(len(allParameters)):
		minValues.append(np.amin(allParameters[i]))
	lowVal = cfg.RRAM.SA1_VAL
	for i in range(len(allParameters)):
		if ((len(rramTensors[i].get_shape().as_list()) == 4) or (len(rramTensors[i].get_shape().as_list()) == 2)):
		# if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name) or ('alp' not in rramTensors[i].name):
			param = allParameters[i]
			dims = len(shapes)
			if dims == 1:
				num = int(math.ceil(shapes[i][0]*percentSA1/100.0))
				x = np.arange(shapes[i][0])
				random.shuffle(x)
				for j in range(num):
					param[x[j]] = lowVal if param[x[j]] > 0 else -1.0*lowVal
			elif dims == 2:
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*percentSA1/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				random.shuffle(x)
				random.shuffle(y)
				for j in range(num):
					param[x[j], y[j]] = lowVal if param[x[j], y[j]] > 0 else -1.0*lowVal
			elif dims == 4:
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]*percentSA1/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				z = np.arange(shapes[i][2])
				k = np.arange(shapes[i][3])
				random.shuffle(x)
				random.shuffle(y)
				random.shuffle(z)
				random.shuffle(k)
				for j in range(num):
					param[x[j], y[j], z[j], k[j]] = lowVal if param[x[j], y[j], z[j], k[j]] > 0 else -1.0*lowVal

			assign_wgts = tf.assign(rramTensors[i], param)
			if sess.run(tf.reduce_all(tf.equal(rramTensors[i], assign_wgts))):
				pass
			else:
				print("Warning Unequal Tensors after Applying Variation")
		else:
			pass
			# print('not adding SA1 for ', rramTensors[i].name)

def addSA0(sess, rramTensors, percentSA0):
	"""
	This function adds the SAF high defects into the crossbar.	"""

	allParameters = [sess.run(v) for v in rramTensors]
	print("Entered addSA0")
	shapes = [v.get_shape().as_list() for v in rramTensors]
	maxValues = []
	for i in range(len(allParameters)):
		maxValues.append(np.amax(allParameters[i]))
	highVal = cfg.RRAM.SA0_VAL
	for i in range(len(allParameters)):
		if ((len(rramTensors[i].get_shape().as_list()) == 4) or (len(rramTensors[i].get_shape().as_list()) == 2)):
		# if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name) or ('alp' not in rramTensors[i].name):
			param = allParameters[i]
			# print("Entered the loop", param)
			dims = len(shapes)
			if dims == 1:
				num = int(math.ceil(shapes[i][0]*percentSA0/100.0))
				x = np.arange(shapes[i][0])
				random.shuffle(x)
				for j in range(num):
					param[x[j]] = highVal
			elif dims == 2:
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*percentSA0/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				random.shuffle(x)
				random.shuffle(y)
				for j in range(num):
					param[x[j], y[j]] = highVal
			elif dims == 4:
				print("Entereed dims=4")
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]*percentSA0/100.0))
				print(num)
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				z = np.arange(shapes[i][2])
				k = np.arange(shapes[i][3])
				random.shuffle(x)
				random.shuffle(y)
				random.shuffle(z)
				random.shuffle(k)
				print(x)
				print(y)
				print(z)
				print(k)
				for j in range(num):
					print(j)
					param[x[j], y[j], z[j], k[j]] = highVal

			assign_wgts = tf.assign(rramTensors[i], param)
			if sess.run(tf.reduce_all(tf.equal(rramTensors[i], assign_wgts))):
				pass
			else:
				print("Warning Unequal Tensors after Applying Variation")
		else:
			pass
			# print('not adding SA0 for ', rramTensors[i].name)

def quantize(sess, rramTensors, levels=32):

	allParameters = [sess.run(v) for v in rramTensors]
	allShapes = [v.get_shape().as_list() for v in rramTensors]
	for i in range(len(allParameters)):
		param = allParameters[i]
		signMat = np.ones(param.shape, dtype=np.float32)
		signMat[np.where(param < 0.0)] = -1.0
		param = np.absolute(param)
		param[np.where(param < cfg.RRAM.SA1_VAL)] = 0.0
		# param = (cfg.RRAM.SA0_VAL-cfg.RRAM.SA1_VAL)*np.ceil(param*levels)/levels + cfg.RRAM.SA1_VAL
		param = fw(param, levels)
		# param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
		# param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
		param = param*signMat
		assign_wgts = tf.assign(rramTensors[i], param)
		if sess.run(tf.reduce_all(tf.equal(rramTensors[i], assign_wgts))):
			pass
		else:
			print("Warning Unequal Tensors after Applying Variation")


def RRAM_fc2d (x, W, b, xbar_size, adc_bits):
	input_shape = x.get_shape().as_list()
	weights_shape = W.get_shape().as_list()
	n_rows = (math.ceil((weights_shape[0])/xbar_size))
	nW_per_xbar = xbar_size
	print("nW_per_xbar, n_rows is",nW_per_xbar,n_rows )
	# res_actual = tf.matmul(x,W) + b
	# res = tf.zeros(shape = res_actual.get_shape().as_list())
	#Based on the xbar_cell bits we add different variation numbers in the var function
	count = 0
	while ((weights_shape[0]-count) >= nW_per_xbar):
		if (count == 0):
			start = 0
		else:
			start = count
		count += nW_per_xbar
		end  = count-1
		print("Start and end are:", start, end)
		W_temp = W[start:end:1,:]
		print("Shape of W_temp is:", W_temp.shape)
		x_temp = x[:,start:end:1]
		print("Shape of x_temp is:", x_temp.shape)
		res_temp = tf.matmul(x_temp,W_temp) + b
		res_temp = activate_ADC(res_temp, adc_bits)
		#Adc Quantization error comes into play
		if (start == 0):
			res = res_temp
		else:
			res = tf.add(res_temp, res)

	if (((weights_shape[0]-count) < nW_per_xbar) & (count != weights_shape[0])):
		start = count
		end = weights_shape[0]-1
		print("Start and end are:", start, end)
		W_temp = W[start:end:1,:]
		print("Shape of W_temp is:", W_temp.shape)
		x_temp = x[:,start:end:1]
		print("Shape of x_temp is:", x_temp.shape)
		res_temp = tf.matmul(x_temp,W_temp) + b
		res_temp = activate_ADC(res_temp, adc_bits)
		#Adc Quantization error comes into play
		if (start == 0):
			res = res_temp
		else:
			res = tf.add(res_temp, res)
	return (res)

def RRAM_conv2d(x, W, xbar_size, adc_bits, strides, padding='SAME'):
	input_shape = x.get_shape().as_list()
	weights_shape = W.get_shape().as_list()
	if xbar_size<=64:
		nif_per_xbar = math.ceil(xbar_size/(weights_shape[0]*weights_shape[1]))
	else:
		nif_per_xbar = math.floor(xbar_size / (weights_shape[0] * weights_shape[1]))
	n_rows = math.ceil(xbar_size/nif_per_xbar)
	print("nif_per_xbar, n_rows is",nif_per_xbar,n_rows )
	# res_actual = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	# res = tf.zeros(shape = tf.shape(res_actual).np())
	#Based on the xbar_cell bits we add different variation numbers in the var function
	count = 0
	while ((weights_shape[2]-count) >= nif_per_xbar):
		if (count == 0):
			start = 0
		else:
			start = count
		count += nif_per_xbar
		end  = count-1
		print("Start and end are:", start, end)
		W_temp = W[:,:,start:end:1,:]
		print("Shape of W_temp is:", W_temp.shape)
		x_temp = x[:,:,:,start:end:1]
		print("Shape of x_temp is:", x_temp.shape)
		res_temp = tf.nn.conv2d(x_temp, W_temp, strides=strides, padding=padding)
		res_temp = activate_ADC(res_temp, adc_bits)
		#Adc Quantization error comes into play
		if (start == 0):
			res = res_temp
		else:
			res = tf.add(res_temp, res)

	if (((weights_shape[2]-count) < nif_per_xbar) & (count != weights_shape[2])):
		start = count
		end = weights_shape[2]-1
		print("Start and end are:", start, end)
		W_temp = W[:,:,start:end:1,:]
		print("Shape of W_temp is:", W_temp.shape)
		x_temp = x[:,:,:,start:end:1]
		print("Shape of x_temp is:", x_temp.shape)
		res_temp = tf.nn.conv2d(x_temp, W_temp, strides=strides, padding=padding)
		res_temp = activate_ADC(res_temp, adc_bits)
		#Adc Quantization error comes into play
		if (start == 0):
			res = res_temp
		else:
			res = tf.add(res_temp, res)
	return (res)
