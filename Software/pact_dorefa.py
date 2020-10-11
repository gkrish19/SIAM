import tensorflow as tf
import numpy as np


def quantize(x, n):
	@tf.custom_gradient
	def _quantize(x):
		return tf.round(x * n) / n, lambda dy: dy
	return _quantize(x)

def fw(x,bitW=8):
	if bitW == 32:
		return x
	if bitW == 1:   # BWN
		E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
		@tf.custom_gradient
		def _sign(x):
			return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy
		return _sign(x)
	x = tf.tanh(x)
	x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
	n = float(2 ** bitW - 1)
	return 2 * quantize(x,n) - 1

def activate(input, num_bits):
	with tf.variable_scope('alpha'):
		alpha = tf.Variable(8.0,dtype=tf.float32,trainable=True,name='alpha')
	num_bits = num_bits
	input = tf.nn.relu(input)
	input = tf.clip_by_value(input, 0, alpha)
	scale = (2**num_bits-1)/alpha
	input = quantize(input,scale)
	return input

def activate_ADC (input, num_bits):
	alpha = tf.Variable(8.0,dtype=tf.float32,trainable=True,name='rram_alpha')
	input = tf.clip_by_value(input, -alpha, alpha)
	scale = (2**(num_bits-1)-1)/(2*alpha)
	input = quantize(input,scale)
	return input































# def quantize(x, k):
#         n = 2 ** k - 1
#         print("n :",n)
#         n= tf.cast(n,tf.float32)
#         print("Float n:",sess.run(n))
#         @tf.custom_gradient
#         def _quantize(x):
#             print("y:",sess.run(tf.round(x * n) / n, lambda dy: dy))
#             return tf.round(x * n) / n, lambda dy: dy
#         return _quantize(x)

# def _quantize(x):
#     n=3.0
#     print(type(x))
#     return tf.round(x[0] * n) / n, lambda dy: dy
# # Initialize the variables
# init = tf.global_variables_initializer()

# sess=tf.Session()
# sess.run(init)
# np.random.seed(0)
# input = {0:np.random.randn(64, 10, 30)}
# q=_quantize(input)
# #print("quantized value:",sess.run(q))
# sess.close()
# def fw(x):
#     if bitW == 32:
#         return x

#     if bitW == 1:   # BWN
#         E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))

#         @tf.custom_gradient
#         def _sign(x):
#             return tf.where(tf.equal(x, 0), tf.ones_like(x), tf.sign(x / E)) * E, lambda dy: dy

#         return _sign(x)

#     x = tf.tanh(x)
#     x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
#     return 2 * quantize(x, bitW) - 1
