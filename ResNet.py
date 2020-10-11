import numpy as np
import tensorflow as tf
from utils import RRAM_fc2d, RRAM_conv2d
from pact_dorefa import *

BN_EPSILON = 0.001

act = []

def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(shape, quant, name, initializer, is_fc_layer=False):
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
    if quant:
        new_variables = fw(new_variables, bitW=8)
    else:
        pass
    return new_variables


def create_variables_b(shape, name, initializer, is_fc_layer=False):
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
    return new_variables


def output_layer(input_layer, num_labels, rram, xbar_size, adc_bits, quant):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.contrib.keras.initializers.he_normal(), quant=quant, name='fc_weights')
    fc_b = create_variables_b(shape=[num_labels], initializer=tf.zeros_initializer(), name='fc_bias')
    if rram:
        fc_h = RRAM_fc2d(x=input_layer, W=fc_w, b=fc_b, xbar_size=xbar_size, adc_bits=adc_bits)
    else:
        fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer


def conv_bn_relu_layer(input_layer, quant, filter_shape, ab, rram, xbar_size, adc_bits, stride):
    out_channel = filter_shape[-1]
    filter = create_variables(shape=filter_shape, quant=quant, name='conv',
                              initializer=tf.contrib.keras.initializers.he_normal())
    if rram:
        conv_layer = RRAM_conv2d(x=input_layer, W=filter, xbar_size=xbar_size, adc_bits=adc_bits,
                                 strides=[1, stride, stride, 1], padding='SAME')
        act.append(conv_layer)
    else:
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        act.append(conv_layer)

    bn_layer = batch_normalization_layer(conv_layer, out_channel)
    output = tf.nn.relu(bn_layer)

    if quant:
        output = activate(output, ab)
    else:
        pass
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, ab, rram, xbar_size, adc_bits, quant):
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(shape=filter_shape, quant=quant, name='conv',
                              initializer=tf.contrib.keras.initializers.he_normal())
    if rram:
        conv_layer = RRAM_conv2d(x=relu_layer, W=filter, xbar_size=xbar_size, adc_bits=adc_bits,
                                 strides=[1, stride, stride, 1], padding='SAME')
        act.append(conv_layer)
    else:
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        act.append(conv_layer)

    if quant:
        conv_layer = activate(conv_layer, ab)
    else:
        pass
    return conv_layer


def residual_block(input_layer, output_channel, quant, ab, rram, xbar_size, adc_bits, first_block=False):
    input_channel = input_layer.get_shape().as_list()[-1]

    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel], quant=quant,
                                      initializer=tf.contrib.keras.initializers.he_normal())

            if rram:
                conv1 = RRAM_conv2d(x=input_layer, W=filter, xbar_size=xbar_size, adc_bits=adc_bits,
                                    strides=[1, 1, 1, 1], padding='SAME')
                act.append(conv1)
            else:
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
                act.append(conv1)

            if quant:
                conv1 = activate(conv1, ab)
            else:
                pass
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, ab=ab, rram=rram,
                                       xbar_size=xbar_size, adc_bits=adc_bits, quant=quant)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, ab=ab, rram=rram,
                                   xbar_size=xbar_size, adc_bits=adc_bits, quant=quant)

    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer
    output = conv2 + padded_input
    return output


def build_ResNet(input_tensor_batch, label_dim, n, wb, ab, quant, rram, xbar_size, adc_bits, reuse):
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, quant, [3, 3, 3, 16], ab, rram, xbar_size, adc_bits, 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' % i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, quant=quant, ab=ab, rram=rram, xbar_size=xbar_size,
                                       adc_bits=adc_bits, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16, rram=rram, ab=ab, xbar_size=xbar_size, adc_bits=adc_bits,
                                       quant=quant)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' % i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32, rram=rram, ab=ab, xbar_size=xbar_size, adc_bits=adc_bits,
                                   quant=quant)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' % i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64, rram=rram, ab=ab, xbar_size=xbar_size, adc_bits=adc_bits,
                                   quant=quant)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])
        if quant:
            global_pool = activate(global_pool, ab)
        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, label_dim, rram=rram, xbar_size=xbar_size, adc_bits=adc_bits,
                              quant=quant)
        layers.append(output)

    return layers[-1], act
