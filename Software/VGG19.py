from utils import *
from pact_dorefa import *
import tensorflow as tf
import numpy as np


def build_VGG19(images, n_classes, is_training, keep_prob, wb, ab, quant, rram, xbar_size, adc_bits):
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    if quant:
        W_conv1_1 = fw(W_conv1_1, wb)
    if rram:
        output = RRAM_conv2d(x=images, W=W_conv1_1, xbar_size=xbar_size, adc_bits=adc_bits,
                                    strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(images, W_conv1_1) + b_conv1_1
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate (output, ab)

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    if quant:
        W_conv1_2 = fw(W_conv1_2, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv1_2, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv1_2) + b_conv1_2
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)
    output = max_pool(output, 2, 2, "pool1")

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    if quant:
        W_conv2_1 = fw(W_conv2_1, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv2_1, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv2_1) + b_conv2_1
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    if quant:
        W_conv2_2 = fw(W_conv2_2, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv2_2, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv2_2) + b_conv2_2
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)
    output = max_pool(output, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    if quant:
        W_conv3_1 = fw(W_conv3_1, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv3_1, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv3_1) + b_conv3_1
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    if quant:
        W_conv3_2 = fw(W_conv3_2, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv3_2, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv3_2) + b_conv3_2
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    if quant:
        W_conv3_3 = fw(W_conv3_3, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv3_3, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv3_3) + b_conv3_3
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_4 = bias_variable([256])
    if quant:
        W_conv3_4 = fw(W_conv3_4, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv3_4, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv3_4) + b_conv3_4
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)
    output = max_pool(output, 2, 2, "pool3")

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    if quant:
        W_conv4_1 = fw(W_conv4_1, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv4_1, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv4_1) + b_conv4_1
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    if quant:
        W_conv4_2 = fw(W_conv4_2, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv4_2, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv4_2) + b_conv4_2
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    if quant:
        W_conv4_3 = fw(W_conv4_3, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv4_3, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv4_3) + b_conv4_3
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_4 = bias_variable([512])
    if quant:
        W_conv4_4 = fw(W_conv4_4, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv4_4, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv4_4) + b_conv4_4
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)
    output = max_pool(output, 2, 2)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    if quant:
        W_conv5_1 = fw(W_conv5_1, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv5_1, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv5_1) + b_conv5_1
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    if quant:
        W_conv5_2 = fw(W_conv5_2, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv5_2, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv5_2) + b_conv5_2
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    if quant:
        W_conv5_3 = fw(W_conv5_3, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv5_3, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv5_3) + b_conv5_3
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)

    W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_4 = bias_variable([512])
    if quant:
        W_conv5_4 = fw(W_conv5_4, wb)
    if rram:
        output = RRAM_conv2d(x=output, W=W_conv5_4, xbar_size=xbar_size, adc_bits=adc_bits,
                             strides=[1, 1, 1, 1], padding='SAME')
    else:
        output = conv2d(output, W_conv5_4) + b_conv5_4
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate(output, ab)
    output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([512])
    if quant:
        W_fc1 = fw(W_fc1, wb)
    if rram:
        output = RRAM_fc2d(x=output, W=W_fc1, b=b_fc1, xbar_size=xbar_size, adc_bits=adc_bits)
    else:
        output = tf.matmul(output, W_fc1) + b_fc1
    output = tf.nn.relu(batch_norm(output, is_training))
    if quant:
        output = activate (output, ab)
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[512, n_classes], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([n_classes])
    if quant:
        W_fc3 = fw(W_fc3, wb)
    if rram:
        output = RRAM_fc2d(x=output, W=W_fc3, b=b_fc3, xbar_size=xbar_size, adc_bits=adc_bits)
    else:
        output = tf.matmul(output, W_fc3) + b_fc3
    output = tf.nn.relu(batch_norm(output, is_training))
    return output
