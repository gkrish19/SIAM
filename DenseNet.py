import tensorflow as tf
import numpy as np
from utils import RRAM_fc2d, RRAM_conv2d
from pact_dorefa import *

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class Dense:

    def __init__(self, act_width, wgt_width, is_training, keep_prob, quant, depth, growth_rate, total_blocks, bc_mode,
                 reduction, model_type, images, classes, rram, xbar_size, adc_bits):

        self.ab = act_width
        self.wb = wgt_width
        self.depth = depth
        self.model_type = model_type
        self.growth_rate = growth_rate
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        self.reduction = reduction
        self.n_classes = classes
        self.images = images
        self._act = []
        self.W = []
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.rram = rram
        self.quant = quant
        self.xbar_size = xbar_size
        self.adc_bits = adc_bits
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

    def composite_function(self, _input, out_features, kernel_size=3):
        with tf.variable_scope("composite_function"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4  # Need for this ?? Can we reduce this ????
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        output = self.avg_pool(output, k=2)  # Quantization handled inside the function
        return output

    def transition_layer_to_classes(self, _input):
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)

        # FC
        features_total = int(output.get_shape()[-1])

        output = tf.reshape(output, [-1, features_total])
        # Activation Appending for NeuroSim
        self._act.append(output)
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])

        if self.rram:
            logits = RRAM_fc2d(x=output, W=W, b=bias, xbar_size=self.xbar_size, adc_bits=self.adc_bits)
        else:
            logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        self._act.append(tf.transpose(_input, (0, 3, 1, 2)))
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')

        if self.rram:
            output = RRAM_conv2d(x=_input, W=kernel, xbar_size=self.xbar_size, adc_bits=self.adc_bits, strides=strides,
                                 padding=padding)
        else:
            output = tf.nn.conv2d(_input, kernel, strides, padding)

        if self.quant:
            output = activate(output, self.ab)
        else:
            pass
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        if self.quant:
            output = activate(output, self.ab)
        else:
            pass
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        w = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.keras.initializers.he_normal())
        # print("The weight shape for conv is ", shape)
        if self.quant:
            w = fw(w, self.wb)
        self.W.append(w)
        return w

    def weight_variable_xavier(self, shape, name):
        w = tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())
        if self.quant:
            w = fw(w, self.wb)
        self.W.append(w)
        return w

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        # First Convolution of the Whole Network
        with tf.variable_scope("Initial_convolution"):

            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            output = self.transition_layer_to_classes(output)

        return output
