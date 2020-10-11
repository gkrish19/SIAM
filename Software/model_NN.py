# Final Code: IEDM 2020
# Author: Gokul Krishnan
# Time and Date: 15th May 2020


import os
import time
import shutil
from datetime import timedelta
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from VGG16 import *
import random
from VGG19 import *
from DenseNet import *
from ResNet import *
from pact_dorefa import *
from array import array
from scipy import stats
TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class NN:
    def __init__(self, data_provider, keep_prob, num_inter_threads, num_intra_threads, model_type, stddevVar, act_width,
                 wgt_width, depth, growth_rate, total_blocks, bc_mode, reduction,
                 weight_decay, nesterov_momentum, dataset, rram, xbar_size, adc_bits,
                 should_save_logs, should_save_model, vat, quant,
                 renew_logs=False,
                 **kwargs):

        self.data_provider = data_provider
        self.vat = vat
        self.quant = quant
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.num_inter_threads = num_inter_threads
        self.num_intra_threads = num_intra_threads
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        self.cross = []
        self.valid = []
        self._act = []
        self.W = []
        self.param = []
        self.init_param = 0
        self.mask_array = []
        self.slots = []
        self.model_type = model_type
        self.stddevVar = stddevVar
        self.ab = act_width
        self.wb = wgt_width
        self.depth = depth
        self.growth_rate = growth_rate
        self.total_blocks = total_blocks
        self.bc_mode = bc_mode
        self.rram = rram
        self.xbar_size = xbar_size
        self.adc_bits = adc_bits
        # compression rate at the transition layers
        self.reduction = reduction


        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Specify the CPU inter and Intra threads used by MKL
        config.intra_op_parallelism_threads = self.num_intra_threads
        config.inter_op_parallelism_threads = self.num_inter_threads
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.initialize_all_variables())
            self.var = [v.name for v in tf.trainable_variables()]
            print("********************Variable_names are******************************")
            print("**************************************************")
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/floatingpoint_model_%s' % (self.model_type)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def q_save_path(self):
        try:
            save_path = self._q_save_path
        except AttributeError:
            save_path = 'saves/fixedpoint_model_%s' % (self.model_type)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._q_save_path = save_path
        return save_path

    @property
    def vat_save_path(self):
        try:
            save_path = self._vat_save_path
        except AttributeError:
            save_path = 'saves/vat_fixedpoint_model_%s_%f' % (self.model_type, self.stddevVar)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._vat_save_path = save_path
        return save_path

    @property
    def rram_save_path(self):
        try:
            save_path = self._rram_save_path
        except AttributeError:
            save_path = 'saves/rram_vat_fixedpoint_model_%s_%f_%d_%d' % (self.model_type, self.stddevVar, self.xbar_size, self.adc_bits)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._rram_save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_type
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        print("Entered Save Model Function")
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def q_save_model(self, global_step=None):
        print("Entered Quantized Save Model Function")
        self.saver.save(self.sess, self.q_save_path, global_step=global_step)

    def vat_save_model(self, global_step=None):
        print("Entered VAT Quantized Save Model Function")
        self.saver.save(self.sess, self.vat_save_path, global_step=global_step)

    def rram_save_model(self, global_step=None):
        print("Entered RRAM VAT Quantized Save Model Function")
        self.saver.save(self.sess, self.rram_save_path, global_step=global_step)

    def load_model(self):
        print("Entered Load Model Function")
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def q_load_model(self):
        print("Entered Quant Load Model Function")
        try:
            self.saver.restore(self.sess, self.q_save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.q_save_path)
        self.saver.restore(self.sess, self.q_save_path)
        print("Successfully load model from save path: %s" % self.q_save_path)

    def vat_load_model(self):
        non_trainable_vars = []
        for v in tf.trainable_variables():
            string = v.name
            if self.rram:
                if 'rram_alpha' in string:
                    pass
                else:
                    non_trainable_vars.append(v)
        print("Entered VAT Load Model Function")
        try:
            self.saver.restore(self.sess, self.vat_save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.vat_save_path)
        self.saver.restore(self.sess, self.vat_save_path)
        print("Successfully load model from save path: %s" % self.vat_save_path)

    def vat_load_model_new(self):
        non_trainable_vars = []
        for v in tf.trainable_variables():
            string = v.name
            if self.rram:
                if 'rram_alpha' in string:
                    pass
                else:
                    non_trainable_vars.append(v)
        print("Entered VAT Load Model Function")
        self.saver_1 = tf.train.Saver(non_trainable_vars)
        try:
            self.saver_1.restore(self.sess, self.vat_save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.vat_save_path)
        self.saver_1.restore(self.sess, self.vat_save_path)
        print("Successfully load model from save path: %s" % self.vat_save_path)

    def rram_load_model(self):
        print("Entered RRAM VAT Load Model Function")
        try:
            self.saver.restore(self.sess, self.rram_save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.rram_save_path)
        self.saver.restore(self.sess, self.rram_save_path)
        print("Successfully load model from save path: %s" % self.rram_save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def augment_image(self, image, pad):
        flip = random.getrandbits(1)
        if flip:
            image = image[:, ::-1, :]
        init_shape = image.shape
        new_shape = [init_shape[0] + pad * 2,
                     init_shape[1] + pad * 2,
                     init_shape[2]]
        zeros_padded = np.zeros(new_shape)
        zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
        # randomly crop to original size
        init_x = np.random.randint(0, pad * 2)
        init_y = np.random.randint(0, pad * 2)
        cropped = zeros_padded[
                  init_x: init_x + init_shape[0],
                  init_y: init_y + init_shape[1],
                  :]
        return cropped

    def augment_all_images(self, initial_images, pad):
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.augment_image(initial_images[i], pad=4)
        return new_images

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def _build_graph(self):
        if self.model_type == 'ResNet20':
            self.n = 3
            if self.quant:
                print("Quantizing the model to fixed-point precision")
            else:
                print("The model will be in floating-point precision")
            output, self.act = build_ResNet(input_tensor_batch=self.images,
                                  label_dim=self.n_classes, n=self.n, wb=self.wb,
                                  ab=self.ab, quant=self.quant, rram=self.rram,
                                  xbar_size=self.xbar_size,
                                  adc_bits=self.adc_bits, reuse=False)
        elif self.model_type == 'ResNet56':
            self.n = 9
            if self.quant:
                print("Quantizing the model to fixed-point precision")
            else:
                print("The model will be in floating-point precision")
            output, seld.act = build_ResNet(input_tensor_batch=self.images,
                                  label_dim=self.n_classes, n=self.n, wb=self.wb,
                                  ab=self.ab, quant=self.quant, rram=self.rram,
                                  xbar_size=self.xbar_size,
                                  adc_bits=self.adc_bits, reuse=False)
        elif self.model_type == 'ResNet32':
            self.n = 5
            if self.quant:
                print("Quantizing the model to fixed-point precision")
            else:
                print("The model will be in floating-point precision")
            output, self.act = build_ResNet(input_tensor_batch=self.images,
                                  label_dim=self.n_classes, n=self.n, wb=self.wb,
                                  ab=self.ab, quant=self.quant, rram=self.rram,
                                  xbar_size=self.xbar_size,
                                  adc_bits=self.adc_bits, reuse=False)

        elif self.model_type == 'ResNet110':
            self.n = 18
            if self.quant:
                print("Quantizing the model to fixed-point precision")
            else:
                print("The model will be in floating-point precision")
            output, self.act = build_ResNet(input_tensor_batch=self.images,
                                                                label_dim=self.n_classes, n=self.n, wb=self.wb,
                                                                ab=self.ab, quant=self.quant, rram=self.rram,
                                                                xbar_size=self.xbar_size,
                                                                adc_bits=self.adc_bits, reuse=False)
        elif self.model_type == 'DenseNet' or self.model_type == 'DenseNet-BC':
            m_type = self.model_type
            graph = Dense(act_width=self.ab, wgt_width=self.wb, keep_prob=self.keep_prob, is_training=self.is_training,
                          model_type=m_type, images=self.images, classes=self.n_classes,
                          quant=self.quant, depth=self.depth, growth_rate=self.growth_rate,
                          total_blocks=self.total_blocks, bc_mode=self.bc_mode, reduction=self.reduction,
                          rram=self.rram, adc_bits=self.adc_bits, xbar_size=self.xbar_size)
            output = graph._build_graph()
            if self.quant:
                print("Quantizing the model to fixed-point precision")
            else:
                print("The model will be in floating-point precision")
        elif self.model_type == 'VGG16':
            output = build_VGG16(images=self.images, n_classes=self.n_classes,
                                 is_training=self.is_training, wb=self.wb, ab=self.ab,
                                 quant=self.quant, rram=self.rram,
                                 xbar_size=self.xbar_size, adc_bits=self.adc_bits,
                                 keep_prob=self.keep_prob)
        elif self.model_type == 'VGG19':
            output = build_VGG19(images=self.images, n_classes=self.n_classes,
                                 is_training=self.is_training, wb=self.wb, ab=self.ab,
                                 quant=self.quant, rram=self.rram,
                                 xbar_size=self.xbar_size, adc_bits=self.adc_bits,
                                 keep_prob=self.keep_prob)
        else:
            print("Please choose one of the defined models")

        prediction = tf.nn.softmax(output)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=output))
        self.l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        print("Adding the Losses")

        self.total_loss = self.cross_entropy + (self.l2 * self.weight_decay)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = self.optimizer.minimize(self.total_loss)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        self.n_epochs = n_epochs
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        if self.quant:
            non_trainable_vars = []
            for v in tf.trainable_variables():
                string = v.name
                if self.rram:
                    if 'rram_alpha' in string:
                        pass
                    else:
                        non_trainable_vars.append(v)
                else:
                    if 'alpha' in string:
                        pass
                    else:
                        non_trainable_vars.append(v)
            print(non_trainable_vars)
            if self.vat & (self.rram == False):
                saver_old = tf.train.Saver()
                path = self.q_save_path
            elif self.rram:
                saver_old = tf.train.Saver(non_trainable_vars)
                path = self.vat_save_path
            else:
                path = self.save_path
                saver_old = tf.train.Saver(non_trainable_vars)
            # sess = tf.Session()
            saver_old.restore(self.sess, path)
            # sess.close()
            print("Successfully load model from save path: %s" % path)

        print("Saving the Initial graph model before we train")
        print("**************************************************")
        w, w_2d = self.get_tensor()
        self.param.clear()
        for i in w:
            self.param.append(self.calc_param(i))
        for i in w_2d:
            self.param.append(self.calc_param2d(i))
        param = np.sum(self.param)
        self.init_param = param
        self.param.clear()
        if (self.should_save_model):
            if (self.quant):
                if (self.vat & (self.rram == False)):
                    self.vat_save_model()
                elif (self.rram):
                    self.rram_save_model()
                else:
                    self.q_save_model()
            else:
                self.save_model()
        total_start_time = time.time()
        g = tf.get_default_graph()
        max = 0

        for epoch in range(1, n_epochs + 1):
            # Training with Variation and the SAF
            if (self.vat):
                addDeviceVariation_SA(self.sess, stddevVar=self.stddevVar)
                print("Added Device variation during training")
            self.global_step = epoch
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("\nDecrease learning rate, new lr = %f" % learning_rate)
            print("Training...")
            loss, acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate, epoch)
            self.cross.append(loss)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                # if ((self.rram ==True) & (self.quant==True) & (self.vat==False)):
                # 	vat=True
                # 	print("Vat is set to True")
                # else:
                # 	vat=False
                loss, acc = self.test(
                    self.data_provider.validation, batch_size, vat=False)
                self.valid.append(loss)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("\nTime per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            if (acc > max):
                if self.should_save_model:
                    print("Saving model at Epoch %d" % (epoch))
                    if (self.quant):
                        if (self.vat & (self.rram == False)):
                            self.vat_save_model()
                        elif (self.rram):
                            self.rram_save_model()
                        else:
                            self.q_save_model()
                    else:
                        self.save_model()
                max = acc
        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate, epoch):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        global grads_vars_test
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            # images = self.augment_all_images(images, pad=4)
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy, self.slots]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy, grads_vars_test = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size, vat=False):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        # Testing with Variation and the SAF
        # if (self.vat):
        if (vat):
            addDeviceVariation_SA(self.sess, stddevVar=self.stddevVar)
            print("Added Device variation during testing")
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def calc_param(self, weights):
        weights = self.sess.run(weights)
        shape = weights.shape
        count = 0
        weights = np.absolute(weights)
        for a in range(shape[3]):
            for b in range(shape[2]):
                for c in range(shape[1]):
                    for d in range(shape[0]):
                        if (weights[d][c][b][a] > 0):
                            count += 1
        return (count)

    def calc_param2d(self, weights):
        weights = self.sess.run(weights)
        shape = weights.shape
        count = 0
        weights = np.absolute(weights)
        for i in range(shape[1]):
            for j in range(shape[0]):
                if (weights[j][i] > 0):
                    count += 1
        return (count)

    def get_tensor(self):
        w = []
        w_2d = []
        for i in tf.trainable_variables():
            if (len(i.get_shape().as_list()) == 4):
                w.append((i))
            elif (len(i.get_shape().as_list()) == 2):
                w_2d.append((i))
        return (w, w_2d)

    def distr(self, data, batch_size):
        # val = []
        # batch = data.next_batch(1)
        # feed_dict = {
        #     self.images: batch[0],
        #     self.labels: batch[1],
        #     self.is_training: False,
        # }
        # fetches = self.act
        # activation = self.sess.run(fetches, feed_dict=feed_dict)
        # activation = np.ndarray.flatten(activation)
        activation = self.sess.run('conv3_3/conv2_in_block/conv:0')
        activation = np.ndarray.flatten(activation)
        # activation = np.unique(activation)
        mu = np.mean(activation)
        sigma = np.std(activation)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(activation, stats.norm.pdf(activation, mu, sigma))
        # plt.plot(act_unique)
        plt.savefig('./activation.png')
        plt.show()

