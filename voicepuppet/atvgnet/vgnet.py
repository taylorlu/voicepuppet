#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import json
import cv2
from tinynet import ThinNet
import sys

from papio.builder import ModelBuilder
from config.configure import YParams


class ImageEncoder1(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor. scale 1/4
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    filters = [64, 64, 128]
    kernel_sizes = [7, 3, 3]
    strides = [1, 2, 2]

    with tf.variable_scope('ImageEncoder1', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, filters[0], kernel_sizes[0], padding='same',
                           strides=strides[0],
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)

      x = tf.layers.conv2d(x, filters[1], kernel_sizes[1], padding='same',
                           strides=strides[1],
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)

      x = tf.layers.conv2d(x, filters[2], kernel_sizes[2], padding='same',
                           strides=strides[2],
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)

      return x

  def __call__(self, inputs):
    image_encoding = self.build_network(inputs)
    return {'ImageEncoder1': image_encoding}


class ImageEncoder2(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor. scale 1/4
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    filters = [256, 512]
    kernel_sizes = [3, 3]
    strides = [2, 2]

    with tf.variable_scope('ImageEncoder2', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, filters[0], kernel_sizes[0], padding='same',
                           strides=strides[0],
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)

      x = tf.layers.conv2d(x, filters[1], kernel_sizes[1], padding='same',
                           strides=strides[1],
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)

      return x

  def __call__(self, inputs):
    image_encoding = self.build_network(inputs)
    return {'ImageEncoder2': image_encoding}


class LandmarkEncoder1(ModelBuilder):
  '''
  inputs: batch of 1D tensor.
  return: batch of 1D tensor.
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable

  def build_network(self, inputs):
    with tf.variable_scope('LandmarkEncoder1', reuse=tf.AUTO_REUSE):
      ## (img_size/16)^2
      landmark_encoding = tf.layers.dense(inputs, 64, activation=tf.nn.elu)
      return landmark_encoding

  def __call__(self, inputs):
    landmark_encoding = self.build_network(inputs)
    return {'LandmarkEncoder1': landmark_encoding}


class LandmarkEncoder2(ModelBuilder):
  '''
  inputs: batch of 3D tensor, when output from LandmarkEncoder1, reshape 1D to 3D.
  return: batch of 3D tensor.
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('LandmarkEncoder2', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, 256, 3, padding='same',
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      return x

  def __call__(self, inputs):
    landmark_encoding = self.build_network(inputs)
    return {'LandmarkEncoder2': landmark_encoding}


class LandmarkAtt(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor, in [0, 1]. scale 4
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('LandmarkAtt', reuse=tf.AUTO_REUSE):
      x = tf.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same',
                                    kernel_initializer=tf.orthogonal_initializer(), use_bias=False)(inputs)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      x = tf.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                                    kernel_initializer=tf.orthogonal_initializer(), use_bias=False)(x)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      x = tf.layers.conv2d(x, 1, 3, padding='same',
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.sigmoid(x)
      return x

  def __call__(self, inputs):
    landmark_att = self.build_network(inputs)
    return {'LandmarkAtt': landmark_att}


class LandmarkFearure(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor.
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('LandmarkFearure', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, 512, 3, padding='same',
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      return x

  def __call__(self, inputs):
    landmark_fearure = self.build_network(inputs)
    return {'LandmarkFearure': landmark_fearure}


class Bottleneck(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor.
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('Bottleneck', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, 128, 3, padding='same',
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      return x

  def __call__(self, inputs):
    bottleneck = self.build_network(inputs)
    return {'Bottleneck': bottleneck}


class GenBase(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor. scale 4
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
    self.thin_resnet = ThinNet(128, is_training=self.trainable)

  def build_network(self, inputs):
    with tf.variable_scope('GenBase', reuse=tf.AUTO_REUSE):
      x, _ = self.thin_resnet(inputs)
      x = tf.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      x = tf.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      return x

  def __call__(self, inputs):
    gen_base = self.build_network(inputs)
    return {'GenBase': gen_base}


class BaseNet(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor. scale 4
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('BaseNet', reuse=tf.AUTO_REUSE):
      x = tf.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      x = tf.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
      x = tf.contrib.layers.batch_norm(x, updates_collections=None, is_training=self.trainable)
      x = tf.nn.elu(x)
      return x

  def __call__(self, inputs):
    basenet = self.build_network(inputs)
    return {'BaseNet': basenet}


class GenColor(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor. in [-1, 1]
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('GenColor', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, 3, 7, padding='same',
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.nn.tanh(x)
      return x

  def __call__(self, inputs):
    gen_color = self.build_network(inputs)
    return {'GenColor': gen_color}


class GenAttention(ModelBuilder):
  '''
  inputs: batch of 3D tensor.
  return: batch of 3D tensor. in [0, 1]
  '''

  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    with tf.variable_scope('GenAttention', reuse=tf.AUTO_REUSE):
      x = tf.layers.conv2d(inputs, 1, 7, padding='same',
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable,
                           kernel_regularizer=self.kernel_regularizer)
      x = tf.nn.sigmoid(x)
      return x

  def __call__(self, inputs):
    gen_attention = self.build_network(inputs)
    return {'GenAttention': gen_attention}


class Conv2dGRUCell(tf.nn.rnn_cell.RNNCell):
  '''
  A GRU cell with convolutions instead of multiplications.
  Refer to https://github.com/carlthome/tensorflow-convlstm-cell
  '''

  def __init__(self, shape, filters, kernel, strides, dilation_rate, activation=tf.tanh, normalize=True,
               data_format='channels_last', trainable=True):
    super(Conv2dGRUCell, self).__init__(_reuse=tf.AUTO_REUSE)
    self._filters = filters
    self._kernel = kernel
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._activation = activation
    self._normalize = normalize
    self.trainable = trainable
    if (data_format == 'channels_last'):
      self._size = tf.TensorShape(shape + [self._filters])
      self._feature_axis = self._size.ndims
      self._data_format = None
    elif (data_format == 'channels_first'):
      self._size = tf.TensorShape([self._filters] + shape)
      self._feature_axis = 0
      self._data_format = 'NC'
    else:
      raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value

    with tf.variable_scope('gates', reuse=tf.AUTO_REUSE):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2

      y = tf.layers.conv2d(inputs, m, 3, padding='same',
                           strides=self._strides,
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable)

      if (self._normalize):
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = tf.contrib.layers.batch_norm(r, updates_collections=None, is_training=self.trainable)
        u = tf.contrib.layers.batch_norm(u, updates_collections=None, is_training=self.trainable)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.ones_initializer(), trainable=self.trainable)
        r, u = tf.split(y, 2, axis=self._feature_axis)
      r, u = tf.sigmoid(r), tf.sigmoid(u)

    with tf.variable_scope('candidate', reuse=tf.AUTO_REUSE):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      y = tf.layers.conv2d(inputs, m, 3, padding='same',
                           strides=self._strides,
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=False,
                           trainable=self.trainable)

      if (self._normalize):
        y = tf.contrib.layers.batch_norm(y, updates_collections=None, is_training=self.trainable)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer(), trainable=self.trainable)
      h = u * h + (1 - u) * self._activation(y)

    return h, h


class Conv2dGRU(ModelBuilder):
  def __init__(self, trainable=True):
    self.trainable = trainable

  def build_network(self, inputs, seq_len):
    with tf.variable_scope('Conv2dGRU', reuse=tf.AUTO_REUSE):
      conv2dGRUCell = Conv2dGRUCell(shape=[inputs.get_shape().as_list()[2], inputs.get_shape().as_list()[3]],
                                    filters=512,
                                    kernel=[3, 3],
                                    strides=1,
                                    dilation_rate=2,
                                    trainable=self.trainable)
      initial_state = conv2dGRUCell.zero_state(tf.shape(inputs)[0], tf.float32)
      gru_output, _ = tf.nn.dynamic_rnn(conv2dGRUCell, inputs, sequence_length=seq_len, dtype=inputs.dtype,
                                        initial_state=initial_state)
      gru_output = tf.contrib.layers.batch_norm(gru_output, updates_collections=None, is_training=self.trainable)
      gru_output = tf.nn.elu(gru_output)
      return gru_output

  def __call__(self, inputs, seq_len):
    gru_output = self.build_network(inputs, seq_len)
    return {'Conv2dGRU': gru_output}


class Generator(ModelBuilder):
  '''Generate image feature sequence'''

  def __init__(self, batch_size, landmark_size, trainable=True):
    self.batch_size = batch_size
    self.landmark_size = landmark_size
    self.trainable = trainable

    self.Image_Encoder1 = ImageEncoder1(self.trainable)
    self.Image_Encoder2 = ImageEncoder2(self.trainable)
    self.Landmark_Encoder1 = LandmarkEncoder1(self.trainable)
    self.Landmark_Encoder2 = LandmarkEncoder2(self.trainable)
    self.Landmark_Fearure = LandmarkFearure(self.trainable)
    self.Conv2d_GRU = Conv2dGRU(self.trainable)
    self.Landmark_Att = LandmarkAtt(self.trainable)
    self.Bottle_Neck = Bottleneck(self.trainable)

  def build_videofeature(self, lmk_atts, gru_output, img_encoding1, example_img):
    with tf.variable_scope('VideoFeature'):
      time = tf.shape(lmk_atts)[1]
      ## reshape to [batch_size*time, w,h,c]
      lmk_atts = tf.reshape(lmk_atts, [-1, lmk_atts.get_shape().as_list()[2],
                                       lmk_atts.get_shape().as_list()[3],
                                       lmk_atts.get_shape().as_list()[4]])
      gru_output = tf.reshape(gru_output, [-1, gru_output.get_shape().as_list()[2],
                                           gru_output.get_shape().as_list()[3],
                                           gru_output.get_shape().as_list()[4]])

      ## tile to [batch_size, 1*time, w,h,c]
      img_encoding1 = tf.tile(tf.expand_dims(img_encoding1, 1), (1, time, 1, 1, 1))
      ## reshape to [batch_size*time, w,h,c]
      img_encoding1 = tf.reshape(img_encoding1, [-1, img_encoding1.get_shape().as_list()[2],
                                                 img_encoding1.get_shape().as_list()[3],
                                                 img_encoding1.get_shape().as_list()[4]])

      Gen_Base = GenBase(self.trainable)
      vt_feature = Gen_Base(gru_output)['GenBase']

      vt_feature = img_encoding1 * (1 - lmk_atts) + vt_feature * lmk_atts

      Base_Net = BaseNet(self.trainable)
      base = Base_Net(vt_feature)['BaseNet']

      Gen_Color = GenColor(self.trainable)
      color = Gen_Color(base)['GenColor']

      Gen_Attention = GenAttention(self.trainable)
      attention = Gen_Attention(base)['GenAttention']

      example_img = tf.tile(tf.expand_dims(example_img, 1), (1, time, 1, 1, 1))
      example_img = tf.reshape(example_img, [-1, example_img.get_shape().as_list()[2],
                                             example_img.get_shape().as_list()[3],
                                             example_img.get_shape().as_list()[4]])

      video_fearure = attention * color + (1 - attention) * example_img

      ## reshape back to [batch_size, time, w,h,c]
      video_fearure = tf.reshape(video_fearure, [self.batch_size, -1,
                                                 video_fearure.get_shape().as_list()[1],
                                                 video_fearure.get_shape().as_list()[2],
                                                 video_fearure.get_shape().as_list()[3]])
      attention = tf.reshape(attention, [self.batch_size, -1,
                                         attention.get_shape().as_list()[1],
                                         attention.get_shape().as_list()[2],
                                         attention.get_shape().as_list()[3]])
      color = tf.reshape(color, [self.batch_size, -1,
                                 color.get_shape().as_list()[1],
                                 color.get_shape().as_list()[2],
                                 color.get_shape().as_list()[3]])

      return video_fearure, attention, color

  def build_sequence(self, time, lmk_feature, lmk_encoding2, ex_lmk_feature, ex_lmk_encoding2, img_feature):
    with tf.variable_scope('Sequence'):
      ## reshape back to [batch_size, time, w,h,c]
      lmk_feature = tf.reshape(lmk_feature, [self.batch_size, -1,
                                             lmk_feature.get_shape().as_list()[1],
                                             lmk_feature.get_shape().as_list()[2],
                                             lmk_feature.get_shape().as_list()[3]])
      lmk_encoding2 = tf.reshape(lmk_encoding2, [self.batch_size, -1,
                                                 lmk_encoding2.get_shape().as_list()[1],
                                                 lmk_encoding2.get_shape().as_list()[2],
                                                 lmk_encoding2.get_shape().as_list()[3]])

      ## tile to [batch_size, 1*time, w,h,c]
      ex_lmk_encoding2 = tf.tile(tf.expand_dims(ex_lmk_encoding2, 1), (1, time, 1, 1, 1))
      lmk_exlmk_encoding2 = tf.concat([lmk_encoding2, ex_lmk_encoding2], -1)
      ## reshape to [batch_size*time, w,h,c]
      lmk_exlmk_encoding2 = tf.reshape(lmk_exlmk_encoding2, [-1, lmk_exlmk_encoding2.get_shape().as_list()[2],
                                                             lmk_exlmk_encoding2.get_shape().as_list()[3],
                                                             lmk_exlmk_encoding2.get_shape().as_list()[4]])
      lmk_att = self.Landmark_Att(lmk_exlmk_encoding2)['LandmarkAtt']

      ## reshape back to [batch_size, time, w,h,c]
      lmk_att = tf.reshape(lmk_att, [self.batch_size, -1,
                                     lmk_att.get_shape().as_list()[1],
                                     lmk_att.get_shape().as_list()[2],
                                     lmk_att.get_shape().as_list()[3]])

      ## tile to [batch_size, 1*time, w,h,c]
      ex_lmk_feature = tf.tile(tf.expand_dims(ex_lmk_feature, 1), (1, time, 1, 1, 1))
      img_feature = tf.tile(tf.expand_dims(img_feature, 1), (1, time, 1, 1, 1))
      ex_lmk_feature = tf.concat([img_feature, lmk_feature - ex_lmk_feature], -1)

      ## reshape to [batch_size*time, w,h,c]
      ex_lmk_feature = tf.reshape(ex_lmk_feature, [-1, ex_lmk_feature.get_shape().as_list()[2],
                                                   ex_lmk_feature.get_shape().as_list()[3],
                                                   ex_lmk_feature.get_shape().as_list()[4]])

      bottleneck = self.Bottle_Neck(ex_lmk_feature)['Bottleneck']
      ## reshape back to [batch_size, time, w,h,c]
      bottleneck = tf.reshape(bottleneck, [self.batch_size, -1,
                                           bottleneck.get_shape().as_list()[1],
                                           bottleneck.get_shape().as_list()[2],
                                           bottleneck.get_shape().as_list()[3]])
      return lmk_att, bottleneck

  def build_network(self, example_img, landmark_seq, example_landmark, seq_len):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
      ## img_encoding1: [batch,w,h,128]
      img_encoding1 = self.Image_Encoder1(example_img)['ImageEncoder1']

      ## img_feature: [batch,w,h,512]
      img_feature = self.Image_Encoder2(img_encoding1)['ImageEncoder2']

      ## 1D landmark encoding reshape to 2D feauture map
      reshape_size = [-1, img_feature.get_shape().as_list()[1], img_feature.get_shape().as_list()[2], 1]

      ## ex_lmk_encoding1: [batch,w,h,1]
      ex_lmk_encoding1 = self.Landmark_Encoder1(example_landmark)['LandmarkEncoder1']
      ex_lmk_encoding1 = tf.reshape(ex_lmk_encoding1, reshape_size)

      ## ex_lmk_encoding2: [batch,w,h,256]
      ex_lmk_encoding2 = self.Landmark_Encoder2(ex_lmk_encoding1)['LandmarkEncoder2']

      ## ex_lmk_feature: [batch,w,h,512]
      ex_lmk_feature = self.Landmark_Fearure(ex_lmk_encoding2)['LandmarkFearure']

      ## [batch_size*time, landmark_size]
      time = tf.shape(landmark_seq)[1]
      landmark_seq = tf.reshape(landmark_seq, [-1, self.landmark_size])
      lmk_encoding1 = self.Landmark_Encoder1(landmark_seq)['LandmarkEncoder1']
      lmk_encoding1 = tf.reshape(lmk_encoding1, reshape_size)

      lmk_encoding2 = self.Landmark_Encoder2(lmk_encoding1)['LandmarkEncoder2']

      lmk_feature = self.Landmark_Fearure(lmk_encoding2)['LandmarkFearure']

      lmk_atts, bottlenecks = self.build_sequence(time, lmk_feature, lmk_encoding2, ex_lmk_feature, ex_lmk_encoding2,
                                                  img_feature)

      ## Conv2dGRU: input/output (batch, time, rows, cols, channels)
      gru_output = self.Conv2d_GRU(bottlenecks, seq_len)['Conv2dGRU']

      feature, attention, color = self.build_videofeature(lmk_atts, gru_output, img_encoding1, example_img)

    return feature, attention, color

  def __call__(self, input_tuple):
    '''
    Arguments:
      input_tuple: tuple contains (example_img, landmark_seq, example_landmark)
        Interpret as the follows:
        example_img: [batch,w,h,3] example image, which is the target we want to drive animation on.
        landmark_seq: [batch,time,136] the landmark sequence which generated by ATNet, but we use ground-truth when training.
        example_landmark: [batch,136] the landmark of the example image.
        seq_len: [batch] sequence length of every sample.
    Returns:
      A dict contains [feature, attention map, color]
    '''
    example_img = input_tuple[0]
    landmark_seq = input_tuple[1]
    example_landmark = input_tuple[2]
    seq_len = input_tuple[3]

    feature, attention, color = self.build_network(example_img, landmark_seq, example_landmark, seq_len)
    return {'Generator': {"Feature": feature, "Attention": attention, "Color": color}}


class DisLandmarkEncoder(ModelBuilder):
  def __init__(self, trainable=True):
    self.trainable = trainable

  def build_network(self, inputs):
    x = tf.layers.dense(inputs, 256, activation=tf.nn.elu, use_bias=False)
    x = tf.layers.dense(x, 512, activation=tf.nn.elu, use_bias=False)
    return x

  def __call__(self, inputs):
    landmark_encoding = self.build_network(inputs)
    return {'DisLandmarkEncoder': landmark_encoding}


class DisImageEncoder(ModelBuilder):
  def __init__(self, trainable=True):
    self.trainable = trainable
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

  def build_network(self, inputs):
    filters = [64, 128, 128, 256]
    kernel_sizes = [3, 3, 3, 3]
    strides = [2, 2, 2, 2]

    x = tf.layers.conv2d(inputs, filters[0], kernel_sizes[0], padding='same',
                         strides=strides[0],
                         kernel_initializer=tf.orthogonal_initializer(),
                         use_bias=False,
                         trainable=self.trainable,
                         kernel_regularizer=self.kernel_regularizer)
    x = tf.layers.conv2d(x, filters[1], kernel_sizes[1], padding='same',
                         strides=strides[1],
                         kernel_initializer=tf.orthogonal_initializer(),
                         use_bias=False,
                         trainable=self.trainable,
                         kernel_regularizer=self.kernel_regularizer)
    x = tf.layers.conv2d(x, filters[2], kernel_sizes[2], padding='same',
                         strides=strides[2],
                         kernel_initializer=tf.orthogonal_initializer(),
                         use_bias=False,
                         trainable=self.trainable,
                         kernel_regularizer=self.kernel_regularizer)
    x = tf.layers.conv2d(x, filters[3], kernel_sizes[3], padding='same',
                         strides=strides[3],
                         kernel_initializer=tf.orthogonal_initializer(),
                         use_bias=False,
                         trainable=self.trainable,
                         kernel_regularizer=self.kernel_regularizer)

    x = tf.reshape(x, [-1, x.get_shape().as_list()[1] * x.get_shape().as_list()[2] * x.get_shape().as_list()[3]])
    x = tf.layers.dense(x, 512, activation=tf.nn.elu, use_bias=False)
    return x

  def __call__(self, inputs):
    dis_image_encoder = self.build_network(inputs)
    return {'DisImageEncoder': dis_image_encoder}


class RNNModule(ModelBuilder):
  def __init__(self, num_units=256, num_layers=1, trainable=True):
    self.num_units = num_units
    self.num_layers = num_layers
    self.keep_prob = 0.75
    if (not trainable):
      self.keep_prob = 1.0

  def build_network(self, inputs, seq_len):
    gru_cell = tf.nn.rnn_cell.GRUCell(self.num_units, kernel_initializer=tf.orthogonal_initializer())
    layers = [tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=self.keep_prob) for _ in range(self.num_layers)]
    cells = tf.nn.rnn_cell.MultiRNNCell(layers)
    initial_state = cells.zero_state(tf.shape(inputs)[0], tf.float32)
    outputs, last_state = tf.nn.dynamic_rnn(cells, inputs,
                                            sequence_length=seq_len,
                                            dtype=tf.float32,
                                            initial_state=initial_state)

    return outputs

  def __call__(self, inputs, seq_len):
    if self.num_units is None:
      self.num_units = inputs.get_shape()[-1]
    outputs = self.build_network(inputs, seq_len)
    return {'RNNModule': outputs}


class Decision(ModelBuilder):
  def __init__(self, trainable=True):
    self.trainable = trainable

  def build_network(self, inputs):
    x = tf.layers.dense(inputs, 1, use_bias=False)
    return x

  def __call__(self, inputs):
    score = self.build_network(inputs)
    return {'Decision': score}


class RnnDense(ModelBuilder):
  def __init__(self, trainable=True):
    self.trainable = trainable

  def build_network(self, inputs):
    x = tf.layers.dense(inputs, 136, activation=tf.nn.tanh, use_bias=False)
    return x

  def __call__(self, inputs):
    rnn_dense = self.build_network(inputs)
    return {'RnnDense': rnn_dense}


class Discriminator(ModelBuilder):
  '''Discriminator module, Adversarial between generator and real image sequence'''

  def __init__(self, batch_size, trainable=True):
    self.batch_size = batch_size
    self.trainable = trainable

  def build_network(self, img_seq, example_landmark, seq_len):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('DisLandmarkEncoder'):
        DisLandmark_Encoder = DisLandmarkEncoder(self.trainable)
        dis_lmk_encoding = DisLandmark_Encoder(example_landmark)['DisLandmarkEncoder']

      with tf.variable_scope('NewFeature'):
        time = tf.shape(img_seq)[1]
        ## reshape to [batch_size*time, w,h,c]
        img_seq = tf.reshape(img_seq, [-1, img_seq.get_shape().as_list()[2],
                                       img_seq.get_shape().as_list()[3],
                                       img_seq.get_shape().as_list()[4]])

        DisImage_Encoder = DisImageEncoder(self.trainable)
        dis_img_encoding = DisImage_Encoder(img_seq)['DisImageEncoder']
        ## reshape back to [batch_size, time, 512]
        dis_img_encoding = tf.reshape(dis_img_encoding, [self.batch_size, -1,
                                                         dis_img_encoding.get_shape().as_list()[-1]])

        ## tile to [batch_size, 1*time, 512]
        dis_lmk_encoding = tf.tile(tf.expand_dims(dis_lmk_encoding, 1), (1, time, 1))
        new_feature = tf.concat([dis_img_encoding, dis_lmk_encoding], axis=-1)

      with tf.variable_scope('RnnModule'):
        RNN_Module = RNNModule(trainable=self.trainable)
        rnn_output = RNN_Module(new_feature, seq_len)['RNNModule']

      with tf.variable_scope('OutSequence'):
        decision = Decision(self.trainable)
        score = decision(rnn_output)['Decision']
        rnn_dense = RnnDense(self.trainable)
        landmark_seq = rnn_dense(rnn_output)['RnnDense']
        ## tile to [batch_size, 1*time, 136]
        example_landmark = tf.tile(tf.expand_dims(example_landmark, 1), (1, time, 1))
        landmark_seq += example_landmark

      with tf.variable_scope('Aggregation'):
        seq_mask = tf.sequence_mask(seq_len, tf.keras.backend.max(seq_len), dtype=tf.float32)
        score = tf.reshape(score, [self.batch_size, -1])
        score = tf.reduce_sum(score * seq_mask, axis=-1) / tf.cast(seq_len, tf.float32)
        score = tf.nn.sigmoid(score)

    return score, landmark_seq

  def __call__(self, input_tuple):
    '''
    Arguments:
      input_tuple: tuple contains (img_seq, example_landmark, seq_len)
        Interpret as the follows:
        img_seq: [batch,time,w,h,c] the image sequence, real or fake.
        example_landmark: [batch,136] the landmark of the example image.
        seq_len: [batch] the sequence length of each video clip.
    Returns:
      A dict contains [discriminator score, landmark sequence]
    '''
    img_seq = input_tuple[0]
    example_landmark = input_tuple[1]
    seq_len = input_tuple[2]
    score, landmark_seq = self.build_network(img_seq, example_landmark, seq_len)
    return {"Discriminator": {'Decision': score, "LandmarkSeq": landmark_seq}}


class VGNet(ModelBuilder):

  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = VGNet.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.training['learning_rate'] = 0.001
    params.training['decay_steps'] = 1000
    params.training['decay_rate'] = 0.95
    params.add_hparam('landmark_size', 136)
    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.learning_rate = params.training['learning_rate']
    self.max_grad_norm = params.training['max_grad_norm']
    self.decay_steps = params.training['decay_steps']
    self.decay_rate = params.training['decay_rate']
    self.drop_rate = params.training['drop_rate']
    self.landmark_size = params.landmark_size
    self.batch_size = params.batch_size

  def build_generator(self, trainable=True):
    self.generator = Generator(self.batch_size, self.landmark_size, trainable=trainable)

  def build_discriminator(self, trainable=True):
    self.discriminator = Discriminator(self.batch_size, trainable=trainable)

  def add_discriminator_cost(self, real_landmark_seq, real_img_seq, example_landmark, example_img, seq_len):
    nodes = {}
    ## real image sequence to discriminator
    input_tuple = (real_img_seq, example_landmark, seq_len)
    node = self.discriminator(input_tuple)
    nodes.update({'Real_node': node})
    real_score = node['Discriminator']['Decision']
    real_lmk_pred = node['Discriminator']['LandmarkSeq']

    ## binary cross entropy loss, real_score to 1.
    bce_loss1 = tf.reduce_mean(-tf.log(real_score))

    seq_mask = tf.sequence_mask(seq_len, tf.keras.backend.max(seq_len), dtype=tf.float32)

    ## mean square of landmark sequence diff, real_lmk_pred to real_landmark_seq.
    lmk_loss1 = tf.keras.losses.MSE(real_lmk_pred, real_landmark_seq)
    lmk_loss1 = tf.math.reduce_mean(tf.math.reduce_sum(lmk_loss1 * seq_mask, axis=-1))
    nodes.update({'Real_bce_loss': bce_loss1})
    nodes.update({'Real_lmk_loss': lmk_loss1})

    ## (example_img, real_landmark_seq, example_landmark) to generator
    input_tuple = (example_img, real_landmark_seq, example_landmark, seq_len)
    node = self.generator(input_tuple)
    nodes.update({'Generator_node': node})
    fake_img_seq = node['Generator']['Feature']

    ## fake image sequence to discriminator
    input_tuple = (fake_img_seq, example_landmark, seq_len)
    node = self.discriminator(input_tuple)
    nodes.update({'Fake_node': node})
    fake_score = node['Discriminator']['Decision']
    fake_lmk_pred = node['Discriminator']['LandmarkSeq']

    ## binary cross entropy loss, fake_score to 0.
    bce_loss2 = tf.reduce_mean(-tf.log(1 - fake_score))

    ## mean square of landmark sequence diff, fake_lmk_pred to real_landmark_seq.
    lmk_loss2 = tf.keras.losses.MSE(fake_lmk_pred, real_landmark_seq)
    lmk_loss2 = tf.math.reduce_mean(tf.math.reduce_sum(lmk_loss2 * seq_mask, axis=-1))
    nodes.update({'Fake_bce_loss': bce_loss2})
    nodes.update({'Fake_lmk_loss': lmk_loss2})

    loss = bce_loss1 + lmk_loss1 + bce_loss2 + lmk_loss2
    nodes.update({'Discriminator_loss': loss})
    return loss, nodes

  def add_generator_cost(self, real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img, seq_len):
    nodes = {}
    ## (example_img, real_landmark_seq, example_landmark) to generator
    input_tuple = (example_img, real_landmark_seq, example_landmark, seq_len)
    node = self.generator(input_tuple)
    nodes.update({'Generator_node': node})
    fake_img_seq = node['Generator']['Feature']
    attention = node['Generator']['Attention']
    nodes.update({'Fake_img_seq': fake_img_seq})
    nodes.update({'Attention': attention})

    ## remove gradient when backpropagating refer to 3.2 in https://www.cs.rochester.edu/u/lchen63/cvpr2019.pdf
    attention = tf.stop_gradient(attention)

    ## fake image sequence to discriminator
    input_tuple = (fake_img_seq, example_landmark, seq_len)
    node = self.discriminator(input_tuple)
    nodes.update({'Discriminator_node': node})
    fake_score = node['Discriminator']['Decision']
    fake_lmk_pred = node['Discriminator']['LandmarkSeq']

    ## binary cross entropy loss, fake_score to 1.
    bce_loss = tf.reduce_mean(-tf.log(fake_score))

    seq_mask = tf.sequence_mask(seq_len, tf.keras.backend.max(seq_len), dtype=tf.float32)

    ## mean square of landmark sequence diff, fake_lmk_pred to real_landmark_seq.
    lmk_loss = tf.keras.losses.MSE(fake_lmk_pred, real_landmark_seq)
    lmk_loss = tf.math.reduce_mean(tf.math.reduce_sum(lmk_loss * seq_mask, axis=-1))
    nodes.update({'Bce_loss': bce_loss})
    nodes.update({'Lmk_loss': lmk_loss})

    ## mean square of image sequence diff, real_img_seq to fake_img_seq.
    seq_diff = tf.math.reduce_sum(
        tf.math.square(real_img_seq - fake_img_seq) * (real_mask_seq + 0.5) * (attention + 0.5), axis=[2, 3, 4])
    pix_loss = tf.math.reduce_mean(tf.math.reduce_sum(seq_diff * seq_mask, axis=-1))
    nodes.update({'Pix_loss': pix_loss})

    loss = bce_loss + lmk_loss + pix_loss
    nodes.update({'Generator_loss': loss})
    return loss, nodes

  def build_eval_op(self, real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img, seq_len):
    nodes = {}
    nodes.update({'Real_landmark_seq': real_landmark_seq})
    nodes.update({'Real_mask_seq': real_mask_seq})
    nodes.update({'Real_img_seq': real_img_seq})
    nodes.update({'Example_landmark': example_landmark})
    nodes.update({'Example_img': example_img})
    nodes.update({'Seq_len': seq_len})

    self.build_generator(trainable=False)
    self.build_discriminator(trainable=False)

    discriminator_cost, node = self.add_discriminator_cost(real_landmark_seq, real_img_seq, example_landmark,
                                                           example_img,
                                                           seq_len)
    nodes.update({'Discriminator': node})
    generator_cost, node = self.add_generator_cost(real_landmark_seq, real_mask_seq, real_img_seq, example_landmark,
                                                   example_img, seq_len)
    nodes.update({'Generator': node})

    return nodes

  def build_train_op(self, real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img, seq_len):
    nodes = {}
    nodes.update({'Real_landmark_seq': real_landmark_seq})
    nodes.update({'Real_mask_seq': real_mask_seq})
    nodes.update({'Real_img_seq': real_img_seq})
    nodes.update({'Example_landmark': example_landmark})
    nodes.update({'Example_img': example_img})
    nodes.update({'Seq_len': seq_len})

    self.build_generator(trainable=True)
    self.build_discriminator(trainable=True)

    discriminator_cost, node = self.add_discriminator_cost(real_landmark_seq, real_img_seq, example_landmark,
                                                           example_img,
                                                           seq_len)
    nodes.update({'Discriminator': node})
    generator_cost, node = self.add_generator_cost(real_landmark_seq, real_mask_seq, real_img_seq, example_landmark,
                                                   example_img, seq_len)
    nodes.update({'Generator': node})

    # returns all variables created(the two variable scopes) and makes trainable true
    tvars = tf.trainable_variables()
    discriminator_vars = [var for var in tvars if 'Discriminator' in var.name]
    generator_vars = [var for var in tvars if 'Generator' in var.name]

    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                    self.decay_steps, self.decay_rate, staircase=True)
    nodes.update({'Global_step': global_step})
    nodes.update({'Lr': lr})

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # train op for discriminator
      D_optimizer = tf.train.AdamOptimizer(lr)
      grads, tvars = zip(*D_optimizer.compute_gradients(discriminator_cost, discriminator_vars))
      grads_clip, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
      train_discriminator_op = D_optimizer.apply_gradients(zip(grads_clip, tvars), global_step=global_step)
      nodes['Train_discriminator'] = tf.tuple([train_discriminator_op, discriminator_cost, global_step, lr])
      nodes['Discriminator_grads'] = grads
      nodes['Discriminator_tvars'] = tvars

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # train op for generator
      G_optimizer = tf.train.AdamOptimizer(lr)
      grads, tvars = zip(*G_optimizer.compute_gradients(generator_cost, generator_vars))
      grads_clip, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
      train_generator_op = G_optimizer.apply_gradients(zip(grads_clip, tvars), global_step=global_step)
      nodes['Train_generator'] = tf.tuple([train_generator_op, generator_cost, global_step, lr])
      nodes['Generator_grads'] = grads
      nodes['Generator_tvars'] = tvars

    return nodes

  def build_inference_op(self, landmark_seq, example_landmark, example_img, seq_len):
    nodes = {}
    nodes.update({'Landmark_seq': landmark_seq})
    nodes.update({'Example_landmark': example_landmark})
    nodes.update({'Example_img': example_img})
    nodes.update({'Seq_len': seq_len})

    self.build_generator(trainable=False)
    input_tuple = (example_img, landmark_seq, example_landmark, seq_len)
    node = self.generator(input_tuple)
    fake_img_seq = node['Generator']['Feature']
    nodes.update({'Fake_img_seq': fake_img_seq})

    return nodes
