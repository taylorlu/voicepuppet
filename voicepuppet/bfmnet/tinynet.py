#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np


def mobilenet_v2_func_blocks(is_training):
  filter_initializer = tf.contrib.layers.xavier_initializer()
  activation_func = tf.nn.relu6
  kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-4)

  def conv2d(inputs, filters, kernel_size, stride, scope=''):
    with tf.variable_scope(scope):
      with tf.variable_scope('conv2d'):
        outputs = tf.layers.conv2d(inputs,
                                   filters,
                                   kernel_size,
                                   strides=stride,
                                   padding='same',
                                   activation=None,
                                   use_bias=False,
                                   kernel_initializer=filter_initializer,
                                   kernel_regularizer=kernel_regularizer)

        outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None, is_training=is_training)
        outputs = tf.nn.relu(outputs)
      return outputs

  def _1x1_conv2d(inputs, filters, stride):
    kernel_size = [1, 1]
    with tf.variable_scope('1x1_conv2d'):
      outputs = tf.layers.conv2d(inputs,
                                 filters,
                                 kernel_size,
                                 strides=stride,
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=filter_initializer,
                                 kernel_regularizer=kernel_regularizer)

      outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None, is_training=is_training)
      # no activation_func
    return outputs

  def expansion_conv2d(inputs, expansion, stride):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 4
    filters = input_shape[3] * expansion

    kernel_size = [1, 1]
    with tf.variable_scope('expansion_1x1_conv2d'):
      outputs = tf.layers.conv2d(inputs,
                                 filters,
                                 kernel_size,
                                 strides=stride,
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=filter_initializer,
                                 kernel_regularizer=kernel_regularizer)

      outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None, is_training=is_training)
      outputs = activation_func(outputs)
    return outputs

  def projection_conv2d(inputs, filters, stride):
    kernel_size = [1, 1]
    with tf.variable_scope('projection_1x1_conv2d'):
      outputs = tf.layers.conv2d(inputs,
                                 filters,
                                 kernel_size,
                                 strides=stride,
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 kernel_initializer=filter_initializer,
                                 kernel_regularizer=kernel_regularizer)

      outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None, is_training=is_training)
      # no activation_func
    return outputs

  def depthwise_conv2d(inputs,
                       depthwise_conv_kernel_size,
                       stride):
    with tf.variable_scope('depthwise_conv2d'):
      outputs = tf.contrib.layers.separable_conv2d(
          inputs,
          None,
          depthwise_conv_kernel_size,
          depth_multiplier=1,
          stride=stride,
          padding='SAME',
          activation_fn=None,
          weights_initializer=filter_initializer,
          weights_regularizer=kernel_regularizer,
          biases_initializer=None)

      outputs = tf.contrib.layers.batch_norm(outputs, updates_collections=None, is_training=is_training)
      outputs = activation_func(outputs)

    return outputs

  def avg_pool2d(inputs, scope=''):
    inputs_shape = inputs.get_shape().as_list()
    assert len(inputs_shape) == 4

    pool_height = inputs_shape[1]
    pool_width = inputs_shape[2]

    with tf.variable_scope(scope):
      outputs = tf.layers.average_pooling2d(inputs,
                                            [pool_height, pool_width],
                                            strides=(1, 1),
                                            padding='valid')

    return outputs

  def inverted_residual_block(inputs,
                              filters,
                              stride,
                              expansion=6,
                              scope=''):

    depthwise_conv_kernel_size = [7, 3]
    pointwise_conv_filters = filters

    with tf.variable_scope(scope):
      net = inputs
      net = expansion_conv2d(net, expansion, stride=1)
      net = depthwise_conv2d(net, depthwise_conv_kernel_size, stride=stride)
      net = projection_conv2d(net, pointwise_conv_filters, stride=1)

      if (stride == [1, 1]):
        if net.get_shape().as_list()[3] != inputs.get_shape().as_list()[3]:
          inputs = _1x1_conv2d(inputs, net.get_shape().as_list()[3], stride=1)

        net = net + inputs
        return net
      else:
        return net

  func_blocks = {}
  func_blocks['conv2d'] = conv2d
  func_blocks['inverted_residual_block'] = inverted_residual_block
  func_blocks['avg_pool2d'] = avg_pool2d
  func_blocks['filter_initializer'] = filter_initializer
  func_blocks['activation_func'] = activation_func

  return func_blocks


class MfccNet:
  def __init__(self, output_channels, is_training=True):
    self.output_channels = output_channels
    self.is_training = is_training

  def build_network(self, inputs):
    func_blocks = mobilenet_v2_func_blocks(is_training=self.is_training)
    _conv2d = func_blocks['conv2d']
    _inverted_residual_block = func_blocks['inverted_residual_block']
    _avg_pool2d = func_blocks['avg_pool2d']

    with tf.variable_scope('MfccNet', [inputs]):
      end_points = {}
      net = inputs

      net = _conv2d(net, 32, [9, 5], stride=[1, 2], scope='block0_0')  # size/2
      end_points['block0'] = net

      net = _inverted_residual_block(net, 64, stride=[1, 1], expansion=1, scope='block1_0')
      end_points['block1'] = net

      net = _inverted_residual_block(net, 64, stride=[1, 1], scope='block2_0')  # size/4
      net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')
      net = _inverted_residual_block(net, 64, stride=[1, 1], scope='block2_1')
      end_points['block2'] = net

      net = _inverted_residual_block(net, 128, stride=[1, 1], scope='block3_0')  # size/8
      net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')
      net = _inverted_residual_block(net, 128, stride=[1, 1], scope='block3_1')
      net = _inverted_residual_block(net, 128, stride=[1, 1], scope='block3_2')
      end_points['block3'] = net

      net = _inverted_residual_block(net, 192, stride=[1, 1], scope='block4_0')  # size/16
      net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')
      net = _inverted_residual_block(net, 192, stride=[1, 1], scope='block4_1')
      net = _inverted_residual_block(net, 192, stride=[1, 1], scope='block4_2')
      net = _inverted_residual_block(net, 192, stride=[1, 1], scope='block4_3')
      end_points['block4'] = net

      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block5_0')
      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block5_1')
      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block5_2')
      end_points['block5'] = net

      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block6_0')  # size/32
      net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')
      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block6_1')
      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block6_2')
      end_points['block6'] = net

      net = _inverted_residual_block(net, 256, stride=[1, 1], scope='block7_0')
      end_points['block7'] = net

      net = _conv2d(net, self.output_channels, [1, 1], stride=[1, 1], scope='block8_0')
      end_points['block8'] = net

      output = net

    return output, end_points

  def __call__(self, inputs):
    return self.build_network(inputs)


class ThinNet:
  def __init__(self, output_channels, is_training=True):
    self.output_channels = output_channels
    self.is_training = is_training

  def build_network(self, inputs):
    func_blocks = mobilenet_v2_func_blocks(is_training=self.is_training)
    _conv2d = func_blocks['conv2d']
    _inverted_residual_block = func_blocks['inverted_residual_block']
    _avg_pool2d = func_blocks['avg_pool2d']

    with tf.variable_scope('ThinNet', [inputs]):
      end_points = {}
      net = inputs

      net = _conv2d(net, 32, [3, 3], stride=[2, 2], scope='block0_0')  # size/2
      end_points['block0'] = net

      net = _inverted_residual_block(net, 16, stride=[1, 1], expansion=1, scope='block1_0')
      end_points['block1'] = net

      net = _inverted_residual_block(net, 24, stride=[1, 1], scope='block2_0')  # size/4
      net = _inverted_residual_block(net, 24, stride=[1, 1], scope='block2_1')
      end_points['block2'] = net

      net = _inverted_residual_block(net, 32, stride=[1, 1], scope='block3_0')  # size/8
      net = _inverted_residual_block(net, 32, stride=[1, 1], scope='block3_1')
      net = _inverted_residual_block(net, 32, stride=[1, 1], scope='block3_2')
      end_points['block3'] = net

      net = _inverted_residual_block(net, 64, stride=[1, 1], scope='block4_0')  # size/16
      net = _inverted_residual_block(net, 64, stride=[1, 1], scope='block4_1')
      net = _inverted_residual_block(net, 64, stride=[1, 1], scope='block4_2')
      net = _inverted_residual_block(net, 64, stride=[1, 1], scope='block4_3')
      end_points['block4'] = net

      net = _inverted_residual_block(net, 96, stride=[1, 1], scope='block5_0')
      net = _inverted_residual_block(net, 96, stride=[1, 1], scope='block5_1')
      net = _inverted_residual_block(net, 96, stride=[1, 1], scope='block5_2')
      end_points['block5'] = net

      net = _inverted_residual_block(net, 160, stride=[1, 1], scope='block6_0')  # size/32
      net = _inverted_residual_block(net, 160, stride=[1, 1], scope='block6_1')
      net = _inverted_residual_block(net, 160, stride=[1, 1], scope='block6_2')
      end_points['block6'] = net

      net = _inverted_residual_block(net, 320, stride=[1, 1], scope='block7_0')
      end_points['block7'] = net

      net = _conv2d(net, self.output_channels, [1, 1], stride=[1, 1], scope='block8_0')
      end_points['block8'] = net

      output = net

    return output, end_points

  def __call__(self, inputs):
    return self.build_network(inputs)


if (__name__ == '__main__'):
  inputs = tf.placeholder(shape=[None, 512, 512, 3], dtype=tf.float32)

  thinNet = ThinNet(136, is_training=True)
  output, end_points = thinNet(inputs)

  print('ThinNet output = {}'.format(output.shape))
  print('ThinNet block1 = {}'.format(end_points['block1'].shape))
  print('ThinNet block2 = {}'.format(end_points['block2'].shape))
  print('ThinNet block3 = {}'.format(end_points['block3'].shape))
  print('ThinNet block4 = {}'.format(end_points['block4'].shape))
  print('ThinNet block5 = {}'.format(end_points['block5'].shape))
  print('ThinNet block6 = {}'.format(end_points['block6'].shape))
  print('ThinNet block7 = {}'.format(end_points['block7'].shape))
  print('ThinNet block8 = {}'.format(end_points['block8'].shape))

  # print('====' * 10)

  # mfccNet = MfccNet(136, is_training=True)
  # output, end_points = mfccNet(inputs)

  # print('MfccNet output = {}'.format(output.shape))
  # print('MfccNet block1 = {}'.format(end_points['block1'].shape))
  # print('MfccNet block2 = {}'.format(end_points['block2'].shape))
  # print('MfccNet block3 = {}'.format(end_points['block3'].shape))
  # print('MfccNet block4 = {}'.format(end_points['block4'].shape))
  # print('MfccNet block5 = {}'.format(end_points['block5'].shape))
  # print('MfccNet block6 = {}'.format(end_points['block6'].shape))
  # print('MfccNet block7 = {}'.format(end_points['block7'].shape))
  # print('MfccNet block8 = {}'.format(end_points['block8'].shape))
