import tensorflow as tf
import numpy as np


def batch_norm(x, axis=-1, trainable=True):
  bn_op = tf.keras.layers.BatchNormalization(axis=axis, name='bn')
  x = bn_op(x, training=trainable)
  if (trainable):
    for operation in bn_op.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, operation)
  return x


class ThinResnet():

  def __init__(self, output_channels):
    self.weight_decay = 1e-4
    self.kernel_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)
    self.output_channels = output_channels
    self.nodes = {}

  def identity_block_2D(self, input_tensor, kernel_sizes, filters, stage, block, trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: list of the kernel size of 3 conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x = input_tensor

    for i, flt in enumerate(filters):
      conv_name = 'conv' + str(stage) + '_' + str(block) + '_' + str(i)
      with tf.variable_scope(conv_name) as scope:
        x = tf.layers.conv2d(x, flt, kernel_sizes[i],
                             padding='same',
                             kernel_initializer=tf.orthogonal_initializer(),
                             use_bias=False,
                             trainable=trainable,
                             kernel_regularizer=self.kernel_regularizer)

        x = batch_norm(x, axis=bn_axis, trainable=trainable)
        x = tf.nn.relu(x)

    with tf.variable_scope('add'):
      x = x + input_tensor
      x = tf.nn.relu(x)
    return x

  def conv_block_2D(self, input_tensor, kernel_sizes, filters, stage, block, strides, trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_sizes: list of the kernel size of 3 conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: list of the stride size of 3 conv layer at main path
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x = input_tensor

    for i, flt in enumerate(filters):
      conv_name = 'conv' + str(stage) + '_' + str(block) + '_' + str(i)
      with tf.variable_scope(conv_name) as scope:
        x = tf.layers.conv2d(x, flt, kernel_sizes[i], padding='same',
                             strides=strides[i],
                             kernel_initializer=tf.orthogonal_initializer(),
                             use_bias=False,
                             trainable=trainable,
                             kernel_regularizer=self.kernel_regularizer)

        x = batch_norm(x, axis=bn_axis, trainable=trainable)
        x = tf.nn.relu(x)

    conv_name = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    with tf.variable_scope(conv_name) as scope:
      shortcut = tf.layers.conv2d(input_tensor, filters[-1], kernel_sizes[-1], padding='same',
                                  strides=strides[1],
                                  kernel_initializer=tf.orthogonal_initializer(),
                                  use_bias=False,
                                  trainable=trainable,
                                  kernel_regularizer=self.kernel_regularizer)

      shortcut = batch_norm(shortcut, axis=bn_axis, trainable=trainable)

    with tf.variable_scope('add'):
      x = x + shortcut
      x = tf.nn.relu(x)
    return x

  def resnet_2D_v1(self, inputs, trainable=True):
    bn_axis = 3

    # ===============================================
    #            Convolution Block 1
    # ===============================================
    with tf.variable_scope('conv1_1'):
      x1 = tf.layers.conv2d(inputs, 64, [7, 7],
                            kernel_initializer=tf.orthogonal_initializer(),
                            use_bias=False,
                            trainable=trainable,
                            kernel_regularizer=self.kernel_regularizer,
                            padding='same')

      x1 = batch_norm(x1, axis=bn_axis, trainable=trainable)

    x1 = tf.nn.relu(x1)
    x1 = tf.layers.max_pooling2d(x1, [4, 1], [4, 1], padding='same')
    self.nodes['ThinResnet_Block1'] = x1

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    kernel_sizes = [[1, 1], [3, 3], [1, 1]]
    filters = [48, 48, 96]
    strides = [[1, 1], [2, 1], [1, 1]]
    x2 = self.conv_block_2D(x1, kernel_sizes, filters, stage=2, block='a', strides=strides, trainable=trainable)
    x2 = self.identity_block_2D(x2, kernel_sizes, filters, stage=2, block='b', trainable=trainable)
    self.nodes['ThinResnet_Block2'] = x2

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    kernel_sizes = [[1, 1], [3, 3], [1, 1]]
    filters = [96, 96, 128]
    strides = [[1, 1], [2, 1], [1, 1]]
    x3 = self.conv_block_2D(x2, kernel_sizes, filters, stage=3, block='a', strides=strides, trainable=trainable)
    x3 = self.identity_block_2D(x3, kernel_sizes, filters, stage=3, block='b', trainable=trainable)
    x3 = self.identity_block_2D(x3, kernel_sizes, filters, stage=3, block='c', trainable=trainable)
    self.nodes['ThinResnet_Block3'] = x3

    # ===============================================
    #            Convolution Section 4
    # ===============================================
    kernel_sizes = [[1, 1], [3, 3], [1, 1]]
    filters = [128, 128, 128]
    strides = [[1, 1], [2, 2], [1, 1]]
    x4 = self.conv_block_2D(x3, kernel_sizes, filters, stage=4, block='a', strides=strides, trainable=trainable)
    x4 = self.conv_block_2D(x4, kernel_sizes, filters, stage=4, block='b', strides=strides, trainable=trainable)
    x4 = self.identity_block_2D(x4, kernel_sizes, filters, stage=4, block='c', trainable=trainable)
    self.nodes['ThinResnet_Block4'] = x4

    # ===============================================
    #            Convolution Section 5
    # ===============================================
    kernel_sizes = [[1, 1], [3, 3], [1, 1]]
    filters = [128, 128, self.output_channels]
    strides = [[1, 1], [2, 1], [1, 1]]
    x5 = self.conv_block_2D(x4, kernel_sizes, filters, stage=5, block='a', strides=strides, trainable=trainable)
    x5 = self.identity_block_2D(x5, kernel_sizes, filters, stage=5, block='b', trainable=trainable)
    x5 = self.identity_block_2D(x5, kernel_sizes, filters, stage=5, block='c', trainable=trainable)
    y = tf.reduce_mean(x5, [1])
    self.nodes['ThinResnet_Output'] = y
    return y

  def __call__(self, inputs, trainable=True):
    return self.resnet_2D_v1(inputs, trainable)


if (__name__ == '__main__'):
  resnet = ThinResnet(512)
  inputs = tf.placeholder(shape=[None, 417, 256], dtype=tf.float32)
  mfccs = tf.expand_dims(inputs, -1)
  mfccs = tf.transpose(mfccs, [0, 2, 1, 3])
  x = resnet.resnet_2D_v1(mfccs)
  y = x.get_shape().as_list()
  print(y)
