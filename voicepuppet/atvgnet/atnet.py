#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os

from papio.builder import ModelBuilder
from tinynet import MfccNet
from config.configure import YParams


class MfccEncoder(ModelBuilder):
  '''
  Use backbone CNN to process mfcc tensor.
  '''

  def __init__(self, mfcc_net, pooling_size, thinresnet_output_channels, batch_size, trainable=True):
    self.mfcc_net = mfcc_net
    self.pooling_size = pooling_size
    self.thinresnet_output_channels = thinresnet_output_channels
    self.batch_size = batch_size
    self.trainable = trainable

  def build_network(self, mfccs):
    mfccs = tf.expand_dims(mfccs, -1)
    features, _ = self.mfcc_net(mfccs)
    mfcc_encoding = tf.layers.max_pooling2d(features, self.pooling_size, self.pooling_size, padding='same')
    mfcc_encoding = tf.reshape(mfcc_encoding, [self.batch_size, -1, self.thinresnet_output_channels])
    return mfcc_encoding

  def __call__(self, inputs):
    mfcc_encoding = self.build_network(inputs)
    return {'MfccEncoder': mfcc_encoding}


class PoseEncoder(ModelBuilder):
  '''
  Encode pose tensor to embedding_size.
  '''

  def __init__(self, embedding_size=128, drop_rate=0.25, trainable=True):
    self.embedding_size = embedding_size
    self.trainable = trainable
    self.drop_rate = drop_rate
    if (not trainable):
      self.drop_rate = 0

  def build_network(self, poses):
    poses = tf.layers.dense(poses, self.embedding_size)
    poses = tf.contrib.layers.batch_norm(poses, updates_collections=None, is_training=self.trainable)
    poses = tf.nn.elu(poses)
    return poses

  def __call__(self, inputs):
    poses = self.build_network(inputs)
    return {'PoseEncoder': poses}


class LandmarkEncoder(ModelBuilder):
  '''
  Encode landmark tensor to embedding_size.
  '''

  def __init__(self, embedding_size=128, drop_rate=0.25, trainable=True):
    self.embedding_size = embedding_size
    self.trainable = trainable
    self.drop_rate = drop_rate
    if (not trainable):
      self.drop_rate = 0

  def build_network(self, example_landmark, seq_len):
    example_landmark = tf.tile(tf.expand_dims(example_landmark, 1), (1, seq_len, 1))
    dense1 = tf.layers.dense(example_landmark, self.embedding_size)
    dense1 = tf.contrib.layers.batch_norm(dense1, updates_collections=None, is_training=self.trainable)
    landmark_f = tf.nn.elu(dense1)
    return landmark_f

  def __call__(self, inputs, seq_len):
    landmark_f = self.build_network(inputs, seq_len)
    return {'LandmarkEncoder': landmark_f}


class RNNModule(ModelBuilder):
  def __init__(self, num_units=None, num_layers=1, drop_rate=0.25, trainable=True):
    self.num_units = num_units
    self.num_layers = num_layers
    self.drop_rate = drop_rate
    if (not trainable):
      self.drop_rate = 0

  def build_network(self, inputs, seq_len):
    gru_cell = tf.contrib.rnn.GRUCell(self.num_units, kernel_initializer=tf.orthogonal_initializer())
    layers = [tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=1 - self.drop_rate) for _ in
              range(self.num_layers)]
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


class LandmarkDecoder(ModelBuilder):
  '''
  Decode landmark tensor back to 136.
  '''

  def __init__(self, component, embedding_size=128, landmark_size=136, eye_index_start=72, eye_lmk_size=24,
               drop_rate=0.25, trainable=True):
    self.component = component
    self.embedding_size = embedding_size
    self.trainable = trainable
    self.eye_lmk_size = eye_lmk_size
    self.padding_begin = eye_index_start
    self.padding_end = landmark_size - eye_index_start - eye_lmk_size
    self.drop_rate = drop_rate
    if (not trainable):
      self.drop_rate = 0

  def build_network(self, inputs, ear):
    dense = tf.layers.dense(inputs, 64)
    dense = tf.contrib.layers.batch_norm(dense, updates_collections=None, is_training=self.trainable)
    dense = tf.nn.elu(dense)
    dense = tf.layers.dense(dense, self.component.shape[0], activation=tf.nn.tanh) * 0.9
    dense_ear = tf.concat([dense, ear], -1)
    dense_ear = tf.layers.dense(dense_ear, 24, activation=tf.nn.tanh) * 0.1
    output_landmarks = tf.matmul(dense, self.component) + tf.pad(dense_ear, [[0, 0], [0, 0], [self.padding_begin,
                                                                                              self.padding_end]])
    return output_landmarks

  def __call__(self, inputs, ear):
    output_landmarks = self.build_network(inputs, ear)
    return {'LandmarkDecoder': output_landmarks}


class ATNet(ModelBuilder):

  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = ATNet.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('thinresnet_scale', [1, 32])
    params.add_hparam('thinresnet_output_channels', 256)
    params.add_hparam('encode_embedding_size', 128)
    params.add_hparam('decode_embedding_size', 128)
    params.add_hparam('rnn_hidden_size', 128)
    params.add_hparam('rnn_layers', 1)
    params.add_hparam('landmark_size', 136)
    params.add_hparam('eye_index_start', 72)
    params.add_hparam('eye_lmk_size', 24)

    params.training['learning_rate'] = 0.001
    params.training['decay_steps'] = 1000
    params.training['decay_rate'] = 0.95

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.mean = np.load(params.mean_file)
    self.component = np.load(params.components_file).astype(np.float32)[:, :6].T  # (6,136)
    self.encode_embedding_size = params.encode_embedding_size
    self.decode_embedding_size = params.decode_embedding_size
    self.rnn_hidden_size = params.rnn_hidden_size
    self.rnn_layers = params.rnn_layers
    self.landmark_size = params.landmark_size
    self.eye_index_start = params.eye_index_start
    self.eye_lmk_size = params.eye_lmk_size
    self.learning_rate = params.training['learning_rate']
    self.max_grad_norm = params.training['max_grad_norm']
    self.decay_steps = params.training['decay_steps']
    self.decay_rate = params.training['decay_rate']
    self.drop_rate = params.training['drop_rate']
    self.batch_size = params.batch_size

    self.sample_rate = params.mel['sample_rate']
    self.num_mel_bins = params.mel['num_mel_bins']
    self.hop_step = params.mel['hop_step']
    self.frame_rate = params.frame_rate
    self.frame_wav_scale = self.sample_rate / self.frame_rate
    self.frame_mfcc_scale = self.frame_wav_scale / self.hop_step
    assert (self.frame_mfcc_scale - int(self.frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

    self.thinresnet_output_channels = params.thinresnet_output_channels
    self.thinresnet_scale = params.thinresnet_scale
    self.thinresnet_pooling_size = [int(math.ceil(float(self.frame_mfcc_scale) / self.thinresnet_scale[0])),
                                    int(math.ceil(float(self.num_mel_bins) / self.thinresnet_scale[1]))]

  def build_network(self, ears, poses, mfccs, example_landmark, seq_len, trainable=True):
    nodes = {}
    example_landmark = tf.matmul(example_landmark, self.component.T)
    if (not trainable):
      self.drop_rate = 0

    with tf.variable_scope('mfcc_encoder', reuse=tf.AUTO_REUSE):
      mfccNet = MfccNet(self.thinresnet_output_channels, is_training=trainable)
      mfccEncoder = MfccEncoder(mfccNet, self.thinresnet_pooling_size, self.thinresnet_output_channels,
                                self.batch_size, trainable=trainable)
      mfcc_output = mfccEncoder(mfccs)['MfccEncoder']
      mfcc_output = tf.layers.dense(mfcc_output, self.encode_embedding_size)
      mfcc_output = tf.contrib.layers.batch_norm(mfcc_output, updates_collections=None, is_training=trainable)
      mfcc_output = tf.nn.elu(mfcc_output)
      nodes.update({'MfccEncoder': mfcc_output})

    with tf.variable_scope('landmark_encoder', reuse=tf.AUTO_REUSE):
      landmarkEncoder = LandmarkEncoder(self.encode_embedding_size, trainable=trainable)
      nodes.update(landmarkEncoder(example_landmark, tf.shape(poses)[1]))

    with tf.variable_scope('pose_encoder', reuse=tf.AUTO_REUSE):
      poseEncoder = PoseEncoder(self.encode_embedding_size, trainable=trainable)
      nodes.update(poseEncoder(poses))

    with tf.variable_scope('rnn_module', reuse=tf.AUTO_REUSE):
      add_embedding = nodes['MfccEncoder'] + nodes['LandmarkEncoder'] + nodes['PoseEncoder']
      rnn_net = RNNModule(num_units=self.rnn_hidden_size, num_layers=self.rnn_layers, trainable=trainable)
      nodes.update(rnn_net(add_embedding, seq_len))

    with tf.variable_scope('landmark_decoder', reuse=tf.AUTO_REUSE):
      landmarkDecoder = LandmarkDecoder(self.component,
                                        self.decode_embedding_size,
                                        self.landmark_size,
                                        self.eye_index_start,
                                        self.eye_lmk_size,
                                        trainable=trainable)
      nodes.update(landmarkDecoder(nodes['RNNModule'], ears))

    return nodes

  def add_cost_function(self, output_landmarks, landmarks, seq_len):
    landmark_mask = tf.sequence_mask(seq_len, tf.keras.backend.max(seq_len), dtype=tf.float32)
    landmark_diff = tf.math.reduce_sum(tf.math.square(landmarks - output_landmarks), axis=-1)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(landmark_diff * landmark_mask, axis=-1))  # frame diff

    video_mask = tf.sequence_mask(seq_len - 1, tf.keras.backend.max(seq_len) - 1,
                                  dtype=tf.float32)  # shorter of 1 to landmark_mask
    video_diff = (output_landmarks[:, 1:, :] - output_landmarks[:, :-1, :]) - (
        landmarks[:, 1:, :] - landmarks[:, :-1, :])
    video_diff = tf.math.reduce_sum(tf.math.square(video_diff), axis=-1)
    video_loss = tf.math.reduce_mean(tf.math.reduce_sum(video_diff * video_mask, axis=-1))  # video diff

    loss += video_loss
    loss += tf.losses.get_regularization_loss()
    return loss

  def build_eval_op(self, landmark, ears, poses, mfccs, example_landmark, seq_len):
    nodes = {}
    nodes.update({'Landmark': landmark})
    nodes.update({'Ears': ears})
    nodes.update({'Poses': poses})
    nodes.update({'Mfccs': mfccs})
    nodes.update({'Example_landmark': example_landmark})
    nodes.update({'Seq_len': seq_len})

    network_dict = self.build_network(ears, poses, mfccs, example_landmark, seq_len, trainable=False)
    nodes.update(network_dict)

    loss = self.add_cost_function(nodes['LandmarkDecoder'], landmark, seq_len)
    nodes.update({'Loss': loss})

    return nodes

  def build_train_op(self, landmark, ears, poses, mfccs, example_landmark, seq_len):
    nodes = {}
    nodes.update({'Landmark': landmark})
    nodes.update({'Ears': ears})
    nodes.update({'Poses': poses})
    nodes.update({'Mfccs': mfccs})
    nodes.update({'Example_landmark': example_landmark})
    nodes.update({'Seq_len': seq_len})

    network_dict = self.build_network(ears, poses, mfccs, example_landmark, seq_len, trainable=True)
    nodes.update(network_dict)

    loss = self.add_cost_function(nodes['LandmarkDecoder'], landmark, seq_len)
    nodes.update({'Loss': loss})

    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                    self.decay_steps, self.decay_rate, staircase=True)
    nodes.update({'Global_step': global_step})
    nodes.update({'Lr': lr})

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer(lr)
      grads, tvars = zip(*optimizer.compute_gradients(loss))
      grads_clip, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
      train_op = optimizer.apply_gradients(zip(grads_clip, tvars), global_step=global_step)
      nodes.update({'Train_op': train_op})
      nodes['Grads'] = grads
      nodes['Tvars'] = tvars

    return nodes

  def build_inference_op(self, ears, poses, mfccs, example_landmark, seq_len):
    nodes = {}
    nodes.update({'Ears': ears})
    nodes.update({'Poses': poses})
    nodes.update({'Mfccs': mfccs})
    nodes.update({'Example_landmark': example_landmark})

    network_dict = self.build_network(ears, poses, mfccs, example_landmark, seq_len, trainable=False)
    nodes.update(network_dict)

    return nodes
