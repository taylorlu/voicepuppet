#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os

from voicepuppet.builder import ModelBuilder
from tinynet import MfccNet
from config.configure import YParams
from utils.bfm_load_data import *


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


class BFMCoeffDecoder(ModelBuilder):
  '''
  Decode basel face model coefficient tensor back to 144.
  '''

  def __init__(self, bfm_coeff_size=64, drop_rate=0.25, trainable=True):
    self.trainable = trainable
    self.bfm_coeff_size = bfm_coeff_size
    self.drop_rate = drop_rate
    if (not trainable):
      self.drop_rate = 0

  def build_network(self, inputs):
    dense = tf.layers.dense(inputs, self.bfm_coeff_size, activation=tf.nn.leaky_relu)
    dense = tf.layers.dropout(dense, rate=self.drop_rate, training=self.trainable)
    output_bfm_coeff = tf.layers.dense(dense, self.bfm_coeff_size, activation=None)
    return output_bfm_coeff

  def __call__(self, inputs):
    output_bfm_coeff = self.build_network(inputs)
    return {'BFMCoeffDecoder': output_bfm_coeff}


class BFMNet(ModelBuilder):

  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = BFMNet.default_hparams(config_path)
    self.facemodel = BFM(self.__params.pretrain_dir)
    mouth_mat = np.load(os.path.join(self.__params.pretrain_dir, 'mouth_idx.npy'))
    self.mouth_mask = np.ones([35709, 3], dtype=np.float32)
    for k in range(mouth_mat.shape[0]):
      self.mouth_mask[mouth_mat[k], ...] = [10, 10, 10]

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('thinresnet_scale', [1, 32])
    params.add_hparam('thinresnet_output_channels', 256)
    params.add_hparam('encode_embedding_size', 256)
    params.add_hparam('rnn_hidden_size', 256)
    params.add_hparam('rnn_layers', 1)
    params.add_hparam('bfm_coeff_size', 64)

    params.training['learning_rate'] = 0.001
    params.training['decay_steps'] = 1000
    params.training['decay_rate'] = 0.95

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.pretrain_dir = params.pretrain_dir
    self.encode_embedding_size = params.encode_embedding_size
    self.rnn_hidden_size = params.rnn_hidden_size
    self.rnn_layers = params.rnn_layers
    self.bfm_coeff_size = params.bfm_coeff_size
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

  def build_network(self, mfccs, seq_len, trainable=True):
    nodes = {}
    if (not trainable):
      self.drop_rate = 0

    with tf.variable_scope('mfcc_encoder', reuse=tf.AUTO_REUSE):
      mfccNet = MfccNet(self.thinresnet_output_channels, is_training=trainable)
      mfccEncoder = MfccEncoder(mfccNet, self.thinresnet_pooling_size, self.thinresnet_output_channels,
                                self.batch_size, trainable=trainable)
      mfcc_output = mfccEncoder(mfccs)['MfccEncoder']
      mfcc_output = tf.layers.dense(mfcc_output, self.encode_embedding_size, activation=tf.nn.leaky_relu)
      mfcc_output = tf.layers.dropout(mfcc_output, rate=self.drop_rate, training=trainable)
      nodes.update({'MfccEncoder': mfcc_output})

    with tf.variable_scope('rnn_module', reuse=tf.AUTO_REUSE):
      c1 = tf.layers.dense(nodes['MfccEncoder'], self.encode_embedding_size, activation=tf.nn.leaky_relu)
      rnn_net = RNNModule(num_units=self.rnn_hidden_size, num_layers=self.rnn_layers, trainable=trainable)
      nodes.update(rnn_net(c1, seq_len))

    with tf.variable_scope('bfm_coeff_decoder', reuse=tf.AUTO_REUSE):
      bfmCoeffDecoder = BFMCoeffDecoder(self.bfm_coeff_size, trainable=trainable)
      nodes.update(bfmCoeffDecoder(nodes['RNNModule']))

    return nodes

  def Shape_formation(self, bfm_coeffs):
    idBase = tf.convert_to_tensor(self.facemodel.idBase, dtype=tf.float32)
    exBase = tf.convert_to_tensor(self.facemodel.exBase, dtype=tf.float32)
    meanshape = tf.convert_to_tensor(self.facemodel.meanshape, dtype=tf.float32)
    face_shape = tf.einsum('ij,aj->ai', idBase, bfm_coeffs[:, :80]) + \
                 tf.einsum('ij,aj->ai', exBase, bfm_coeffs[:, 80:144]) + \
                 meanshape

    face_shape = tf.reshape(face_shape, [1, -1, 3])
    # re-center face shape
    face_shape = face_shape - tf.reduce_mean(tf.reshape(meanshape, [1, -1, 3]), axis=1, keepdims=True)

    return face_shape

  def add_cost_function(self, output_bfm_coeffs, bfm_coeffs, seq_len):
    # bfm_coeff_mask = tf.sequence_mask(seq_len, tf.keras.backend.max(seq_len), dtype=tf.float32)
    # bfm_coeff_diff = tf.math.reduce_sum(tf.math.square(bfm_coeffs[:, :, 80:144] - output_bfm_coeffs[:, :, :]), axis=-1)
    # loss_coeff = tf.math.reduce_mean(tf.math.reduce_sum(bfm_coeff_diff * bfm_coeff_mask, axis=-1))  # frame diff

    # video_mask = tf.sequence_mask(seq_len - 1, tf.keras.backend.max(seq_len) - 1,
    #                               dtype=tf.float32)  # shorter of 1 to landmark_mask
    # video_diff = (output_bfm_coeffs[:, 1:, :] - output_bfm_coeffs[:, :-1, :]) - (
    #     bfm_coeffs[:, 1:, 80:144] - bfm_coeffs[:, :-1, 80:144])
    # video_diff = tf.math.reduce_sum(tf.math.square(video_diff), axis=-1)
    # video_loss = tf.math.reduce_mean(tf.math.reduce_sum(video_diff * video_mask, axis=-1))  # video diff

    output_bfm_coeffs = tf.concat([bfm_coeffs[:, :, :80], output_bfm_coeffs], -1)
    output_bfm_coeffs = tf.reshape(output_bfm_coeffs, [-1, output_bfm_coeffs.shape[-1]])

    bfm_coeffs = tf.reshape(bfm_coeffs, [-1, bfm_coeffs.shape[-1]])
    face_shape = self.Shape_formation(bfm_coeffs)
    face_shape = tf.reshape(face_shape, [self.batch_size, -1, 35709*3])

    output_bfm_coeffs = tf.reshape(output_bfm_coeffs, [-1, output_bfm_coeffs.shape[-1]])
    output_face_shape = self.Shape_formation(output_bfm_coeffs)
    output_face_shape = tf.reshape(output_face_shape, [self.batch_size, -1, 35709*3])

    vertice_mask = tf.tile(tf.expand_dims(self.mouth_mask, 0), (tf.keras.backend.max(seq_len), 1, 1))
    vertice_mask = tf.tile(tf.expand_dims(vertice_mask, 0), (self.batch_size, 1, 1, 1))
    vertice_mask = tf.reshape(vertice_mask, [self.batch_size, -1, 35709*3])

    bfm_coeff_mask = tf.sequence_mask(seq_len, tf.keras.backend.max(seq_len), dtype=tf.float32)
    bfm_coeff_diff = tf.math.reduce_sum(tf.math.square(face_shape - output_face_shape)*vertice_mask, axis=-1)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(bfm_coeff_diff * bfm_coeff_mask, axis=-1))  # frame diff

    video_mask = tf.sequence_mask(seq_len - 1, tf.keras.backend.max(seq_len) - 1,
                                  dtype=tf.float32)  # shorter of 1 to landmark_mask
    video_diff = (output_face_shape[:, 1:, :] - output_face_shape[:, :-1, :]) - (
        face_shape[:, 1:, :] - face_shape[:, :-1, :])
    video_diff = tf.math.reduce_sum(tf.math.square(video_diff)*vertice_mask[:,:-1,:], axis=-1)
    video_loss = tf.math.reduce_mean(tf.math.reduce_sum(video_diff * video_mask, axis=-1))  # video diff

    loss += video_loss
    loss += tf.losses.get_regularization_loss()
    return loss

  def build_eval_op(self, bfm_coeff_seq, mfccs, seq_len):
    nodes = {}
    nodes.update({'BFM_coeff_seq': bfm_coeff_seq})
    nodes.update({'Mfccs': mfccs})
    nodes.update({'Seq_len': seq_len})

    ## Only preserve the identity[80] and expression[64] coefficients
    # bfm_coeff_seq = bfm_coeff_seq[:, :, 80:144]

    network_dict = self.build_network(mfccs, seq_len, trainable=False)
    nodes.update(network_dict)

    loss = self.add_cost_function(nodes['BFMCoeffDecoder'], bfm_coeff_seq, seq_len)
    nodes.update({'Loss': loss})

    return nodes

  def build_train_op(self, bfm_coeff_seq, mfccs, seq_len):
    nodes = {}
    nodes.update({'BFM_coeff_seq': bfm_coeff_seq})
    nodes.update({'Mfccs': mfccs})
    nodes.update({'Seq_len': seq_len})

    ## Only preserve the identity[80] and expression[64] coefficients
    # bfm_coeff_seq = bfm_coeff_seq[:, :, 80:144]

    network_dict = self.build_network(mfccs, seq_len, trainable=True)
    nodes.update(network_dict)

    loss = self.add_cost_function(nodes['BFMCoeffDecoder'], bfm_coeff_seq, seq_len)
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

  def build_inference_op(self, mfccs, seq_len):
    nodes = {}
    nodes.update({'Mfccs': mfccs})

    network_dict = self.build_network(mfccs, seq_len, trainable=False)
    nodes.update(network_dict)

    return nodes
