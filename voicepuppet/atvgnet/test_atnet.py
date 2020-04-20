#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Test for ATNet architectures."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from optparse import OptionParser
import tensorflow as tf
import numpy as np
import os
from atnet import ATNet
from tinynet import MfccNet


class ArchitectureTest(tf.test.TestCase):

  def testATNet(self):
    config_path = 'config/params.yml'
    with tf.Graph().as_default():
      time = 100

      ### ATNet setting
      atnet = ATNet(config_path)
      params = atnet.params
      params.batch_size = 2
      atnet.set_params(params)

      seq_len = np.random.uniform(1, 100, params.batch_size).astype(np.int32)
      time = max(seq_len)

      ## landmark: [batch_size, time, 68*2]
      landmark = tf.random.uniform([params.batch_size, time, params.landmark_size], minval=-1, maxval=1,
                                   dtype=tf.float32)
      ## ears: [batch_size, 1]
      ears = tf.random.uniform([params.batch_size, time, 1], minval=0, maxval=1, dtype=tf.float32)
      ## poses: [batch_size, 3]
      poses = tf.random.uniform([params.batch_size, time, 3], minval=-1, maxval=1, dtype=tf.float32)
      ## mfccs: [batch_size, time*frame_mfcc_scale, num_mel_bins]
      mfccs = tf.random.uniform([params.batch_size, time * 5, 80], dtype=tf.float32)
      ## example_landmark: [batch_size, 68*2]
      example_landmark = tf.random.uniform([params.batch_size, params.landmark_size], minval=-1, maxval=1,
                                           dtype=tf.float32)
      ## seq_len: [batch_size], in rational size
      seq_len = tf.convert_to_tensor(seq_len, dtype=tf.int32)

      def check_nodes(nodes):
        ## Test input tensor
        self.assertAllEqual(nodes['Landmark'].shape, landmark.shape.as_list())
        self.assertAllEqual(nodes['Ears'].shape, ears.shape.as_list())
        self.assertAllEqual(nodes['Poses'].shape, poses.shape.as_list())
        self.assertAllEqual(nodes['Mfccs'].shape, mfccs.shape.as_list())
        self.assertAllEqual(nodes['Example_landmark'].shape, example_landmark.shape.as_list())
        self.assertAllEqual(nodes['Seq_len'].shape, seq_len.shape.as_list())

        ## Test MfccEncoder output tensor
        self.assertAllEqual(nodes['MfccEncoder'].shape, [params.batch_size, time, params.encode_embedding_size])
        ## Test LandmarkEncoder output tensor
        self.assertAllEqual(nodes['LandmarkEncoder'].shape, [params.batch_size, time, params.encode_embedding_size])
        ## Test PoseEncoder output tensor
        self.assertAllEqual(nodes['PoseEncoder'].shape, [params.batch_size, time, params.encode_embedding_size])
        ## Test RNNModule output tensor
        self.assertAllEqual(nodes['RNNModule'].shape, [params.batch_size, time, params.rnn_hidden_size])
        ## Test LandmarkDecoder output tensor
        self.assertAllEqual(nodes['LandmarkDecoder'].shape, [params.batch_size, time, params.landmark_size])

        ## Test LandmarkDecoder output value range
        self.assertAllGreaterEqual(nodes['LandmarkDecoder'], -2)
        self.assertAllLessEqual(nodes['LandmarkDecoder'], 2)

      ################## 1. Test train stage ##################
      nodes = atnet.build_train_op(landmark, ears, poses, mfccs, example_landmark, seq_len)
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run([nodes['Landmark'], nodes['Ears'], nodes['Poses'], nodes['Mfccs'], nodes['Example_landmark'],
                           nodes['Seq_len'], nodes['MfccEncoder'], nodes['LandmarkEncoder'], nodes['PoseEncoder'],
                           nodes['RNNModule'], nodes['LandmarkDecoder']])

        nodes = {}
        nodes.update({'Landmark': result[0]})
        nodes.update({'Ears': result[1]})
        nodes.update({'Poses': result[2]})
        nodes.update({'Mfccs': result[3]})
        nodes.update({'Example_landmark': result[4]})
        nodes.update({'Seq_len': result[5]})
        nodes.update({'MfccEncoder': result[6]})
        nodes.update({'LandmarkEncoder': result[7]})
        nodes.update({'PoseEncoder': result[8]})
        nodes.update({'RNNModule': result[9]})
        nodes.update({'LandmarkDecoder': result[10]})
        check_nodes(nodes)

      ################## 2. Test evaluate stage ##################
      nodes = atnet.build_eval_op(landmark, ears, poses, mfccs, example_landmark, seq_len)
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run([nodes['Landmark'], nodes['Ears'], nodes['Poses'], nodes['Mfccs'], nodes['Example_landmark'],
                           nodes['Seq_len'], nodes['MfccEncoder'], nodes['LandmarkEncoder'], nodes['PoseEncoder'],
                           nodes['RNNModule'], nodes['LandmarkDecoder']])

        nodes = {}
        nodes.update({'Landmark': result[0]})
        nodes.update({'Ears': result[1]})
        nodes.update({'Poses': result[2]})
        nodes.update({'Mfccs': result[3]})
        nodes.update({'Example_landmark': result[4]})
        nodes.update({'Seq_len': result[5]})
        nodes.update({'MfccEncoder': result[6]})
        nodes.update({'LandmarkEncoder': result[7]})
        nodes.update({'PoseEncoder': result[8]})
        nodes.update({'RNNModule': result[9]})
        nodes.update({'LandmarkDecoder': result[10]})
        check_nodes(nodes)


if (__name__ == '__main__'):
  tf.test.main()
