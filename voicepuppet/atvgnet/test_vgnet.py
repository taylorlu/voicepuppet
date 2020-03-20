#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Test for ATNet architectures."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import random
from vgnet import VGNet


class ArchitectureTest(tf.test.TestCase):

  def testVGNet(self):
    config_path = 'config/params.yml'
    with tf.Graph().as_default():
      img_size = 128

      ### VGNet setting
      vgnet = VGNet(config_path)
      params = vgnet.params
      params.batch_size = 2
      vgnet.set_params(params)

      seq_len = np.random.uniform(1, 100, params.batch_size).astype(np.int32)
      time = max(seq_len)

      ## real_landmark_seq: [batch_size, time, 68*2]
      real_landmark_seq = tf.random.uniform([params.batch_size, time, params.landmark_size], minval=-1, maxval=1,
                                            dtype=tf.float32)
      ## real_mask_seq: [batch_size, time, img_size, img_size, 1]
      real_mask_seq = tf.random.uniform([params.batch_size, time, img_size, img_size, 1], minval=0, maxval=1,
                                        dtype=tf.float32)
      ## real_img_seq: [batch_size, time, img_size, img_size, 3]
      real_img_seq = tf.random.uniform([params.batch_size, time, img_size, img_size, 3], minval=-1, maxval=1,
                                       dtype=tf.float32)
      ## example_landmark: [batch_size, 68*2]
      example_landmark = tf.random.uniform([params.batch_size, params.landmark_size], minval=-1, maxval=1,
                                           dtype=tf.float32)
      ## example_img: [batch_size, img_size, img_size, 3]
      example_img = tf.random.uniform([params.batch_size, img_size, img_size, 3], minval=-1, maxval=1, dtype=tf.float32)
      ## seq_len: [batch_size], in rational size
      seq_len = tf.convert_to_tensor(seq_len, dtype=tf.int32)

      def check_nodes(nodes):
        ## Test input tensors' shape
        self.assertAllEqual(nodes['Real_landmark_seq'].shape, real_landmark_seq.shape.as_list())
        self.assertAllEqual(nodes['Real_mask_seq'].shape, real_mask_seq.shape.as_list())
        self.assertAllEqual(nodes['Real_img_seq'].shape, real_img_seq.shape.as_list())
        self.assertAllEqual(nodes['Example_landmark'].shape, example_landmark.shape.as_list())
        self.assertAllEqual(nodes['Example_img'].shape, example_img.shape.as_list())
        self.assertAllEqual(nodes['Seq_len'].shape, seq_len.shape.as_list())

        ## Test Discriminator tensors' shape
        self.assertAllEqual(nodes['Discriminator']['Real_node']['Discriminator']['Decision'].shape, [params.batch_size])
        self.assertAllEqual(nodes['Discriminator']['Real_node']['Discriminator']['LandmarkSeq'].shape,
                            [params.batch_size, time, params.landmark_size])
        self.assertAllEqual(nodes['Discriminator']['Fake_node']['Discriminator']['Decision'].shape,
                            [params.batch_size])
        self.assertAllEqual(nodes['Discriminator']['Fake_node']['Discriminator']['LandmarkSeq'].shape,
                            [params.batch_size, time, params.landmark_size])
        self.assertAllEqual(nodes['Discriminator']['Generator_node']['Generator']['Color'].shape,
                            [params.batch_size, time, img_size, img_size, 3])
        self.assertAllEqual(nodes['Discriminator']['Generator_node']['Generator']['Attention'].shape,
                            [params.batch_size, time, img_size, img_size, 1])
        self.assertAllEqual(nodes['Discriminator']['Generator_node']['Generator']['Feature'].shape,
                            [params.batch_size, time, img_size, img_size, 3])

        ## Test Generator tensors' shape
        self.assertAllEqual(nodes['Generator']['Discriminator_node']['Discriminator']['Decision'].shape,
                            [params.batch_size])
        self.assertAllEqual(nodes['Generator']['Discriminator_node']['Discriminator']['LandmarkSeq'].shape,
                            [params.batch_size, time, params.landmark_size])
        self.assertAllEqual(nodes['Generator']['Generator_node']['Generator']['Color'].shape,
                            [params.batch_size, time, img_size, img_size, 3])
        self.assertAllEqual(nodes['Generator']['Generator_node']['Generator']['Attention'].shape,
                            [params.batch_size, time, img_size, img_size, 1])
        self.assertAllEqual(nodes['Generator']['Generator_node']['Generator']['Feature'].shape,
                            [params.batch_size, time, img_size, img_size, 3])

        ## Test input tensors' value range
        self.assertAllGreaterEqual(nodes['Real_landmark_seq'], -1)
        self.assertAllLessEqual(nodes['Real_landmark_seq'], 1)
        self.assertAllGreaterEqual(nodes['Real_mask_seq'], 0)
        self.assertAllLessEqual(nodes['Real_mask_seq'], 1)
        self.assertAllGreaterEqual(nodes['Real_img_seq'], -1)
        self.assertAllLessEqual(nodes['Real_img_seq'], 1)
        self.assertAllGreaterEqual(nodes['Example_landmark'], -1)
        self.assertAllLessEqual(nodes['Example_landmark'], 1)
        self.assertAllGreaterEqual(nodes['Example_img'], -1)
        self.assertAllLessEqual(nodes['Example_img'], 1)
        self.assertAllGreaterEqual(nodes['Seq_len'], 1)
        self.assertAllLessEqual(nodes['Seq_len'], time)

        ## Test Discriminator tensors' value range
        self.assertAllGreaterEqual(nodes['Discriminator']['Real_node']['Discriminator']['Decision'], 0)
        self.assertAllLessEqual(nodes['Discriminator']['Real_node']['Discriminator']['Decision'], 1)
        self.assertAllGreaterEqual(nodes['Discriminator']['Real_node']['Discriminator']['LandmarkSeq'], -2)
        self.assertAllLessEqual(nodes['Discriminator']['Real_node']['Discriminator']['LandmarkSeq'], 2)
        self.assertAllGreaterEqual(nodes['Discriminator']['Fake_node']['Discriminator']['Decision'], 0)
        self.assertAllLessEqual(nodes['Discriminator']['Fake_node']['Discriminator']['Decision'], 1)
        self.assertAllGreaterEqual(nodes['Discriminator']['Fake_node']['Discriminator']['LandmarkSeq'], -2)
        self.assertAllLessEqual(nodes['Discriminator']['Fake_node']['Discriminator']['LandmarkSeq'], 2)
        self.assertAllGreaterEqual(nodes['Discriminator']['Generator_node']['Generator']['Color'], -1)
        self.assertAllLessEqual(nodes['Discriminator']['Generator_node']['Generator']['Color'], 1)
        self.assertAllGreaterEqual(nodes['Discriminator']['Generator_node']['Generator']['Attention'], 0)
        self.assertAllLessEqual(nodes['Discriminator']['Generator_node']['Generator']['Attention'], 1)
        self.assertAllGreaterEqual(nodes['Discriminator']['Generator_node']['Generator']['Feature'], -1)
        self.assertAllLessEqual(nodes['Discriminator']['Generator_node']['Generator']['Feature'], 1)

        ## Test Generator tensors' value range
        self.assertAllGreaterEqual(nodes['Generator']['Discriminator_node']['Discriminator']['Decision'], 0)
        self.assertAllLessEqual(nodes['Generator']['Discriminator_node']['Discriminator']['Decision'], 1)
        self.assertAllGreaterEqual(nodes['Generator']['Discriminator_node']['Discriminator']['LandmarkSeq'], -2)
        self.assertAllLessEqual(nodes['Generator']['Discriminator_node']['Discriminator']['LandmarkSeq'], 2)
        self.assertAllGreaterEqual(nodes['Generator']['Generator_node']['Generator']['Color'], -1)
        self.assertAllLessEqual(nodes['Generator']['Generator_node']['Generator']['Color'], 1)
        self.assertAllGreaterEqual(nodes['Generator']['Generator_node']['Generator']['Attention'], 0)
        self.assertAllLessEqual(nodes['Generator']['Generator_node']['Generator']['Attention'], 1)
        self.assertAllGreaterEqual(nodes['Generator']['Generator_node']['Generator']['Feature'], -1)
        self.assertAllLessEqual(nodes['Generator']['Generator_node']['Generator']['Feature'], 1)

      def walkDict(aDict, key_list, value_list, path=()):
        ## visit the nodes dict into key and value list, while keep the hierarchy
        for k in aDict:
          if type(aDict[k]) != dict:
            if ('_grads' in k or '_tvars' in k):
              continue
            key_list.append(path + (k,))
            value_list.append(aDict[k])
          else:
            walkDict(aDict[k], key_list, value_list, path + (k,))

      ################## 1. Test train stage ##################
      nodes = vgnet.build_train_op(real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img,
                                   seq_len)

      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        ## visit the nodes dict into key and value list, while keep the hierarchy
        key_list = []
        value_list = []
        walkDict(nodes, key_list, value_list)

        result = sess.run(value_list)

        ## replace the tensor in nodes by numpy matrix after sess.run
        for i, tensor in enumerate(result):
          node = nodes
          for key in key_list[i]:
            node = node[key]
          node = tensor

        ## test the nodes' shapes and values
        check_nodes(nodes)

      ################## 2. Test evaluate stage ##################
      nodes = vgnet.build_eval_op(real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img,
                                  seq_len)
      with self.session() as sess:
        sess.run(tf.global_variables_initializer())
        ## visit the nodes dict into key and value list, while keep the hierarchy
        key_list = []
        value_list = []
        walkDict(nodes, key_list, value_list)

        result = sess.run(value_list)

        ## replace the tensor in nodes by numpy matrix after sess.run
        for i, tensor in enumerate(result):
          node = nodes
          for key in key_list[i]:
            node = node[key]
          node = tensor

        ## test the nodes' shapes and values
        check_nodes(nodes)


if (__name__ == '__main__'):
  tf.test.main()
