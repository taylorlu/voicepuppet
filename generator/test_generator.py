import tensorflow as tf
import os
import numpy as np
from generator import ATNetDataGenerator
from generator import VGNetDataGenerator


class GeneratorTest(tf.test.TestCase):

  def testATNetGenerator(self):
    config_path = 'config/params.yml'
    batch_size = 2
    landmark_size = 136
    ### Generator for training setting
    generator = ATNetDataGenerator(config_path)
    params = generator.params
    params.dataset_path = params.train_dataset_path
    params.batch_size = batch_size
    generator.set_params(params)
    dataset = generator.get_dataset()

    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)

    iterator = dataset.make_one_shot_iterator()
    landmark, ears, poses, mfccs, example_landmark, seq_len = sess.run(iterator.get_next())

    frame_mfcc_scale = params.mel['sample_rate'] / params.frame_rate / params.mel['hop_step']

    assert (frame_mfcc_scale - int(frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

    ## Test seq_len value range
    self.assertAllGreaterEqual(seq_len, params.min_squence_len)
    self.assertAllLessEqual(seq_len, params.max_squence_len)

    max_seq_len = np.max(seq_len)

    ## Test seq_len shape, [batch_size]
    self.assertAllEqual(seq_len.shape, [params.batch_size])
    ## Test landmark shape, [batch_size, padding_time, landmark_size]
    self.assertAllEqual(landmark.shape, [params.batch_size, max_seq_len, landmark_size])
    ## Test ears shape, [batch_size, padding_time, 1]
    self.assertAllEqual(ears.shape, [params.batch_size, max_seq_len, 1])
    ## Test poses shape, [batch_size, padding_time, 3]
    self.assertAllEqual(poses.shape, [params.batch_size, max_seq_len, 3])
    ## Test mfccs shape, [batch_size, padding_time, num_mel_bins]
    self.assertAllEqual(mfccs.shape, [params.batch_size, max_seq_len * frame_mfcc_scale, params.mel['num_mel_bins']])
    ## Test example_landmark shape, [batch_size, landmark_size]
    self.assertAllEqual(example_landmark.shape, [params.batch_size, landmark_size])

    ## Test the range of value, landmark [-1, 1]
    self.assertAllGreaterEqual(landmark, -1)
    self.assertAllLessEqual(landmark, 1)
    self.assertAllGreaterEqual(example_landmark, -1)
    self.assertAllLessEqual(example_landmark, 1)

    ## Test the range of value, ears [0, 1]
    self.assertAllGreaterEqual(ears, 0)
    self.assertAllLessEqual(ears, 1)

    ## Test the range of value, poses [-1, 1]
    self.assertAllGreaterEqual(poses, -1)
    self.assertAllLessEqual(poses, 1)

  def testVGNetGenerator(self):
    config_path = 'config/params.yml'
    batch_size = 2
    landmark_size = 136
    ### Generator for training setting
    generator = VGNetDataGenerator(config_path)
    params = generator.params
    params.dataset_path = params.train_dataset_path
    params.batch_size = batch_size
    generator.set_params(params)
    dataset = generator.get_dataset()

    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)

    iterator = dataset.make_one_shot_iterator()
    real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img, seq_len = sess.run(
      iterator.get_next())

    ## Test seq_len value range
    self.assertAllGreaterEqual(seq_len, params.min_squence_len)
    self.assertAllLessEqual(seq_len, params.max_squence_len)

    max_seq_len = np.max(seq_len)

    ## Test seq_len shape, [batch_size]
    self.assertAllEqual(seq_len.shape, [params.batch_size])
    ## Test real_landmark_seq shape, [batch_size, padding_time, landmark_size]
    self.assertAllEqual(real_landmark_seq.shape, [params.batch_size, max_seq_len, landmark_size])
    ## Test real_mask_seq shape, [batch_size, padding_time, img_height, img_width, 1]
    self.assertAllEqual(real_mask_seq.shape, [params.batch_size, max_seq_len, params.img_size, params.img_size, 1])
    ## Test real_img_seq shape, [batch_size, padding_time, img_height, img_width, 3]
    self.assertAllEqual(real_img_seq.shape, [params.batch_size, max_seq_len, params.img_size, params.img_size, 3])
    ## Test example_landmark shape, [batch_size, 136]
    self.assertAllEqual(example_landmark.shape, [params.batch_size, landmark_size])
    ## Test example_img shape, [batch_size, img_height, img_width, 3]
    self.assertAllEqual(example_img.shape, [params.batch_size, params.img_size, params.img_size, 3])

    ## Test the range of value, real_landmark_seq [-1, 1]
    self.assertAllGreaterEqual(real_landmark_seq, -1)
    self.assertAllLessEqual(real_landmark_seq, 1)
    self.assertAllGreaterEqual(example_landmark, -1)
    self.assertAllLessEqual(example_landmark, 1)

    ## Test the range of value, real_mask_seq [0, 1]
    self.assertAllGreaterEqual(real_mask_seq, 0)
    self.assertAllLessEqual(real_mask_seq, 1)

    ## Test the range of value, real_img_seq [-1, 1]
    self.assertAllGreaterEqual(real_img_seq, -1)
    self.assertAllLessEqual(real_img_seq, 1)
    self.assertAllGreaterEqual(example_img, -1)
    self.assertAllLessEqual(example_img, 1)


if (__name__ == '__main__'):
  tf.test.main()
