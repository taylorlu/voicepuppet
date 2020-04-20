#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
import subprocess
from pixrefer import PixReferNet
from voicepuppet.bfmnet.bfmnet import BFMNet
from generator.loader import *
from generator.generator import DataGenerator
from utils.bfm_load_data import *
from utils.bfm_visual import *
from utils.utils import *
import scipy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if (__name__ == '__main__'):

  cmd_parser = OptionParser(usage="usage: %prog [options] --config_path <>")
  cmd_parser.add_option('--config_path', type="string", dest="config_path",
                        help='the config yaml file')

  opts, argv = cmd_parser.parse_args()

  if (opts.config_path is None):
    logger.error('Please check your parameters.')
    exit(0)

  config_path = opts.config_path

  if (not os.path.exists(config_path)):
    logger.error('config_path not exists')
    exit(0)

  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  mkdir('output')
  for file in os.listdir('output'):
    os.system('rm -rf output/{}'.format(file))

  batch_size = 1
  img_size = 512
  image_loader = ImageLoader()
  root = '/media/dong/DiskData/gridcorpus/todir_vid2vid/vid1/05'
  bg_img = cv2.resize(cv2.imread('/home/dong/Downloads/bg.jpg'), (img_size, img_size)).astype(np.float32)/255.0

  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    with tf.variable_scope('recognition'):
      ### Vid2VidNet setting
      vid2vidnet = PixReferNet(config_path)
      params = vid2vidnet.params
      params.batch_size = 1
      params.add_hparam('is_training', False)
      vid2vidnet.set_params(params)

      inputs_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 6])
      fg_inputs_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
      targets_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
      vid2vid_nodes = vid2vidnet.build_inference_op(inputs_holder, fg_inputs_holder, targets_holder)

    variables_to_restore = tf.global_variables()
    rec_varlist = {v.name[12:][:-2]: v 
                            for v in variables_to_restore if v.name[:11]=='recognition'}

    rec_saver = tf.train.Saver(var_list=rec_varlist)

    sess.run(tf.global_variables_initializer())
    rec_saver.restore(sess, 'ckpt_pixrefer/pixrefernet-20000')

    inputs = np.zeros([1, img_size, img_size, 6], dtype=np.float32)
    fg_inputs = np.zeros([1, img_size, img_size, 3], dtype=np.float32)

    img = image_loader.get_data(os.path.join(root, '{}.jpg'.format(0)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs[0, :, :, 0:3] = img[:, img_size:img_size*2, :]
    fg_inputs[0, :, :, 0:3] = img[:, :img_size, :] * img[:, img_size*2:, :]

    for index in range(4, 195):
      img = image_loader.get_data(os.path.join(root, '{}.jpg'.format(index)))
      if (img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs[0, ..., 3:6] = img[:, img_size:img_size*2, :]

        frames, last = sess.run([vid2vid_nodes['Outputs'], vid2vid_nodes['Outputs_FG']], 
          feed_dict={inputs_holder: inputs, fg_inputs_holder: fg_inputs, targets_holder: bg_img[np.newaxis, ...]})

        cv2.imwrite('output/_{}.jpg'.format(index), cv2.cvtColor((frames[0,...]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        # cv2.imshow('', last[0, ...])
        # cv2.waitKey(0)


  #     cv2.imwrite('output/_{}.jpg'.format(i), cv2.cvtColor((frames[0,...]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

  # cmd = 'ffmpeg -i output/_%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp2.mp4'
  # subprocess.call(cmd, shell=True)

  # cmd = 'ffmpeg -i output/%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp.mp4'
  # subprocess.call(cmd, shell=True)
