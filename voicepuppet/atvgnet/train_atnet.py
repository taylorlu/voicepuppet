#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
from atnet import ATNet
from dataset.generator import ATNetDataGenerator
from plot import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mkdir(dirname):
  if not os.path.isdir(dirname):
    os.makedirs(dirname)


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

  os.environ["CUDA_VISIBLE_DEVICES"] = '1'

  batch_size = 16
  ### Generator for training setting
  train_generator = ATNetDataGenerator(config_path)
  params = train_generator.params
  params.dataset_path = params.train_dataset_path
  params.batch_size = batch_size
  train_generator.set_params(params)
  train_dataset = train_generator.get_dataset()

  ### Generator for evaluation setting
  eval_generator = ATNetDataGenerator(config_path)
  params = eval_generator.params
  params.dataset_path = params.eval_dataset_path
  params.batch_size = batch_size
  eval_generator.set_params(params)
  eval_dataset = eval_generator.get_dataset()

  sess = tf.Session()
  tf.train.start_queue_runners(sess=sess)

  train_iter = train_dataset.make_one_shot_iterator()
  eval_iter = eval_dataset.make_one_shot_iterator()

  ### ATNet setting
  atnet = ATNet(config_path)
  params = atnet.params
  epochs = params.training['epochs']
  params.add_hparam('max_to_keep', 10)
  params.add_hparam('save_dir', 'ckpt_atnet')
  params.add_hparam('save_name', 'atnet')
  params.add_hparam('save_step', 1000)
  params.add_hparam('eval_step', 1000)
  params.add_hparam('summary_step', 100)
  params.add_hparam('eval_visual_dir', 'log/eval_atnet')
  params.add_hparam('summary_dir', 'log/summary_atnet')
  params.batch_size = batch_size
  atnet.set_params(params)
  mean = np.load(params.mean_file)

  mkdir(params.save_dir)
  mkdir(params.eval_visual_dir)
  mkdir(params.summary_dir)

  train_nodes = atnet.build_train_op(*train_iter.get_next())
  eval_nodes = atnet.build_eval_op(*eval_iter.get_next())
  sess.run(tf.global_variables_initializer())

  # Restore from save_dir
  if ('checkpoint' in os.listdir(params.save_dir)):
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(params.save_dir))

  tf.summary.scalar("loss", train_nodes['Loss'])
  tf.summary.scalar("lr", train_nodes['Lr'])
  grads = train_nodes['Grads']
  tvars = train_nodes['Tvars']
  # Add histograms for gradients.
  for i, grad in enumerate(grads):
    if grad is not None:
      var = tvars[i]
      if ('BatchNorm' not in var.op.name):
        tf.summary.histogram(var.op.name + '/gradients', grad)

  merge_summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(params.summary_dir, graph=sess.graph)

  for i in range(epochs):
    ### Run training
    result = sess.run([train_nodes['Train_op'],
                       merge_summary_op,
                       train_nodes['Loss'],
                       train_nodes['Lr'],
                       train_nodes['Global_step'],
                       train_nodes['Mfccs'],
                       train_nodes['Poses'],
                       train_nodes['Ears'],
                       train_nodes['Seq_len'],
                       train_nodes['Landmark'],
                       train_nodes['Example_landmark']])
    _, summary, loss, lr, global_step, mfccs, poses, ears, seq_len, landmark, example_landmark = result
    print('Step {}: Loss= {:.3f}, Lr= {:.2e}'.format(global_step, loss, lr))

    if (global_step % params.summary_step == 0):
      summary_writer.add_summary(summary, global_step)

    ### Run evaluation
    if (global_step % params.eval_step == 0):
      result = sess.run([eval_nodes['Loss'],
                         eval_nodes['Seq_len'],
                         eval_nodes['Landmark'],
                         eval_nodes['LandmarkDecoder']])
      loss, seq_len, real_lmk_seq, lmk_seq = result

      print('\r\nEvaluation >>> Loss= {:.3f}'.format(loss))
      plot_lmk_seq(params.eval_visual_dir, global_step, mean, seq_len, real_lmk_seq, lmk_seq)

    ### Save checkpoint
    if (global_step % params.save_step == 0):
      tf.train.Saver(max_to_keep=params.max_to_keep, var_list=tf.global_variables()).save(sess,
                                                                                          os.path.join(params.save_dir,
                                                                                                       params.save_name),
                                                                                          global_step=global_step)
