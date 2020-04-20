#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from vgnet import VGNet
from dataset.generator import VGNetDataGenerator
from optparse import OptionParser
import logging
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

  os.environ["CUDA_VISIBLE_DEVICES"] = '2'

  batch_size = 4
  ### Generator for training setting
  train_generator = VGNetDataGenerator(config_path)
  params = train_generator.params
  params.dataset_path = params.train_dataset_path
  params.batch_size = batch_size
  train_generator.set_params(params)
  train_dataset = train_generator.get_dataset()

  ### Generator for evaluation setting
  eval_generator = VGNetDataGenerator(config_path)
  params = eval_generator.params
  params.dataset_path = params.eval_dataset_path
  params.batch_size = batch_size
  eval_generator.set_params(params)
  eval_dataset = eval_generator.get_dataset()

  sess = tf.Session()
  tf.train.start_queue_runners(sess=sess)

  train_iter = train_dataset.make_one_shot_iterator()
  eval_iter = eval_dataset.make_one_shot_iterator()

  ### VGNet setting
  vgnet = VGNet(config_path)
  params = vgnet.params
  epochs = params.training['epochs']
  params.add_hparam('max_to_keep', 10)
  params.add_hparam('save_dir', 'ckpt_vgnet')
  params.add_hparam('save_name', 'vgnet')
  params.add_hparam('save_step', 1000)
  params.add_hparam('eval_step', 1000)
  params.add_hparam('summary_step', 100)
  params.add_hparam('alternative', 1000)
  params.add_hparam('eval_visual_dir', 'log/eval_vgnet')
  params.add_hparam('summary_dir', 'log/summary_vgnet')
  params.batch_size = batch_size
  vgnet.set_params(params)
  mean = np.load(params.mean_file)

  mkdir(params.save_dir)
  mkdir(params.eval_visual_dir)
  mkdir(params.summary_dir)

  train_nodes = vgnet.build_train_op(*train_iter.get_next())
  eval_nodes = vgnet.build_eval_op(*eval_iter.get_next())
  sess.run(tf.global_variables_initializer())

  # Restore from save_dir
  if ('checkpoint' in os.listdir(params.save_dir)):
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(params.save_dir, latest_filename=None))

  # Add summary when training
  discriminator_summary = []
  discriminator_summary.append(tf.summary.scalar("real_bce_loss", train_nodes['Discriminator']['Real_bce_loss']))
  discriminator_summary.append(tf.summary.scalar("real_lmk_loss", train_nodes['Discriminator']['Real_lmk_loss']))
  discriminator_summary.append(tf.summary.scalar("fake_bce_loss", train_nodes['Discriminator']['Fake_bce_loss']))
  discriminator_summary.append(tf.summary.scalar("fake_lmk_loss", train_nodes['Discriminator']['Fake_lmk_loss']))
  discriminator_summary.append(
      tf.summary.scalar("discriminator_loss", train_nodes['Discriminator']['Discriminator_loss']))

  generator_summary = []
  generator_summary.append(tf.summary.scalar("bce_loss", train_nodes['Generator']['Bce_loss']))
  generator_summary.append(tf.summary.scalar("lmk_loss", train_nodes['Generator']['Lmk_loss']))
  generator_summary.append(tf.summary.scalar("pix_loss", train_nodes['Generator']['Pix_loss']))
  generator_summary.append(tf.summary.scalar("generator_loss", train_nodes['Generator']['Generator_loss']))

  # Add gradient to summary
  grads = train_nodes['Discriminator_grads']
  tvars = train_nodes['Discriminator_tvars']
  for i, grad in enumerate(grads):
    if grad is not None:
      var = tvars[i]
      if('BatchNorm' not in var.name):
        discriminator_summary.append(tf.summary.histogram(var.op.name + '/gradients', grad))

  grads = train_nodes['Generator_grads']
  tvars = train_nodes['Generator_tvars']
  for i, grad in enumerate(grads):
    if grad is not None:
      var = tvars[i]
      if('BatchNorm' not in var.name):
        generator_summary.append(tf.summary.histogram(var.op.name + '/gradients', grad))

  discriminator_summary_op = tf.summary.merge(discriminator_summary)
  generator_summary_op = tf.summary.merge(generator_summary)
  lr_summary_op = tf.summary.scalar("lr", train_nodes['Lr'])

  summary_writer = tf.summary.FileWriter(params.summary_dir, graph=sess.graph)

  # Run epoch
  for i in range(epochs):
    if ((i // params.alternative) % 2 == 0):
      ### Run discriminator training
      result = sess.run([train_nodes['Train_discriminator'],
                         discriminator_summary_op,
                         train_nodes['Lr'],
                         train_nodes['Global_step'],
                         train_nodes['Discriminator']['Real_bce_loss'],
                         train_nodes['Discriminator']['Real_lmk_loss'],
                         train_nodes['Discriminator']['Fake_bce_loss'],
                         train_nodes['Discriminator']['Fake_lmk_loss'],
                         train_nodes['Discriminator']['Discriminator_loss']])
      _, summary, lr, global_step, real_bce_loss, real_lmk_loss, fake_bce_loss, fake_lmk_loss, discriminator_loss = result
      print(
        'Step {}: Lr= {:.2e}, Discriminator_loss= {:.3f}, [Real_bce_loss= {:.3f}, Real_lmk_loss= {:.3f}, Fake_bce_loss= {:.3f}, Fake_lmk_loss= {:.3f}]'.format(
            global_step, lr, discriminator_loss, real_bce_loss, real_lmk_loss, fake_bce_loss, fake_lmk_loss))

    else:
      ### Run generator training
      result = sess.run([train_nodes['Train_generator'],
                         generator_summary_op,
                         train_nodes['Lr'],
                         train_nodes['Global_step'],
                         train_nodes['Generator']['Bce_loss'],
                         train_nodes['Generator']['Lmk_loss'],
                         train_nodes['Generator']['Pix_loss'],
                         train_nodes['Generator']['Generator_loss']])
      _, summary, lr, global_step, bce_loss, lmk_loss, pix_loss, generator_loss = result
      print(
      'Step {}: Lr= {:.2e}, Generator_loss= {:.3f}, [Bce_loss= {:.3f}, Lmk_loss= {:.3f}, Pix_loss= {:.3f}]'.format(
          global_step, lr,
          generator_loss,
          bce_loss, lmk_loss, pix_loss))

    if (global_step % params.summary_step == 0):
      summary_writer.add_summary(summary, global_step)

    ### Run evaluation
    if (global_step % params.eval_step == 0):
      result = sess.run([eval_nodes['Real_landmark_seq'],
                         eval_nodes['Real_mask_seq'],
                         eval_nodes['Real_img_seq'],
                         eval_nodes['Example_landmark'],
                         eval_nodes['Example_img'],
                         eval_nodes['Seq_len'],
                         eval_nodes['Generator']['Fake_img_seq'],
                         eval_nodes['Generator']['Attention'],
                         eval_nodes['Generator']['Generator_loss'],
                         eval_nodes['Discriminator']['Discriminator_loss']])
      real_landmark_seq, real_mask_seq, real_img_seq, example_landmark, example_img, seq_len, fake_img_seq, attention, generator_loss, discriminator_loss = result

      print('\r\nEvaluation >>> Generator_loss= {:.3f}, Discriminator_loss= {:.3f}'.format(generator_loss,
                                                                                           discriminator_loss))
      plot_image_seq(params.eval_visual_dir, global_step, mean, seq_len, real_landmark_seq, real_mask_seq, real_img_seq,
                     fake_img_seq, attention)

    ### Save checkpoint
    if (global_step % params.save_step == 0):
      tf.train.Saver(max_to_keep=params.max_to_keep, var_list=tf.global_variables()).save(sess,
                                                                                          os.path.join(params.save_dir,
                                                                                                       params.save_name),
                                                                                          global_step=global_step)
