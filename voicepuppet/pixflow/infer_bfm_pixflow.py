#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
import subprocess
from pixflow import PixFlowNet
from voicepuppet.bfmnet.bfmnet import BFMNet
from generator.loader import *
from generator.generator import DataGenerator
from utils.bfm_load_data import *
from utils.bfm_visual import *
from utils.utils import *
import scipy
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def alignto_bfm_coeff(model_dir, img, xys):
  from PIL import Image
  import tensorflow as tf

  def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    return graph_def

  # read standard landmarks for preprocessing images
  lm3D = load_lm3d(model_dir)

  # build reconstruction model
  with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
    images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
    graph_def = load_graph(os.path.join(model_dir, "FaceReconModel.pb"))
    tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

    # output coefficients of R-Net (dim = 257) 
    coeff = graph.get_tensor_by_name('resnet/coeff:0')

    with tf.Session() as sess:
      ps = map(lambda x: int(x), xys)

      left_eye_x = int(round((ps[72] + ps[74] + ps[76] + ps[78] + ps[80] + ps[82]) / 6))
      left_eye_y = int(round((ps[73] + ps[75] + ps[77] + ps[79] + ps[81] + ps[83]) / 6))
      right_eye_x = int(round((ps[84] + ps[86] + ps[88] + ps[90] + ps[92] + ps[94]) / 6))
      right_eye_y = int(round((ps[85] + ps[87] + ps[89] + ps[91] + ps[93] + ps[95]) / 6))
      nose_x = int(round(ps[60]))
      nose_y = int(round(ps[61]))
      left_mouse_x = int(round(ps[96]))
      left_mouse_y = int(round(ps[97]))
      right_mouse_x = int(round(ps[108]))
      right_mouse_y = int(round(ps[109]))

      lmk5 = np.array(
          [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [nose_x, nose_y], [left_mouse_x, left_mouse_y],
           [right_mouse_x, right_mouse_y]])

      image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      # preprocess input image
      input_img, lm_new, transform_params = Preprocess(image, lmk5, lm3D)
      bfmcoeff = sess.run(coeff, feed_dict={images: input_img})
      return bfmcoeff, input_img, transform_params

angles = np.array([[0, 0, 0]], dtype=np.float32)
shift = 0.005

def render_face(center_x, center_y, ratio, bfmcoeff, img, transform_params, facemodel):
  ratio *= transform_params[2]
  tx = -int((transform_params[3] / ratio))
  ty = -int((transform_params[4] / ratio))
  global angles, shift

  # angles[0][0] += shift
  # angles[0][1] += shift
  # angles[0][2] += shift
  # if (angles[0][1] > 0.03 or angles[0][1] < -0.03):
  #   shift = -shift

  face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d = Reconstruction_rotation(
    bfmcoeff, facemodel, angles)
  face_projection2 = np.concatenate([face_projection, z_buffer], axis=2)
  face_projection = np.squeeze(face_projection2, (0))

  shape = np.squeeze(face_projection2, (0))
  color = np.squeeze(face_color, (0))
  color = np.clip(color, 0, 255).astype(np.int32)

  new_image = np.zeros((224 * 224 * 3), dtype=np.uint8)
  face_mask = np.zeros((224 * 224), dtype=np.uint8)

  vertices = shape.reshape(-1).astype(np.float32).copy()
  triangles = (facemodel.tri - 1).reshape(-1).astype(np.int32).copy()
  colors = color.reshape(-1).astype(np.float32).copy()
  depth_buffer = (np.zeros((224 * 224)) - 99999.0).astype(np.float32)
  mesh_core_cython.render_colors_core(new_image, face_mask, vertices, triangles, colors, depth_buffer,
                                      facemodel.tri.shape[0], 224, 224, 3)
  new_image = new_image.reshape([224, 224, 3])

  new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
  new_image = cv2.resize(new_image, (
  int(round(new_image.shape[0] / ratio)), int(round(new_image.shape[1] / ratio))))

  back_new_image = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
  center_face_x = new_image.shape[1] // 2
  center_face_y = new_image.shape[0] // 2

  ry = center_y - center_face_y + new_image.shape[0] - ty
  rx = center_x - center_face_x + new_image.shape[1] - tx
  back_new_image[center_y - center_face_y - ty:ry, center_x - center_face_x - tx:rx, :] = new_image
  return back_new_image


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

  image_file, audio_file = argv

  mkdir('output')
  for file in os.listdir('output'):
    os.system('rm -rf output/{}'.format(file))

  batch_size = 1
  ### Generator for inference setting
  infer_generator = DataGenerator(config_path)
  params = infer_generator.params
  params.batch_size = batch_size
  infer_generator.set_params(params)
  wav_loader = WavLoader(sr=infer_generator.sample_rate)
  pcm = wav_loader.get_data(audio_file)
  facemodel = BFM(params.pretrain_dir)

  pad_len = int(1 + pcm.shape[0] / infer_generator.frame_wav_scale)
  # calculate the rational length of pcm in order to keep the alignment of mfcc and landmark sequence.
  pcm_length = infer_generator.hop_step * (pad_len * infer_generator.frame_mfcc_scale - 1) + infer_generator.win_length
  if (pcm.shape[0] < pcm_length):
    pcm = np.pad(pcm, (0, pcm_length - pcm.shape[0]), 'constant', constant_values=(0))
  pcm_slice = pcm[:pcm_length][np.newaxis, :]

  mfcc = infer_generator.extract_mfcc(pcm_slice)
  img_size = 512
  img = cv2.imread(image_file)[:, :512, :]
  img, img_landmarks, img_cropped, lmk_cropped, center_x, center_y, ratio = get_mxnet_sat_alignment(params.pretrain_dir, img)
  bfmcoeff, input_img, transform_params = alignto_bfm_coeff(params.pretrain_dir, img_cropped, lmk_cropped)

  img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
  face3d_refer = img[:, 512:512*2, :]
  fg_refer = img[:, :512, :] * img[:, 512*2:, :]
  img = img[:, :512, :]

  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    seq_len = tf.convert_to_tensor([pad_len], dtype=tf.int32)
    ear = np.random.rand(1, pad_len, 1).astype(np.float32)/100
    ear = tf.convert_to_tensor(ear, dtype=tf.float32)

    with tf.variable_scope('localization'):
      ### BFMNet setting
      bfmnet = BFMNet(config_path)
      params = bfmnet.params
      params.batch_size = 1
      bfmnet.set_params(params)

      bfmnet_nodes = bfmnet.build_inference_op(ear, mfcc, seq_len)

    with tf.variable_scope('recognition'):
      ### Vid2VidNet setting
      vid2vidnet = PixFlowNet(config_path)
      params = vid2vidnet.params
      params.batch_size = 1
      vid2vidnet.set_params(params)

      inputs_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 6])
      targets_holder = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 6])
      vid2vid_nodes = vid2vidnet.build_inference_op(inputs_holder, targets_holder)

    variables_to_restore = tf.global_variables()
    loc_varlist = {v.name[13:][:-2]: v 
                            for v in variables_to_restore if v.name[:12]=='localization'}
    rec_varlist = {v.name[12:][:-2]: v 
                            for v in variables_to_restore if v.name[:11]=='recognition'}

    loc_saver = tf.train.Saver(var_list=loc_varlist)
    rec_saver = tf.train.Saver(var_list=rec_varlist)

    sess.run(tf.global_variables_initializer())
    loc_saver.restore(sess, 'ckpt_bfmnet_new3/bfmnet-40000')
    rec_saver.restore(sess, 'ckpt_pixflow3/pixflownet-50000')

    ### Run inference
    bfm_coeff_seq = sess.run(bfmnet_nodes['BFMCoeffDecoder'])
    bfmcoeff = np.tile(bfmcoeff[:, np.newaxis, :], [1, bfm_coeff_seq.shape[1], 1])

    bfm_coeff_seq = np.concatenate([bfmcoeff[:, :, :80], bfm_coeff_seq[:, :, :], bfmcoeff[:, :, 144:]], axis=2)
    
    inputs = np.zeros([1, img_size, img_size, 6], dtype=np.float32)
    inputs[0, ..., 0:3] = face3d_refer

    for i in range(bfm_coeff_seq.shape[1]):
      face3d = render_face(center_x+random.randint(0, 0), center_y+random.randint(0, 0), ratio, bfm_coeff_seq[0, i:i + 1, ...], img, transform_params, facemodel)
      # cv2.imwrite('output/{}.jpg'.format(i), face3d)
      face3d = cv2.cvtColor(face3d, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

      inputs[0, ..., 0:3] = face3d

      bg_img = np.zeros([1, img_size, img_size, 6], dtype=np.float32)
      bg_img[0, ..., :3] = cv2.resize(cv2.imread('background/{}.jpg'.format(i+1)), (img_size, img_size)).astype(np.float32)/255.0
      bg_img[0, ..., 3:] = bg_img[0, ..., :3]

      # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
      frames = sess.run(vid2vid_nodes['Outputs'], 
        feed_dict={inputs_holder: inputs, targets_holder: bg_img})

      cv2.imwrite('output/{}.jpg'.format(i), cv2.cvtColor((frames[0,..., :3]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    # image_loader = ImageLoader()
    # for index in range(4, 195):
    #   img = image_loader.get_data(os.path.join('/media/dong/DiskData/gridcorpus/todir_vid2vid/vid1/05', '{}.jpg'.format(index)))
    #   face3d = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:, img_size:img_size*2, :]

    #   inputs[0, ..., 3:6] = inputs[0, ..., 6:9]
    #   inputs[0, ..., 6:9] = face3d

    #   frames, last = sess.run([vid2vid_nodes['Outputs'], vid2vid_nodes['Outputs_FG']], 
    #     feed_dict={inputs_holder: inputs, fg_inputs_holder: fg_inputs, targets_holder: np.tile(bg_img, (1, 1, 3))[np.newaxis, ...]})
    #   fg_inputs[0, ..., 3:6] = last

    #   cv2.imwrite('output/{}.jpg'.format(index), cv2.cvtColor((last[0,...]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
