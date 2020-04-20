#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
import subprocess
from generator.loader import *
from bfmnet import BFMNet
from generator.generator import DataGenerator
from utils.bfm_load_data import *
from utils.bfm_visual import *
from utils.utils import *
import scipy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# #########################################################################################################
# facemodel = BFM('../allmodels')
# def visual_3dface(root, name):
#   mkdir('output')
#   for file in os.listdir('output'):
#     os.system('rm -rf output/{}'.format(file))

#   bfmcoeff_loader = BFMCoeffLoader()
#   bfm_coeff_seq = bfmcoeff_loader.get_data(os.path.join(root, 'bfmcoeff.txt'))
#   audio_file = os.path.join(root, 'audio.wav')
#   id_coeff = np.mean(bfm_coeff_seq[:, :80], 0)

#   for i in range(bfm_coeff_seq.shape[0]):
#     bfm_coeff_seq[i, :80] = id_coeff

#   for i in range(bfm_coeff_seq.shape[0]):
#     face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, _ = Reconstruction(
#         bfm_coeff_seq[i:i + 1, ...], facemodel)
#     if(i>300):
#       break
#     shape = np.squeeze(face_shape, (0))
#     color = np.squeeze(face_color, (0))
#     color = np.clip(color, 0, 255).astype(np.int32)
#     shape[:, :2] = 112 - shape[:, :2] * 112
#     shape *=3

#     img_size = 672
#     new_image = np.zeros((img_size * img_size * 3), dtype=np.uint8)
#     face_mask = np.zeros((img_size * img_size), dtype=np.uint8)

#     vertices = shape.reshape(-1).astype(np.float32).copy()
#     triangles = (facemodel.tri - 1).reshape(-1).astype(np.int32).copy()
#     colors = color.reshape(-1).astype(np.float32).copy()
#     depth_buffer = (np.zeros((img_size * img_size)) - 99999.0).astype(np.float32)
#     mesh_core_cython.render_colors_core(new_image, face_mask, vertices, triangles, colors, depth_buffer,
#                                         facemodel.tri.shape[0], img_size, img_size, 3)
#     new_image = new_image.reshape([img_size, img_size, 3])

#     new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

#     cv2.imwrite('output/{}.jpg'.format(i), new_image)
#     print(i)

#   cmd = 'ffmpeg -i output/%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y {}'.format(name)
#   subprocess.call(cmd, shell=True)

# root = '/media/dong/DiskData/gridcorpus/todir/vid1'
# for folder in os.listdir(root):
#   name = os.path.join(root, folder+'.mp4')
#   visual_3dface(os.path.join(root, folder), name)
# sys.exit(0)
# #########################################################################################################

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

  image_file, audio_file = argv

  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
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

  pad_len = int(1 + pcm.shape[0] / infer_generator.frame_wav_scale)
  # calculate the rational length of pcm in order to keep the alignment of mfcc and landmark sequence.
  pcm_length = infer_generator.hop_step * (pad_len * infer_generator.frame_mfcc_scale - 1) + infer_generator.win_length
  if (pcm.shape[0] < pcm_length):
    pcm = np.pad(pcm, (0, pcm_length - pcm.shape[0]), 'constant', constant_values=(0))
  pcm_slice = pcm[:pcm_length][np.newaxis, :]

  ears = np.ones([1, pad_len, 1], dtype=np.float32)*0.9
  for i in range(pad_len//2):
    ears[0, i, 0] = 0.2
  ears = tf.convert_to_tensor(ears, dtype=tf.float32)
  mfcc = infer_generator.extract_mfcc(pcm_slice)
  img = cv2.imread(image_file)

  _, _, img_cropped, lmk_cropped, center_x, center_y, ratio = get_mxnet_sat_alignment(params.pretrain_dir, img)
  bfmcoeff, input_img, transform_params = alignto_bfm_coeff(params.pretrain_dir, img_cropped, lmk_cropped)
  ratio *= transform_params[2]
  tx = -int(round(transform_params[3] / ratio))
  ty = -int(round(transform_params[4] / ratio))

  seq_len = tf.convert_to_tensor([pad_len], dtype=tf.int32)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  ### BFMNet setting
  bfmnet = BFMNet(config_path)
  params = bfmnet.params
  params.batch_size = batch_size
  bfmnet.set_params(params)
  facemodel = BFM(params.pretrain_dir)

  infer_nodes = bfmnet.build_inference_op(ears, mfcc, seq_len)
  sess.run(tf.global_variables_initializer())

  # Restore from save_dir
  tf.train.Saver().restore(sess, 'ckpt_bfmnet/bfmnet-65000')

  ### Run inference
  bfm_coeff_seq = sess.run(infer_nodes['BFMCoeffDecoder'])
  bfmcoeff = np.tile(bfmcoeff[:, np.newaxis, :], [1, bfm_coeff_seq.shape[1], 1])

  bfm_coeff_seq = np.concatenate([bfmcoeff[:, :, :80], bfm_coeff_seq[:, :, :], bfmcoeff[:, :, 144:]], axis=2)
  merge_images = []

  ### step 2: generate tuple image sequence
  angles = np.array([[0, 0, 0]], dtype=np.float32)
  shift = 0.04
  for i in range(bfm_coeff_seq.shape[1]):
    angles[0][1] += shift
    if (angles[0][1] > 0.8 or angles[0][1] < -0.8):
      shift = -shift

    face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, _ = Reconstruction(
        bfm_coeff_seq[0, i:i + 1, ...], facemodel)

    shape = np.squeeze(face_shape, (0))
    color = np.squeeze(face_color, (0))
    color = np.clip(color, 0, 255).astype(np.int32)
    shape[:, :2] = 112 - shape[:, :2] * 112
    shape *=3

    img_size = 672
    new_image = np.zeros((img_size * img_size * 3), dtype=np.uint8)
    face_mask = np.zeros((img_size * img_size), dtype=np.uint8)

    vertices = shape.reshape(-1).astype(np.float32).copy()
    triangles = (facemodel.tri - 1).reshape(-1).astype(np.int32).copy()
    colors = color.reshape(-1).astype(np.float32).copy()
    depth_buffer = (np.zeros((img_size * img_size)) - 99999.0).astype(np.float32)
    mesh_core_cython.render_colors_core(new_image, face_mask, vertices, triangles, colors, depth_buffer,
                                        facemodel.tri.shape[0], img_size, img_size, 3)
    new_image = new_image.reshape([img_size, img_size, 3])

    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite('output/{}.jpg'.format(i), new_image)

  cmd = 'ffmpeg -i output/%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp.mp4'
  subprocess.call(cmd, shell=True)
