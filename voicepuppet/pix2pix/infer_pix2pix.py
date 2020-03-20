#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
import subprocess
from pix2pix import Pix2PixNet
from voicepuppet.bfmnet.bfmnet import BFMNet
from generator.loader import *
from generator.generator import DataGenerator
from utils.bfm_load_data import *
from utils.bfm_visual import *
from utils.utils import *
import scipy

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
shift = 0.001

def render_face(center_x, center_y, ratio, bfmcoeff, img, transform_params, facemodel):
  ratio *= transform_params[2]
  tx = -int((transform_params[3] / ratio))
  ty = -int((transform_params[4] / ratio))
  global angles, shift

  # angles[0][1] += shift
  # if (angles[0][1] > 0.01 or angles[0][1] < -0.01):
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
  facemodel = BFM(params.pretrain_dir)

  pad_len = int(1 + pcm.shape[0] / infer_generator.frame_wav_scale)
  # calculate the rational length of pcm in order to keep the alignment of mfcc and landmark sequence.
  pcm_length = infer_generator.hop_step * (pad_len * infer_generator.frame_mfcc_scale - 1) + infer_generator.win_length
  if (pcm.shape[0] < pcm_length):
    pcm = np.pad(pcm, (0, pcm_length - pcm.shape[0]), 'constant', constant_values=(0))
  pcm_slice = pcm[:pcm_length][np.newaxis, :]

  mfcc = infer_generator.extract_mfcc(pcm_slice)
  img = cv2.imread(image_file)
  img = cv2.resize(img[:, 72:72 + 576, :], (256, 256))

  img, img_landmarks, img_cropped, lmk_cropped, center_x, center_y, ratio = get_mxnet_sat_alignment(params.pretrain_dir, img)
  bfmcoeff, input_img, transform_params = alignto_bfm_coeff(params.pretrain_dir, img_cropped, lmk_cropped)

  seq_len = tf.convert_to_tensor([pad_len], dtype=tf.int32)

  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

    ### BFMNet setting
    bfmnet = BFMNet(config_path)
    params = bfmnet.params
    params.batch_size = batch_size
    bfmnet.set_params(params)

    infer_nodes = bfmnet.build_inference_op(mfcc, seq_len)
    sess.run(tf.global_variables_initializer())

    # Restore from save_dir
    tf.train.Saver().restore(sess, 'ckpt_bfmnet/bfmnet-31000')

    ### Run inference
    bfm_coeff_seq = sess.run(infer_nodes['BFMCoeffDecoder'])
    bfmcoeff = np.tile(bfmcoeff[:, np.newaxis, :], [1, bfm_coeff_seq.shape[1], 1])

    bfm_coeff_seq = np.concatenate([bfmcoeff[:, :, :80], bfm_coeff_seq[:, :, :], bfmcoeff[:, :, 144:]], axis=2)
    merge_images = []

    for i in range(bfm_coeff_seq.shape[1]):
      face3d = render_face(center_x, center_y, ratio, bfm_coeff_seq[0, i:i + 1, ...], img, transform_params, facemodel)

      merge_image = np.zeros((256, 512, 3), dtype=np.uint8)
      merge_image[:, 0:256, :] = cv2.resize(cv2.imread('/home/dong/Downloads/bg.jpg'), (256,256))
      merge_image[:, 256:512, :] = face3d

      cv2.imwrite('output/{}.jpg'.format(i), merge_image)
      merge_image = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)
      merge_images.append(merge_image)

  tf.reset_default_graph()
  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess2:
    ### Pix2PixNet setting
    pix2pixnet = Pix2PixNet(config_path)
    params = pix2pixnet.params
    params.batch_size = 1
    pix2pixnet.set_params(params)

    inputs_holder = tf.placeholder(tf.float32, shape=[None, 256, 256, 3*3])
    targets_holder = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    infer_nodes = pix2pixnet.build_inference_op(inputs_holder, targets_holder)

    sess2.run(tf.global_variables_initializer())
    # Restore from save_dir
    tf.train.Saver().restore(sess2, 'ckpt_pix2pixnet/pix2pixnet-35000')

    merge_images = np.array(merge_images)/255.0
    targets = merge_images[:, :, :256, :]
    inputs = merge_images[:, :, 256:512, :]

    ## padding 2 empty frames before image sequence.
    inputs = np.concatenate([np.zeros([2, inputs.shape[1], inputs.shape[2], inputs.shape[3]], dtype=inputs.dtype), inputs], axis=0)
    for i in range(merge_images.shape[0]):
      input_slice = inputs[i: i + 3, ...]
      input_slice = input_slice.transpose((1, 2, 0, 3))
      input_slice = input_slice.reshape([256, 256, 9])

      frames = sess2.run(infer_nodes['Outputs'], 
        feed_dict={inputs_holder: input_slice[np.newaxis, ...], targets_holder: targets[i:i+1, ...]})

      cv2.imwrite('output/_{}.jpg'.format(i), cv2.cvtColor((frames[0,...]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

  cmd = 'ffmpeg -i output/_%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp2.mp4'
  subprocess.call(cmd, shell=True)

  cmd = 'ffmpeg -i output/%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp.mp4'
  subprocess.call(cmd, shell=True)
