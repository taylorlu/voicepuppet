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

  mfcc = infer_generator.extract_mfcc(pcm_slice)
  img = cv2.imread(image_file)
  img = cv2.resize(img[:, 72:72 + 576, :], (256, 256))

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

  infer_nodes = bfmnet.build_inference_op(mfcc, seq_len)
  sess.run(tf.global_variables_initializer())

  # Restore from save_dir
  tf.train.Saver().restore(sess, 'ckpt_bfmnet/bfmnet-31000')

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

    # face_shape, face_texture, face_color = Reconstruction_rotation(bfm_coeff_seq[0, i:i + 1, ...], facemodel, angles)
    face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, _ = Reconstruction(
        bfm_coeff_seq[0, i:i + 1, ...], facemodel)
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
    face_mask = face_mask.reshape([224, 224])

    face_mask = np.squeeze(face_mask > 0).astype(np.float32)
    face_mask = scipy.ndimage.binary_fill_holes(face_mask)
    face_mask = face_mask.astype(np.float32)

    face_mask = cv2.resize(face_mask, (
    int(round(face_mask.shape[0] / ratio)), int(round(face_mask.shape[1] / ratio))))

    kernel = np.ones((5, 5), np.uint8)
    face_mask = cv2.erode(face_mask, kernel).astype(np.float32)

    face_mask = cv2.GaussianBlur((face_mask * 255).astype(np.uint8), (11, 11), cv2.BORDER_DEFAULT)
    face_mask = face_mask.astype(np.float32) / 255

    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    new_image = cv2.resize(new_image, (
    int(round(new_image.shape[0] / ratio)), int(round(new_image.shape[1] / ratio))))

    back_new_image = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
    center_face_x = new_image.shape[1] / 2
    center_face_y = new_image.shape[0] / 2

    ry = center_y - center_face_y + new_image.shape[0] - ty
    rx = center_x - center_face_x + new_image.shape[1] - tx
    cuty = 256 - ry
    cutx = 256 - rx
    if (cuty < 0):
      ry = 256
      new_image = new_image[:cuty, ...]
      face_mask = face_mask[:cuty, ...]
    if (cutx < 0):
      rx = 256
      new_image = new_image[:, :cutx, :]
      face_mask = face_mask[:, :cutx]
    back_new_image[center_y - center_face_y - ty:ry, center_x - center_face_x - tx:rx, :] = new_image

    back_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    back_mask[center_y - center_face_y - ty:ry, center_x - center_face_x - tx:rx] = face_mask

    back_mask2 = back_mask.copy()
    back_mask2[back_mask2 > 0] = 1
    result = img * (1 - back_mask[:, :, np.newaxis]) + back_new_image * back_mask[:, :, np.newaxis]
    result = result * back_mask2[:, :, np.newaxis]

    merge_image = np.zeros((256, 768, 3), dtype=np.uint8)
    merge_image[:, 0:256, :] = img
    merge_image[:, 256:512, :] = result

    merge_image[:, 512:, :] = np.tile(back_mask[:, :, np.newaxis] * 255, (1, 1, 3))

    cv2.imwrite('output/{}.jpg'.format(i), merge_image)
    merge_image = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)
    merge_images.append(merge_image)

  ### step 3: generate video with background
  with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess2:
    with tf.gfile.FastGFile('pix2pix_output/frozen_model.pb', 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')

      sess2.run(tf.global_variables_initializer())

      final_tensor = sess2.graph.get_tensor_by_name('convert_image_1:0')
      merge_images = np.array(merge_images)
      frames = sess2.run(final_tensor, feed_dict={"Placeholder:0": merge_images})
      for i in range(frames.shape[0]):
        cv2.imwrite('output/_{}.jpg'.format(i), cv2.cvtColor(frames[i,...], cv2.COLOR_BGR2RGB))

  cmd = 'ffmpeg -i output/%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp.mp4'
  subprocess.call(cmd, shell=True)

  cmd = 'ffmpeg -i output/_%d.jpg -i ' + audio_file + ' -c:v libx264 -c:a aac -strict experimental -y temp2.mp4'
  subprocess.call(cmd, shell=True)
