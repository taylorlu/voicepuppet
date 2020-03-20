#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
import logging
import sys

from atnet import ATNet
from vgnet import VGNet
from dataset.loader import *
from config.configure import YParams
from plot import *


alignment_handler = None
MXDetectorHandler_prefix = '/Users/donglu/workspace/deep-face-alignment/models/model-sat2d3-cab'
wav_file = '/Users/donglu/Downloads/cctv_cut.wav'
img_path = '/Users/donglu/Desktop/kanghui.jpg'

def extract_mfcc(pcm, params):
  # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
  pcm = tf.convert_to_tensor(pcm, dtype=tf.float32)
  stfts = tf.signal.stft(pcm, frame_length=params.mel['win_length'], frame_step=params.mel['hop_step'], fft_length=params.mel['fft_length'])
  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1].value
  lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(params.mel['num_mel_bins'],
                                                                      num_spectrogram_bins,
                                                                      params.mel['sample_rate'],
                                                                      lower_edge_hertz,
                                                                      upper_edge_hertz)
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, axes=[[2], [0]])
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  return log_mel_spectrograms

class MXDetectorHandler:
  def __init__(self, prefix, epoch, ctx_id, mx):
    if ctx_id>=0:
      ctx = mx.gpu(ctx_id)
    else:
      ctx = mx.cpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(prefix, "model"), epoch)
    all_layers = sym.get_internals()
    sym = all_layers['heatmap_output']
    image_size = (128, 128)
    self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model


def face_alignment(image):
  import mxnet as mx
  global alignment_handler
  global MXDetectorHandler_prefix
  if(alignment_handler is None):
    alignment_handler = MXDetectorHandler(prefix=MXDetectorHandler_prefix, epoch=0, ctx_id=-1, mx=mx)

  import dlib
  dlib_detector = dlib.get_frontal_face_detector()

  def crop_expand_dlib(image, rect, ratio=1.5):
    ## rect: [left, right, top, bottom]
    mean = [(rect[2] + rect[3]) // 2, (rect[0] + rect[1]) // 2]
    ## mean: [y, x]
    half_crop_size = int((rect[1] + rect[3] - rect[0] - rect[2]) * ratio // 4)

    # padding if the crop area outside of image.
    if (mean[0] - half_crop_size < 0):
      image = cv2.copyMakeBorder(image, 0, 0, half_crop_size - mean[0], 0, cv2.BORDER_CONSTANT, 0)
    if (mean[0] + half_crop_size > image.shape[1]):
      image = cv2.copyMakeBorder(image, 0, 0, 0, mean[0] + half_crop_size - image.shape[1], cv2.BORDER_CONSTANT, 0)
    if (mean[1] - half_crop_size < 0):
      image = cv2.copyMakeBorder(image, half_crop_size - mean[1], 0, 0, 0, cv2.BORDER_CONSTANT, 0)
    if (mean[1] + half_crop_size > image.shape[0]):
      image = cv2.copyMakeBorder(image, 0, mean[1] + half_crop_size - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, 0)

    left = mean[1] - half_crop_size
    right = mean[1] + half_crop_size
    top = mean[0] - half_crop_size
    buttom = mean[0] + half_crop_size

    if (left < 0):
      left = 0
    if (top < 0):
      top = 0

    return image, [left, right, top, buttom]

  def crop_expand_alignment(img, xys, out_img_size=224, ratio=1.3):
    xys = np.array(map(lambda x: int(x), xys))
    max_x = max(xys[::2])
    max_y = max(xys[1::2])
    min_x = min(xys[::2])
    min_y = min(xys[1::2])
    width = int((max_x - min_x) * ratio)
    height = int((max_y - min_y) * ratio)
    height = width

    center_x = (max_x + min_x) // 2
    center_y = (max_y + min_y) // 2

    left = center_x - width / 2
    top = center_y - height / 2
    img = img[top:top + height, left:left + width]

    xys[::2] -= left
    xys[1::2] -= top
    xys[::2] = xys[::2] * out_img_size / width
    xys[1::2] = xys[1::2] * out_img_size / height

    img = cv2.resize(img, (out_img_size, out_img_size))
    xys = np.array(list(map(lambda x: float(x)/out_img_size, xys)))

    return img, xys

  img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  rects = dlib_detector(img_gray, 0)
  if (len(rects) != 1):
    return None

  rect = [rects[0].left(), rects[0].right(), rects[0].top(), rects[0].bottom()]
  image, rect = crop_expand_dlib(image, rect)  # dlib region is too small
  ## rect: [left, right, top, bottom]

  img = cv2.cvtColor(image[rect[2]:rect[3], rect[0]:rect[1]], cv2.COLOR_BGR2RGB)
  crop_width = img.shape[1]
  crop_height = img.shape[0]

  img = cv2.resize(img, (128, 128))
  img = np.transpose(img, (2, 0, 1))  # 3*128*128, RGB
  input_blob = np.zeros((1, 3, 128, 128), dtype=np.uint8)
  input_blob[0] = img
  data = mx.nd.array(input_blob)
  db = mx.io.DataBatch(data=(data,))
  alignment_handler.model.forward(db, is_train=False)
  alabel = alignment_handler.model.get_outputs()[-1].asnumpy()[0]

  img_landmarks = []
  for j in xrange(alabel.shape[0]):
    a = cv2.resize(alabel[j], (128, 128))
    ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    ## ind: [y, x]

    origin_x = rect[0] + ind[1] * crop_width / 128
    origin_y = rect[2] + ind[0] * crop_height / 128

    img_landmarks.append(str(origin_x))
    img_landmarks.append(str(origin_y))

  image, img_landmarks = crop_expand_alignment(image, img_landmarks)
  return image, img_landmarks

def test_atnet(config_path):
  global wav_file
  global img_path
  img = cv2.imread(img_path)
  example_img, example_lmk = face_alignment(img)

  params = YParams(config_path, 'default')
  sample_rate = params.mel['sample_rate']
  hop_step = params.mel['hop_step']
  win_length = params.mel['win_length']
  frame_rate = params.frame_rate
  mean = np.load(params.mean_file)
  component = np.load(params.components_file)

  example_lmk = np.dot((example_lmk - mean), component[:,:20])
  example_lmk *= np.array([1.5, 1.0, 1.0, 1.0, 1.0, 2.0,  1.0,2.0,1.0,1.0, 1,1,1,1,1, 1,1,1,1,1])
  example_lmk = np.dot(example_lmk, component[:,:20].T)

  wav_loader = WavLoader(sr=sample_rate)

  pose = np.ones([1000,3], dtype=np.float32)*0.0
  ear = np.ones([1000,1], dtype=np.float32)*0.6
  ear[40:75,:] = np.ones([35,1], dtype=np.float32)*0.2

  pcm = wav_loader.get_data(wav_file)

  frame_wav_scale = sample_rate / frame_rate
  frame_mfcc_scale = frame_wav_scale / hop_step

  assert (frame_mfcc_scale - int(frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

  frame_mfcc_scale = int(frame_mfcc_scale)
  min_len = min(ear.shape[0], pose.shape[0], pcm.shape[0]//frame_wav_scale)

  g1 = tf.Graph()
  with g1.as_default():

    ear = tf.convert_to_tensor(ear[np.newaxis, :min_len, :], dtype=tf.float32)
    pose = tf.convert_to_tensor(pose[np.newaxis, :min_len, :], dtype=tf.float32)
    seq_len = tf.convert_to_tensor(np.array([min_len]), dtype=tf.int32)
    example_landmark = tf.convert_to_tensor(example_lmk[np.newaxis, :], dtype=tf.float32)

    pcm_length = hop_step * (min_len * frame_mfcc_scale - 1) + win_length
    if (pcm.shape[0] < pcm_length):
      pcm = np.pad(pcm, (0, pcm_length - pcm.shape[0]), 'constant', constant_values=(0))
    elif(pcm.shape[0] > pcm_length):
      pcm = pcm[:pcm_length]
    mfcc = extract_mfcc(pcm[np.newaxis, :], params)

    atnet = ATNet(config_path)
    params = atnet.params
    params.batch_size = 1
    atnet.set_params(params)

    infer_nodes = atnet.build_inference_op(ear, pose, mfcc, example_landmark, seq_len)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, 'ckpt_atnet/atnet-80000')
    lmk_seq = sess.run(infer_nodes['LandmarkDecoder'])
    save_lmkseq_video(lmk_seq, mean, "atnet.avi", wav_file)

  return example_img, example_lmk, lmk_seq

def test_vgnet(config_path, example_img, example_landmark, lmk_seq):
  example_img = cv2.resize(example_img, (128, 128)).astype(np.float32)[np.newaxis, ...]
  example_img /= 256.0
  example_img = (example_img - 0.5) / 0.5

  params = YParams(config_path, 'default')

  g2 = tf.Graph()
  with g2.as_default():
    example_landmark = tf.convert_to_tensor(example_landmark[np.newaxis, :], dtype=tf.float32)
    example_img = tf.convert_to_tensor(example_img, dtype=tf.float32)
    seq_len = tf.convert_to_tensor(np.array([lmk_seq.shape[1]]), dtype=tf.int32)
    lmk_seq = tf.convert_to_tensor((lmk_seq), dtype=tf.float32)

    vgnet = VGNet(config_path)
    params = vgnet.params
    params.batch_size = 1
    vgnet.set_params(params)

    infer_nodes = vgnet.build_inference_op(lmk_seq, example_landmark, example_img, seq_len)

    sess = tf.Session(graph=g2)
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, 'ckpt_vgnet/vgnet-70000')
    img_seq = sess.run(infer_nodes['Fake_img_seq'])

  save_imgseq_video(img_seq, "vgnet.mp4", wav_file)


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

  example_img, example_landmark, lmk_seq = test_atnet(config_path)
  test_vgnet(config_path, example_img, example_landmark, lmk_seq)


  # lmk_seq = []
  # example_image = None
  # example_landmark = None
  # params = YParams(config_path, 'default')
  # mean = np.load(params.mean_file)
  # component = np.load(params.components_file)

  # wav_file = '/Users/donglu/Downloads/cctv_cut.wav'
  # cap = cv2.VideoCapture('/Users/donglu/Downloads/cctv_cut.mp4')
  # if (cap.isOpened()):
  #   success, image = cap.read()
  #   idx = 0
  #   while (success):
  #     idx += 1
  #     if(idx==100):
  #       break
  #     [h, w, c] = image.shape
  #     if c > 3:
  #       image = image[:, :, :3]
  #     example_img, example_lmk = face_alignment(image)
  #     example_lmk = np.dot((example_lmk - mean), component[:,:20])
  #     example_lmk *= np.array([1.5, 1.0, 1.0, 1.0, 1.0, 2.0,  1.0,2.0,1.0,1.0, 1,1,1,1,1, 1,1,1,1,1])
  #     example_lmk = np.dot(example_lmk, component[:,:20].T)
  #     if(example_image is None):
  #       example_image = example_img
  #     if(example_landmark is None):
  #       example_landmark = example_lmk
  #     lmk_seq.append(example_lmk)

  #     success, image = cap.read()
  # cap.release()
  # lmk_seq = np.array(lmk_seq)[np.newaxis,...]
  # save_lmkseq_video(lmk_seq, mean, "atnet.avi", wav_file)

  # test_vgnet(config_path, example_image, example_landmark, lmk_seq)


