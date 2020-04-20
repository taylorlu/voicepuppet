#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import os
import cv2
import subprocess


def strokeline_lookup():
  '''
  the strokeline index of 68 points.
  '''
  Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
           [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
           [66, 67], [67, 60]]

  Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
          [33, 34], [34, 35], [27, 31], [27, 35]]

  leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
  rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

  leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
  rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

  other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
           [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
           [12, 13], [13, 14], [14, 15], [15, 16]]

  faceLmarkLookups = []
  faceLmarkLookups.append(Mouth)
  faceLmarkLookups.append(Nose)
  faceLmarkLookups.append(leftBrow)
  faceLmarkLookups.append(rightBrow)
  faceLmarkLookups.append(leftEye)
  faceLmarkLookups.append(rightEye)
  faceLmarkLookups.append(other)
  return faceLmarkLookups


def plot_lmk_seq(save_dir, step, mean, seq_len, real_lmk_seq, lmk_seq):
  '''
  merge 128x128 images to a large 9*10 grid picture.
  '''

  ## 9*10 block
  block_x = 10
  block_y = 9
  img_size = 128

  faceLmarkLookups = strokeline_lookup()

  def merge_seq(lmk_seq, big_img, time, h_index):

    for i in range(time):
      back_img = np.ones((img_size, img_size), dtype=np.uint8) * 255
      lmk = (((lmk_seq[0, i, ...] + mean)/2+0.5) * img_size).astype(np.int32)
      for k in range(68):
        cv2.circle(back_img, (int(lmk[k * 2]), int(lmk[k * 2 + 1])), 1, [0], -1)

      for part in faceLmarkLookups:
        for idx in part:
          cv2.line(back_img, (int(lmk[idx[0] * 2]), int(lmk[idx[0] * 2 + 1])),
                   (int(lmk[idx[1] * 2]), int(lmk[idx[1] * 2 + 1])), (0), 1)

      big_img[(i // block_x + h_index) * img_size: (i // block_x + h_index + 1) * img_size,
      (i % block_x) * img_size: (i % block_x + 1) * img_size] = back_img

    return big_img

  ### We only pick the first sequence of the batch, trim length of 30.
  if (seq_len[0] > 30):
    time = 30
  else:
    time = seq_len[0]

  big_img = np.zeros((img_size * block_y, img_size * block_x), dtype=np.uint8)
  big_img = merge_seq(real_lmk_seq, big_img, time, 0)
  big_img = merge_seq(lmk_seq, big_img, time, 3)

  cv2.imwrite('{}/atnet_{}.jpg'.format(save_dir, step), big_img)


def plot_image_seq(save_dir, step, mean, seq_len, real_lmk_seq, real_mask_seq, real_img_seq, fake_img_seq,
                   attention_seq):
  '''
  merge 2 sequence of image and attention map to a large image (9*10 grid picture).
  '''

  ## 9*10 block
  block_x = 10
  block_y = 9
  img_size = real_img_seq.shape[2]

  ### We only pick the first sequence of the batch, trim length of 30.
  if (seq_len[0] > 30):
    time = 30
  else:
    time = seq_len[0]

  big_img = 255 * np.ones((img_size * block_y, img_size * block_x, 4), dtype=np.uint8)

  for i in range(time):
    real_img = (((real_img_seq[0, i, ...] * 0.5) + 0.5) * 256).astype(np.uint8)
    fake_img = (((fake_img_seq[0, i, ...] * 0.5) + 0.5) * 256).astype(np.uint8)
    real_mask = (((real_mask_seq[0, i, ...] + 1) / 2) * 255).astype(np.uint8)
    attention_img = (attention_seq[0, i, ...] * 256).astype(np.uint8)

    lmk = (((real_lmk_seq[0, i, ...] + mean)/2+0.5) * img_size).astype(np.int32)
    for k in range(68):
      cv2.circle(real_img, (int(lmk[k * 2]), int(lmk[k * 2 + 1])), 1, [255, 255, 0], 1)

    real_img = np.concatenate([real_img, real_mask], axis=-1)

    big_img[i // block_x * img_size: (i // block_x + 1) * img_size,
    (i % block_x) * img_size: (i % block_x + 1) * img_size,
    :] = real_img

    big_img[(i // block_x + 3) * img_size: (i // block_x + 1 + 3) * img_size,
    (i % block_x) * img_size: (i % block_x + 1) * img_size,
    :-1] = fake_img

    big_img[(i // block_x + 6) * img_size: (i // block_x + 1 + 6) * img_size,
    (i % block_x) * img_size: (i % block_x + 1) * img_size,
    :] = cv2.merge((attention_img, attention_img, attention_img, attention_img))

  cv2.imwrite('{}/vgnet_{}.png'.format(save_dir, step), big_img)


def save_lmkseq_video(lmk_seq, mean, output_file, wav_file=None):
  img_size = 480
  seq_len = lmk_seq.shape[1]
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  output_movie = cv2.VideoWriter('temp.avi', fourcc, 25, (img_size, img_size), isColor=False)
  faceLmarkLookups = strokeline_lookup()

  for i in range(seq_len):
    back_img = np.ones((img_size, img_size), dtype=np.uint8) * 255
    lmk = (((lmk_seq[0, i, ...] + mean)/2+0.5) * img_size).astype(np.int32)
    for k in range(68):
      cv2.circle(back_img, (int(lmk[k * 2]), int(lmk[k * 2 + 1])), 1, [0], -1)

    for part in faceLmarkLookups:
      for idx in part:
        cv2.line(back_img, (int(lmk[idx[0] * 2]), int(lmk[idx[0] * 2 + 1])),
                 (int(lmk[idx[1] * 2]), int(lmk[idx[1] * 2 + 1])), (0), 1)

    output_movie.write(back_img)

  if (wav_file is not None):
    cmd = 'ffmpeg -y -i temp.avi -i ' + wav_file + ' -c:v copy -c:a aac -strict experimental ' + output_file
    subprocess.call(cmd, shell=True)
    os.remove('temp.avi')


def save_imgseq_video(img_seq, output_file, wav_file=None):
  def mkdir(dirname):
    if not os.path.isdir(dirname):
      os.makedirs(dirname)

  img_size = 128
  seq_len = img_seq.shape[1]
  mkdir('temp')

  for i in range(seq_len):
    real_img = (((img_seq[0, i, ...] * 0.5) + 0.5) * 256).astype(np.uint8)
    cv2.imwrite('temp/{}.jpg'.format(i), real_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

  if (wav_file is not None):
    cmd = 'ffmpeg -i temp/%d.jpg -i ' + wav_file + ' -c:v libx264 -c:a aac -strict experimental -y -vf format=yuv420p ' + output_file
    subprocess.call(cmd, shell=True)
    cmd = 'rm -rf temp temp.avi'
    subprocess.call(cmd, shell=True)
