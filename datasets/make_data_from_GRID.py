#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import sys
from optparse import OptionParser
import cv2
import numpy as np
import math
from subprocess import Popen, PIPE
import shutil
from PIL import Image
import scipy

from utils import bfm_load_data
from utils import reconstruct_mesh
from utils import utils
import mesh_core_cython
from models import *


class Schedule:
  tasks = {}

  def __init__(self, step, *args):
    self.tasks = {
      1: self.ear_compute,
      2: self.extract_audio,
      3: self.mxnet_sat_alignment,
      4: self.alignto_bfm_coeff,
      5: self.deep_video_portraits,
      6: self.deep_video_portraits_v2
    }
    function = self.tasks.get(step, self.raise_err)
    function(args)

  def raise_err(self, *args):
    if (len(args) == 0):
      print(
          "Error: The steps [1-6]:\n1.ear_compute.\n2.extract_audio.\n3.mxnet_sat_alignment.\n4.alignto_bfm_coeff.\n5.deep_video_portraits.\n6.deep_video_portraits_v2.\n")
    else:
      print("Error: Please check the args.\n")

  ############# STEP 1: EAR result compute from landmark result. #############
  def ear_compute(self, args):
    if (len(args) != 1):
      self.raise_err("ear_compute")
      return

    to_dir, = args
    print('>>> Compute EAR result. <<<\n')
    # walk to_dir, do 'EAR result compute'.
    for root, subdirs, files in os.walk(to_dir):
      if not subdirs:
        print('EAR compute: {}'.format(root))

        if (not os.path.exists(os.path.join(root, "_landmark.txt"))):
          continue

        landmarkfile = open(os.path.join(root, "_landmark.txt"))
        lines = landmarkfile.readlines()
        earfile = open(os.path.join(root, "ear.txt"), "w")
        for idx, line in enumerate(lines):
          ps = line.split(',')
          ps = map(lambda x: int(x), ps)

          EAR1 = float((math.sqrt((ps[74] - ps[82]) ** 2 + (ps[75] - ps[83]) ** 2) + math.sqrt(
              (ps[76] - ps[80]) ** 2 + (ps[77] - ps[81]) ** 2))) / math.sqrt(
              (ps[72] - ps[78]) ** 2 + (ps[73] - ps[79]) ** 2)
          EAR2 = float((math.sqrt((ps[86] - ps[94]) ** 2 + (ps[87] - ps[95]) ** 2) + math.sqrt(
              (ps[88] - ps[92]) ** 2 + (ps[89] - ps[93]) ** 2))) / math.sqrt(
              (ps[84] - ps[90]) ** 2 + (ps[85] - ps[91]) ** 2)
          EAR = (EAR1 + EAR2) / 2

          earfile.write('{}\n'.format(EAR))

        landmarkfile.close()
        earfile.close()

  ############# STEP 2: Extract Audio in Video. #############
  def extract_audio(self, args):
    if (len(args) != 2):
      self.raise_err("extract_audio")
      return

    src_dir, to_dir = args
    print('>>> Extract Audio. <<<\n')
    # walk src_dir, do 'Extract Audio'.
    for root, subdirs, files in os.walk(src_dir):
      if not subdirs:
        print('Extract Audio: {}'.format(root))
        pid = root.split('/video/mpg_6000')[0]
        pid = pid.split('/')[-1]
        for f in files:
          if(not (f.endswith('.mp4') or f.endswith('.mpg'))):
            continue
          # s2/lgbm3p.mpg
          movie_id = f.split('.')[0]

          src_movie_file = os.path.join(root, f)
          to_extract_dir = os.path.join(to_dir, pid, movie_id)
          if (os.path.exists(os.path.join(to_extract_dir, 'audio.wav'))):
            continue
          cmd = 'ffmpeg -i {} -vn {}'.format(src_movie_file, os.path.join(to_extract_dir, 'audio.wav'))
          proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
          try:
            buff = proc.communicate()
          except Exception as e:
            print('Extract Audio FAILED.\n')
            sys.exit(1)
            proc.stdout.close()

  ############# STEP 3: 'landmark detect' by mxnet SAT model #############
  def mxnet_sat_alignment(self, args):
    if (len(args) != 3):
      self.raise_err("mxnet SAT alignment")
      return

    src_dir, to_dir, model_dir = args

    # walk movie dir, do 'face crop' and 'pose estimate'.
    for root, subdirs, files in os.walk(src_dir):
      if not subdirs:
        pid = root.split('/video/mpg_6000')[0]
        pid = pid.split('/')[-1]
        for f in files:
          if(not (f.endswith('.mp4') or f.endswith('.mpg'))):
            continue

          movie_id = f.split('.')[0]
          src_movie_file = os.path.join(root, f)
          to_crop_dir = os.path.join(to_dir, pid, movie_id)
          utils.mkdir(to_crop_dir)
          print('mxnet SAT: {}'.format(to_crop_dir))

          if (os.path.exists(os.path.join(to_crop_dir, "landmark.txt"))):
            continue

          labelfile = open(os.path.join(to_crop_dir, 'landmark.txt'), 'w')

          cap = cv2.VideoCapture(src_movie_file)
          if (cap.isOpened()):
            idx = 0
            success, image = cap.read()
            while (success):
              [h, w, c] = image.shape
              if c > 3:
                image = image[:, :, :3]

              _, _, img_cropped, lmk_cropped, center_x, center_y, ratio = utils.get_mxnet_sat_alignment(model_dir, image)

              labelfile.write('{}\n'.format(','.join(['{:.2f}'.format(v) for v in lmk_cropped])))
              cv2.imwrite(os.path.join(to_crop_dir, '{}.jpg'.format(idx)), img_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
              idx += 1
              success, image = cap.read()
            labelfile.close()
          cap.release()

  def get_bfm_coeff(self, lm3D, sess, images, coeff, img, ps):
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
    input_img, lm_new, transform_params = bfm_load_data.Preprocess(image, lmk5, lm3D)
    bfmcoeff = sess.run(coeff, feed_dict={images: input_img})
    return bfmcoeff, input_img, transform_params

  ############# STEP 4: Deep3DFaceReconstruction #############
  def alignto_bfm_coeff(self, args):
    if (len(args) != 2):
      self.raise_err("Deep3DFaceReconstruction")
      return

    to_dir, model_dir = args
    print('>>> Deep3DFaceReconstruction. <<<\n')

    from PIL import Image
    import tensorflow as tf

    def load_graph(graph_filename):
      with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

      return graph_def

    # read standard landmarks for preprocessing images
    lm3D = bfm_load_data.load_lm3d(model_dir)
    facemodel = bfm_load_data.BFM(model_dir)

    # build reconstruction model
    ctx_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0])
    if (ctx_id >= 0):
      device = tf.device('/gpu:{}'.format(ctx_id))
    else:
      device = tf.device('/cpu:0')

    with tf.Graph().as_default() as graph, device:
      images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
      graph_def = load_graph(os.path.join(model_dir, "FaceReconModel.pb"))
      tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

      # output coefficients of R-Net (dim = 257) 
      coeff = graph.get_tensor_by_name('resnet/coeff:0')
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:

        # walk to_dir, feed face image, output coeff
        for root, subdirs, files in os.walk(to_dir):
          if not subdirs:
            print('Deep3DFaceReconstruction: {}'.format(root))

            if (not os.path.exists(os.path.join(root, "landmark.txt"))):
              continue

            if (os.path.exists(os.path.join(root, "bfmcoeff.txt"))):
              continue

            count = 0
            for file in files:
              if (file.endswith('.jpg')):
                count += 1

            lines = open(os.path.join(root, "landmark.txt")).readlines()

            bfmcoeff_file = open(os.path.join(root, "bfmcoeff.txt"), 'w')
            for i in range(count):
              xys = lines[i].split(',')
              ps = map(lambda x: float(x), xys)
              img = cv2.imread(os.path.join(root, "{}.jpg".format(i)))
              bfmcoeff, _, _ = self.get_bfm_coeff(lm3D, sess, images, coeff, img, ps)

              face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, translation = reconstruct_mesh.Reconstruction(
                  bfmcoeff, facemodel)

              shape = np.squeeze(face_shape, (0))
              color = np.squeeze(face_color, (0))
              color = np.clip(color, 0, 255).astype(np.int32)
              shape[:, :2] = 112 - shape[:, :2] * 112

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
              cv2.imwrite('out/{}.jpg'.format(i), new_image)

              bfmcoeff_file.write('{}\n'.format(','.join(bfmcoeff[0].astype(str))))
            bfmcoeff_file.close()

  ############# STEP 5: DeepVideoPortraits #############
  def deep_video_portraits(self, args):
    if (len(args) != 3):
      self.raise_err("DeepVideoPortraits")
      return
    import tensorflow as tf

    src_dir, to_dir, model_dir = args
    facemodel = bfm_load_data.BFM(model_dir)

    def load_graph(graph_filename):
      with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

      return graph_def

    lm3D = bfm_load_data.load_lm3d(model_dir)

    # build reconstruction model
    ctx_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    if (ctx_id >= 0):
      device = tf.device('/gpu:{}'.format(ctx_id))
    else:
      device = tf.device('/cpu:0')

    with tf.Graph().as_default() as graph, device:
      images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
      graph_def = load_graph(os.path.join(model_dir, "FaceReconModel.pb"))
      tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

      # output coefficients of R-Net (dim = 257) 
      coeff = graph.get_tensor_by_name('resnet/coeff:0')
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:

        # walk movie dir, do 'face crop' and 'pose estimate'.
        for root, subdirs, files in os.walk(src_dir):
          if not subdirs:
            pid = root.split('/video/mpg_6000')[0]
            pid = pid.split('/')[-1]
            exps = ['s16', 's27', 's28', 's31', 's5', 's7', 's9']  # Wear glasses
            if (pid in exps):
              continue
            mov_per_idx = 0
            for f in files:
              if(not (f.endswith('.mp4') or f.endswith('.mpg'))):
                continue
              mov_per_idx += 1
              if (mov_per_idx > 100):  # Only pick up 100 videos per person.
                break

              movie_id = f.split('.')[0]
              movie_file = os.path.join(root, f)
              to_crop_dir = os.path.join(to_dir, pid, movie_id)
              utils.mkdir(to_crop_dir)

              cap = cv2.VideoCapture(movie_file)
              if (cap.isOpened()):
                idx = 0
                success, img = cap.read()
                while (success):
                  img = cv2.resize(img[:, 72:72 + 576, :], (256, 256))
                  lmk_results = utils.get_mxnet_sat_alignment(model_dir, img)
                  if (lmk_results is None):
                    os.system('rm -rf {}'.format(to_crop_dir))
                    break
                  # image, img_landmarks, center_x, center_y, ratio = lmk_results
                  img, img_landmarks, img_cropped, lmk_cropped, center_x, center_y, ratio = lmk_results

                  bfmcoeff, input_img, transform_params = self.get_bfm_coeff(lm3D, sess, images, coeff, img_cropped,
                                                                                     lmk_cropped)
                  ratio *= transform_params[2]
                  tx = -int(round(transform_params[3] / ratio))
                  ty = -int(round(transform_params[4] / ratio))

                  face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, translation = reconstruct_mesh.Reconstruction(
                    bfmcoeff, facemodel)
                  face_projection2 = np.concatenate([face_projection, z_buffer], axis=2)
                  face_projection = np.squeeze(face_projection2, (0))

                  shape = np.squeeze(face_projection2, (0))
                  color = np.squeeze(face_color, (0))
                  color = np.clip(color, 0, 255).astype(np.int32)

                  ## Rasterize vertices to face image, [input => shape, color], [output => new_image, face_mask]
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

                  ## Fill the hole of mouth region
                  face_mask = np.squeeze(face_mask > 0).astype(np.float32)
                  face_mask = scipy.ndimage.binary_fill_holes(face_mask)
                  face_mask = face_mask.astype(np.float32)

                  face_mask = cv2.resize(face_mask, (
                  int(round(face_mask.shape[0] / ratio)), int(round(face_mask.shape[1] / ratio))))

                  kernel = np.ones((5, 5), np.uint8)
                  face_mask = cv2.erode(face_mask, kernel).astype(np.float32)

                  face_mask = cv2.GaussianBlur((face_mask * 255).astype(np.uint8), (21, 21), cv2.BORDER_DEFAULT)
                  face_mask = face_mask.astype(np.float32) / 255

                  new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
                  new_image = cv2.resize(new_image, (
                  int(round(new_image.shape[0] / ratio)), int(round(new_image.shape[1] / ratio))))

                  ## Redraw the reconstructed face to origin image
                  back_new_image = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
                  center_face_x = new_image.shape[1] / 2
                  center_face_y = new_image.shape[0] / 2

                  ry = center_y - center_face_y + new_image.shape[0] - ty
                  rx = center_x - center_face_x + new_image.shape[1] - tx
                  back_new_image[center_y - center_face_y - ty:ry, center_x - center_face_x - tx:rx, :] = new_image

                  back_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                  back_mask[center_y - center_face_y - ty:ry, center_x - center_face_x - tx:rx] = face_mask

                  back_mask2 = back_mask.copy()
                  back_mask2[back_mask2 > 0] = 1
                  result = img * (1 - back_mask[:, :, np.newaxis]) + back_new_image * back_mask[:, :, np.newaxis]
                  result = result * back_mask2[:, :, np.newaxis]

                  ## merge images to 1*3 grid image
                  merge_image = np.zeros((256, 768, 3), dtype=np.uint8)
                  merge_image[:, 0:256, :] = img
                  merge_image[:, 256:512, :] = result
                  merge_image[:, 512:, :] = np.tile(back_mask[:, :, np.newaxis] * 255, (1, 1, 3))

                  cv2.imwrite('{}/{}.jpg'.format(to_crop_dir, idx), merge_image)

                  idx += 1
                  success, img = cap.read()
              cap.release()

  ############# STEP 6: DeepVideoPortraits_v2 #############
  def deep_video_portraits_v2(self, args):
    if (len(args) != 3):
      self.raise_err("DeepVideoPortraits")
      return

    import torch
    from torchvision import transforms
    from torch.autograd import Variable
    from skimage.morphology import remove_small_objects, remove_small_holes
    import tensorflow as tf

    src_dir, to_dir, model_dir = args

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    ## face segmentation model
    net_init_params = {'Dropout': 0.4, 'pretrained': False, 'num_classes': 1, 'num_filters': 32}
    seg_model = UnetMobilenetV2(**net_init_params)
    seg_model = seg_model.train().cpu()
    state = torch.load(os.path.join(model_dir, 'mobilenetV2_model_checkpoint_metric.pth'))
    seg_model.load_state_dict(state['state_dict'])
    seg_model.eval()

    ## image matting model
    checkpoint = torch.load(os.path.join(model_dir, 'BEST_checkpoint.tar'))
    dim_model = checkpoint['model'].module
    dim_model = dim_model.to(torch.device('cuda'))
    dim_model.eval()

    ## Rasterize bfmcoeff to face image and move to original position
    def render_face(center_x, center_y, ratio, bfmcoeff, img, transform_params, facemodel):
      ratio *= transform_params[2]
      tx = -int((transform_params[3] / ratio))
      ty = -int((transform_params[4] / ratio))

      face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, translation = reconstruct_mesh.Reconstruction(
        bfmcoeff, facemodel)
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

    def predict_mask(imgs, biggest_side=320, denoise_borders=True):
      if imgs.ndim < 4:
        imgs = np.expand_dims(imgs, axis=0)
      l, h, w, c = imgs.shape
      w_n, h_n = w, h
      if biggest_side is not None: 
        w_n = int(w/h * min(biggest_side, h))
        h_n = min(biggest_side, h)

      wd, hd = w_n % 32, h_n % 32
      if wd != 0: w_n += 32 - wd
      if hd != 0: h_n += 32 - hd

      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
      all_predictions = []
      for i in range(imgs.shape[0]):
        img = norm(cv2.resize(imgs[i], (w_n, h_n), interpolation=cv2.INTER_LANCZOS4))
        img = img.unsqueeze_(0)
        img = img.type(torch.FloatTensor)
        output = seg_model(Variable(img))
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        output = torch.sigmoid(output)
        output = output.cpu().data.numpy()
        y_pred = np.squeeze(output[0])
        y_pred = remove_small_holes(remove_small_objects(y_pred > .3))
        y_pred = (y_pred * 255).astype(np.uint8)
        y_pred = cv2.resize(y_pred, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        _,alpha = cv2.threshold(y_pred.astype(np.uint8),0,255,cv2.THRESH_BINARY)
        b, g, r = cv2.split(imgs[i])
        bgra = [r,g,b, alpha]
        y_pred = cv2.merge(bgra,4)
        if denoise_borders:
          y_pred[:, :, -1] = cv2.morphologyEx(y_pred[:, :, -1], cv2.MORPH_OPEN, kernel)
        all_predictions.append(y_pred)
      return all_predictions

    def load_graph(graph_filename):
      with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

      return graph_def

    facemodel = bfm_load_data.BFM(model_dir)
    lm3D = bfm_load_data.load_lm3d(model_dir)

    # build reconstruction model
    ctx_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    if (ctx_id >= 0):
      device = tf.device('/gpu:{}'.format(ctx_id))
    else:
      device = tf.device('/cpu:0')

    with tf.Graph().as_default() as graph, device:
      images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
      graph_def = load_graph(os.path.join(model_dir, "FaceReconModel.pb"))
      tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

      # output coefficients of R-Net (dim = 257) 
      coeff = graph.get_tensor_by_name('resnet/coeff:0')
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      with tf.Session(config=config) as sess:

        # walk movie dir, do 'face crop' and 'pose estimate'.
        for root, subdirs, files in os.walk(src_dir):
          if not subdirs:
            pid = root.split('/video/mpg_6000')[0]
            pid = pid.split('/')[-1]
            exps = ['s16', 's27', 's28', 's31', 's5', 's7', 's9']  # Wear glasses
            if (pid in exps):
              continue
            mov_per_idx = 0
            for f in files:
              if(not (f.endswith('.mp4') or f.endswith('.mpg'))):
                continue
              mov_per_idx += 1
              if (mov_per_idx > 100):  # Only pick up 100 videos per person.
                break

              movie_id = f.split('.')[0]
              movie_file = os.path.join(root, f)
              to_crop_dir = os.path.join(to_dir, pid, movie_id)
              utils.mkdir(to_crop_dir)

              cap = cv2.VideoCapture(movie_file)
              if (cap.isOpened()):
                idx = 0
                success, img = cap.read()
                while (success):
                  img = img[:, 72:72 + 576, :]
                  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                  out = predict_mask(img)[0]
                  rgba = np.array(out)

                  alpha = rgba[...,3]
                  rgb = rgba[...,:3]

                  kernel = np.ones((30, 30), np.uint8)
                  mask2 = cv2.erode(alpha, kernel)
                  kernel = np.ones((10, 10), np.uint8)
                  mask3 = cv2.dilate(alpha, kernel)
                  mask = mask3 - mask2
                  trimap = mask2 + mask // 2

                  x = torch.zeros((1, 4, img.shape[0], img.shape[1]), dtype=torch.float)
                  image = rgb
                  image = transforms.ToPILImage()(image)
                  image = data_transforms(image)
                  x[0:, 0:3, :, :] = image
                  x[0:, 3, :, :] = torch.from_numpy(trimap / 255.)

                  # Move to GPU, if available
                  with torch.no_grad():
                    pred = dim_model(Variable(x.type(torch.FloatTensor).to(torch.device('cuda')))).cpu().data.numpy()[0]
                    pred[trimap == 0] = 0.0
                    pred[trimap == 255] = 1.0

                  alpha = np.tile(pred[:,:,np.newaxis]*255, [1,1,3]).astype(np.uint8)
                  alpha = cv2.resize(alpha, (256, 256))

                  img = cv2.resize(img, (256, 256))
                  lmk_results = utils.get_mxnet_sat_alignment(model_dir, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                  if (lmk_results is None):
                    os.system('rm -rf {}'.format(to_crop_dir))
                    break
                  img, img_landmarks, img_cropped, lmk_cropped, center_x, center_y, ratio = lmk_results

                  bfmcoeff, input_img, transform_params = self.get_bfm_coeff(lm3D, sess, images, coeff, img_cropped,
                                                                                     lmk_cropped)
                  face3d = render_face(center_x, center_y, ratio, bfmcoeff, img, transform_params, facemodel)

                  merge_image = np.zeros((256, 768, 3), dtype=np.uint8)
                  merge_image[:, 0:256, :] = img
                  merge_image[:, 256:512, :] = face3d
                  merge_image[:, 512:, :] = alpha
                  cv2.imwrite('{}/{}.jpg'.format(to_crop_dir, idx), merge_image)

                  idx += 1
                  success, img = cap.read()
              cap.release()


if __name__ == '__main__':
  cmd_parser = OptionParser(
      usage="usage: %prog [options] <src_dir >to_dir >model_dir")
  cmd_parser.add_option('--gpu', type=str, dest="gpu", default='-1',
                        help='visible gpu id, -1 for cpu\n')
  cmd_parser.add_option('--step', type=int, dest="step",
                        help='The steps [1-6]:\n1.ear_compute.\n2.extract_audio.\n3.mxnet_sat_alignment.\n4.alignto_bfm_coeff.\n5.deep_video_portraits.\n6.deep_video_portraits_v2\n')

  opts, argv = cmd_parser.parse_args()
  # print opts, argv
  if len(argv) and argv[0] == '':
    argv.pop(0)

  os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

  Schedule(opts.step, *(argv))
