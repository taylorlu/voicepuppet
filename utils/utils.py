import mxnet as mx
import cv2
import numpy as np
import os
import math

alignment_handler = None
dlib_detector = None

def mkdir(dirname):
  if(not os.path.isdir(dirname)):
    os.makedirs(dirname)

class MXDetectorHandler:
  '''
  face 2D landmark alignment by mxnet, refer to https://github.com/deepinx/deep-face-alignment
  '''
  def __init__(self, prefix, epoch, mx, name='model'):
    ctx_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    if (ctx_id >= 0):
      ctx = mx.gpu(ctx_id)
    else:
      ctx = mx.cpu()

    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(prefix, name), epoch)
    all_layers = sym.get_internals()
    sym = all_layers['heatmap_output']
    image_size = (128, 128)
    self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model


def get_mxnet_sat_alignment(model_dir, image):
  '''
  Arguments:
    model_dir: The folder contains mxnet pretrained model.
    image: The image contains at least 1 face inside, we only detect the first face.
  Returns:
    image: The image input.
    img_landmarks: The 68 landmarks' coordinates in image.
    img: The face area expand by sat alignment, resize to out_img_size=224.
    lmk_cropped: The 68 landmarks' coordinates in img.
    center_x: the x position of the face center in image.
    center_y: the y position of the face center in image.
    ratio: The return image size / original face area size(before resize).
  '''
  global alignment_handler, dlib_detector

  if (alignment_handler is None):
    alignment_handler = MXDetectorHandler(prefix=model_dir, epoch=0, mx=mx, name='model-sat')

  import dlib
  if (dlib_detector is None):
    dlib_detector = dlib.get_frontal_face_detector()

  def crop_expand_dlib(image, rect, ratio=1.5):
    ## rect: [left, right, top, bottom]
    mean = [(rect[2] + rect[3]) / 2.0, (rect[0] + rect[1]) / 2.0]
    ## mean: [y, x]
    width = rect[1] - rect[0]
    height = rect[3] - rect[2]

    max_ratio = min([(image.shape[0] - mean[0])/(height/2), (image.shape[1] - mean[1])/(width/2), mean[0]/(height/2), mean[1]/(width/2)])
    if(max_ratio<ratio):
      ratio = max_ratio

    width = int((rect[1] - rect[0]) * ratio)
    height = int((rect[3] - rect[2]) * ratio)

    left = int(math.ceil(mean[1] - width // 2))
    top = int(math.ceil(mean[0] - height // 2))

    return image, [left, left + width, top, top + height]

  def crop_expand_alignment(img, xys, out_img_size=224, ratio=1.3):
    xys = xys.copy()
    max_x = max(xys[::2])
    max_y = max(xys[1::2])
    min_x = min(xys[::2])
    min_y = min(xys[1::2])

    center_x = int(round((max_x + min_x) / 2))
    center_y = int(round((max_y + min_y) / 2))
    width = max_x - min_x
    height = max_y - min_y
    height = width

    max_ratio = min([(img.shape[0] - center_y)/(height/2), (img.shape[1] - center_x)/(width/2), center_y/(height/2), center_x/(width/2)])
    if(max_ratio<ratio):
      ratio = max_ratio

    width = int((max_x - min_x) * ratio)
    height = int((max_y - min_y) * ratio)
    height = width

    left = int(round(center_x - width / 2))
    top = int(round(center_y - height / 2))
    img = img[top:top + height, left:left + width]

    xys[::2] -= left
    xys[1::2] -= top
    xys[::2] = xys[::2] * out_img_size / width
    xys[1::2] = xys[1::2] * out_img_size / height

    img = cv2.resize(img, (out_img_size, out_img_size))

    return img, xys, center_x, center_y, float(out_img_size) / width

  img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  rects = dlib_detector(img_gray, 0)
  if (len(rects) != 1):
    return

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
  for j in range(alabel.shape[0]):
    a = cv2.resize(alabel[j], (128, 128))
    ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    ## ind: [y, x]

    origin_x = rect[0] + ind[1] * crop_width / 128.0
    origin_y = rect[2] + ind[0] * crop_height / 128.0

    img_landmarks.append(origin_x)
    img_landmarks.append(origin_y)

  img_landmarks = np.array(img_landmarks)
  img, lmk_cropped, center_x, center_y, ratio = crop_expand_alignment(image, img_landmarks)
  return image, img_landmarks, img, lmk_cropped, center_x, center_y, ratio
