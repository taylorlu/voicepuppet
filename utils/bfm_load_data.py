import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat
from array import array
import os


# define facemodel for reconstruction
class BFM():
  def __init__(self, model_dir='BFM'):
    model_path = os.path.join(model_dir, 'BFM_model_front.mat')
    model = loadmat(model_path)
    self.meanshape = model['meanshape']  # mean face shape
    self.idBase = model['idBase']  # identity basis
    self.exBase = model['exBase']  # expression basis
    self.meantex = model['meantex']  # mean face texture
    self.texBase = model['texBase']  # texture basis
    self.point_buf = model[
      'point_buf']  # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
    self.tri = model['tri']  # vertex index for each triangle face, starts from 1
    self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1  # 68 face landmark index, starts from 0


# load expression basis
def LoadExpBasis(model_dir='BFM'):
  n_vertex = 53215
  Expbin = open(os.path.join(model_dir, 'Exp_Pca.bin'), 'rb')
  exp_dim = array('i')
  exp_dim.fromfile(Expbin, 1)
  expMU = array('f')
  expPC = array('f')
  expMU.fromfile(Expbin, 3 * n_vertex)
  expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)

  expPC = np.array(expPC)
  expPC = np.reshape(expPC, [exp_dim[0], -1])
  expPC = np.transpose(expPC)

  expEV = np.loadtxt(os.path.join(model_dir, 'std_exp.txt'))

  return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(model_dir='BFM'):
  original_BFM = loadmat(os.path.join(model_dir, '01_MorphableModel.mat'))
  shapePC = original_BFM['shapePC']  # shape basis
  shapeEV = original_BFM['shapeEV']  # corresponding eigen value
  shapeMU = original_BFM['shapeMU']  # mean face
  texPC = original_BFM['texPC']  # texture basis
  texEV = original_BFM['texEV']  # eigen value
  texMU = original_BFM['texMU']  # mean texture

  expPC, expEV = LoadExpBasis()

  # transfer BFM09 to our face model

  idBase = shapePC * np.reshape(shapeEV, [-1, 199])
  idBase = idBase / 1e5  # unify the scale to decimeter
  idBase = idBase[:, :80]  # use only first 80 basis

  exBase = expPC * np.reshape(expEV, [-1, 79])
  exBase = exBase / 1e5  # unify the scale to decimeter
  exBase = exBase[:, :64]  # use only first 64 basis

  texBase = texPC * np.reshape(texEV, [-1, 199])
  texBase = texBase[:, :80]  # use only first 80 basis

  # our face model is cropped align face landmarks which contains only 35709 vertex.
  # original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
  # thus we select corresponding vertex to get our face model.

  index_exp = loadmat(os.path.join(model_dir, 'BFM_front_idx.mat'))
  index_exp = index_exp['idx'].astype(np.int32) - 1  # starts from 0 (to 53215)

  index_shape = loadmat(os.path.join(model_dir, 'BFM_exp_idx.mat'))
  index_shape = index_shape['trimIndex'].astype(np.int32) - 1  # starts from 0 (to 53490)
  index_shape = index_shape[index_exp]

  idBase = np.reshape(idBase, [-1, 3, 80])
  idBase = idBase[index_shape, :, :]
  idBase = np.reshape(idBase, [-1, 80])

  texBase = np.reshape(texBase, [-1, 3, 80])
  texBase = texBase[index_shape, :, :]
  texBase = np.reshape(texBase, [-1, 80])

  exBase = np.reshape(exBase, [-1, 3, 64])
  exBase = exBase[index_exp, :, :]
  exBase = np.reshape(exBase, [-1, 64])

  meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
  meanshape = meanshape[index_shape, :]
  meanshape = np.reshape(meanshape, [1, -1])

  meantex = np.reshape(texMU, [-1, 3])
  meantex = meantex[index_shape, :]
  meantex = np.reshape(meantex, [1, -1])

  # other info contains triangles, region used for computing photometric loss,
  # region used for skin texture regularization, and 68 landmarks index etc.
  other_info = loadmat(os.path.join(model_dir, 'facemodel_info.mat'))
  frontmask2_idx = other_info['frontmask2_idx']
  skinmask = other_info['skinmask']
  keypoints = other_info['keypoints']
  point_buf = other_info['point_buf']
  tri = other_info['tri']
  tri_mask2 = other_info['tri_mask2']

  # save our face model
  savemat(os.path.join(model_dir, 'BFM_model_front.mat'),
          {'meanshape': meanshape, 'meantex': meantex, 'idBase': idBase, 'exBase': exBase, 'texBase': texBase,
           'tri': tri, 'point_buf': point_buf, 'tri_mask2': tri_mask2 \
            , 'keypoints': keypoints, 'frontmask2_idx': frontmask2_idx, 'skinmask': skinmask})


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(model_dir='BFM'):
  Lm3D = loadmat(os.path.join(model_dir, 'similarity_Lm3D_all.mat'))
  Lm3D = Lm3D['lm']

  # calculate 5 facial landmarks using 68 landmarks
  lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
  Lm3D = np.stack(
      [Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :],
       Lm3D[lm_idx[6], :]], axis=0)
  Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

  return Lm3D


# save 3D face to obj file
def save_obj(path, v, f, c):
  with open(path, 'w') as file:
    for i in range(len(v)):
      file.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
    # file.write('v %f %f %f\n'%(v[i,0],v[i,1],v[i,2]))

    file.write('\n')

    for i in range(len(f)):
      file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

  file.close()


# calculating least sqaures problem
def POS(xp, x):
  npts = xp.shape[1]

  A = np.zeros([2 * npts, 8])

  A[0:2 * npts - 1:2, 0:3] = x.transpose()
  A[0:2 * npts - 1:2, 3] = 1

  A[1:2 * npts:2, 4:7] = x.transpose()
  A[1:2 * npts:2, 7] = 1;

  b = np.reshape(xp.transpose(), [2 * npts, 1])

  k, _, _, _ = np.linalg.lstsq(A, b)

  R1 = k[0:3]
  R2 = k[4:7]
  sTx = k[3]
  sTy = k[7]
  s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
  t = np.stack([sTx, sTy], axis=0)

  return t, s


def process_img(img, lm, t, s):
  w0, h0 = img.size
  img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0 / 2, 0, 1, h0 / 2 - t[1]))
  w = (w0 / s * 102).astype(np.int32)
  h = (h0 / s * 102).astype(np.int32)
  img = img.resize((w, h), resample=Image.BILINEAR)
  lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) / s * 102

  # crop the image to 224*224 from image center
  left = (w / 2 - 112).astype(np.int32)
  right = left + 224
  up = (h / 2 - 112).astype(np.int32)
  below = up + 224

  img = img.crop((left, up, right, below))
  img = np.array(img)
  img = img[:, :, ::-1]
  img = np.expand_dims(img, 0)
  lm = lm - np.reshape(np.array([(w / 2 - 112), (h / 2 - 112)]), [1, 2])

  return img, lm, t[0] - w0 / 2, h0 / 2 - t[1]


# resize and crop input images before sending to the R-Net
def Preprocess(img, lm, lm3D):
  w0, h0 = img.size

  # change from image plane coordinates to 3D sapce coordinates(X-Y plane)
  lm = np.stack([lm[:, 0], h0 - 1 - lm[:, 1]], axis=1)

  # calculate translation and scale factors using 5 facial landmarks and standard landmarks
  t, s = POS(lm.transpose(), lm3D.transpose())
  #   print('t = {}, s = {}'.format(t,s))

  # processing the image
  img_new, lm_new, t0, t1 = process_img(img, lm, t, s)
  lm_new = np.stack([lm_new[:, 0], 223 - lm_new[:, 1]], axis=1)
  trans_params = np.array([w0, h0, 102.0 / s, t0, t1])

  return img_new, lm_new, trans_params
