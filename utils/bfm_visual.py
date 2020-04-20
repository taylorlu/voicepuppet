import numpy as np
import cv2
from PIL import Image
import os

from bfm_load_data import *
from reconstruct_mesh import *
import mesh_core_cython


def isPointInTri(point, tri_points):
  ''' Judge whether the point is in the triangle
  Method:
      http://blackpawn.com/texts/pointinpoly/
  Args:
      point: [u, v] or [x, y]
      tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
  Returns:
      bool: true for in triangle
  '''
  tp = tri_points

  # vectors
  v0 = tp[:, 2] - tp[:, 0]
  v1 = tp[:, 1] - tp[:, 0]
  v2 = point - tp[:, 0]

  # dot products
  dot00 = np.dot(v0.T, v0)
  dot01 = np.dot(v0.T, v1)
  dot02 = np.dot(v0.T, v2)
  dot11 = np.dot(v1.T, v1)
  dot12 = np.dot(v1.T, v2)

  # barycentric coordinates
  if dot00 * dot11 - dot01 * dot01 == 0:
    inverDeno = 0
  else:
    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01)

  u = (dot11 * dot02 - dot01 * dot12) * inverDeno
  v = (dot00 * dot12 - dot01 * dot02) * inverDeno

  # check if point in triangle
  return (u >= 0) & (v >= 0) & (u + v < 1)


def render_texture(vertices, colors, triangles, h, w, c=3):
  ''' render mesh by z buffer
  Args:
      vertices: 3 x nver
      colors: 3 x nver
      triangles: 3 x ntri
      h: height
      w: width
  '''
  # initial
  image = np.zeros((h, w, c), dtype=np.uint8)

  depth_buffer = np.zeros([h, w]) - 999999.
  # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
  tri_depth = (vertices[2, triangles[0, :]] + vertices[2, triangles[1, :]] + vertices[2, triangles[2, :]]) / 3.
  tri_tex = (colors[:, triangles[0, :]] + colors[:, triangles[1, :]] + colors[:, triangles[2, :]]) / 3.

  for i in range(triangles.shape[1]):
    tri = triangles[:, i]  # 3 vertex indices

    # the inner bounding box
    umin = max(int(np.ceil(np.min(vertices[0, tri]))), 0)
    umax = min(int(np.floor(np.max(vertices[0, tri]))), w - 1)

    vmin = max(int(np.ceil(np.min(vertices[1, tri]))), 0)
    vmax = min(int(np.floor(np.max(vertices[1, tri]))), h - 1)

    if umax < umin or vmax < vmin:
      continue

    for u in range(umin, umax + 1):
      for v in range(vmin, vmax + 1):
        if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u, v], vertices[:2, tri]):
          depth_buffer[v, u] = tri_depth[i]
          image[v, u, :] = tri_tex[:, i]
  return image


def plot_bfm_coeff_seq(save_dir, facemodel, step, seq_len, real_bfm_coeff_seq, bfm_coeff_seq, id_coeff=None, texture_coeff=None):
  ## 9*10 block
  block_x = 10
  block_y = 9
  img_size = 224

  def merge_seq(bfm_coeff_seq, big_img, time, h_index):

    for i in range(time):
      face_shape, face_texture, face_color, face_projection, z_buffer, landmarks_2d, translation = Reconstruction(
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


      # shape = np.squeeze(face_shape, (0))
      # color = np.squeeze(face_color, (0))
      # color = np.clip(color, 0, 255).astype(np.int32)
      # shape[:, :2] = 112 - shape[:, :2] * 112

      # new_image = render_texture(shape.T, color.T, (facemodel.tri - 1).astype(int).T, 224, 224, c=3)
      new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

      big_img[(i // block_x + h_index) * img_size: (i // block_x + h_index + 1) * img_size,
      (i % block_x) * img_size: (i % block_x + 1) * img_size] = new_image

    return big_img

  ### We only pick the first sequence of the batch, trim length of 30.
  if (seq_len[0] > 30):
    time = 30
  else:
    time = seq_len[0]

  ### We only pick the first sequence of the batch, trim length of 30.
  if (seq_len[0] > 30):
    time = 30
  else:
    time = seq_len[0]

  big_img = np.zeros((img_size * block_y, img_size * block_x, 3), dtype=np.uint8)
  big_img = merge_seq(real_bfm_coeff_seq, big_img, time, 0)

  if(id_coeff is None or texture_coeff is None):
    bfm_coeff_seq = np.concatenate([real_bfm_coeff_seq[:, :, :80], bfm_coeff_seq[:, :, :], real_bfm_coeff_seq[:, :, 144:]], axis=2)
  else:
    bfm_coeff_seq = np.concatenate([np.tile(id_coeff, (1, real_bfm_coeff_seq.shape[1], 1)), bfm_coeff_seq[:, :, :], np.tile(texture_coeff, (1, real_bfm_coeff_seq.shape[1], 1)), real_bfm_coeff_seq[:, :, 224:]], axis=2)

  big_img = merge_seq(bfm_coeff_seq, big_img, time, 3)

  cv2.imwrite('{}/bfmnet_{}.jpg'.format(save_dir, step), big_img)

