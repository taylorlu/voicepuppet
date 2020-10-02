import os
import numpy as np
import librosa
import cv2
from scipy.io import wavfile
import resampy


class Loader:
  ### root_path: None if the file_path is full path
  def __init__(self, root_path=None):
    self.root_path = root_path

  ### load txt data, each line split by comma, default float format
  ### file_path: file name in root_path, or full path.
  ### return: numpy array(float32)
  def get_text_data(self, file_path):
    if (self.root_path):
      file_path = os.path.join(self.root_path, file_path)

    with open(file_path) as f:
      lines = f.readlines()
      data_list = []
      for line in lines:
        pts = line.strip().split(',')
        if (len(pts) != 0):
          pts = list(map(lambda x: np.float32(x), pts))
          data_list.append(np.array(pts))

    return np.array(data_list)

  ### load binary data of pickle format.
  ### file_path: file name in root_path, or full path.
  ### return: numpy array(float32)
  def get_bin_data(self, file_path):
    if (self.root_path):
      file_path = os.path.join(self.root_path, file_path)

    if (file_path.endswith('.npy') or file_path.endswith('.npz')):
      data = np.load(file_path)
    return data


class EarLoader(Loader):

  def get_data(self, file_path):
    data = self.get_text_data(file_path)
    return data


class PoseLoader(Loader):

  def get_data(self, file_path):
    data = self.get_text_data(file_path)
    return data


class LandmarkLoader(Loader):
  def __init__(self, root_path=None, norm_size=128):
    Loader.__init__(self, root_path)
    self.norm_size = norm_size

  def get_data(self, file_path):
    data = self.get_text_data(file_path).astype(np.float32)
    data /= self.norm_size
    return data


class BFMCoeffLoader(Loader):

  def get_data(self, file_path):
    data = self.get_text_data(file_path)
    return data


class ImageLoader(Loader):
  def __init__(self, root_path=None, resize=None):
    Loader.__init__(self, root_path)
    self.resize = resize

  def get_data(self, file_path):
    if (self.root_path):
      file_path = os.path.join(self.root_path, file_path)

    data = cv2.imread(file_path).astype(np.float32)
    if (self.resize is not None):
      data = cv2.resize(data, (self.resize[0], self.resize[1]))
    data /= 255.0
    return data


class WavLoader(Loader):
  def __init__(self, root_path=None, sr=16000):
    self.sr = sr
    Loader.__init__(self, root_path)

  def get_data(self, file_path):
    if (self.root_path):
      file_path = os.path.join(self.root_path, file_path)

    data, _ = librosa.load(file_path, sr=self.sr)
    return data


class AudioLoader(Loader):
  def __init__(self, root_path=None, sr=16000):
    self.sr = sr
    Loader.__init__(self, root_path)

  def get_data(self, file_path):
    if (self.root_path):
      file_path = os.path.join(self.root_path, file_path)

    rate, data = wavfile.read(file_path)
    if data.ndim != 1:
        data = data[:,0]

    data = resampy.resample(data.astype(np.float32), rate, self.sr)
    return data
