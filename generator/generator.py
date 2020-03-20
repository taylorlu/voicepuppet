import tensorflow as tf
import os
import numpy as np
from loader import *
import random
from optparse import OptionParser
import logging
import sys
import math
import python_speech_features
import librosa
import subprocess
from config.configure import YParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataGenerator:
  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = DataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.sample_rate = params.mel['sample_rate']
    self.num_mel_bins = params.mel['num_mel_bins']
    self.win_length = params.mel['win_length']
    self.hop_step = params.mel['hop_step']
    self.fft_length = params.mel['fft_length']
    self.frame_rate = params.frame_rate
    self.frame_wav_scale = self.sample_rate / self.frame_rate
    self.frame_mfcc_scale = self.frame_wav_scale / self.hop_step

    assert (self.frame_mfcc_scale - int(self.frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

    self.frame_mfcc_scale = int(self.frame_mfcc_scale)

  def iterator(self):
    raise NotImplementError('iterator not implemented.')

  def get_dataset(self):
    raise NotImplementError('get_dataset not implemented.')

  def extract_mfcc(self, pcm):
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    pcm = tf.convert_to_tensor(pcm, dtype=tf.float32)
    stfts = tf.signal.stft(pcm, frame_length=self.win_length, frame_step=self.hop_step, fft_length=self.fft_length)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins,
                                                                        num_spectrogram_bins,
                                                                        self.sample_rate,
                                                                        lower_edge_hertz,
                                                                        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, axes=[[2], [0]])
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms

  def ear_compute(self, landmarks):
    ears = []
    for ps in landmarks:
      ps = map(lambda x: float(x), ps)

      EAR1 = float((math.sqrt((ps[74] - ps[82]) ** 2 + (ps[75] - ps[83]) ** 2) + math.sqrt(
          (ps[76] - ps[80]) ** 2 + (ps[77] - ps[81]) ** 2))) / math.sqrt(
          (ps[72] - ps[78]) ** 2 + (ps[73] - ps[79]) ** 2)
      EAR2 = float((math.sqrt((ps[86] - ps[94]) ** 2 + (ps[87] - ps[95]) ** 2) + math.sqrt(
          (ps[88] - ps[92]) ** 2 + (ps[89] - ps[93]) ** 2))) / math.sqrt(
          (ps[84] - ps[90]) ** 2 + (ps[85] - ps[91]) ** 2)
      EAR = (EAR1 + EAR2) / 2
      ears.append([EAR])

    return np.array(ears)

  def split_bfmcoeff(self, coeff):
    id_coeff = coeff[:80]  # identity(shape) coeff of dim 80
    ex_coeff = coeff[80:144]  # expression coeff of dim 64
    tex_coeff = coeff[144:224]  # texture(albedo) coeff of dim 80
    angle = coeff[224:227]  # ruler angle(x,y,z) for rotation of dim 3
    gamma = coeff[227:254]  # lighting coeff for 3 channel SH function of dim 27
    translation = coeff[254:]  # translation coeff of dim 3

    return id_coeff, ex_coeff, tex_coeff, angle, gamma, translation

  def pose_compute(self, bfmcoeffs):
    poses = []
    for bfmcoeff in bfmcoeffs:
      _, _, _, angle, _, _ = self.split_bfmcoeff(bfmcoeff)
      poses.append(angle)

    return np.array(poses)


class ATNetDataGenerator(DataGenerator):
  def __init__(self, config_path):

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = ATNetDataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('dataset_path', params.train_dataset_path)

    ### Image sequence length when training
    params.add_hparam('max_squence_len', 70)
    params.add_hparam('min_squence_len', 30)
    params.add_hparam('shuffle_bufsize', 100)
    params.add_hparam('img_size', 224)
    params.add_hparam('batch_size', 32)

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.data_list = open(params.dataset_path).readlines()
    self.sample_rate = params.mel['sample_rate']
    self.num_mel_bins = params.mel['num_mel_bins']
    self.win_length = params.mel['win_length']
    self.hop_step = params.mel['hop_step']
    self.fft_length = params.mel['fft_length']
    self.frame_rate = params.frame_rate
    self.frame_wav_scale = self.sample_rate / self.frame_rate
    self.frame_mfcc_scale = self.frame_wav_scale / self.hop_step

    assert (self.frame_mfcc_scale - int(self.frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

    self.frame_mfcc_scale = int(self.frame_mfcc_scale)
    self.shuffle_bufsize = params.shuffle_bufsize

    self.bfmcoeff_name = params.sample_file['bfmcoeff_name']
    self.landmark_name = params.sample_file['landmark_name']
    self.wav_name = params.sample_file['wav_name']
    self.img_size = [params.img_size, params.img_size]
    self.mean = np.load(params.mean_file)
    self.component = np.load(params.components_file)

    self.max_squence_len = params.max_squence_len
    self.min_squence_len = params.min_squence_len

    self.batch_size = params.batch_size

  def iterator(self):
    bfmcoeff_loader = BFMCoeffLoader()
    landmark_loader = LandmarkLoader(norm_size=1)
    wav_loader = WavLoader(sr=self.sample_rate)

    random.shuffle(self.data_list)

    for line in self.data_list:
      folder, img_count = line.strip().split('|')
      img_count = int(img_count)

      bfmcoeff = bfmcoeff_loader.get_data(os.path.join(folder, self.bfmcoeff_name))
      landmark = landmark_loader.get_data(os.path.join(folder, self.landmark_name))
      pcm = wav_loader.get_data(os.path.join(folder, self.wav_name))

      if (bfmcoeff is not None and
          landmark is not None and
          pcm is not None and
          img_count > 0):
        if (bfmcoeff.shape[0] == img_count and
            landmark.shape[0] == img_count):

          pose = self.pose_compute(bfmcoeff)
          ear = self.ear_compute(landmark)

          # renorm to (-1,1) by svd, enhance the mouth, eye and pose feature.
          landmark /= self.img_size[0]
          landmark -= 0.5
          landmark *= 2
          landmark = np.dot((landmark - self.mean), self.component[:, :6])
          landmark *= 2 * np.array([0.5, 0.5, 0.5, 1.3, 0.5, 0.5])
          landmark = np.dot(landmark, self.component[:, :6].T)

          if (self.min_squence_len > img_count):
            continue
          if (img_count < self.max_squence_len):
            rnd_len = random.randint(self.min_squence_len, img_count)
          else:
            rnd_len = random.randint(self.min_squence_len, self.max_squence_len)

          rnd_len = 25
          slice_cnt = img_count // rnd_len

          for i in range(slice_cnt):
            landmark_slice = landmark[i * rnd_len: (i + 1) * rnd_len, :]
            ear_slice = ear[i * rnd_len: (i + 1) * rnd_len, :]
            pose_slice = pose[i * rnd_len: (i + 1) * rnd_len, :]
            # calculate the rational length of pcm in order to keep the alignment of mfcc and landmark sequence.
            pcm_start = int(i * rnd_len * self.frame_wav_scale)
            pcm_length = self.hop_step * (rnd_len * self.frame_mfcc_scale - 1) + self.win_length
            if (pcm.shape[0] < pcm_start + pcm_length):
              pcm = np.pad(pcm, (0, pcm_start + pcm_length - pcm.shape[0]), 'constant', constant_values=(0))
            pcm_slice = pcm[pcm_start: pcm_start + pcm_length]
            rnd_idx = random.randint(0, landmark_slice.shape[0] - 1)
            yield landmark_slice, ear_slice, pose_slice, pcm_slice, landmark_slice[rnd_idx, :], landmark_slice.shape[0]

  def process_data(self, landmark, ear, pose, pcm, example_landmark, seq_len):
    mfcc = self.extract_mfcc(pcm)
    return landmark, ear, pose, mfcc, example_landmark, seq_len

  def get_dataset(self):
    self.set_params(self.__params)

    dataset = tf.data.Dataset.from_generator(
        self.iterator,
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=([None, 136], [None, 1], [None, 3], [None], [136], [])
    )

    dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
    dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([None, 136], [None, 1], [None, 3], [None], [136], []))
    dataset = dataset.map(
        self.process_data,
        num_parallel_calls=4)

    return dataset


class VGNetDataGenerator(DataGenerator):
  def __init__(self, config_path):

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = VGNetDataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('dataset_path', params.train_dataset_path)

    ### Image sequence length when training
    params.add_hparam('max_squence_len', 30)
    params.add_hparam('min_squence_len', 20)
    params.add_hparam('shuffle_bufsize', 100)
    params.add_hparam('img_size', 128)
    params.add_hparam('batch_size', 4)

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.data_list = open(params.dataset_path).readlines()
    self.shuffle_bufsize = params.shuffle_bufsize
    self.max_squence_len = params.max_squence_len
    self.min_squence_len = params.min_squence_len
    self.batch_size = params.batch_size
    self.img_size = [params.img_size, params.img_size]
    self.landmark_name = params.sample_file['landmark_name']
    self.mean = np.load(params.mean_file)
    self.component = np.load(params.components_file)

  def iterator(self):
    landmark_loader = LandmarkLoader(norm_size=224.0)
    image_loader = ImageLoader(resize=self.img_size)

    def face_region_bylmk(landmark):
      if ((landmark < 1).all()):  # In [0,1]
        landmark *= self.img_size[0]
        landmark = landmark.astype(np.int32)
      landmark = cv2.convexHull(landmark)
      mask = np.zeros(self.img_size, dtype=np.uint8)
      mask = cv2.fillConvexPoly(mask, landmark.astype(np.int32), color=(255))
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
      mask = cv2.dilate(mask, kernel).astype(np.float32) / 256.0
      return mask

    random.shuffle(self.data_list)

    for line in self.data_list:
      folder, img_count = line.strip().split('|')
      img_count = int(img_count)

      imgs = []
      for i in range(img_count):
        img = image_loader.get_data(os.path.join(folder, '{}.jpg'.format(i)))
        if (img is not None):
          imgs.append(img)
      imgs = np.array(imgs)

      landmarks = landmark_loader.get_data(os.path.join(folder, self.landmark_name))

      if (landmarks is not None and
          img_count > 0):
        if (landmarks.shape[0] == img_count and
            imgs.shape[0] == img_count):

          masks = []
          for landmark in np.split(landmarks, landmarks.shape[0]):
            mask = face_region_bylmk(landmark.copy().reshape([-1, 2]))
            masks.append(mask)
          masks = np.array(masks)[..., np.newaxis]

          # renorm to (-1,1) by svd, enhance the mouth, eye and pose feature.
          landmarks -= 0.5
          landmarks *= 2
          landmarks = np.dot((landmarks - self.mean), self.component[:, :6])
          landmarks *= 2 * np.array([0.5, 0.5, 0.5, 1.3, 0.5, 0.5])
          landmarks = np.dot(landmarks, self.component[:, :6].T)

          ## Generate random length in [min_squence_len, max_squence_len]
          if (self.min_squence_len > img_count):
            continue
          if (img_count < self.max_squence_len):
            rnd_len = random.randint(self.min_squence_len, img_count)
          else:
            rnd_len = random.randint(self.min_squence_len, self.max_squence_len)

          rnd_len = 15
          slice_cnt = img_count // rnd_len

          for i in range(slice_cnt):
            landmark_slice = landmarks[i * rnd_len: (i + 1) * rnd_len, :]
            mask_slice = masks[i * rnd_len: (i + 1) * rnd_len, ...]
            img_slice = imgs[i * rnd_len: (i + 1) * rnd_len, ...]
            rnd_idx = random.randint(0, landmark_slice.shape[0] - 1)
            yield landmark_slice, mask_slice, img_slice, landmark_slice[rnd_idx, :], img_slice[rnd_idx, ...], \
                  landmark_slice.shape[
                    0]

  def get_dataset(self):
    self.set_params(self.__params)

    dataset = tf.data.Dataset.from_generator(
        self.iterator,
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        output_shapes=(
          [None, 136], [None, self.img_size[0], self.img_size[1], 1], [None, self.img_size[0], self.img_size[1], 3],
          [136], [self.img_size[0], self.img_size[1], 3],
          [])
    )

    dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
    dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([None, 136], [None, self.img_size[0], self.img_size[1], 1],
                                                  [None, self.img_size[0], self.img_size[1], 3], [136],
                                                  [self.img_size[0], self.img_size[1], 3], []))

    return dataset


class BFMNetDataGenerator(DataGenerator):
  def __init__(self, config_path):

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = BFMNetDataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('dataset_path', params.train_dataset_path)

    ### Image sequence length when training
    params.add_hparam('max_squence_len', 30)
    params.add_hparam('min_squence_len', 20)
    params.add_hparam('shuffle_bufsize', 100)
    params.add_hparam('batch_size', 16)

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.data_list = open(params.dataset_path).readlines()
    self.sample_rate = params.mel['sample_rate']
    self.num_mel_bins = params.mel['num_mel_bins']
    self.win_length = params.mel['win_length']
    self.hop_step = params.mel['hop_step']
    self.fft_length = params.mel['fft_length']
    self.frame_rate = params.frame_rate
    self.frame_wav_scale = self.sample_rate / self.frame_rate
    self.frame_mfcc_scale = self.frame_wav_scale / self.hop_step

    assert (self.frame_mfcc_scale - int(self.frame_mfcc_scale) == 0), "sample_rate/hop_step must divided by frame_rate."

    self.frame_mfcc_scale = int(self.frame_mfcc_scale)
    self.shuffle_bufsize = params.shuffle_bufsize

    self.landmark_name = params.sample_file['landmark_name']
    self.wav_name = params.sample_file['wav_name']
    self.bfmcoeff_name = params.sample_file['bfmcoeff_name']

    self.max_squence_len = params.max_squence_len
    self.min_squence_len = params.min_squence_len

    self.batch_size = params.batch_size

  def iterator(self):
    bfmcoeff_loader = BFMCoeffLoader()
    wav_loader = WavLoader(sr=self.sample_rate)

    random.shuffle(self.data_list)

    for line in self.data_list:
      folder, img_count = line.strip().split('|')
      img_count = int(img_count)

      bfmcoeffs = bfmcoeff_loader.get_data(os.path.join(folder, self.bfmcoeff_name))
      pcm = wav_loader.get_data(os.path.join(folder, self.wav_name))

      if (bfmcoeffs is not None and
          pcm is not None and
          img_count > 0):
        if (bfmcoeffs.shape[0] == img_count):

          # if (self.min_squence_len > img_count):
          #   continue
          # if (img_count < self.max_squence_len):
          #   rnd_len = random.randint(self.min_squence_len, img_count)
          # else:
          #   rnd_len = random.randint(self.min_squence_len, self.max_squence_len)

          rnd_len = 20
          intervals = librosa.effects.split(pcm, top_db=20)
          sil_rm_start = intervals[0][0] // self.frame_wav_scale
          pcm = pcm[intervals[0][0]:]
          bfmcoeffs = bfmcoeffs[sil_rm_start:, :]
          img_count = img_count - sil_rm_start

          slice_cnt = img_count // rnd_len

          for i in range(slice_cnt):
            bfmcoeff_slice = bfmcoeffs[i * rnd_len: (i + 1) * rnd_len, :]
            # calculate the rational length of pcm in order to keep the alignment of mfcc and bfmcoeff sequence.
            pcm_start = int(i * rnd_len * self.frame_wav_scale)
            pcm_length = self.hop_step * (rnd_len * self.frame_mfcc_scale - 1) + self.win_length
            if (pcm.shape[0] < pcm_start + pcm_length):
              pcm = np.pad(pcm, (0, pcm_start + pcm_length - pcm.shape[0]), 'constant', constant_values=(0))
            pcm_slice = pcm[pcm_start: pcm_start + pcm_length]
            yield bfmcoeff_slice, pcm_slice, bfmcoeff_slice.shape[0]

  def process_data(self, bfmcoeff, pcm, seq_len):
    mfcc = self.extract_mfcc(pcm)
    return bfmcoeff, mfcc, seq_len

  def get_dataset(self):
    self.set_params(self.__params)

    dataset = tf.data.Dataset.from_generator(
        self.iterator,
        output_types=(tf.float32, tf.float32, tf.int32),
        output_shapes=([None, 257], [None], [])
    )

    dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
    dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([None, 257], [None], []))
    dataset = dataset.map(
        self.process_data,
        num_parallel_calls=4)

    return dataset


class Pix2PixDataGenerator(DataGenerator):
  def __init__(self, config_path):

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = Pix2PixDataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('dataset_path', params.train_dataset_path)

    ### Image sequence length when training
    params.add_hparam('shuffle_bufsize', 100)
    params.add_hparam('batch_size', 16)

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.data_list = open(params.dataset_path).readlines()
    self.shuffle_bufsize = params.shuffle_bufsize
    self.batch_size = params.batch_size

  def iterator(self):
    image_loader = ImageLoader()

    random.shuffle(self.data_list)

    for line in self.data_list:
      folder, img_count = line.strip().split('|')
      img_count = int(img_count)

      if (img_count > 0):
        imgs = []
        for i in range(img_count):
          img = image_loader.get_data(os.path.join(folder, '{}.jpg'.format(i)))
          if (img is not None):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = np.array(imgs)

        if (imgs.shape[0] == img_count):
          targets = imgs[:, :, :256, :]
          inputs = imgs[:, :, 256:512, :]
          masks = imgs[:, :, 512:, :]
          ## padding 2 empty frames before image sequence.
          inputs = np.concatenate([np.zeros([2, inputs.shape[1], inputs.shape[2], inputs.shape[3]], dtype=inputs.dtype), inputs], axis=0)
          for i in range(img_count):
            input_slice = inputs[i: i + 3, ...]
            input_slice = input_slice.transpose((1, 2, 0, 3))
            input_slice = input_slice.reshape([256, 256, 9])
            yield input_slice, targets[i, ...], masks[i, ...]

  def get_dataset(self):
    self.set_params(self.__params)

    dataset = tf.data.Dataset.from_generator(
        self.iterator,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=([256, 256, 3*3], [256, 256, 3], [256, 256, 3])
    )

    dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
    dataset = dataset.padded_batch(self.batch_size,
                                   padded_shapes=([256, 256, 3*3], [256, 256, 3], [256, 256, 3]))

    return dataset


class Audio2ExpNetDataGenerator(DataGenerator):
  def __init__(self, config_path):

    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = Audio2ExpNetDataGenerator.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('dataset_path', params.train_dataset_path)

    ### Image sequence length when training
    params.add_hparam('max_squence_len', 30)
    params.add_hparam('min_squence_len', 20)
    params.add_hparam('shuffle_bufsize', 100)
    params.add_hparam('batch_size', 16)

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.data_list = open(params.dataset_path).readlines()
    self.sample_rate = params.mel['sample_rate']
    self.num_mel_bins = params.mel['num_mel_bins']
    self.win_length = params.mel['win_length']
    self.hop_step = params.mel['hop_step']
    self.frame_rate = params.frame_rate
    self.deepspeech_model = params.deepspeech_model
    self.frame_feature_scale = self.sample_rate / self.hop_step / self.frame_rate / 2

    self.shuffle_bufsize = params.shuffle_bufsize

    self.landmark_name = params.sample_file['landmark_name']
    self.wav_name = params.sample_file['wav_name']
    self.bfmcoeff_name = params.sample_file['bfmcoeff_name']

    self.max_squence_len = params.max_squence_len
    self.min_squence_len = params.min_squence_len

    self.batch_size = params.batch_size

  # def process_data(self, bfmcoeff, features, seq_len):
  #   return bfmcoeff, features, seq_len

  def get_dataset(self):
    # self.set_params(self.__params)

    with tf.gfile.GFile(self.deepspeech_model, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

      graph = tf.get_default_graph()
      tf.import_graph_def(graph_def, name="deepspeech")
      input_tensor = graph.get_tensor_by_name('deepspeech/input_node:0')
      seq_length = graph.get_tensor_by_name('deepspeech/input_lengths:0')
      layer_6 = graph.get_tensor_by_name('deepspeech/logits:0')

      os.environ["CUDA_VISIBLE_DEVICES"] = '0'
      # self.sess = tf.Session(graph=self.graph)
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(graph=graph, config=config)
      # with tf.Session(graph=graph, config=config) as sess:

      # with tf.Session(graph=self.graph) as sess:
      def iterator():

        def interpolate_features(features, input_rate, output_rate, output_len=None):
          num_features = features.shape[1]
          input_len = features.shape[0]
          seq_len = input_len / float(input_rate)
          if output_len is None:
            output_len = int(seq_len * output_rate)
          input_timestamps = np.arange(input_len) / float(input_rate)
          output_timestamps = np.arange(output_len) / float(output_rate)
          output_features = np.zeros((output_len, num_features))
          for feat in range(num_features):
            output_features[:, feat] = np.interp(output_timestamps,
                                                 input_timestamps,
                                                 features[:, feat])
          return output_features

        def audioToInputVector(audio, fs):
          numcontext = 9
          # Get mfcc coefficients
          features = python_speech_features.mfcc(audio, samplerate=fs, numcep=self.num_mel_bins, winlen=0.025, winstep=0.01)

          # We only keep every second feature (BiRNN stride = 2)
          features = features[::2]

          # One stride per time step in the input
          num_strides = len(features)

          # Add empty initial and final contexts
          empty_context = np.zeros((numcontext, self.num_mel_bins), dtype=features.dtype)
          features = np.concatenate((empty_context, features, empty_context))

          # Create a view into the array with overlapping strides of size
          # numcontext (past) + 1 (present) + numcontext (future)
          window_size = 2 * numcontext + 1
          train_inputs = np.lib.stride_tricks.as_strided(
              features,
              (num_strides, window_size, self.num_mel_bins),
              (features.strides[0], features.strides[0], features.strides[1]),
              writeable=False)

          # Flatten the second and third dimensions
          train_inputs = np.reshape(train_inputs, [num_strides, -1])

          train_inputs = np.copy(train_inputs)
          train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

          # Return results
          return train_inputs

        def proProcessVector(features):
          # One stride per time step in the input
          num_strides = len(features)

          # Add empty initial and final contexts
          left_empty_context = np.zeros((4, features.shape[1]), dtype=features.dtype)
          right_empty_context = np.zeros((3, features.shape[1]), dtype=features.dtype)
          features = np.concatenate((left_empty_context, features, right_empty_context))

          # Create a view into the array with overlapping strides of size
          window_size = 8
          train_inputs = np.lib.stride_tricks.as_strided(
              features,
              (num_strides, window_size, features.shape[1]),
              (features.strides[0], features.strides[0], features.strides[1]),
              writeable=False)

          # Return results
          return train_inputs

        bfmcoeff_loader = BFMCoeffLoader()
        audio_loader = AudioLoader(sr=self.sample_rate)

        random.shuffle(self.data_list)

        # with tf.Session(graph=self.graph) as sess:
        for line in self.data_list:
          folder, img_count = line.strip().split('|')
          img_count = int(img_count)

          bfmcoeffs = bfmcoeff_loader.get_data(os.path.join(folder, self.bfmcoeff_name))
          pcm = audio_loader.get_data(os.path.join(folder, self.wav_name))

          if (bfmcoeffs is not None and
              pcm is not None and
              img_count > 0):
            if (bfmcoeffs.shape[0] == img_count):

              if (self.min_squence_len > img_count):
                continue
              if (img_count < self.max_squence_len):
                rnd_len = random.randint(self.min_squence_len, img_count)
              else:
                rnd_len = random.randint(self.min_squence_len, self.max_squence_len)

              rnd_len = 25
              slice_cnt = img_count // rnd_len

              input_vector = audioToInputVector(pcm, self.sample_rate)
              network_output = sess.run(layer_6, feed_dict={input_tensor: input_vector[np.newaxis, ...],
                                                seq_length: [input_vector.shape[0]]})

              # Resample network output to self.frame_rate fps
              audio_len_s = float(pcm.shape[0]) / self.sample_rate
              num_frames = int(round(audio_len_s * self.frame_rate))
              network_output = interpolate_features(network_output[:, 0], self.frame_rate*self.frame_feature_scale, self.frame_rate,
                                                    output_len=num_frames)
              network_output = np.squeeze(network_output)
              if(network_output.shape[0] < img_count):
                network_output = np.pad(network_output, ([0, img_count - network_output.shape[0]], [0, 0]), 'constant', constant_values=(0))

              for i in range(slice_cnt):
                bfmcoeff_slice = bfmcoeffs[i * rnd_len: (i + 1) * rnd_len, :]
                length = rnd_len
                start = int(i * length)

                network_output_slice = network_output[start: start + length]
                network_output_slice = proProcessVector(network_output_slice)

                yield bfmcoeff_slice, network_output_slice, bfmcoeff_slice.shape[0]

      dataset = tf.data.Dataset.from_generator(
          iterator,
          output_types=(tf.float32, tf.float32, tf.int32),
          output_shapes=([None, 257], [None, 8, 29], [])
      )

      dataset = dataset.shuffle(self.shuffle_bufsize).repeat()
      dataset = dataset.padded_batch(self.batch_size,
                                     padded_shapes=([None, 257], [None, 8, 29], []))
      # dataset = dataset.map(
      #     self.process_data,
      #     num_parallel_calls=4)

    return dataset
