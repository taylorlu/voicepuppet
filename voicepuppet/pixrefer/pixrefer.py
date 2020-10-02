import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from voicepuppet.builder import ModelBuilder
from config.configure import YParams
import vgg_simple as vgg
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class PixReferNet(ModelBuilder):

  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = PixReferNet.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('separable_conv', False)
    params.add_hparam('ngf', 64)
    params.add_hparam('ndf', 64)
    params.add_hparam('l1_weight', 500.0)
    params.add_hparam('gan_weight', 1.0)

    params.training['learning_rate'] = 0.0003
    params.training['beta1'] = 0.5
    params.training['decay_rate'] = 0.999

    return params

  @property
  def params(self):
    return self.__params

  def set_params(self, params):
    self.learning_rate = params.training['learning_rate']
    self.beta1 = params.training['beta1']
    self.decay_rate = params.training['decay_rate']
    self.decay_steps = params.training['decay_steps']
    self.batch_size = params.batch_size

    self.separable_conv = params.separable_conv
    self.ngf = params.ngf
    self.ndf = params.ndf
    self.l1_weight = params.l1_weight
    self.gan_weight = params.gan_weight
    if(params.is_training):
      self.sess = params.sess
      self.vgg_model_path = params.vgg_model_path

  def build_network(self, inputs, fg_inputs, targets, trainable=True):

    def discrim_conv(batch_input, out_channels, stride):
      padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
      return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                              kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def gen_conv(batch_input, out_channels):
      # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
      initializer = tf.random_normal_initializer(0, 0.02)
      if self.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
      else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)

    def gen_deconv(batch_input, out_channels):
      # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
      initializer = tf.random_normal_initializer(0, 0.02)
      if self.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
      else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=initializer)

    def lrelu(x, a):
      with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(inputs):
      return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                           gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    def create_discriminator(discrim_inputs, discrim_targets):
      n_layers = 3
      layers = []

      # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
      input = tf.concat([discrim_inputs, discrim_targets], axis=3)

      # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
      with tf.variable_scope("layer_1"):
        convolved = discrim_conv(input, self.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

      # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
      # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
      # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
      for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
          out_channels = self.ndf * min(2 ** (i + 1), 8)
          stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
          convolved = discrim_conv(layers[-1], out_channels, stride=stride)
          normalized = batchnorm(convolved)
          rectified = lrelu(normalized, 0.2)
          layers.append(rectified)

      # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
      with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

      return layers[-1]

    def create_target_discriminator(discrim_inputs):
      n_layers = 3
      layers = []

      # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
      with tf.variable_scope("layer_1"):
        convolved = discrim_conv(discrim_inputs, self.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

      # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
      # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
      # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
      for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
          out_channels = self.ndf * min(2 ** (i + 1), 8)
          stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
          convolved = discrim_conv(layers[-1], out_channels, stride=stride)
          normalized = batchnorm(convolved)
          rectified = lrelu(normalized, 0.2)
          layers.append(rectified)

      # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
      with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

      return layers[-1]

    def create_generator(generator_inputs, generator_fg_inputs, generator_outputs_channels):
      layers = []

      # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
      with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, self.ngf)
        layers.append(output)

      layer_specs = [
        self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        self.ngf * 2,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        self.ngf * 4,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
      ]

      for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
          rectified = lrelu(layers[-1], 0.2)
          # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
          convolved = gen_conv(rectified, out_channels)
          output = batchnorm(convolved)
          layers.append(output)

      fg_layers = []
      # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
      with tf.variable_scope("encoder_fg_1"):
        output = gen_conv(generator_fg_inputs, self.ngf)
        fg_layers.append(output)

      layer_specs = [
        self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        self.ngf * 2,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        self.ngf * 4,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
      ]

      for out_channels in layer_specs:
        with tf.variable_scope("encoder_fg_%d" % (len(fg_layers) + 1)):
          rectified = lrelu(fg_layers[-1], 0.2)
          # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
          convolved = gen_conv(rectified, out_channels)
          output = batchnorm(convolved)
          fg_layers.append(output)

      merged_layers = [tf.concat([layers[-1], fg_layers[-1]], axis=3)]

      layer_specs = [
        self.ngf * 4,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        self.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        self.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        self.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
      ]

      for out_channels in layer_specs:
        with tf.variable_scope("merged_encoder_%d" % (len(merged_layers) + 1)):
          rectified = lrelu(merged_layers[-1], 0.2)
          # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
          convolved = gen_conv(rectified, out_channels)
          output = batchnorm(convolved)
          merged_layers.append(output)

      layer_specs = [
        (self.ngf * 8),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (self.ngf * 8),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (self.ngf * 4),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (self.ngf * 4),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
      ]

      num_encoder_layers = len(merged_layers)
      for decoder_layer, out_channels in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("merged_decoder_%d" % (skip_layer + 1)):
          if decoder_layer == 0:
            # first decoder layer doesn't have skip connections
            # since it is directly connected to the skip_layer
            input = merged_layers[-1]
          else:
            input = tf.concat([merged_layers[-1], merged_layers[skip_layer]], axis=3)

          rectified = tf.nn.relu(input)
          # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
          output = gen_deconv(rectified, out_channels)
          output = batchnorm(output)

          merged_layers.append(output)

      layer_specs = [
        (self.ngf * 2),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (self.ngf * 2),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (self.ngf),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
      ]

      num_encoder_layers = len(layers)
      for decoder_layer, out_channels in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("merged2_decoder_%d" % (skip_layer + 1)):
          input = tf.concat([merged_layers[-1], layers[skip_layer]], axis=3)

          rectified = tf.nn.relu(input)
          # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
          output = gen_deconv(rectified, out_channels)
          output = batchnorm(output)

          merged_layers.append(output)

      # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
      with tf.variable_scope("decoder_1"):
        input = tf.concat([merged_layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

      return layers[-1]

    nodes = {}
    with tf.variable_scope("generator"):
      output = create_generator(inputs, fg_inputs[..., :3], generator_outputs_channels=4)
      rgb = output[:,:,:,:3]
      alpha = (output[:,:,:,3:]+1)/2
      alpha = tf.tile(alpha, [1,1,1,3])
      output = rgb * alpha + targets * (1 - alpha)
      output_fg = rgb * alpha + alpha - 1

      nodes.update({'Outputs': output})
      nodes.update({'Alphas': alpha})
      nodes.update({'Outputs_FG': output_fg})

    if(trainable):
      # create two copies of discriminator, one for real pairs and one for fake pairs
      # they share the same underlying variables
      with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
          predict_real = create_discriminator(inputs[..., 3:], fg_inputs[..., 3:])
        with tf.variable_scope("discriminator", reuse=True):
          predict_real2 = create_discriminator(inputs[..., :3], fg_inputs[..., :3])
          predict_real = (predict_real + predict_real2)/2
          nodes.update({'Predict_real': predict_real})

      with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
          predict_fake = create_discriminator(inputs[..., 3:], output_fg)
          nodes.update({'Predict_fake': predict_fake})

      # with tf.name_scope("real_target_discriminator"):
      #   with tf.variable_scope("target_discriminator"):
      #     predict_real = create_target_discriminator(fg_inputs[:, 384:, :, 3:])
      #     nodes.update({'Predict_real_target': predict_real})

      # with tf.name_scope("fake_target_discriminator"):
      #   with tf.variable_scope("target_discriminator", reuse=True):
      #     predict_fake = create_target_discriminator(output_fg[:, 384:, :, :])
      #     nodes.update({'Predict_fake_target': predict_fake})

      with tf.name_scope("vgg_perceptual"):
        with slim.arg_scope(vgg.vgg_arg_scope()):

          f1, f2, f3, f4, exclude = vgg.vgg_16(tf.concat([fg_inputs[..., 3:], output_fg], axis=0))
          gen_f, img_f = tf.split(f3, 2, 0)
          content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))

          vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
          init_fn = slim.assign_from_checkpoint_fn(self.vgg_model_path, vgg_vars)
          init_fn(self.sess)
          nodes.update({'Perceptual_loss': content_loss})

    return nodes

  def add_cost_function(self, predict_real, predict_fake, perceptual_loss, targets, outputs, alphas, masks):
    nodes = {}
    with tf.name_scope("discriminator_loss"):
      # minimizing -tf.log will try to get inputs to 1
      # predict_real => 1
      # predict_fake => 0
      discrim_loss = tf.reduce_mean(-(tf.log(predict_real + 1e-12)*2 + tf.log(1 - predict_fake + 1e-12)))
      # discrim_loss += tf.reduce_mean(-(tf.log(predict_real_target + 1e-12)*2 + tf.log(1 - predict_fake_target + 1e-12)))
      nodes.update({'Discrim_loss': discrim_loss})

    with tf.name_scope("generator_loss"):
      # predict_fake => 1
      # abs(targets - outputs) => 0
      gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + 1e-12))
      # gen_loss_GAN += tf.reduce_mean(-tf.log(predict_fake_target + 1e-12))
      gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
      gen_loss_L1 += tf.reduce_mean(tf.abs(masks - alphas))
      gen_loss_L1 += tf.reduce_mean(perceptual_loss)
      gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 * self.l1_weight
      nodes.update({'Gen_loss_GAN': gen_loss_GAN})
      nodes.update({'Gen_loss_L1': gen_loss_L1})
      nodes.update({'Gen_loss': gen_loss})
      return nodes

  def build_train_op(self, inputs, fg_inputs, targets, masks):

    def preprocess(image):
      with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

    def deprocess(image):
      with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

    nodes = {}
    nodes.update({'Inputs': inputs})
    nodes.update({'FGInputs': fg_inputs})
    nodes.update({'Targets': targets})
    nodes.update({'Masks': masks})
    inputs = preprocess(inputs)
    fg_inputs = preprocess(fg_inputs)
    targets = preprocess(targets)

    network_dict = self.build_network(inputs, fg_inputs, targets, trainable=True)
    nodes.update(network_dict)

    loss_dict = self.add_cost_function(nodes['Predict_real'], 
                                       nodes['Predict_fake'], 
                                       nodes['Perceptual_loss'], 
                                       targets, 
                                       nodes['Outputs'], 
                                       nodes['Alphas'], 
                                       nodes['Masks'])
    nodes.update(loss_dict)
    nodes.update({"Outputs": deprocess(nodes['Outputs'])})

    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                    self.decay_steps, self.decay_rate, staircase=True)
    nodes.update({'Global_step': global_step})
    nodes.update({'Lr': lr})

    with tf.name_scope("discriminator_train"):
      discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
      discrim_optim = tf.train.AdamOptimizer(lr, self.beta1)
      discrim_grads_and_vars = discrim_optim.compute_gradients(nodes['Discrim_loss'], var_list=discrim_tvars)
      discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars, global_step=global_step)

    with tf.name_scope("generator_train"):
      with tf.control_dependencies([discrim_train]):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(lr, self.beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(nodes['Gen_loss'], var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=global_step)

    nodes.update({'Train_op': gen_train})
    nodes.update({'Discrim_grads_and_vars': discrim_grads_and_vars})
    nodes.update({'Gen_grads_and_vars': gen_grads_and_vars})
    return nodes

  def build_inference_op(self, inputs, fg_inputs, targets):
    def preprocess(image):
      with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

    def deprocess(image):
      with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

    nodes = {}
    nodes.update({'Inputs': inputs})
    nodes.update({'FGInputs': fg_inputs})
    nodes.update({'Targets': targets})
    inputs = preprocess(inputs)
    fg_inputs = preprocess(fg_inputs)
    targets = preprocess(targets)

    network_dict = self.build_network(inputs, fg_inputs, targets, trainable=False)
    nodes.update(network_dict)
    nodes.update({"Outputs": deprocess(nodes['Outputs'])})
    nodes.update({"Outputs_FG": deprocess(nodes['Outputs_FG'] + nodes['Alphas'] -1)})

    return nodes
