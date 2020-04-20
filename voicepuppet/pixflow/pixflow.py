import tensorflow as tf
import numpy as np
import os
from voicepuppet.builder import ModelBuilder
from config.configure import YParams

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class PixFlowNet(ModelBuilder):

  def __init__(self, config_path):
    if (not os.path.exists(config_path)):
      logger.error('config_path not exists.')
      exit(0)

    self.__params = PixFlowNet.default_hparams(config_path)

  @staticmethod
  def default_hparams(config_path, name='default'):
    params = YParams(config_path, name)
    params.add_hparam('separable_conv', False)
    params.add_hparam('ngf', 64)
    params.add_hparam('ndf', 48)
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

  def build_network(self, inputs, fg_inputs, trainable=True):

    def discrim_conv(batch_input, out_channels, stride):
      padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
      return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                              kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def gen_conv(batch_input, out_channels, kernel_size=4):
      # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
      initializer = tf.random_normal_initializer(0, 0.02)
      if self.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=kernel_size, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
      else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=kernel_size, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)

    def gen_deconv(batch_input, out_channels, kernel_size=4):
      # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
      initializer = tf.random_normal_initializer(0, 0.02)
      if self.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=kernel_size, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
      else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=kernel_size, strides=(2, 2), padding="same",
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

    def resnet(batch_input, out_channels):
      initializer = tf.random_normal_initializer(0, 0.02)
      convolved = tf.layers.conv2d(batch_input, out_channels, kernel_size=3, strides=(1, 1), padding="same",
                                kernel_initializer=initializer)
      normalized = batchnorm(convolved)
      rectified = lrelu(normalized, 0.2)

      if(trainable):
        rectified = tf.nn.dropout(rectified, keep_prob=0.5)

      convolved = tf.layers.conv2d(rectified, out_channels, kernel_size=3, strides=(1, 1), padding="same",
                                kernel_initializer=initializer)
      normalized = batchnorm(convolved)

      return batch_input + normalized

    def encoder_net(batch_input):
      layers = []
      # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
      with tf.variable_scope("encoder_conv"):
        output = gen_conv(batch_input, self.ngf, kernel_size=7)
        layers.append(output)

      layer_specs = [
        self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        self.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        self.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
      ]

      for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers))):
          rectified = lrelu(layers[-1], 0.2)
          # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
          convolved = gen_conv(rectified, out_channels)
          output = batchnorm(convolved)
          layers.append(output)
      return layers[-1]

    def diffnet(batch_input):
      layers = []
      with tf.variable_scope("diff_conv"):
        output = gen_conv(batch_input, self.ngf, kernel_size=7)
        layers.append(output)

      layer_specs = [
        self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        self.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        self.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
      ]

      for out_channels in layer_specs:
        with tf.variable_scope("diff_%d" % (len(layers))):
          rectified = lrelu(layers[-1], 0.2)
          # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
          convolved = gen_conv(rectified, out_channels)
          output = batchnorm(convolved)
          layers.append(output)
      return layers[-1]

    def decoder_net(batch_input, generator_outputs_channels):
      layers = [batch_input]
      layer_specs = [
        self.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        self.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
      ]

      for out_channels in layer_specs:
        with tf.variable_scope("post_resnet_%d" % (len(layers))):
          output = resnet(layers[-1], out_channels)
          layers.append(output)

      layer_specs = [
        (self.ngf * 8),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (self.ngf * 4),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (self.ngf * 2),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
      ]

      for decoder_layer, out_channels in enumerate(layer_specs):
        with tf.variable_scope("decoder_%d" % (decoder_layer)):

          rectified = tf.nn.relu(layers[-1])
          # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
          output = gen_deconv(rectified, out_channels)
          output = batchnorm(output)
          layers.append(output)

      # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
      with tf.variable_scope("final"):
        rectified = tf.nn.relu(layers[-1])
        output = gen_deconv(rectified, generator_outputs_channels, kernel_size=7)
        output = tf.tanh(output)

      return output

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

    def create_generator(generator_inputs, generator_fg_inputs, generator_outputs_channels):
      with tf.variable_scope("encoder_net"):
        encode_feat = encoder_net(generator_fg_inputs[..., :3])

      with tf.variable_scope("diffnet"):
        feat0 = diffnet(generator_inputs[..., :3])
      with tf.variable_scope("diffnet", reuse=True):
        feat1 = diffnet(generator_inputs[..., 3:])
      diff_feat = feat1 - feat0

      pre_resnet_layers = [encode_feat]
      layer_specs = [
        self.ngf * 8,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        self.ngf * 8,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
      ]
      for out_channels in layer_specs:
        with tf.variable_scope("pre_resnet_%d" % (len(pre_resnet_layers))):
          output = resnet(pre_resnet_layers[-1], out_channels)
          pre_resnet_layers.append(output)

      diff_resnet_layers = [diff_feat]
      layer_specs = [
        self.ngf * 8,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        self.ngf * 8,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
      ]
      for out_channels in layer_specs:
        with tf.variable_scope("diff_resnet_%d" % (len(diff_resnet_layers))):
          output = resnet(diff_resnet_layers[-1], out_channels)
          diff_resnet_layers.append(output)

      with tf.variable_scope("decoder_net"):
        added_layers = pre_resnet_layers[-1] + diff_resnet_layers[-1]
        output = decoder_net(added_layers, generator_outputs_channels)
      return output

    nodes = {}
    with tf.variable_scope("generator"):
      output = create_generator(inputs, fg_inputs, generator_outputs_channels=4)
      rgb = output[..., :3]
      alpha = (output[..., 3:]+1)/2
      alpha = tf.tile(alpha, [1,1,1,3])
      # output = rgb * alpha + targets[..., 3:] * (1 - alpha)
      output = rgb * alpha + alpha - 1

      nodes.update({'Outputs': output})
      nodes.update({'Alphas': alpha})

    if(trainable):
      # create two copies of discriminator, one for real pairs and one for fake pairs
      # they share the same underlying variables
      with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
          # inputs = tf.reshape(inputs, [self.batch_size, inputs.shape[1], inputs.shape[2], 2, 3])
          # inputs = tf.transpose(inputs, [3, 0, 1, 2, 4])
          # inputs = tf.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], 3])

          # foregrounds = tf.reshape(fg_inputs, [self.batch_size, fg_inputs.shape[1], fg_inputs.shape[2], 2, 3])
          # foregrounds = tf.transpose(foregrounds, [3, 0, 1, 2, 4])
          # foregrounds = tf.reshape(foregrounds, [-1, foregrounds.shape[2], foregrounds.shape[3], 3])

          predict_real = create_discriminator(inputs[..., 3:], fg_inputs[..., 3:])
          nodes.update({'Predict_real': predict_real})

      with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
          # predict_fake = create_discriminator(inputs, tf.concat([fg_inputs[..., 3:], output], axis=0))
          predict_fake = create_discriminator(inputs[..., 3:], output)
          nodes.update({'Predict_fake': predict_fake})

    return nodes

  def add_cost_function(self, predict_real, predict_fake, fg_inputs, outputs, alphas, masks):
    nodes = {}
    with tf.name_scope("discriminator_loss"):
      # minimizing -tf.log will try to get inputs to 1
      # predict_real => 1
      # predict_fake => 0
      discrim_loss = tf.reduce_mean(-(tf.log(predict_real + 1e-12) + tf.log(1 - predict_fake + 1e-12)))
      nodes.update({'Discrim_loss': discrim_loss})

    with tf.name_scope("generator_loss"):
      # predict_fake => 1
      # abs(targets - outputs) => 0
      gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + 1e-12))
      gen_loss_L1 = tf.reduce_mean(tf.abs(fg_inputs[..., 3:] - outputs))
      gen_loss_L1 += tf.reduce_mean(tf.abs(masks - alphas))
      gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 * self.l1_weight
      nodes.update({'Gen_loss_GAN': gen_loss_GAN})
      nodes.update({'Gen_loss_L1': gen_loss_L1})
      nodes.update({'Gen_loss': gen_loss})
      return nodes

  def build_train_op(self, inputs, fg_inputs, masks):

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
    nodes.update({'FG_Inputs': fg_inputs})
    nodes.update({'Masks': masks})
    inputs = preprocess(inputs)
    fg_inputs = preprocess(fg_inputs)

    network_dict = self.build_network(inputs, fg_inputs, trainable=True)
    nodes.update(network_dict)

    loss_dict = self.add_cost_function(nodes['Predict_real'], nodes['Predict_fake'], fg_inputs, nodes['Outputs'], nodes['Alphas'], nodes['Masks'])
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

  def build_inference_op(self, inputs, targets):
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
    nodes.update({'Targets': targets})
    inputs = preprocess(inputs)
    targets = preprocess(targets)

    network_dict = self.build_network(inputs, targets, trainable=False)
    nodes.update(network_dict)
    nodes.update({"Outputs": deprocess(nodes['Outputs'])})

    return nodes
