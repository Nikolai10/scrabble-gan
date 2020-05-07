import tensorflow as tf
from tensorflow.keras import layers


class ConditionalBatchNorm(layers.Layer):
    """CBN layer as described in https://arxiv.org/pdf/1707.00683.pdf"""

    def __init__(self, name, cbn_idx, conditioning_vector, k_reg):
        super(ConditionalBatchNorm, self).__init__(name='CBN' + '_' + name + '_' + str(cbn_idx))
        self.conditioning_vector = conditioning_vector
        self.k_reg = k_reg

    def call(self, inputs, **kwargs):
        net = layers.BatchNormalization(scale=False, center=False)(inputs)
        num_channels = net.shape[-1]

        # scale
        gamma = layers.Dense(units=num_channels, use_bias=False, activation='linear', kernel_regularizer=self.k_reg,
                             kernel_initializer=tf.initializers.orthogonal())(self.conditioning_vector)
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels])
        net *= gamma

        # shift
        beta = layers.Dense(units=num_channels, use_bias=False, activation='linear', kernel_regularizer=self.k_reg,
                            kernel_initializer=tf.initializers.orthogonal())(self.conditioning_vector)
        beta = tf.reshape(beta, [-1, 1, 1, num_channels])
        net += beta
        return net

    def get_config(self):
        config = super(ConditionalBatchNorm, self).get_config()
        config.update({'cbn_idx': self.cbn_idx, 'conditioning_vector': self.conditioning_vector, 'k_reg': self.k_reg})
        return config


class ResNetBlockUp(layers.Layer):
    def __init__(self, name, output_dim, is_last_block, conditioning_vector, k_reg):
        super(ResNetBlockUp, self).__init__(name='ResNetBlockUp' + '_' + name)

        self.nm = name
        self.output_dim = output_dim
        self.is_last_block = is_last_block
        self.conditioning_vector = conditioning_vector
        self.k_reg = k_reg

    def call(self, inputs, **kwargs):
        net = inputs

        # CBN 1
        net = ConditionalBatchNorm(self.nm, 1, self.conditioning_vector, self.k_reg).call(net)
        net = tf.nn.relu(net)

        # in order to match SrabbleGAN dimension
        up_stride = (2, 1) if self.is_last_block else (2, 2)

        # Upsample
        net = layers.Conv2DTranspose(self.output_dim, (3, 3), strides=up_stride, kernel_regularizer=self.k_reg,
                                     kernel_initializer=tf.initializers.orthogonal(), padding='same', use_bias=True)(
            net)

        # CBN 2
        net = ConditionalBatchNorm(self.nm, 2, self.conditioning_vector, self.k_reg).call(net)
        net = tf.nn.relu(net)
        # Conv
        net = layers.Conv2D(self.output_dim, (3, 3), strides=(1, 1), kernel_initializer=tf.initializers.orthogonal(),
                            kernel_regularizer=self.k_reg, padding='same', use_bias=True)(net)

        # Skip connection
        shortcut = layers.Conv2DTranspose(self.output_dim, (1, 1), strides=up_stride,
                                          kernel_initializer=tf.initializers.orthogonal(),
                                          kernel_regularizer=self.k_reg,
                                          padding='same', use_bias=True)(inputs)
        net += shortcut
        return net

    def get_config(self):
        config = super(ResNetBlockUp, self).get_config()
        config.update(
            {'output_dim': self.output_dim, 'is_last_block': self.is_last_block,
             'conditioning_vector': self.conditioning_vector, 'k_reg': self.k_reg})
        return config


class ResNetBlockDown(layers.Layer):

    def __init__(self, name, output_dim, is_last_block, k_reg):
        super(ResNetBlockDown, self).__init__(name='ResNetBlockDown' + '_' + name)
        self.nm = name
        self.output_dim = output_dim
        self.is_last_block = is_last_block
        self.k_reg = k_reg

    def call(self, inputs, **kwargs):
        net = inputs

        # Conv1
        net = tf.nn.relu(net)
        net = layers.Conv2D(self.output_dim, (3, 3), strides=(1, 1), kernel_initializer=tf.initializers.orthogonal(),
                            kernel_regularizer=self.k_reg, padding='same', use_bias=True)(net)

        # Conv2 (downsample)
        net = tf.nn.relu(net)
        net = layers.Conv2D(self.output_dim, (3, 3), strides=(1, 1), kernel_initializer=tf.initializers.orthogonal(),
                            kernel_regularizer=self.k_reg, padding='same', use_bias=True)(net)
        if not self.is_last_block:
            net = tf.nn.pool(net, window_shape=[2, 2], pooling_type="AVG", padding="SAME", strides=[2, 2])

        # Skip connection
        shortcut = layers.Conv2D(self.output_dim, (1, 1), strides=(1, 1),
                                 kernel_initializer=tf.initializers.orthogonal(),
                                 kernel_regularizer=self.k_reg, padding='same', use_bias=True)(inputs)
        if not self.is_last_block:
            shortcut = tf.nn.pool(shortcut, window_shape=[2, 2], pooling_type="AVG", padding="SAME", strides=[2, 2])
        net += shortcut
        return net

    def get_config(self):
        config = super(ResNetBlockDown, self).get_config()
        config.update({'output_dim': self.output_dim, 'is_last_block': self.is_last_block, 'k_reg': self.k_reg})
        return config
