import tensorflow as tf
from tensorflow.keras import layers


class NonLocalBlock(layers.Layer):
    """Self-attention (non-local) block.
    This method is used to exactly reproduce SAGAN and ignores Gin settings on
    weight initialization and spectral normalization.
    based on non_local_block
    - https://github.com/google/compare_gan/blob/master/compare_gan/architectures/arch_ops.py
    - https://github.com/brain-research/self-attention-gan/blob/master/non_local.py
    """

    def __init__(self, name, k_reg):
        super(NonLocalBlock, self).__init__(name='NonLocalBlock' + '_' + name)
        self.k_reg = k_reg
        b_init = tf.zeros_initializer()
        self.sigma = tf.Variable(
            initial_value=b_init(shape=[], dtype="float32"), trainable=True
        )

    def build(self, input_shape):
        h, w, num_channels = input_shape.as_list()[1:]
        self.h = h
        self.w = w
        self.num_channels = num_channels
        self.num_channels_attn = self.num_channels // 8
        self.num_channels_g = self.num_channels // 2
        self.conv2d_1 = layers.Conv2D(filters=self.num_channels_attn, kernel_size=(1, 1), use_bias=False, strides=(1, 1),
                              padding='same', kernel_initializer=tf.initializers.orthogonal(),
                              kernel_regularizer=self.k_reg, name="conv2d_theta")
        self.conv2d_2 = layers.Conv2D(filters=self.num_channels_attn, kernel_size=(1, 1), use_bias=False, strides=(1, 1),
                            padding='same', kernel_initializer=tf.initializers.orthogonal(),
                            kernel_regularizer=self.k_reg, name="conv2d_phi")
        self.conv2d_3 = layers.Conv2D(filters=self.num_channels_g, kernel_size=(1, 1), use_bias=False, strides=(1, 1),
                          padding='same', kernel_initializer=tf.initializers.orthogonal(),
                          kernel_regularizer=self.k_reg, name="conv2d_g")
        self.conv2d_4 = layers.Conv2D(filters=self.num_channels, kernel_size=(1, 1), use_bias=False, strides=(1, 1),
                               padding='same', kernel_initializer=tf.initializers.orthogonal(),
                               kernel_regularizer=self.k_reg, name="conv2d_attn_g")

    def _spatial_flatten(self, inputs):
        shape = inputs.shape
        return tf.reshape(inputs, (tf.shape(inputs)[0], -1, shape[3]))

    def call(self, input):
  
        # Theta path
        theta = self.conv2d_1(input)
        theta = self._spatial_flatten(theta)

        # Phi path
        phi = self.conv2d_2(input)
        phi = layers.MaxPool2D(pool_size=[2, 2], strides=2)(phi)
        phi = self._spatial_flatten(phi)

        # attention map
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        # G path
        g = self.conv2d_3(input)
        g = layers.MaxPool2D(pool_size=[2, 2], strides=2)(g)
        g = self._spatial_flatten(g)

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [tf.shape(attn_g)[0], self.h, -1, self.num_channels_g])
        attn_g = self.conv2d_4(attn_g)

        return self.sigma * attn_g + input

    def get_config(self):
        config = super(NonLocalBlock, self).get_config()
        config.update({'k_reg': self.k_reg, 'sigma': self.sigma.numpy(), 'h': self.h, 'w': self.w, 
        'num_channels': self.num_channels, 'num_channels_attn': self.num_channels_attn, 'num_channels_g': self.num_channels_g, 
        'conv2d_1': self.conv2d_1, 'conv2d_2': self.conv2d_2, 'conv2d_3': self.conv2d_3, 'conv2d_4': self.conv2d_4})
        return config


# filterbank as described in https://arxiv.org/pdf/2003.10557.pdf
# equal to tf.keras.layers.Embedding + Reshape
class SpatialEmbedding(layers.Layer):

    def __init__(self, vocab_size, filter_dim):
        super(SpatialEmbedding, self).__init__(name='SpatialEmbedding')
        self.vocab_size = vocab_size
        self.filter_dim = filter_dim

    def build(self, input_shape):
        self.kernel = self.add_weight("filter_bank",
                                      shape=[self.vocab_size, self.filter_dim[0], self.filter_dim[1]],
                                      trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.kernel, inputs)

    def get_config(self):
        config = super(SpatialEmbedding, self).get_config()
        config.update({'vocab_size': self.vocab_size, 'filter_dim': self.filter_dim})
        return config


@tf.keras.utils.register_keras_serializable(package='Custom', name='spectral_norm')
def spectral_norm(w, power_iteration=1):
    """
    based on https://github.com/taki0112/Spectral_Normalization-Tensorflow
    :param w:
    :param power_iteration:
    :return:
    """

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.random.normal([1, w_shape[-1]])

    u_hat = u
    v_hat = None
    for i in range(power_iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    w_norm = w / sigma
    w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

