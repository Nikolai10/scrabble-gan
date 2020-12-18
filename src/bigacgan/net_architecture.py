import os

import tensorflow as tf
from tensorflow.keras import layers
from src.bigacgan.arch_ops import NonLocalBlock, SpatialEmbedding
from src.bigacgan.resnet_ops import ResNetBlockUp, ResNetBlockDown


def make_recognizer(input_dim, sequence_length, output_classes, gen_path, vis_model=True):
    """
    Build (fully convolutional) CRNN network based on https://arxiv.org/abs/1507.05717

    :param input_dim:
    :param sequence_length:
    :param output_classes:
    :param gen_path:
    :param vis_model:
    :return:
    """

    h, w, c = input_dim
    w = None
    # define input layer
    inp_imgs = layers.Input(shape=(h, w, c), name='input_images')

    # Convolutional Layers
    # ============================================= 1st layer ==============================================#
    conv_1 = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', strides=1)(inp_imgs)
    pool_1 = layers.MaxPool2D(pool_size=(2, 2))(conv_1)
    # ============================================= 2nd layer ==============================================#
    conv_2 = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', strides=1)(pool_1)
    pool_2 = layers.MaxPool2D(pool_size=(2, 2))(conv_2)
    # ============================================= 3rd layer ==============================================#
    conv_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=1)(pool_2)
    # ============================================= 4th layer ==============================================#
    conv_4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=1)(conv_3)
    # "In the 3rd and the 4th max-pooling layers, we adopt 1x2 sized rectangular pooling windows instead"
    pool_4 = layers.MaxPool2D(pool_size=(2, 1))(conv_4)
    # ============================================= 5th layer ==============================================#
    conv_5 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', strides=1)(pool_4)
    # "two batch normalization layers are inserted after the 5th and 6th convolutional layers respectively."
    batch_norm_5 = layers.BatchNormalization()(conv_5)
    # ============================================= 6th layer ==============================================#
    conv_6 = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', strides=1)(
        batch_norm_5)
    batch_norm_6 = layers.BatchNormalization()(conv_6)
    pool_6 = layers.MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    # ============================================= 7th layer ==============================================#
    conv_7 = layers.Conv2D(filters=512, kernel_size=2, activation='relu', padding='valid', strides=1)(pool_6)
    # ========================================== Map-to-Sequence ===========================================#
    # (None, 1, X, 512) -> (None, X, 512)
    map_to_seq = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 1))(conv_7)

    # Per frame predictions (skip RNN layers -> avoid learning implicit language model)
    per_frame_predictions = tf.keras.layers.Dense(output_classes, activation='softmax')(map_to_seq)

    def ctc_loss(args):
        """
        to better understand the meaning of the params:
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/ctc_batch_cost?version=stable
        :return:
        """
        y_true, y_pred, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    labels = layers.Input(name='y_true', shape=[sequence_length], dtype='float32')
    input_length = layers.Input(shape=[1], dtype=tf.int32, name='input_length')
    label_length = layers.Input(name='label_length', shape=[1], dtype=tf.int32)

    # ========================================== Transcription layer ===================================#
    loss_out = tf.keras.layers.Lambda(ctc_loss, output_shape=(1,), name='ctc')(
        [labels, per_frame_predictions, input_length, label_length])

    model = tf.keras.Model(inputs=[inp_imgs, labels, input_length, label_length], outputs=loss_out)

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'Recognizer.png')

    return model


def make_generator(latent_dim, input_dim, embed_y, gen_path, kernel_reg, blocks_with_attention, vocab_size,
                   vis_model=True):
    """
    (fully convolutional) Generator based on

    - https://arxiv.org/pdf/1809.11096.pdf
    - https://github.com/google/compare_gan/blob/master/compare_gan/architectures/resnet_biggan_deep.py
    - https://arxiv.org/pdf/2003.10557.pdf

    (1) instead of using the unpooling operation for upsampling, we use layers.Conv2DTranspose as described in
        https://www.tensorflow.org/tutorials/generative/dcgan
    (2) output values are within [-1, 1] rather than [0, 1]

    TODOs:
    - skip connection     [ok]
    - orthogonal init     [ok]
    - hierarchical_z      [ok]
    - CBN                 [ok]
    - shared_embedding    [ok] -> filter bank
    - spectral norm       [ok] -> optional
    - self-attention      [ok] -> optional

    :param latent_dim:
    :param input_dim:
    :param embed_y:
    :param gen_path:
    :param kernel_reg:
    :param blocks_with_attention:
    :param vocab_size:
    :param vis_model:
    :return:
    """

    h, _, c = input_dim

    in_channels, out_channels = get_in_out_channels_gen(h)
    num_blocks = len(in_channels)

    # each upsample-round doubles spatial map; compute seed_size_h based on num_blocks to match final target dimension
    seed_size_h = int(h / 2 ** num_blocks)
    seed_size_w = seed_size_h

    # input latent vec + label
    z = layers.Input(shape=(latent_dim,))
    y = layers.Input(shape=(None,), dtype=tf.int32)

    # retrieve spatial embedding
    spatial_embedding = SpatialEmbedding(vocab_size=vocab_size, filter_dim=embed_y)
    se_layer = spatial_embedding(y)

    # hierarchical z
    z_per_block = tf.split(z, num_blocks + 1, axis=1)
    z0, z_per_block = z_per_block[0], z_per_block[1:]
    z0 = tf.reshape(z0, (-1, 1, 1, z0.shape[1]))

    # matrix multiply each "char-patch/ spatial embedding" with z0
    net = tf.linalg.matmul(tf.tile(z0, [1, tf.shape(se_layer)[1], 1, 1]), se_layer)
    net = tf.squeeze(net, axis=2)

    # align image patches side by side (without rearranging the pixel values within the image patches)
    net = tf.reshape(net, (tf.shape(net)[0], 512, seed_size_h, seed_size_w, -1))
    net = tf.reshape(net, (tf.shape(net)[0], -1, 512, seed_size_h))
    net = tf.transpose(net, (0, 3, 1, 2))

    # ResNetBlock "up"
    for block_idx in range(num_blocks):
        name = "B{}".format(block_idx + 1)
        is_last_block = block_idx == num_blocks - 1
        net = ResNetBlockUp(name, out_channels[block_idx], is_last_block, z_per_block[block_idx], kernel_reg).call(net)
        if name in blocks_with_attention:
            net = NonLocalBlock(name, kernel_reg)(net)

    net = layers.BatchNormalization()(net)
    net = tf.nn.relu(net)
    net = layers.Conv2D(c, (3, 3),
                        strides=(1, 1),
                        kernel_initializer=tf.initializers.orthogonal(),
                        kernel_regularizer=kernel_reg,
                        padding='same')(net)

    net = tf.nn.tanh(net)
    # define model
    model = tf.keras.Model([z, y], net)

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=gen_path + 'Generator.png')
    return model


def make_discriminator(gen_path, input_dim, kernel_reg, blocks_with_attention, vis_model=True):
    """
     (fully convolutional) Discriminator based on

    - https://arxiv.org/pdf/1809.11096.pdf and
    - https://github.com/google/compare_gan/blob/master/compare_gan/architectures/resnet_biggan_deep.py
    - https://arxiv.org/pdf/2003.10557.pdf

    - orthogonal init     [ok]
    - skip connection     [ok]
    - project-y           here not necessary (use auxiliary classifier instead)
    - shared_embedding    here not necessary (use auxiliary classifier instead)
    - spectral norm       [ok]
    - self-attention      [ok]

    :param gen_path:
    :param input_dim:
    :param kernel_reg:
    :param blocks_with_attention:
    :param vis_model:
    :return:
    """

    h, w, c = input_dim
    w = None
    in_channels, out_channels = get_in_out_channels_disc(colors=c, resolution=h)
    num_blocks = len(in_channels)

    x = layers.Input(shape=(h, w, c))
    net = x

    # ResNetBlock "down"
    for block_idx in range(num_blocks):
        name = "B{}".format(block_idx + 1)
        is_last_block = block_idx == num_blocks - 1
        net = ResNetBlockDown(name, out_channels[block_idx], is_last_block, kernel_reg).call(net)
        if name in blocks_with_attention:
            net = NonLocalBlock(name, kernel_reg)(net)

    # Final part
    net = tf.nn.relu(net)
    net_h = layers.GlobalAveragePooling2D()(net)
    out_logit = layers.Dense(units=1,
                             use_bias=False,
                             activation='linear',
                             kernel_regularizer=kernel_reg,
                             kernel_initializer=tf.initializers.orthogonal())(net_h)

    out = out_logit
    # define model
    model = tf.keras.Model([x], [out])

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'Discriminator.png')

    return model


def make_gan(g_model, d_model, r_model, gen_path, vis_model=True):
    """
    Update G through composite model

    :param g_model:
    :param d_model:
    :param r_model:
    :param gen_path:
    :param vis_model:
    :return:
    """

    d_model.trainable = False
    r_model.trainable = False

    input_length = layers.Input(shape=[1], dtype=tf.int32, name='input_length')
    label_length = layers.Input(name='label_length', shape=[1], dtype=tf.int32)

    # run D + R on image generated by G
    gan_output_d = d_model(g_model.output, training=True)
    gan_output_r = r_model([g_model.output, g_model.input[1], input_length, label_length], training=True)

    # define composite model
    model = tf.keras.Model(g_model.input + [input_length, label_length], [g_model.output, gan_output_d, gan_output_r])

    if vis_model:
        model.summary()
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'ScrabbleGAN.png')

    return model


# we use the params suggested by https://arxiv.org/pdf/2003.10557.pdf
def get_in_out_channels_gen(resolution=32):
    ch = 64
    if resolution == 32:
        channel_multipliers = [8, 4, 2, 1]
    else:
        raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [ch * c for c in channel_multipliers[:-1]]
    out_channels = [ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels


def get_in_out_channels_disc(colors=1, resolution=32):
    ch = 64
    if colors not in [1, 3]:
        raise ValueError("Unsupported color channels: {}".format(colors))
    if resolution == 32:
        channel_multipliers = [1, 8, 16, 16]
    else:
        raise ValueError("Unsupported resolution: {}".format(resolution))
    out_channels = [ch * c for c in channel_multipliers]
    in_channels = [colors] + out_channels[:-1]
    return in_channels, out_channels

