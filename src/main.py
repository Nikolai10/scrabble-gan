# import sys
# sys.path.extend(['/home/ubuntu/workspace/scrabble-gan'])

import os
import random

import gin
import numpy as np
import tensorflow as tf

from src.bigacgan.arch_ops import spectral_norm
from src.bigacgan.data_utils import load_prepare_data, train, make_gif, load_random_word_list
from src.bigacgan.net_architecture import make_generator, make_discriminator, make_recognizer, make_gan
from src.bigacgan.net_loss import hinge, not_saturating

gin.external_configurable(hinge)
gin.external_configurable(not_saturating)
gin.external_configurable(spectral_norm)

from src.dinterface.dinterface import init_reading

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@gin.configurable
def setup_optimizer(g_lr, d_lr, r_lr, beta_1, beta_2, loss_fn, disc_iters):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    recognizer_optimizer = tf.keras.optimizers.Adam(learning_rate=r_lr, beta_1=beta_1, beta_2=beta_2)
    return generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters


@gin.configurable('shared_specs')
def get_shared_specs(epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention):
    return epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention


@gin.configurable('io')
def setup_io(base_path, checkpoint_dir, gen_imgs_dir, model_dir, raw_dir, read_dir, input_dim, buf_size, n_classes,
             seq_len, char_vec, bucket_size):
    gen_path = base_path + gen_imgs_dir
    ckpt_path = base_path + checkpoint_dir
    m_path = base_path + model_dir
    raw_dir = base_path + raw_dir
    read_dir = base_path + read_dir
    return input_dim, buf_size, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec


def main():
    # init params
    gin.parse_config_file('scrabble_gan.gin')
    epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention = get_shared_specs()
    in_dim, buf_size, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec = setup_io()

    # convert IAM Handwriting dataset (words) to GAN format
    if not os.path.exists(read_dir):
        print('converting iamDB-Dataset to GAN format...')
        init_reading(raw_dir, read_dir, in_dim, bucket_size)

    # load random words into memory (used for word generation by G)
    random_words = load_random_word_list(read_dir, bucket_size, char_vec)

    # load and preprocess dataset (python generator)
    train_dataset = load_prepare_data(in_dim, batch_size, read_dir, char_vec, bucket_size)

    # init generator, discriminator and recognizer
    generator = make_generator(latent_dim, in_dim, embed_y, gen_path, kernel_reg, g_bw_attention, n_classes)
    discriminator = make_discriminator(gen_path, in_dim, kernel_reg, d_bw_attention)
    recognizer = make_recognizer(in_dim, seq_len, n_classes + 1, gen_path)

    # build composite model (update G through composite model)
    gan = make_gan(generator, discriminator, recognizer, gen_path)

    # init optimizer for both generator, discriminator and recognizer
    generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters = setup_optimizer()

    # purpose: save and restore models
    checkpoint_prefix = os.path.join(ckpt_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     recognizer_optimizer=recognizer_optimizer,
                                     generator=generator,
                                     discriminator=discriminator,
                                     recognizer=recognizer)

    # reuse this seed + labels overtime to visualize progress in the animated GIF
    seed = tf.random.normal([num_gen, latent_dim])
    random_bucket_idx = random.randint(4, bucket_size - 1)
    labels = np.array([random.choice(random_words[random_bucket_idx]) for _ in range(num_gen)], np.int32)

    # start training
    train(train_dataset, generator, discriminator, recognizer, gan, checkpoint, checkpoint_prefix, generator_optimizer,
          discriminator_optimizer, recognizer_optimizer, [seed, labels], buf_size, batch_size, epochs, m_path,
          latent_dim, gen_path, loss_fn, disc_iters, random_words, bucket_size, char_vec)

    # use imageio to create an animated gif using the images saved during training.
    make_gif(gen_path)


if __name__ == "__main__":
    main()
