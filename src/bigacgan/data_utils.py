import glob
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import imageio
import matplotlib.pyplot as plt
import random


def load_prepare_data(input_dim, batch_size, reading_dir, char_vector, bucket_size):
    """
    load data into tensor (python generator)

    (1) read buckets into memory
    (2) compute bucket_weights
    (3) create python generator

    :param input_dim:
    :param batch_size:
    :param reading_dir:
    :param char_vector:
    :param bucket_size:
    :return:
    """

    h, w, c = input_dim

    data_buckets = {}
    bucket_weights = {}
    number_samples = 0

    # (1) read buckets into memory
    for i in range(1, bucket_size + 1, 1):

        imgs = []
        labels = []

        reading_dir_bucket = reading_dir + str(i) + '/'
        file_list = os.listdir(reading_dir_bucket)
        file_list = [fi for fi in file_list if fi.endswith(".txt")]

        for file in file_list:
            with open(reading_dir_bucket + file, 'r', encoding='utf8') as f:
                # 'auto' -> [0, 20, 19, 14]
                label = [char_vector.index(char) for char in f.readline()]
                img = cv2.imread(os.path.join(reading_dir_bucket, os.path.splitext(file)[0] + '.png'), 0)
                imgs.append(img)
                labels.append(label)
                number_samples += 1

        data_buckets[i] = (imgs, labels)

    # (2) compute bucket_weights
    for i in range(1, bucket_size + 1, 1):
        bucket_weights[i] = len(data_buckets[i][1]) / number_samples

    # (3) create python generator
    while True:
        # select random bucket (follow transcription length distribution)
        random_bucket_idx = np.random.choice(bucket_size, 1, p=[value for value in bucket_weights.values()]) + 1
        random_bucket_idx = int(random_bucket_idx[0])

        image_batch = []
        label_batch = []

        for i in range(batch_size):
            # retrieve random samples from bucket of size batch_size
            sample_idx = random.randint(0, len(data_buckets[random_bucket_idx][1]) - 1)
            image_batch.append(data_buckets[random_bucket_idx][0][sample_idx])
            label_batch.append(data_buckets[random_bucket_idx][1][sample_idx])

        # convert to numpy array
        image_batch = np.array(image_batch).astype('float32')
        label_batch = np.array(label_batch).astype(np.int32)

        # normalize images to [-1, 1]
        image_batch = image_batch.reshape(-1, h, int((h / 2) * random_bucket_idx), c)
        image_batch = (image_batch - 127.5) / 127.5

        yield (image_batch, label_batch)


def train(dataset, generator, discriminator, recognizer, composite_gan, checkpoint, checkpoint_prefix,
          generator_optimizer, discriminator_optimizer, recognizer_optimizer, seed_labels, buffer_size, batch_size,
          epochs, model_path, latent_dim, gen_path, loss_fn, disc_iters, random_words, bucket_size, char_vector):
    """
    Whole training procedure

    :param dataset:                     python generator
    :param generator:                   generator model
    :param discriminator:               discriminator model
    :param recognizer:                  Auxiliary Classifier (CRNN)
    :param composite_gan:               Composite GAN
    :param checkpoint:                  tf.train.Checkpoint
    :param checkpoint_prefix:           directory path where to store checkpoints
    :param generator_optimizer:         generator optimizer
    :param discriminator_optimizer:     discriminator optimizer
    :param recognizer_optimizer:        recognizer optimizer
    :param seed_labels:                 noise vector + random labels
    :param buffer_size:                 buffer size (number of training samples)
    :param batch_size:                  batch size
    :param epochs:                      number of epochs
    :param model_path:                  directory path where to store trained model
    :param latent_dim:                  noise vector
    :param gen_path:                    directory path where to store generated images
    :param loss_fn:                     loss function
    :param disc_iters:                  take <disc_iters> D steps per G step
    :param random_words:                list of random words
    :param bucket_size:                 max sequence length
    :param char_vector:                 valid vocabulary represented as array of chars/ string
    :return:
    """

    batch_per_epoch = int(buffer_size / batch_size) + 1

    print('training...')
    for epoch_idx in range(epochs):
        start = time.time()

        for batch_idx in range(batch_per_epoch):
            image_batch, label_batch = next(dataset)

            train_step(epoch_idx, batch_idx, batch_per_epoch, image_batch, label_batch, discriminator, recognizer,
                       composite_gan, generator_optimizer, discriminator_optimizer, recognizer_optimizer, batch_size,
                       latent_dim, loss_fn, disc_iters, random_words, bucket_size)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch_idx + 1, seed_labels, gen_path, char_vector)

        # Save the model every 5 epochs
        if (epoch_idx + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch_idx + 1, time.time() - start))

    # save generator model
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    generator.save(model_path + 'generator_{}'.format(epochs), save_format='tf')


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function
def train_step(epoch_idx, batch_idx, batch_per_epoch, images, labels, discriminator, recognizer, composite_gan,
               generator_optimizer, discriminator_optimizer, recognizer_optimizer, batch_size, latent_dim, loss_fn,
               disc_iters, random_words, bucket_size):
    """
    Single training loop

    :param epoch_idx:                   current epoch idx
    :param batch_idx:                   current batch idx
    :param batch_per_epoch:             number of batches per epoch
    :param images:                      batch of images
    :param labels:                      batch of labels
    :param discriminator:               discriminator model
    :param recognizer:                  Auxiliary Classifier (CRNN)
    :param composite_gan:               Composite GAN
    :param generator_optimizer:         generator optimizer
    :param discriminator_optimizer:     discriminator optimizer
    :param recognizer_optimizer:        recognizer optimizer
    :param batch_size:                  batch size
    :param latent_dim:                  noise vector
    :param loss_fn:                     loss function (gin configurable)
    :param disc_iters:                  take <disc_iters> D steps per G step
    :param random_words:                list of random words
    :param bucket_size:                 max sequence length
    :return:
    """

    # generate latent points + random sequence labels from word list
    noise = tf.random.normal([batch_size, latent_dim])
    random_bucket_idx = random.randint(0, bucket_size - 1)
    fake_labels = np.array([random.choice(random_words[random_bucket_idx]) for _ in range(batch_size)], np.int32)

    # obtain shapes
    batch_size_real = images.shape[0]
    sequence_length_real = len(labels[0])
    sequence_length_fake = random_bucket_idx + 1

    # compute loss & update gradients
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as rec_tape:
        # generate images + compute D(fake) + R(fake)
        inp_len_fake = -1 + sequence_length_fake * 4
        gen_images, d_fake_logits, r_fake_logits = composite_gan(
            [noise, fake_labels, np.array([[inp_len_fake]] * batch_size),
             np.array([[sequence_length_fake]] * batch_size)], training=True)

        # compute D(real)
        d_real_logits = discriminator([images], training=True)

        # compute R(real)
        inp_len_real = -1 + sequence_length_real * 4
        r_real_logits = recognizer([images, labels, np.array([[inp_len_real]] * batch_size_real),
                                    np.array([[sequence_length_real]] * batch_size_real)], training=True)

        # compute losses
        d_loss, d_loss_real, d_loss_fake, g_loss = loss_fn(d_real_logits, d_fake_logits)

        # apply gradient balancing (optional)
        g_loss_balanced = g_loss + r_fake_logits
        # g_loss_balanced = apply_gradient_balancing(r_fake_logits, g_loss, alpha=1)

        # compute stats
        r_loss_fake_mean = tf.reduce_mean(r_fake_logits)
        r_loss_real_mean = tf.reduce_mean(r_real_logits)
        g_loss_mean = tf.reduce_mean(g_loss)
        d_loss_mean = tf.reduce_mean(d_loss)
        d_loss_real_mean = tf.reduce_mean(d_loss_real)
        d_loss_fake_mean = tf.reduce_mean(d_loss_fake)
        g_loss_balanced_mean = tf.reduce_mean(g_loss_balanced)

    tf.print('>%d, %d/%d, d=%.3f, d_real=%.3f, d_fake=%.3f, g_trad=%.3f, r_loss_fake=%.3f, g_loss=%.3f, r=%.3f' % (
        epoch_idx + 1, batch_idx + 1, batch_per_epoch, d_loss_mean, d_loss_real_mean, d_loss_fake_mean, g_loss_mean,
        r_loss_fake_mean, g_loss_balanced_mean, r_loss_real_mean))

    # compute and apply gradients of D and R
    discriminator.trainable = True
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    recognizer.trainable = True
    gradients_of_recognizer = rec_tape.gradient(r_real_logits, recognizer.trainable_variables)
    recognizer_optimizer.apply_gradients(zip(gradients_of_recognizer, recognizer.trainable_variables))

    # take disc_iters (default 1) D steps per G step
    if (batch_idx + 1) % disc_iters == 0:
        # compute and apply gradients of G
        recognizer.trainable = False
        discriminator.trainable = False
        gradients_of_generator = gen_tape.gradient(g_loss_balanced, composite_gan.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, composite_gan.trainable_variables))


def apply_gradient_balancing(r_fake_logits, g_loss, alpha=1):
    """
    gradient balancing based on https://arxiv.org/pdf/2003.10557.pdf

    :param r_fake_logits:   output of tf.keras.backend.ctc_batch_cost
    :param g_loss:          output of tf.nn.sigmoid_cross_entropy_with_logits
    :param alpha:           controls relative importance of L_R and L_D
    :return:
    """

    r_loss_fake_std = tf.math.reduce_std(r_fake_logits)
    g_loss_std = tf.math.reduce_std(g_loss)
    return g_loss + alpha * (((g_loss_std) / (r_loss_fake_std)) * r_fake_logits)


def generate_and_save_images(model, epoch, test_input, gen_path, char_vector):
    """
    Generate and save predictions + ground truth label

    :param model:
    :param epoch:
    :param test_input:
    :param gen_path:
    :param char_vector:
    :return:
    """

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0
    labels = test_input[1]

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.text(0, -1, "".join([char_vector[label] for label in labels[i]]))
        plt.axis('off')

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    plt.savefig(gen_path + 'image_at_epoch_{:04d}.png'.format(epoch))


def make_gif(gen_path):
    """
    Use imageio to create an animated gif using the images saved during training; based on
    https://www.tensorflow.org/tutorials/generative/dcgan

    :param gen_path:
    :return:
    """
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    anim_file = gen_path + 'biggan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(gen_path + 'image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def load_random_word_list(reading_dir, bucket_size, char_vector):
    """
    Helper to produce random word list; encode each word into vector defined by char_vector
    e.g. 'auto' -> [0, 20, 19, 14]

    :param reading_dir:         input dir of random_words.txt
    :param bucket_size:         max transcription length of random word
    :param char_vector:         index of char within charvector represents char encoding
    :return:
    """

    random_words = []
    for i in range(bucket_size):
        random_words.append([])

    random_words_path = os.path.dirname(os.path.dirname(os.path.dirname(reading_dir))) + '/'
    with open(os.path.join(random_words_path, 'random_words.txt'), 'r') as fi_random_word_list:
        for word in fi_random_word_list:
            word = word.strip()
            bucket = len(word)

            if bucket <= bucket_size:
                random_words[bucket - 1].append([char_vector.index(char) for char in word])

    return random_words
