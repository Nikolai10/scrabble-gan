# import sys
# sys.path.extend(['/home/ubuntu/workspace/scrabble-gan'])

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    latent_dim = 128
    char_vec = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    path_to_saved_model = '/home/ubuntu/workspace/scrabble-gan/res/out/big_ac_gan/model/generator_15'

    # number of samples to generate
    batch_size = 10
    # sample string
    sample_string = 'machinelearning'
    # load trained model
    imported_model = tf.saved_model.load(path_to_saved_model)

    # inference loop
    for idx in range(1):
        fake_labels = []
        words = [sample_string] * 10
        noise = tf.random.normal([batch_size, latent_dim])
        # encode words
        for word in words:
            fake_labels.append([char_vec.index(char) for char in word])
        fake_labels = np.array(fake_labels, np.int32)

        # run inference process
        predictions = imported_model([noise, fake_labels], training=False)
        # transform values into range [0, 1]
        predictions = (predictions + 1) / 2.0

        # plot results
        for i in range(predictions.shape[0]):
            plt.subplot(10, 1, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            # plt.text(0, -1, "".join([char_vec[label] for label in fake_labels[i]]))
            plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
