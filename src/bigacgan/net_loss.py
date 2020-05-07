import tensorflow as tf


def not_saturating(d_real_logits, d_fake_logits):
    """
    Returns the discriminator and generator loss for Non-saturating loss; based on
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py

    :param d_real_logits:
    :param d_fake_logits:
    :return:
    """

    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits),
                                                          name="cross_entropy_d_real")
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
                                                          name="cross_entropy_d_fake")
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
                                                     name="cross_entropy_g")
    return d_loss, d_loss_real, d_loss_fake, g_loss


def hinge(d_real_logits, d_fake_logits):
    """
    Returns the discriminator and generator loss for the hinge loss; based on
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py

    :param d_real_logits: logits for real points, shape [batch_size, 1].
    :param d_fake_logits: logits for fake points, shape [batch_size, 1].
    :return:
    """
    d_loss_real = tf.nn.relu(1.0 - d_real_logits)
    d_loss_fake = tf.nn.relu(1.0 + d_fake_logits)
    d_loss = d_loss_real + d_loss_fake
    g_loss = - d_fake_logits
    return d_loss, d_loss_real, d_loss_fake, g_loss
