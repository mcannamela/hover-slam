import tensorflow as tf


def build_true_generator(x, y, h, k, r, a):

    if x.shape.ndims != 1:
        raise RuntimeError("x must have rank 1")
    if y.shape.ndims !=1:
        raise RuntimeError("y must have rank 1")

    circ = tf.square(x - h) + tf.expand_dims(tf.square(y - k), 1) < tf.square(r)
    return tf.cast(circ, tf.float32) * a