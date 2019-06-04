import tensorflow as tf


def parse_fn(dataset, input_size=(64, 64)):
    x = tf.cast(dataset['image'], tf.float32)
    x = tf.image.resize(x, input_size)
    x = tf.clip_by_value(x, 0, 255)
    x = x / 127.5 - 1
    return x