import tensorflow as tf


def parse_fn(dataset, input_size=(64, 64)):
    x = tf.cast(dataset['image'], tf.float32)
    crop_size = 108
    h, w, _ = x.shape
    x = tf.image.crop_to_bounding_box(x, (h-crop_size)//2, (w-crop_size)//2, crop_size, crop_size)
    x = tf.image.resize(x, input_size)
    x = x / 127.5 - 1
    return x
