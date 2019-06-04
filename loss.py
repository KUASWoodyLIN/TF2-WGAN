import tensorflow as tf


def discriminator_loss(real, fake):
    r_loss = - tf.reduce_mean(real)
    f_loss = tf.reduce_mean(fake)
    return r_loss, f_loss


def generator_loss(fake):
    f_loss = - tf.reduce_mean(fake)
    return f_loss


def gradient_penalty(f, real, fake):
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = (alpha * a) + ((1 - alpha) * b)
        inter.set_shape(a.shape)
        return inter

    # 生成生成图片与真实图片之间的插值
    x = _interpolate(real, fake)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = f(x)
    grad = tape.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)

    return gp

