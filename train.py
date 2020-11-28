import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import cv2
from utils.dataset import parse_fn
from utils.losses import generator_loss, discriminator_loss, gradient_penalty
from utils.models import Generator, Discriminator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset = 'celeb_a'     # 'cifar10', 'fashion_mnist', 'mnist'
log_dirs = 'logs_wgan_2'
batch_size = 64
# learning rate
lr = 0.0001
# Random vector size
z_dim = 128
# Critic updates per generator update
n_dis = 5
# Gradient penalty weight
gradient_penalty_weight = 10.0


# Load datasets and setting
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
train_data, info = tfds.load(dataset, split='train+validation+test', with_info=True)
train_data = train_data.shuffle(1000)
train_data = train_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
train_data = train_data.batch(batch_size, drop_remainder=True)      # 如果最後一批資料小於batch_size，則捨棄該批資料
train_data = train_data.prefetch(buffer_size=AUTOTUNE)

# Create networks
generator = Generator((1, 1, z_dim))
discriminator = Discriminator((64, 64, 3))
generator.summary()
discriminator.summary()

# Create optimizers
g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)


@tf.function
def train_generator():
    with tf.GradientTape() as tape:
        # sample data
        random_vector = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        # create image
        fake_img = generator(random_vector, training=True)
        # predict real or fake
        fake_logit = discriminator(fake_img, training=True)
        # calculate generator loss
        g_loss = generator_loss(fake_logit)

    gradients = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return g_loss


@tf.function
def train_discriminator(real_img):
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        fake_img = generator(z, training=True)

        real_logit = discriminator(real_img, training=True)
        fake_logit = discriminator(fake_img, training=True)

        real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
        gp = gradient_penalty(partial(discriminator, training=True), real_img, fake_img)

        d_loss = (real_loss + fake_loss) + gp * gradient_penalty_weight

    D_grad = t.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(D_grad, discriminator.trainable_variables))

    return real_loss + fake_loss, gp


def combine_images(images, col=10, row=10):
    images = (images + 1) / 2
    images = images.numpy()
    b, h, w, _ = images.shape
    images_combine = np.zeros(shape=(h*col, w*row, 3))
    for y in range(col):
        for x in range(row):
            images_combine[y*h:(y+1)*h, x*w:(x+1)*w] = images[x+y*row]
    return images_combine


def train_wgan():
    # Create tensorboard logs
    model_dir = log_dirs + '/models/'
    os.makedirs(model_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dirs)

    # Create fixed Random vector for sampling
    sample_random_vector = tf.random.normal((100, 1, 1, z_dim))
    for epoch in range(25):
        for step, real_img in enumerate(train_data):
            # training discriminator
            d_loss, gp = train_discriminator(real_img)
            # save discriminator loss
            with summary_writer.as_default():
                tf.summary.scalar('discriminator_loss', d_loss, d_optimizer.iterations)
                tf.summary.scalar('gradient_penalty', gp, d_optimizer.iterations)

            # training generator
            if d_optimizer.iterations.numpy() % n_dis == 0:
                g_loss = train_generator()
                # save generator loss
                with summary_writer.as_default():
                    tf.summary.scalar('generator_loss', g_loss, g_optimizer.iterations)
                print('G Loss: {:.2f}\tD loss: {:.2f}\tGP Loss {:.2f}'.format(g_loss, d_loss, gp))

                # save sample
                if g_optimizer.iterations.numpy() % 100 == 0:
                    x_fake = generator(sample_random_vector, training=False)
                    save_img = combine_images(x_fake)
                    # save fake images
                    with summary_writer.as_default():
                        tf.summary.image(dataset, [save_img], step=g_optimizer.iterations)

        # save model
        if epoch != 0:
            generator.save_weights(model_dir + "generator-epochs-{}.h5".format(epoch))


if __name__ == '__main__':
    train_wgan()
    # for epoch in range(1):
    #     print('Start of epoch {}'.format(epoch))
    #     for step, real_img in enumerate(train_data):
    #         print(real_img.shape)
    #         save = (real_img.numpy()[0] + 1) * 127.5
    #         cv2.imwrite('test.png', save[..., ::-1])
    #         assert real_img.shape[0] == 64