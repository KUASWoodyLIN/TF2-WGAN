import tensorflow_addons as tfa
import tensorflow.keras as keras


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        # return tfa.layers.LayerNormalization
        return keras.layers.LayerNormalization


def Generator(input_shape=(1, 1, 128),
              norm='batch_norm',
              name='Generator'):
    Norm = _get_norm_layer(norm)
    inputs = keras.Input(shape=input_shape)

    # 1: 1x1 -> 4x4
    x = keras.layers.Conv2DTranspose(512, 4, strides=1, padding='valid', use_bias=False)(inputs)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)

    # 2: 4x4 -> 8x8
    x = keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)
    # 3: 8x8 -> 16x16
    x = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)
    # 4: 16x16 -> 32x32
    x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)
    # 5: 32x32 -> 64x64
    x = keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', use_bias=False)(x)
    outputs = keras.layers.Activation('tanh')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def Discriminator(input_shape=(64, 64, 3),
                  norm='batch_norm',
                  name='Discriminator'):
    Norm = _get_norm_layer(norm)

    inputs = keras.Input(shape=input_shape)

    # 1: 64x64 -> 32x32
    x = keras.layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = keras.layers.LeakyReLU()(x)
    # 2: 32x32 -> 16x16
    x = keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)
    # 3: 16x16 -> 8x8
    x = keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)
    # 4: 8x8 -> 4x4
    x = keras.layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = Norm()(x)
    x = keras.layers.LeakyReLU()(x)
    # 5: 4x4 -> 1x1
    outputs = keras.layers.Conv2D(1, 4, strides=1, padding='valid')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


if __name__ == '__main__':
    G = Generator()
    D = Discriminator((28, 28, 3))
    G.summary()
    D.summary()
