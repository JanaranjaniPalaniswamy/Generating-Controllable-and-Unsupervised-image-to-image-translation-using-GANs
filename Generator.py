import tensorflow as tf
from tensorflow.keras import layers


class Generator:
    generator = None

    def __init__(self):
        self.generator = tf.keras.Sequential()

        self.generator.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same',
                                         input_shape=[256, 256, 3]))
        assert self.generator.output_shape == (None, 256, 256, 128)
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())

        self.generator.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        print(self.generator.output_shape)
        assert self.generator.output_shape == (None, 256, 256, 64)
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())

        self.generator.add(layers.Conv2D(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        assert self.generator.output_shape == (None, 256, 256, 3)

