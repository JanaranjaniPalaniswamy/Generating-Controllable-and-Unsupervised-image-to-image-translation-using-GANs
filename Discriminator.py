import tensorflow as tf
from tensorflow.keras import layers


class Discriminator:
    discriminator = None

    def __init__(self):
        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                             input_shape=[256, 256, 3]))
        self.discriminator.add(layers.LeakyReLU())
        self.discriminator.add(layers.Dropout(0.3))

        self.discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.discriminator.add(layers.LeakyReLU())
        self.discriminator.add(layers.Dropout(0.3))

        self.discriminator.add(layers.Flatten())
        self.discriminator.add(layers.Dense(1))
