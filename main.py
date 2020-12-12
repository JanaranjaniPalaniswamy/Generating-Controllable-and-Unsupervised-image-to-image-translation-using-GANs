import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time

if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # loading dataset from file
    dataset_real = glob.glob("/bigpool/export/users/datasets_faprak2020/CelebAMask-HQ/CelebA-HQ-img/*")
    dataset_comic = glob.glob("/bigpool/export/users/datasets_faprak2020/facemaker/*")

    # Count of dataset
    dataset_real_count = 30000
    dataset_comic_count = 26020

    # Training count
    dataset_real_training_count = int(dataset_real_count * 0.8)
    dataset_comic_training_count = int(dataset_comic_count * 0.8)

    # Split training and test dataset
    train_comic, train_real = tf.data.Dataset.from_tensor_slices(dataset_comic[0:dataset_comic_training_count]), tf.data.Dataset.from_tensor_slices(dataset_real[0:dataset_real_training_count])
    test_comic, test_real = tf.data.Dataset.from_tensor_slices(dataset_comic[dataset_comic_training_count:dataset_comic_count]), tf.data.Dataset.from_tensor_slices(dataset_real[dataset_real_training_count:dataset_real_count])

    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    def print_image(subplot, name, image):
        plt.subplot(subplot)
        plt.title(name)
        plt.imshow(image * 0.5 + 0.5)

    def train_parse_func(filename):
        # Read the image
        image_string = tf.io.read_file(filename)

        # Decode the image
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        print_image(121, 'Before Normalizing', image_decoded)

        # Normalize the image
        image = tf.cast(image_decoded, tf.float32)
        image = (image / 255)

        print_image(122, 'After Normalizing', image)

        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        print_image(123, 'After resizing to 286x286', image)

        # randomly cropping to 256 x 256 x 3
        image = tf.image.random_crop(
            image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

        print_image(124, 'After random crop', image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        print(125, 'After random mirroring', image)

        return image

    def test_parse_func(filename):
        # Read the image
        image_string = tf.io.read_file(filename)

        # Decode the image
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        print_image(121, 'Before Normalizing', image_decoded)

        # Normalize the image
        image = tf.cast(image_decoded, tf.float32)
        image = (image / 255)

        print_image(122, 'After Normalizing', image)

        return image

    train_comic = train_comic.map(
        train_parse_func, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_real = train_real.map(
        train_parse_func, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_comic = test_comic.map(
        test_parse_func, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_real = test_real.map(
        test_parse_func, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    print("Done Pre processing")


    def generator_model():
        generator = tf.keras.Sequential()
        generator.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())

        generator.add(layers.Reshape((7, 7, 256)))
        assert generator.output_shape == (None, 7, 7, 256)

        generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert generator.output_shape == (None, 7, 7, 128)
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())

        generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert generator.output_shape == (None, 14, 14, 64)
        generator.add(layers.BatchNormalization())
        generator.add(layers.LeakyReLU())

        generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert generator.output_shape == (None, 28, 28, 1)

        return generator

    def discriminator_model():
        discriminator = tf.keras.Sequential()
        discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))

        discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        discriminator.add(layers.LeakyReLU())
        discriminator.add(layers.Dropout(0.3))

        discriminator.add(layers.Flatten())
        discriminator.add(layers.Dense(1))

        return discriminator

    generator = generator_model()
    discriminator = discriminator_model()

    sample_comic = next(iter(train_comic))
    sample_real = next(iter(train_real))

    to_real = generator(sample_comic[0])

    plt.figure(figsize=(8, 8))
    contrast = 8

    imgs = [sample_comic, to_real]
    title = ['Comic', 'To Zebra']

    for i in range(len(imgs)):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        if i % 2 == 0:
            plt.imshow(imgs[i][0] * 0.5 + 0.5)
        else:
            plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
    plt.show()

    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.title('Is a real zebra?')
    plt.imshow(discriminator(sample_real)[0, ..., -1], cmap='RdBu_r')

    plt.show()

    LAMBDA = 10

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(generated):
        return loss_obj(tf.ones_like(generated), generated)

    def discriminator_loss(real, generated):
        real_loss = loss_obj(tf.ones_like(real), real)
        generated_loss = loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(generator=generator,
                               discriminator=discriminator,
                               generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    EPOCHS = 40

    def generate_images(model, test_input):
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    @tf.function
    def train_step(real_x, real_y):

        with tf.GradientTape(persistent=True) as tape:
            # Generator translates X -> Y

            fake_y = generator(real_x, training=True)

            disc_real_y = discriminator(real_y, training=True)

            disc_fake_y = discriminator(fake_y, training=True)

            # calculate the loss
            gen_loss = generator_loss(disc_fake_y)
            disc_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_gradients = tape.gradient(gen_loss,
                                              generator.trainable_variables)

        discriminator_gradients = tape.gradient(disc_loss,
                                                  discriminator.trainable_variables)

        # Apply the gradients to the optimizer
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                  generator.trainable_variables))

        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                      discriminator.trainable_variables))

    for epoch in range(EPOCHS):
        start = time.time()

        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_comic, train_real)):
            train_step(image_x, image_y)
            if n % 10 == 0:
                print('.', end='')
            n += 1

        generate_images(generator, sample_comic)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))

        for inp in test_comic.take(5):
            generate_images(generator, inp)
