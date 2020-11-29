import tensorflow as tf

if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset, metadata = None  # Load Dataset here

    train_comic, train_real = dataset['trainA'], dataset['trainB']
    test_comic, test_real = dataset['testA'], dataset['testB']

    BUFFER_SIZE = 1000
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256


    def random_crop(image):
        cropped_image = tf.image.random_crop(
            image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

        return cropped_image


    # normalizing the images to [0, 1]
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 255)
        return image


    def random_jitter(image):
        # resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image


    def preprocess_image_train(image, label):
        image = random_jitter(image)
        image = normalize(image)
        return image


    def preprocess_image_test(image, label):
        image = normalize(image)
        return image


    train_comic = train_comic.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_real = train_real.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_comic = test_comic.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_real = test_real.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
