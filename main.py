import tensorflow as tf
import glob

if __name__ == '__main__':
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # loading dataset from file
    dataset_real = glob.glob("/bigpool/export/users/datasets_faprak2020/CelebAMask-HQ/CelebA-HQ-img/*.jpg")
    dataset_comic = glob.glob("/bigpool/export/users/datasets_faprak2020/facemaker/*")

    dataset_real_count = 30000
    dataset_comic_count = 26020

    dataset_real_training_count = int(dataset_real_count * 0.8)
    dataset_comic_training_count = int(dataset_comic_count * 0.8)

    train_comic, train_real = tf.data.Dataset.from_tensor_slices(dataset_comic[0:dataset_comic_training_count]), tf.data.Dataset.from_tensor_slices(dataset_real[0:dataset_real_training_count])
    test_comic, test_real = tf.data.Dataset.from_tensor_slices(dataset_comic[dataset_comic_training_count:dataset_comic_count]), tf.data.Dataset.from_tensor_slices(dataset_real[dataset_real_training_count:dataset_real_count])

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
        image = tf.image.resize(image, [286, 286, 3],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image


    def preprocess_image_train(image):
        image = random_jitter(image)
        image = normalize(image)
        return image


    def preprocess_image_test(image):
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
