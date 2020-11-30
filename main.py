import tensorflow as tf
import glob
import matplotlib.pyplot as plt

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
