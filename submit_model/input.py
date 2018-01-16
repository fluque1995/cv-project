from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


IMAGE_SIZE = 64

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000
TRAIN_FILE = '../train.txt'
TEST_FILE = '../test.txt'


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size):
    
    if not eval_data:
        filenames = []
        labels = []
        with open(TRAIN_FILE) as f:
            content = f.readlines()
            for row in content:
                row_split = row.split(' ')
                filenames.append(data_dir + row_split[0] + '.jpg')
                labels.append(int(row_split[1]))
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = []
        labels = []
        with open(TEST_FILE) as f:
            content = f.readlines()
            for row in content:
                row_split = row.split(' ')
                filenames.append(data_dir + row_split[0] + '.jpg')
                labels.append(int(row_split[1]))
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    [files_queue, labels_queue] = tf.train.slice_input_producer(
        [filenames, labels]
    )

    read_input = tf.read_file(files_queue)
    reshaped_image = tf.image.decode_jpeg(read_input, channels=3)

    height = IMAGE_SIZE
    width = IMAGE_SIZE


    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)


    float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, 3])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, labels_queue,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def single_input(file_name):
    read_input = tf.read_file(file_name)
    reshaped_image = tf.image.decode_jpeg(read_input, channels=3)

    height = IMAGE_SIZE
    width = IMAGE_SIZE
    resized_image = tf.image.resize_images(reshaped_image, [height, width])
    
    float_image = tf.image.per_image_standardization(resized_image)

    float_image = tf.expand_dims(float_image, 0)
    float_image = tf.tile(float_image, [128, 1, 1, 1])
    return float_image
