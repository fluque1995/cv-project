from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../images/',
                           """Path to the CIFAR-10 data directory.""")

IMAGE_SIZE = input.IMAGE_SIZE
NUM_CLASSES = input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(
          name,
          shape,
          initializer=initializer,
          dtype=tf.float32
        )
    return var


def inputs(eval_data):

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images, labels = input.inputs(eval_data=eval_data,
                                  data_dir=data_dir,
                                  batch_size=FLAGS.batch_size)
    return images, labels


def inference(images):
    # conv
    with tf.variable_scope('conv') as scope:
        kernel = _variable_on_cpu(
            'weights',
            shape=[5, 5, 3, 64],
            initializer=tf.truncated_normal_initializer(
                stddev=0.1,
                dtype=tf.float32
            )
        )
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool')

    # dense
    with tf.variable_scope('dense') as scope:
        reshape = tf.reshape(pool1, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_on_cpu(
            'weights',
            shape=[dim, 384],
            initializer=tf.truncated_normal_initializer(
                stddev=0.04,
                dtype=tf.float32
            )
        )

        biases = _variable_on_cpu('biases', [384],
                                  tf.constant_initializer(0.1))
        local = tf.nn.relu(
            tf.matmul(reshape, weights) + biases, name=scope.name
        )

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_on_cpu(
            'weights',
            shape=[384, NUM_CLASSES],
            initializer=tf.truncated_normal_initializer(
                stddev=1/192.0,
                dtype=tf.float32
            )
        )

        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local, weights), biases,
                                name=scope.name)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(0.001)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op
