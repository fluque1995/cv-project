from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import model, input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './model_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './model_train',
                           """Directory where to read model checkpoints.""")

def evaluate(file_name, real_label):

    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        if eval_data:
            print("Everything OK. Testing...")
        image, label = input.single_input(file_name, real_label)
        logits = model.inference(image)[0]
        predicted_class = tf.argmax(logits, axis=0)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

            variable_averages = tf.train.ExponentialMovingAverage(
                model.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            evaluation = sess.run(predicted_class)
            print(sess.run(logits))
            real_class = "cat" if real_label == 0 else "dog"
            prediction = "cat" if evaluation == 0 else "dog"
            print("Real class: {}".format(real_class))
            print("Predicted class: {}".format(prediction))

def main(argv=None):
    filename=argv[1]
    label=int(argv[2])
    evaluate(filename, label)


if __name__ == '__main__':
        tf.app.run()
