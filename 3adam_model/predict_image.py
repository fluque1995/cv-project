import tensorflow as tf

import model
import input
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './model_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './model_train',
                           """Directory where to read model checkpoints.""")


def evaluate(file_name):

    with tf.Graph().as_default():
        eval_data = FLAGS.eval_data == 'test'
        if eval_data:
            print("Everything OK. Testing...")

        display_image = plt.imread(file_name)
        image = input.single_input(file_name)

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
            prediction = "cat" if evaluation == 0 else "dog"
            photo_label = "Predicted class: {}".format(prediction)
            plt.imshow(display_image)
            plt.suptitle(photo_label)
            plt.show()


def main(argv=None):
    filename = argv[1]
    evaluate(filename)


if __name__ == '__main__':
        tf.app.run()
