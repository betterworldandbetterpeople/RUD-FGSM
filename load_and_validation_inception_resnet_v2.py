from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim

# images_arr = np.genfromtxt('labels', dtype='str')
# labels = np.asarray(list(map(int, images_arr[:, 1]))).reshape((1000, 1))
# images_arr = images_arr[:, 0].reshape((1000, 1))
# print(images_arr)
# print(labels)

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', 'adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', 'ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', 'ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', 'inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', 'inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', 'ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', 'resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '/data/ml/Non-Targeted-Adversarial-Attacks/dataset/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '/data/ml/Non-Targeted-Adversarial-Attacks/prediction', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 1.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 1, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')

FLAGS = tf.flags.FLAGS

def graph(x, y, i, grad):
    num_classes = 1001
    x = x
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=num_classes, is_training=False)
    pred = tf.argmax(end_points_res_v2['Predictions'],
        1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y

    return x, y, i, grad


def save_prediction(data, filenames, output_dir):
  for i, filename in enumerate(filenames):
    with tf.gfile.Open(os.path.join(output_dir, 'normal_prediction'), 'a') as f:
      f.write(filename + ' '+ str(data[0])+'\n')

def load_images(input_dir, batch_shape):
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath, 'rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

def stop(x, y, i, grad):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)

def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        _, pred, _, _ = graph(x_input, y, i, grad)
        # Run computation
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        with tf.Session() as sess:
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                prediction = sess.run(pred, feed_dict={x_input: images})
                save_prediction(prediction, filenames, FLAGS.output_dir)
if __name__ == '__main__':
  tf.app.run()