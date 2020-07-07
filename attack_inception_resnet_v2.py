from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from nets import inception_resnet_v2
import time


tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', 'models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '/data/ml/Attack_new/dataset/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '/data/ml/Attack_new/adv_images/0.8_NAG_adv_images_inception_resnet_v2', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 1.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 0.8, 'Momentum.')

FLAGS = tf.flags.FLAGS
slim = tf.contrib.slim


def load_images(input_dir, batch_shape):
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath, 'rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
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

def save_images(images, filenames, output_dir):
  for i, filename in enumerate(filenames):
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
            x + momentum *grad, num_classes=num_classes, is_training=False)
    pred = tf.argmax(end_points_res_v2['Predictions'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)
    logits = (logits_res_v2) / 7.25
    auxlogits = (end_points_res_v2['AuxLogits']) / 6.25
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)

def main(_):
  eps = 2.0 * FLAGS.max_epsilon / 255.0

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    with tf.Session() as sess:
      s1.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()
