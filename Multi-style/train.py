# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains the multi-style transfer model on styles preprocessed by extract_style.py"""

import ast
import os
import src.image_utils as image_utils
import src.learning as learning
import src.model as model
import src.vgg as vgg
import tensorflow.compat.v1 as tf
import tf_slim as slim
import wandb

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1.0}'
DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 1e-4, "vgg_16/conv2": 1e-4, "vgg_16/conv3": 1e-4, "vgg_16/conv4": 1e-4}')

flags = tf.app.flags
flags.DEFINE_float('clip_gradient_norm', 0, 'Clip gradients to this norm')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter servers. If 0, parameters are handled locally by the worker.')
flags.DEFINE_integer('num_styles', None, 'Number of styles.')
flags.DEFINE_float('alpha', 1.0, 'Width multiplier')
flags.DEFINE_integer('save_summaries_secs', 60, 'Frequency at which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 60, 'Frequency at which the model is saved, in seconds.')
flags.DEFINE_integer('task', 0,
                     'Task ID. Used when training with multiple workers to identify each worker.')
flags.DEFINE_integer('train_steps', 20000, 'Number of training steps.')
flags.DEFINE_string('content_weights', DEFAULT_CONTENT_WEIGHTS, 'Content weights')
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_string('style_dataset_file', None, 'Style dataset file.')
flags.DEFINE_string('style_weights', DEFAULT_STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_string('train_dir', None, 'Directory for checkpoints and summaries.')
flags.DEFINE_string('image_dir', None, 'Directory of training images.')
FLAGS = flags.FLAGS


def main(unused_argv=None):
  with tf.Graph().as_default():
    # Force all input processing onto CPU in order to reserve the GPU for the forward inference and back-propagation
    device = '/cpu:0' if not FLAGS.ps_tasks else '/job:worker/cpu:0'
    # Load target style features
    features = image_utils.load_style_features(os.path.expanduser(FLAGS.style_dataset_file))
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, worker_device=device)):
      # Load batch of content images and select random style (new selection for each graph execution)
      inputs = image_utils.load_batch(FLAGS.image_dir, FLAGS.batch_size, FLAGS.image_size)
      style_labels, style_gram_matrices = image_utils.get_random_style(features)
      
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      # Process style and weight flags
      num_styles = FLAGS.num_styles
      content_weights = ast.literal_eval(FLAGS.content_weights)
      style_weights = ast.literal_eval(FLAGS.style_weights)

      # Define the model
      stylized_inputs = model.transform(inputs, alpha=FLAGS.alpha,
          normalizer_params={
              'labels': style_labels,
              'num_categories': num_styles,
              'center': True,
              'scale': True })

      # Compute losses
      total_loss, loss_dict = learning.total_loss(inputs, stylized_inputs, style_gram_matrices, content_weights, style_weights)
      for key, value in loss_dict.items():
        tf.summary.scalar(key, value)

      # Set up training
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      train_op = slim.learning.create_train_op(total_loss, optimizer, clip_gradient_norm=FLAGS.clip_gradient_norm, summarize_gradients=False)

      # Restore pre-trained VGG-16 parameters from checkpoint
      init_fn_vgg = slim.assign_from_checkpoint_fn(vgg.checkpoint_file(), slim.get_variables('vgg_16'))

      # Run training
      slim.learning.train(
          train_op=train_op,
          logdir=os.path.expanduser(FLAGS.train_dir),
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=FLAGS.train_steps,
          init_fn=init_fn_vgg,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  wandb.init(tensorboard=True, config=flags.FLAGS, settings=wandb.Settings(start_method='fork'))
  console_entry_point()
