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

""" Transforms input images into the specified styles and saves outputs to output directory. """

import ast
import os
import src.image_utils as image_utils
import src.model as model
import src.ops as ops
import numpy as np
import tensorflow.compat.v1 as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

flags = tf.flags
flags.DEFINE_float('alpha', 1.0, 'Width multiplier the model was trained on.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from.')
flags.DEFINE_string('input_image', None, 'Input image file.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('style_dir', None, 'Directory containing {"style_name" : syle_label} dictionary.')
flags.DEFINE_string('output_basename', None, 'Output base name.')
flags.DEFINE_string('which_styles', None, 'A list of styles in which to transform the input image e.g. ["monet"].')
FLAGS = flags.FLAGS

# Loads a model checkpoint
def _load_checkpoint(sess, checkpoint):
  model_saver = tf.train.Saver(tf.global_variables())
  checkpoint = os.path.expanduser(checkpoint)
  if tf.gfile.IsDirectory(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
    tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))
  model_saver.restore(sess, checkpoint)


# Loads styles learned in the above model
def _load_style_labels(style_dict_dir, which_styles):
  labels = []
  dict = (np.load(style_dict_dir, allow_pickle=True)).item()
  for name in which_styles :
    labels.append(dict[name])
  return labels, len(dict.keys())


 # Transforms images in desired styles and saves them to the output directory
def _transform_images(input_image, which_styles, output_dir):
  which_styles, num_styles = _load_style_labels(FLAGS.style_dir, FLAGS.which_styles)
  with tf.Graph().as_default(), tf.Session() as sess:
    stylized_images = model.transform(
        tf.concat([input_image for _ in range(len(which_styles))], 0),
        alpha=FLAGS.alpha,
        normalizer_params={
            'labels': tf.constant(which_styles),
            'num_categories': num_styles,
            'center': True,
            'scale': True
        })
    _load_checkpoint(sess, FLAGS.checkpoint)
    stylized_images = stylized_images.eval()
    for which, stylized_image in zip(which_styles, stylized_images):
      image_utils.save_np_image(
          stylized_image[None, ...],
          '{}/{}_{}.png'.format(output_dir, FLAGS.output_basename, which))



def main(unused_argv=None):
  image = np.expand_dims(image_utils.load_np_image(os.path.expanduser(FLAGS.input_image)), 0)

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  which_styles = ast.literal_eval(FLAGS.which_styles)
  if isinstance(which_styles, list):
    _transform_images(image, which_styles, output_dir)
  else:
    raise ValueError('--which_styles must be a list of styles')


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
