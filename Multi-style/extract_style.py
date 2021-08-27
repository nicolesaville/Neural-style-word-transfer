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

"""
Calculates Gram matrices for input images from extractions from a pre-trained VGG-16. 
Then computes an average Gram matrix for each style and saves to the output directory.
"""
from collections import defaultdict
import os
import numpy as np
import src.image_utils as image_utils
import src.learning as learning
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

flags = tf.compat.v1.app.flags
flags.DEFINE_string('style_files', None, 'Style image files.')
flags.DEFINE_string('output_file', None, 'Where to save the dataset.')
flags.DEFINE_bool('compute_gram_matrices', True, 'Whether to compute Gram matrices or not.')
FLAGS = flags.FLAGS

# Load style images
def _parse_style_files(style_dir):
  style_files = [style_dir + "/" + filename for filename in os.listdir(style_dir)]
  if not style_files:
    raise ValueError('No image files found in {}'.format(style_files))
  return style_files

# Extract features from VGG-16, compute Gram matrices and average per-style
def main():

    style_layers = ['vgg_16/conv1', 'vgg_16/conv2', 'vgg_16/conv3', 'vgg_16/conv4', 'vgg_16/conv5']

    for style_label, style_dir in enumerate(os.listdir(FLAGS.style_files)):
      feature = defaultdict(list)
      feature['label'] = style_label
      feature['name']  = style_dir 
      style_files = _parse_style_files(os.path.join(FLAGS.style_files, style_dir))
      for style_file in style_files:
        print("Processing style ", style_file)
        style_image = image_utils.load_np_image(style_file)
        with tf.Graph().as_default():
            style_end_points = learning.precompute_gram_matrices(tf.expand_dims(tf.cast(style_image, dtype=tf.float32), 0), final_endpoint='pool5')
            for name in style_layers:
                feature[name].append(style_end_points[name])
      for name in style_layers:
        feature[name] = tf.make_ndarray(tf.make_tensor_proto(tf.keras.layers.average(feature[name])))
      np.save(FLAGS.output_file + str(style_dir), dict(feature))
      print(style_dir, " features saved")  

main()