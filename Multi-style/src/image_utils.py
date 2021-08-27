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


"""Image-related functions for style transfer."""
import io
import os
import tempfile
import numpy as np 
import random
import skimage.io
import imageio
import tensorflow.compat.v1 as tf

def load_batch(image_dir, batch_size, image_size):
  """ 
  Loads random batch from image dir, resizes and normalises 
  Returns 4D tensor of shape [batch size, image size, image size, 3]
  """
  image_files = [image_dir + filename for filename in os.listdir(image_dir)]
  file_batch = random.choices(image_files, k=batch_size)
  image_batch = [imageio.imread(file, pilmode='RGB') for file in file_batch]
  image_batch = [tf.convert_to_tensor(image.astype(np.float32)) for image in image_batch]
  image_batch = [center_crop_resize_image(image, image_size) for image in image_batch]
  image_batch = tf.stack(image_batch)
  image_batch = tf.reshape(image_batch, shape=[batch_size, image_size, image_size, 3])
  print("image batch created")
  return image_batch


def load_style_features(feature_dir, train_dir):
  """
  Loads preprocessed style features and saves style dictionary with model checkpoints
  """
  features = [(np.load(feature_dir + file, allow_pickle=True)).item() for file in os.listdir(feature_dir)]
  name_label_dict = {}
  for feature in features:
    name_label_dict[feature['name']] = feature['label']
  np.save(train_dir + "name_dict", name_label_dict)

  return features


def get_random_style(features, batch_size=1):
  """
  Selects a random style and returns its unique label and average Gram matrices for each style layer
  """
  style_layers = ['vgg_16/conv1', 'vgg_16/conv2', 'vgg_16/conv3', 'vgg_16/conv4', 'vgg_16/conv5']

  feature = random.choice(features)
  label = feature['label']
  gram_matrices = [feature[style_layer] for style_layer in style_layers]

  image_label_gram_matrices = tf.train.batch( [label] + gram_matrices, batch_size=batch_size)
  label = image_label_gram_matrices[:1]
  gram_matrices = image_label_gram_matrices[1:]

  gram_matrices = dict((style_layer, gram_matrix) for style_layer, gram_matrix in zip(style_layers, gram_matrices))

  return label, gram_matrices


def load_np_image(image_file):
  """Loads an image as a numpy array.

  Args:
    image_file: str. Image file.

  Returns:
    A 3-D numpy array of shape [image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  return np.float32(load_np_image_uint8(image_file) / 255.0)


def load_np_image_uint8(image_file):
  """Loads an image as a numpy array.

  Args:
    image_file: str. Image file.

  Returns:
    A 3-D numpy array of shape [image_size, image_size, 3] and dtype uint8,
    with values in [0, 255].
  """
  with tempfile.NamedTemporaryFile() as f:
    f.write(tf.gfile.GFile(image_file, 'rb').read())
    f.flush()
    image = skimage.io.imread(f.name)
    # Workaround for black-and-white images
    if image.ndim == 2:
      image = np.tile(image[:, :, None], (1, 1, 3))
    return image


def save_np_image(image, output_file, save_format='jpeg'):
  """Saves an image to disk.

  Args:
    image: 3-D numpy array of shape [image_size, image_size, 3] and dtype
        float32, with values in [0, 1].
    output_file: str, output file.
    save_format: format for saving image (eg. jpeg).
  """
  image = np.uint8(image * 255.0)
  buf = io.BytesIO()
  skimage.io.imsave(buf, np.squeeze(image, 0), format=save_format)
  buf.seek(0)
  f = tf.gfile.GFile(output_file, 'w')
  f.write(buf.getvalue())
  f.close()


def load_image(image_file, image_size=None):
  """Loads an image and center-crops it to a specific size.

  Args:
    image_file: str. Image file.
    image_size: int, optional. Desired size. If provided, crops the image to
        a square and resizes it to the requested size. Defaults to None.

  Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  image = tf.constant(np.uint8(load_np_image(image_file) * 255.0))
  if image_size is not None:
    # Center-crop into a square and resize to image_size
    small_side = int(min(image.shape[0], image.shape[1]))
    image = tf.image.resize_image_with_crop_or_pad(
        image, small_side, small_side)
    image = tf.image.resize_images(image, [image_size, image_size])
  image = tf.cast(image, dtype=tf.float32) / 255.0

  return tf.expand_dims(image, 0)




# The following functions are copied over from tf.slim.preprocessing.vgg_preprocessing
def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.strided_slice instead of crop_to_bounding box as it accepts tensors
  # to define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.strided_slice(image, offsets, offsets + cropped_shape,
                             strides=tf.ones_like(offsets))
  return tf.reshape(image, cropped_shape)

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.cast(height, dtype=tf.float32)
  width = tf.cast(width, dtype=tf.float32)
  smallest_side = tf.cast(smallest_side, dtype=tf.float32)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image or a 4-D batch of images `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D or 4-D tensor containing the resized image(s).
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  input_rank = len(image.shape)
  if input_rank == 3:
    image = tf.expand_dims(image, 0)

  shape = tf.shape(image)
  height = shape[1]
  width = shape[2]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  if input_rank == 3:
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
  else:
    resized_image.set_shape([None, None, None, 3])
  return resized_image

def center_crop_resize_image(image, image_size):
  """Center-crop into a square and resize to image_size.

  Args:
    image: A 3-D image `Tensor`.
    image_size: int, Desired size. Crops the image to a square and resizes it
      to the requested size.

  Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  shape = tf.shape(image)
  small_side = tf.minimum(shape[0], shape[1])
  image = tf.image.resize_image_with_crop_or_pad(image, small_side, small_side)
  image = tf.cast(image, dtype=tf.float32) / 255.0

  image = tf.image.resize_images(image, tf.constant([image_size, image_size]))

  return image #tf.expand_dims(image, 0)

def resize_image(image, image_size):
  """Resize input image preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    image_size: int, desired size of the smallest size of image after resize.

  Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
  """
  image = _aspect_preserving_resize(image, image_size)
  image = tf.cast(image, dtype=tf.float32) / 255.0

  return tf.expand_dims(image, 0)
