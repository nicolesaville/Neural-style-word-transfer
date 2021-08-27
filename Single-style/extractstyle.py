from __future__ import print_function
from train import VGG_PATH
import src.vgg as vgg
from src.utils import get_img
import tensorflow as tf, numpy as np, os
import os

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
DEVICES = 'CUDA_VISIBLE_DEVICES'
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
style_dir = 'data/style_imgs/sketch_imgs/'
save_dir = 'data/style_grams/sketch/'
style_features = {}

# Extract style features and calucalte Gram matrices for a set of images in a single style
with tf.Graph().as_default(), tf.device('/cpu:0'), tf.compat.v1.Session() as sess:
        style_image = tf.compat.v1.placeholder(tf.float32, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(VGG_PATH, style_image_pre)
        for img_path in  os.listdir(style_dir):
            style_target = get_img(style_dir + img_path)    
            style_pre = np.array([style_target])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={style_image:style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[layer] = style_features[layer].append(gram)

# Average over Gram matrices in each layer to get average style targets
for layer in style_features.keys():
    style_features[layer] = tf.keras.layers.average(style_features[layer])
    np.save(save_dir + layer, style_features[layer])