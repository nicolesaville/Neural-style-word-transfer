from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img
import wandb

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'


def optimise(content_targets, style_dir, content_weight, style_weight, tv_weight, vgg_path, 
             epochs=100, batch_size=4, save_path='saver/fns.ckpt', learning_rate=1e-3, debug=False):

    # Trim training data
    mod = len(content_targets) % batch_size
    if mod > 0:
        content_targets = content_targets[:-mod] 
        print("Train set has been trimmed slightly to", len(content_targets) )
    style_features = {}
    batch_shape = (batch_size,256,256,3)

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        # Load precomputed average style features
        for layer in STYLE_LAYERS:
            style_features[layer] = np.load(style_dir + layer + ".npy")

        X_content = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # Compute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        # Transform content image and compute content loss
        preds = transform.net(X_content/255.0)
        preds_pre = vgg.preprocess(preds)
        net = vgg.net(vgg_path, preds_pre)
        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size )

        # Compute style loss
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(a=feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # Compute total variation loss
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        loss = content_loss + style_loss + tv_loss

        # Run train steps
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.compat.v1.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(0, epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                iterations += 1
                assert X_batch.shape[0] == batch_size
                feed_dict = {X_content:X_batch}
                train_step.run(feed_dict=feed_dict)

                # Log and save
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                should_print = iterations * batch_size >= num_examples
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {X_content:X_batch}
                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    saver = tf.compat.v1.train.Saver()
                    res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)

            wandb.log({"Epoch": epoch, "Iteration" : iterations, "Loss": _loss, "style_loss": _style_loss, "content_loss": _content_loss, "tv_loss": _tv_loss})
            



def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
