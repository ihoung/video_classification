#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a CNN model to classifying 10 digits.

Example Usage:
---------------
python3 train.py \
    --images_path: Path to the training images (directory).
    --model_output_path: Path to model.ckpt.
"""

import cv2
import numpy as np
import os
import tensorflow as tf
import time
import model

import tensorflow.contrib.slim as slim

TRAIN_LOG_DIR = os.path.join('Log/train/',
                                      time.strftime('%Y.%m.%d_%H-%M-%S', time.localtime(time.time())))
TRAIN_LIST = 'train.txt'
MODEL_PATH = 'inception_resnet_v2_2016_08_30.ckpt'
MODEL_OUTPUT_PATH = 'checkpoint/ckpt3/'
STEP = 3500

# flags = tf.app.flags
# flags.DEFINE_string('train_list_path', None, 'Path to image training list.')
# flags.DEFINE_string('model_path', None, 'Path to original model checkpoint.')
# flags.DEFINE_string('model_output_path', None, 'Path to model checkpoint output.')
# FLAGS = flags.FLAGS


def get_train_data(trainlist):
    """Get the training images from images_path.

    Args:
        images_path: Path to trianing images.

    Returns:
        images: A list of images.
        lables: A list of integers representing the classes of images.

    Raises:
        ValueError: If images_path is not exist.
    """
    if not os.path.exists(trainlist):
        raise ValueError('Train data is not exist.')

    images = []
    labels = []
    count = 0
    lines = open(trainlist, 'r')
    lines = list(lines)
    for line in lines:
        image_file, label = line.strip('\n').split('::')
        count += 1
        if count % 100 == 0:
            print('Load {} images.'.format(count))
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def next_batch_set(images, labels, batch_size=64):
    """Generate a batch training data.

    Args:
        images: A 4-D array representing the training images.
        labels: A 1-D array representing the classes of images.
        batch_size: An integer.

    Return:
        batch_images: A batch of images.
        batch_labels: A batch of labels.
    """
    indices = np.random.choice(len(images), batch_size)
    batch_images = images[indices]
    batch_labels = labels[indices]
    return batch_images, batch_labels


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='inputs')
    labels = tf.placeholder(tf.int64, shape=[None], name='labels')

    cls_model = model.InceptionResNetModel(is_training=True, num_classes=4)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)

    checkpoint_exclude_scopes = ['Logits', 'AuxLogits', 'Predict']
    variables_to_restore = slim.get_variables_to_restore(exclude=checkpoint_exclude_scopes)
    # variables_to_restore = []
    # for var in slim.get_model_variables():
    #     excluded = False
    #     for exclusion in checkpoint_exclude_scopes:
    #         if var.op.name.startswith(exclusion):
    #             excluded = True
    #     if not excluded:
    #         variables_to_restore.append(var)

    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    classes = postprocessed_dict['classes']
    classes_ = tf.identity(classes, name='classes')
    acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)
    summary_op = tf.summary.merge_all()

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(5e-4, global_step, 100, 0.9)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step)

    # saver_restore = tf.train.Saver(var_list=variables_to_restore)
    init_fn = slim.assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)
    saver = tf.train.Saver(tf.global_variables())

    images, targets = get_train_data(TRAIN_LIST)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
        sess.run(init)
        # saver_restore.restore(sess, MODEL_PATH)
        init_fn(sess)

        for i in range(STEP):
            batch_images, batch_labels = next_batch_set(images, targets)
            train_dict = {inputs: batch_images, labels: batch_labels}

            _, loss_, acc_, summary = sess.run([train_step, loss, acc, summary_op], feed_dict=train_dict)

            train_text = 'step: {}, loss: {}, acc: {}'.format(
                i + 1, loss_, acc_)
            print(train_text)
            train_summary_writer.add_summary(summary, i + 1)
            if (i+1) % 500 == 0:
                saver.save(sess, MODEL_OUTPUT_PATH+'InceptionResNet.ckpt', i+1)
