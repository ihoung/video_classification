import cv2
import numpy as np
import os
import time
import tensorflow as tf

import model

ORIGINAL_TRAIN_LOG_DIR = os.path.join('Log/original_train/',
                                      time.strftime('%Y.%m.%d_%H-%M-%S', time.localtime(time.time())))
FINETUNE_TRAIN_LOG_DIR = os.path.join('Log/finetuning_train/',
                                      time.strftime('%Y.%m.%d_%H-%M-%S', time.localtime(time.time())))

flags = tf.app.flags
flags.DEFINE_string('train_list_path', 'train.txt', 'Path to training images.')
flags.DEFINE_string('model_output_path', './check_point/original/ckpt1', 'Path to model checkpoint.')
flags.DEFINE_string('finetune_path', None, 'Path to fine tuning model data.')
FLAGS = flags.FLAGS


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
        images.append(image)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def next_batch_set(images, labels, batch_size=128):
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


def main(_):
    inputs = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')

    cls_model = model.ResNet152Model(is_training=True, num_classes=4)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
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
    learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_step = optimizer.minimize(loss, global_step)

    saver = tf.train.Saver()

    images, targets = get_train_data(FLAGS.train_list_path)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if FLAGS.finetune_path is None:
        with tf.Session(config=config) as sess:
            train_summary_writer = tf.summary.FileWriter(ORIGINAL_TRAIN_LOG_DIR, sess.graph)
            sess.run(init)

            for i in range(6000):
                batch_images, batch_labels = next_batch_set(images, targets)
                train_dict = {inputs: batch_images, labels: batch_labels}

                _, loss_, acc_, summary = sess.run([train_step, loss, acc, summary_op], feed_dict=train_dict)

                train_text = 'step: {}, loss: {}, acc: {}'.format(
                    i + 1, loss_, acc_)
                print(train_text)
                train_summary_writer.add_summary(summary, i+1)

            saver.save(sess, FLAGS.model_output_path)

    if FLAGS.finetune_path:
        with tf.Session(config=config) as sess:
            train_summary_writer = tf.summary.FileWriter(FINETUNE_TRAIN_LOG_DIR, sess.graph)
            sess.run(init)
            saver.restore(sess, FLAGS.finetune_path)

            for i in range(6000):
                batch_images, batch_labels = next_batch_set(images, targets)
                train_dict = {inputs: batch_images, labels: batch_labels}

                _, loss_, acc_, summary = sess.run([train_step, loss, acc, summary_op], feed_dict=train_dict)

                train_text = 'step: {}, loss: {}, acc: {}'.format(
                    i + 1, loss_, acc_)
                print(train_text)
                train_summary_writer.add_summary(summary, i + 1)

            saver.save(sess, FLAGS.model_output_path)


if __name__ == '__main__':
    tf.app.run()
