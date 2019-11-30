import cv2
import numpy as np
import os
import tensorflow as tf
import model

CHECK_POINT_PATH = ''
TEST_LIST_PATH = 'test.txt'


def get_evaluated_img(test_list_path):
    if not os.path.exists(test_list_path):
        raise ValueError('Train data is not exist.')
    lines = open(test_list_path, 'r')
    lines = list(lines)
    for line in lines:
        image_file, label = line.strip('\n').split('::')
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        yield image, label


if __name__ == '__main__':
    true_num = 0
    count = 0
    with tf.Session() as sess:
        ckpt_path = CHECK_POINT_PATH
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        classes = tf.get_default_graph().get_tensor_by_name('classes:0')

        for image, label in get_evaluated_img(TEST_LIST_PATH):
            if image is None or label is None:
                break
            count += 1
            predicted_label = sess.run(classes,
                                       feed_dict={inputs: image})
            if predicted_label[0] == label:
                true_num += 1
        accuracy = true_num / count
        print('Accuracy: {}'.format(accuracy))
