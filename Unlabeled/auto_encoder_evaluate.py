import random
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import auto_encoder
import video_frames_process as vfp

CHECK_POINT_PATH = 'checkpoint/ckpt2/train.ckpt'
VIDEO_PATH = 'VideoData'


def get_videos(root_path):
    filelist = []
    for root, dirs, files in os.walk(root_path):
        if len(dirs) != 0:
            continue
        for file in files:
            if file is '.DS_Store':
                continue
            filepath = os.path.join(root, file)
            filelist.append(filepath)
    return filelist


def get_evaluated_frames(vidoe_list, choices_num, fixed_size=128):
    frame_list = []
    video_choices = random.sample(vidoe_list, choices_num)
    frames, noisy_frames = vfp.get_frame_list(video_choices, fixed_size, fixed_size)
    return frames, noisy_frames


def plot_some(*img_lists):
    plt.figure(figsize=(15, 9))
    for i, img_list in enumerate(img_lists):
        for j, array in enumerate(img_list):
            plt.subplot(len(img_lists), len(img_list), i*len(img_list)+j+1)
            plt.imshow(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    video_list = get_videos(VIDEO_PATH)
    frames, noisy_frames = get_evaluated_frames(video_list, 5)

    inputs = tf.placeholder(tf.float32, [None, 128, 128, 3], name='inputs')

    encoded = auto_encoder.encoder(inputs)
    outputs = auto_encoder.decoder(encoded)

    loss = auto_encoder.loss(outputs, inputs)

    restore_saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        restore_saver.restore(sess, CHECK_POINT_PATH)
        outputs_, loss_ = sess.run([outputs, loss], feed_dict={inputs: noisy_frames})
        print('loss: {}'.format(loss_))

# 显示图像
    plot_some(frames.astype(np.uint8),
              noisy_frames.astype(np.uint8),
              outputs_.astype(np.uint8))

