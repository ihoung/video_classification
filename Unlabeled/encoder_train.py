import os
import time
import random
import tensorflow as tf
import auto_encoder
import video_frames_process as vfp

VIDEO_PATH = 'VideoData'
TRAIN_LOG_DIR = os.path.join('Log/train/', time.strftime('%Y.%m.%d %H-%M-%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = 'checkpoint/ckpt2/'
STEP = 10000
LEARNING_RATE = 1e-4


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


# def get_next_batch(filelist, size=128):
#     frames, noisy_frames = vfp.get_frame_list(filelist, size, size)
#     batch_frames, batch_noisy_frames = vfp.get_batch(frames, noisy_frames)
#     yield batch_frames, batch_noisy_frames


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, [None, 128, 128, 3], name='inputs')
    original_img = tf.placeholder(tf.float32, [None, 128, 128, 3], name='original_img')

    encoded = auto_encoder.encoder(inputs)
    outputs = auto_encoder.decoder(encoded)

    loss = auto_encoder.loss(outputs, original_img)
    tf.summary.scalar('loss', loss)

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
        sess.run(init)

        video_list = get_videos(VIDEO_PATH)
        frame_list, noisy_frame_list = vfp.get_frame_list(video_list)
        for step in range(STEP):
            batch_frames, batch_noisy_frames = vfp.get_batch(frame_list, noisy_frame_list)
            train_dict = {inputs: batch_noisy_frames, original_img: batch_frames}
            _, loss_, summary = sess.run([train_step, loss, summary_op], feed_dict=train_dict)
            print('step: {}, loss: {};'.format(step+1, loss_))
            train_summary_writer.add_summary(summary, step+1)

        saver.save(sess, TRAIN_CHECK_POINT + 'train.ckpt')
