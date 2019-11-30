import os
import random

VIDEO_PATH = 'VideoData'
IMAGE_PATH = 'Video2Images'
TRAIN_LIST_NAME = 'train.txt'
TEST_LIST_NAME = 'test.txt'
RATIO = 0.25


def video2list(path):
    label_map = dict()
    with open(TRAIN_LIST_NAME, 'w') as f1, open(TEST_LIST_NAME, 'w') as f2:
        for root, dirs, files in os.walk(path):
            for i in range(len(dirs)):
                label_map[dirs[i]] = i
            if len(files) == 0 or files == ['.DS_Store']:
                continue
            dirname = root.split(os.sep)[-1]
            try:
                label = label_map[dirname]
            except:
                return False
            video_indices = list(range(len(files)))
            random.shuffle(video_indices)
            test_video_indices = video_indices[:int(len(video_indices)*RATIO)]
            train_video_indices = video_indices[int(len(video_indices)*RATIO):]
            for i in train_video_indices:
                filename = '.'.join(files[i].split('.')[:-1])
                filepath = os.sep.join([IMAGE_PATH, dirname, filename])
                line = '{}::{}\n'.format(filepath, label)
                f1.write(line)
            for i in test_video_indices:
                filename = '.'.join(files[i].split('.')[:-1])
                filepath = os.sep.join([IMAGE_PATH, dirname, filename])
                line = '{}::{}\n'.format(filepath, label)
                f2.write(line)
    return True


if __name__ == '__main__':
    # if not os.path.exists(TRAIN_LIST_NAME):
    #     os.mknod(TRAIN_LIST_NAME)
    # if not os.path.exists(TEST_LIST_NAME):
    #     os.mknod(TEST_LIST_NAME)
    video2list(VIDEO_PATH)