import os
import random

IMAGE_PATH = 'ImageData'
TRAIN_LIST_NAME = 'train.txt'
TEST_LIST_NAME = 'test.txt'
RATIO = 0.15


def image2list(path):
    label_map = dict()
    with open(TRAIN_LIST_NAME, 'w') as f1, open(TEST_LIST_NAME, 'w') as f2:
        for root, dirs, files in os.walk(path):
            if not len(dirs) == 0:
                for i in range(len(dirs)):
                    label_map[dirs[i]] = i
            if len(files) == 0 or files == ['.DS_Store']:
                continue
            dirname = root.split(os.sep)[-1]
            try:
                label = label_map[dirname]
            except:
                return False
            if '.DS_Store' in files:
                files.remove('.DS_Store')
            video_indices = list(range(len(files)))
            random.shuffle(video_indices)
            test_video_indices = video_indices[:int(len(video_indices)*RATIO)]
            train_video_indices = video_indices[int(len(video_indices)*RATIO):]
            for i in train_video_indices:
                filepath = os.path.join(root, files[i])
                line = '{}::{}\n'.format(filepath, label)
                f1.write(line)
            for i in test_video_indices:
                filepath = os.path.join(root, files[i])
                line = '{}::{}\n'.format(filepath, label)
                f2.write(line)
    return True


if __name__ == '__main__':
    image2list(IMAGE_PATH)