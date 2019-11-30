import os
import cv2
import numpy as np
import argparse

IMAGE_SIZE = 256
TRAIN_PATH = 'ImageData'
NPY_FILE = 'imgs_mean.npy'


def crop_mean(train_path, img_size):
    sum_b = sum_g = sum_r = np.zeros((img_size, img_size), dtype=np.float32)
    filenum = 0
    for root, dirs, files in os.walk(train_path):
        for file in files:
            if file == '.DS_Store':
                continue
            filepath = os.path.join(root, file)
            try:
                img = cv2.imread(filepath)
                img = cv2.resize(img, (img_size, img_size))
                sum_b = sum_b + img[:, :, 0]
                sum_g = sum_g + img[:, :, 1]
                sum_r = sum_r + img[:, :, 2]
            except:
                return False
            finally:
                filenum += 1
    sum_b /= filenum
    sum_g /= filenum
    sum_r /= filenum
    means = np.array([sum_b, sum_g, sum_r], dtype=np.float32)
    np.save(NPY_FILE, means)
    return means


if __name__ == '__main__':
    # parse = argparse.ArgumentParser()
    # parse.add_argument('trainpath', type=str)
    # args = parse.parse_args()
    means = crop_mean(TRAIN_PATH, IMAGE_SIZE)
