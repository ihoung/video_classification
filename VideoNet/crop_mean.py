import os
import numpy as np
import cv2

IMAGE_LIST = 'train.txt'
IMAGE_SIZE = 112
NPY_FILE = 'crop_mean.npy'


def crop_mean(listfile, img_size):
    videofiles = []
    with open(listfile, 'r') as f:
        line = f.readline()
        while line:
            line = line.rstrip().split('::')[0]
            videofiles.append(line)
            line = f.readline()
    sum_b = sum_g = sum_r = np.zeros((img_size, img_size), dtype=np.float32)
    filenum = 0
    for videofile in videofiles:
        filelist = os.listdir(videofile)
        for file in filelist:
            if not os.path.splitext(file)[1] == '.jpg':
                continue
            imagefile = os.path.join(videofile, file)
            try:
                img = cv2.imread(imagefile)
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
    mean = np.array([sum_b, sum_g, sum_r], dtype=np.float32)
    np.save(NPY_FILE, mean)
    return True


if __name__ == '__main__':
    crop_mean(IMAGE_LIST, IMAGE_SIZE)
