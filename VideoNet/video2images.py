import os
import cv2

INTERVAL = 5
VIDEO_PATH = 'VideoData'
IMAGE_PATH = os.sep.join([os.getcwd(), 'Video2Images'])


def video2images(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file == '.DS_Store':
                continue
            filePath = os.sep.join([root, file])
            fileName = '.'.join(file.split('.')[:-1])
            imgDir = os.sep.join([IMAGE_PATH, root.split(os.sep)[-1], fileName])
            if not os.path.exists(imgDir):
                os.makedirs(imgDir)
            try:
                cap = cv2.VideoCapture()
                cap.open(filePath)
                num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                count = 0
                for i in range(num):
                    if i % INTERVAL == 0:
                        count += 1
                        ret, frame = cap.read()
                        img_name = fileName + '_' + str(count) + '.jpg'
                        img_path = os.sep.join([imgDir, img_name])
                        cv2.imwrite(img_path, frame)
                cap.release()
            except:
                return False
    return True


if __name__ == '__main__':
    video2images(VIDEO_PATH)
