import numpy as np
import math
import cv2
import os


class FrameShot:
    content = dict()
    center = np.zeros((22,), dtype=np.float)


def similarity(x1, x2):
    sh = ss = sv = 0
    alpha_h, alpha_s, alpha_v = (0.5, 0.3, 0.2)
    for i in range(12):
        sh += min(x1[i], x2[i])
    for i in range(12, 17):
        ss += min(x1[i], x2[i])
    for i in range(17, 22):
        sv += min(x1[i], x2[i])
    return sh*alpha_h + ss*alpha_s + sv*alpha_v


def find_max_entropy_id(frameshot):
    n_id = 0
    for key in frameshot.content.keys():
        sh = ss = sv = smax = 0.0
        for i in range(12):
            if frameshot.content[key][i] != 0.0:
                sh += -frameshot.content[key][i] * math.log(frameshot.content[key][i], 2)
        for i in range(12, 17):
            if frameshot.content[key][i] != 0.0:
                ss += -frameshot.content[key][i] * math.log(frameshot.content[key][i], 2)
        for i in range(17, 22):
            if frameshot.content[key][i] != 0.0:
                sv += -frameshot.content[key][i] * math.log(frameshot.content[key][i], 2)
        s = 0.5*sh + 0.3*ss + 0.2*sv
        if s > smax:
            smax = s
            n_id = key
    return n_id


def combine(shotlist, i, j):
    for key in shotlist[j].content.keys():
        shotlist[i].content[key] = shotlist[j].content.get(key)
    shotlist.remove(shotlist[j])


def video_processing(infile, outpath):
    # 初始化一个VideoCapture对象
    cap = cv2.VideoCapture()
    if cap.open(infile):
        colorbar = list()
        # 获取视频帧数
        fps, n_frames = (int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # 对每帧HSV处理
        for n_frame in range(n_frames):
            if n_frame % int(n_frames/10) == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
                ret, frame = cap.read()
            else:
                continue
            frame = np.array(frame, dtype=np.uint8)
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color = np.zeros((22,), dtype=np.float)
            frame_height, frame_width, n_channel = (frame.shape[0],
                                                    frame.shape[1],
                                                    frame.shape[2])
            h = s = v = np.zeros((frame_height, frame_width), dtype=np.uint8)
            for i in range(frame_height):
                for j in range(frame_width):
                    h[i][j] = frame[i][j][0] / 21
                    if h[i][j] > 11:
                        h[i][j] = 11
                    s[i][j] = frame[i][j][1] / 51
                    if s[i][j] > 4:
                        s[i][j] = 4
                    v[i][j] = frame[i][j][2] / 51
                    if v[i][j] > 4:
                        v[i][j] = 4
                    color[h[i][j]] += 1
                    color[12 + s[i][j]] += 1
                    color[17 + v[i][j]] += 1
            for n_color in range(len(color)):
                color[n_color] /= float(frame_height * frame_width)
            colorbar.append(color)

        threshold = 0.8
        shotlist = list()

        # 将第一帧放入第一个聚类
        first = FrameShot()
        first.content[0] = colorbar[0]
        first.center = colorbar[0]
        shotlist.append(first)

        count = rmax = index = 0
        shotnum = 1
        for color in colorbar:
            # 计算相似度最大的
            for i in range(shotnum):
                ratio = similarity(color, shotlist[i].center)
                if ratio>rmax:
                    rmax = ratio
                    index = i
            # 若最大的相似度小于阈值，则创建一个新聚类
            if rmax<threshold:
                shotnum += 1
                newshot = FrameShot()
                newshot.center = color
                newshot.content[count] = color
                shotlist.append(newshot)
            else:
                shotlist[index].center = (color+sum(shotlist[index].content.values())) / \
                                         (len(shotlist[index].content)+1)
                shotlist[index].content[count] = color
            count += 1
        for i in range(len(shotlist)):
            if len(shotlist[i].content) < 10 and i > 0:
                combine(shotlist, i-1, i)
                i -= 1

        for n_shot in range(len(shotlist)):
            frame_id = find_max_entropy_id(shotlist[n_shot])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id * int(n_frames/10))
            ret, keyframe = cap.read()
            # 保存关键帧
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            imagename = '{}.jpg'.format(n_shot)
            imagepath = os.path.join(outpath, imagename)
            cv2.imwrite(imagepath, keyframe)
    # 释放资源
    cap.release()


if __name__ == '__main__':
    # 遍历原视频文件夹，获取文件目录
    for root, dirs, files in os.walk('OriginalVideo'):
        for filename in files:
            filepath = os.path.join(root, filename)  # 视频文件路径
            dirname = root.split(os.sep)[-1]  # 视频的类文件夹名
            # 创建当前视频的关键帧保存路径
            frame_path = os.path.join(os.curdir, 'KeyFrame', dirname, filename)
            video_processing(filepath, frame_path)
