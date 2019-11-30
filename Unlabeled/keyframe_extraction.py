import numpy as np
import cv2
import os
import time
import tensorflow as tf
import auto_encoder
from abc import abstractmethod

CHECK_POINT_PATH = ''
FRAMES_PER_SECOND = 5

class FrameCluster:
    def __init__(self, content=set(), center=None):
        self.content = content  # 存放feature_list的索引
        self.center = center


class Feature:
    def __init__(self, data=None, feature_type=None):
        self.data = data
        self.feature_type = feature_type

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass

    @abstractmethod
    def get_feature(self, frame):
        pass

    @abstractmethod
    def get_similarity(self, other):
        pass


class ColorHistFeature(Feature):
    def __init__(self, data=None):
        super().__init__(data=data, feature_type='colorHist')

    def __add__(self, other):
        return ColorHistFeature(self.data + other.data)

    def __truediv__(self, value):
        return ColorHistFeature(self.data / value)

    def get_feature(self, frame):
        self.data = cv2.calcHist([frame], [0], None, [256], [0, 255])

    def get_similarity(self, other):
        if type(other) != type(self):
            return 0
        similarity = cv2.compareHist(self.data, other.data, cv2.HISTCMP_CORREL)
        return similarity


class CNNFeature(Feature):
    def __init__(self, data=None):
        super().__init__(data=data, feature_type='cnn')

    def __add__(self, other):
        return CNNFeature(self.data + other.data)

    def __truediv__(self, value):
        return CNNFeature(self.data / value)

    def get_feature(self, frame):
        frame = cv2.resize(frame, (128, 128))
        frame = np.expand_dims(frame, axis=0)
        inputs = tf.placeholder(tf.float32, [1, 128, 128, 3])
        encoded = auto_encoder.encoder(inputs)
        vars = tf.global_variables()
        variables_to_restore = [var for var in vars if 'Encoder' in var.name]
        restore_saver = tf.train.Saver(var_list=variables_to_restore)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            restore_saver.restore(sess, CHECK_POINT_PATH)
            feature = sess.run(encoded, feed_dict={inputs: frame})
        self.data = np.reshape(feature, 256)

    def get_similarity(self, other):
        if type(other) != type(self):
            return 0
        similarity = np.corrcoef(self.data, other.data)[0, 1]
        return similarity


def find_closest(feature_list, cluster):
    pos = 0
    rmax = 0.0
    for feature_pos in cluster.content:
        r = feature_list[feature_pos].get_similatity(cluster.center)
        if r > rmax:
            rmax = r
            pos = feature_pos
    return pos


def merge_cluster(clusterList, from_cluster, to_cluster):
    clusterList[to_cluster].content = clusterList[to_cluster].content | clusterList[from_cluster].content
    from_len = len(clusterList[from_cluster].content)
    to_len = len(clusterList[to_cluster].content)
    clusterList[to_cluster].center = (clusterList[from_cluster].center*from_len
                                      + clusterList[to_cluster].center*to_len) / (from_len + to_len)
    clusterList.pop(from_cluster)


def feature_clustering(FEATURE_LIST, feature_indices, threshold, final_num):
    feature_indices_ = sorted(list(feature_indices))
    clusterList = list()

    # 将第一帧放入第一个聚类
    firstCluster = FrameCluster()
    firstCluster.content.add(feature_indices_[0])
    firstCluster.center = FEATURE_LIST[feature_indices_[0]]
    clusterList.append(firstCluster)

    num_cluster = 1
    for feature_num in feature_indices_:
        index = rmax = 0
        # 计算相似度最大的
        for i in range(num_cluster):
            simi = FEATURE_LIST[feature_num].get_similatity(clusterList[i].center)
            if simi > rmax:
                rmax = simi
                index = i
        # 若最大的相似度小于阈值，则创建一个新聚类
        if rmax < threshold:
            num_cluster += 1
            newCluster = FrameCluster()
            newCluster.center = FEATURE_LIST[feature_num]
            newCluster.content.add(feature_num)
            clusterList.append(newCluster)
        else:
            sumresult = Feature()
            for feature_pos in clusterList[index].content:
                sumresult = sumresult + FEATURE_LIST[feature_pos]
            clusterList[index].center = (FEATURE_LIST[feature_num]+sumresult) / (len(clusterList[index].content)+1)
            clusterList[index].content.add(feature_num)

    '''
    cluster merge here
    '''
    if final_num < len(clusterList):
        min_cluster_len = len(clusterList[0].content)
        min_index = 0
        for index in range(1, len(clusterList)):
            cluster_len = len(clusterList[index].content)
            if cluster_len < min_cluster_len:
                min_cluster_len = cluster_len
                min_index = index
        smax = 0
        closest = 0
        min_center = clusterList[min_index].center
        for i in range(len(clusterList)):
            if i == min_index:
                continue
            sim = clusterList[i].center.get_similarity(min_center)
            if smax < sim:
                smax = sim
                closest = i
        merge_cluster(clusterList, min_index, closest)

    return clusterList


def get_feature_list(filepath, method='colorHist'):
    # 初始化一个VideoCapture对象
    cap = cv2.VideoCapture()
    if cap.open(filepath):
        feature_list = list()
        # 获取视频帧率、帧数、尺寸
        fps, n_frames, frame_height, frame_width = (int(cap.get(cv2.CAP_PROP_FPS)),
                                                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("total frames number:", n_frames)
        print("fps:", fps)
        # 对每帧HSV处理
        for n_frame in range(n_frames):
            if FRAMES_PER_SECOND and FRAMES_PER_SECOND <= fps:
                if n_frame % int(fps/FRAMES_PER_SECOND) == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
                    ret, frame = cap.read()
                else:
                    continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
                ret, frame = cap.read()
            frame = np.array(frame, dtype=np.uint8)
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_feature = Feature()
            if method == 'colorHist':
                frame_feature = ColorHistFeature()
                frame_feature.get_feature(grayFrame)
            elif method == 'cnn':
                frame_feature = CNNFeature()
                frame_feature.get_feature(frame)
            feature_list.append(frame_feature)
        # 释放资源
        cap.release()
        return feature_list


def get_keyframes_feature_pos(feature_list, cluster_list):
    pos_list = []
    for cluster in cluster_list:
        pos = find_closest(feature_list, cluster)
        pos_list.append(pos)
    return pos_list


if __name__ == '__main__':
    start_time = time.clock()
    end_time = time.clock()
    print('processing time: {}'.format(end_time - start_time))
