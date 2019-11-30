import numpy as np
import cv2
import random

SAMPLING_RATE = 1
NOISE_RATE = 0.5


# def get_video_frames(filepath, fix_height, fix_width):
#     frame_list = []
#     cap = cv2.VideoCapture()
#     if cap.open(filepath):
#         fps, n_frames, frame_height, frame_width = (int(cap.get(cv2.CAP_PROP_FPS)),
#                                                     int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
#                                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                                                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#         for i in range(n_frames):
#             _, frame = cap.read()
#             if i % SAMPLING_RATE == 0:
#                 frame = np.array(frame, dtype=np.float32)
#                 if len(frame.shape) != 3:
#                     continue
#                 frame = cv2.resize(frame, (fix_width, fix_height))
#                 frame_list.append(frame)
#         cap.release()
#     noisy_frame_list = add_frame_noise(frame_list)
#     frame_list = np.array(frame_list)
#     noisy_frame_list = np.array(noisy_frame_list)
#     return frame_list, noisy_frame_list


def get_frame_list(filelist, fix_height=128, fix_width=128):
    frame_list = []
    for file in filelist:
        cap = cv2.VideoCapture()
        if cap.open(file):
            fps, n_frames, frame_height, frame_width = (int(cap.get(cv2.CAP_PROP_FPS)),
                                                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            frame = np.zeros(1, dtype=np.float32)
            while len(frame.shape) != 3:
                pos = random.randrange(n_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _, frame = cap.read()
                frame = np.array(frame, dtype=np.float32)
            frame = cv2.resize(frame, (fix_width, fix_height))
            frame_list.append(frame)
            cap.release()
    noisy_frame_list = add_frame_noise(frame_list)
    frame_list = np.array(frame_list, dtype=np.float32)
    noisy_frame_list = np.array(noisy_frame_list, dtype=np.float32)
    return frame_list, noisy_frame_list


def add_frame_noise(inputs):
    if isinstance(inputs, list):
        outputs = []
        for i in range(len(inputs)):
            if not isinstance(inputs[i], np.ndarray):
                raise Exception('Inputs must be a  numpy array or a list of numpy arrays.')
            else:
                frame_height, frame_width, channel = inputs[i].shape
                frame_noisy = inputs[i] + NOISE_RATE * np.random.randn(frame_height, frame_width, channel)
                outputs.append(frame_noisy)
        return outputs
    elif isinstance(inputs, np.ndarray):
        frame_height, frame_width = inputs.shape
        frame_noisy = inputs + NOISE_RATE * np.random.randn(frame_height, frame_width)
        return frame_noisy
    else:
        raise Exception('Inputs must be a  numpy array or a list of numpy arrays.')


# def get_batch(frame_list, noisy_frame_list, batch_size=64):
#         frame_num = len(frame_list)
#         fill_num = batch_size % frame_num
#         fill_indices = random.sample(range(frame_num), fill_num)
#         indices = list(range(frame_num)) * (batch_size//frame_num) + fill_indices
#         batch_frame = frame_list[indices]
#         batch_noisy_frame = noisy_frame_list[indices]
#         return batch_frame, batch_noisy_frame


def get_batch(frame_list, noisy_frame_list, batch_size=64):
    if len(frame_list) != len(noisy_frame_list):
        raise Exception('Original frame list size is not equal to noisy list.')
    indices = np.random.choice(len(frame_list), batch_size)
    batch_frame = frame_list[indices]
    batch_noisy_frame = noisy_frame_list[indices]
    return batch_frame, batch_noisy_frame

