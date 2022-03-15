import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torch


rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



class DataLoaderDemo(data.Dataset):
    def __init__(self, base_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = base_folder
        self.transform = transform
        # self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        # self.setup()
        self.samples = self.get_all_samples()
        
    def get_all_samples(self):
        data = []
        label = []
        for sub_dir in ['abnorm']:
            data_list = []
            sub_path = os.path.join(self.dir, sub_dir)
            video_list = os.listdir(sub_path)
            video_list.sort()
            for video in video_list:
                video_path = os.path.join(sub_path, video)
                frames = os.listdir(video_path)
                frames.sort()
                tmp_list = []
                for frame in frames:
                    frame_path = os.path.join(video_path, frame)
                    tmp_list.append(frame_path)
                data_list.append(tmp_list)
            if sub_dir == 'norm':
                label_list = [1] * len(data_list)
            else:
                label_list = [0] * len(data_list)
            data += data_list
            label += label_list
        
        return data, label

    def __getitem__(self, index):
        image_path = self.samples[0][index]
        label = self.samples[1][index]
        batch = []
        for i in range(self._time_step + self._num_pred):
            frame_name = image_path[i]
            image = np_load_frame(frame_name, self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))
        
        return np.concatenate(batch, axis=0), np.array(label)

    def __len__(self):
        return len(self.samples[0])


def collate_demo(batch):
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
    )

