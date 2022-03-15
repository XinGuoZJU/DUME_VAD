import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torch
import random

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



class DataLoader(data.Dataset):
    def __init__(self, base_folder, flag, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = base_folder
        self.label_file = os.path.join(base_folder, flag + '.txt')
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        # self.setup()
        self.samples = self.get_all_samples()
        
    def get_all_samples(self):
        with open(self.label_file, 'r') as op: file_list = op.readlines()
        norm_frames = []
        abnorm_frames = []
        for line in file_list:
            file_path, all_frame_num = line.strip().split()
            prefix = file_path.split('.')[0].split('/')
            video_name = prefix[2]
            frame_id = int(prefix[3])
            
            if prefix[0] == 'norm' or prefix[0] == 'norm2norm':
                if frame_id - self._time_step >= 0 and  frame_id + self._num_pred <= int(all_frame_num):  # 0,1,2,  3(fram_id),5 
                    norm_frames.append(['/'.join(prefix[:-1]), frame_id])
            elif prefix[0] == 'abnorm2abnorm':
                if frame_id - self._time_step >= 0 and  frame_id + self._num_pred <= int(all_frame_num):  # 0,1,2,  3(fram_id),5 
                    abnorm_frames.append(['/'.join(prefix[:-1]), frame_id])
       
        return norm_frames, abnorm_frames


    def __getitem__(self, index):
        anchor_video = self.samples[0][index]
        pos_video = random.sample(self.samples[1], 1)[0]
        neg_video = random.sample(self.samples[0], 1)[0]

        anchor_data = self.get_img(anchor_video)
        pos_data = self.get_img(pos_video)
        neg_data = self.get_img(neg_video)
        
        return np.concatenate(anchor_data, axis=0), np.concatenate(pos_data, axis=0), np.concatenate(neg_data, axis=0)


    def get_img(self, video):
        batch = []
        video_path = os.path.join(self.dir, video[0])
        frame_id = video[1]

        for i in range(-self._time_step, self._num_pred):
            frame_name = os.path.join(video_path, str(frame_id + i).zfill(6) + '.jpg')
            image = np_load_frame(frame_name, self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))
        return batch
    
    def __len__(self):
        return len(self.samples[0])


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        default_collate([b[2] for b in batch]),
    )

