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
        frames = []
        for line in file_list:
            file_path, all_frame_num = line.strip().split()
            prefix = file_path.split('.')[0].split('/')
            if prefix[0] == 'norm' or prefix[0] == 'norm2norm':
                cls_label = 0
                rec_label = 0
            elif prefix[0] == 'abnorm2abnorm':
                cls_label = 1
                rec_label = 1
            elif prefix[0] == 'norm2abnorm':
                cls_label = 0
                rec_label = 1
            elif prefix[0] == 'abnorm2norm':
                cls_label = 1
                rec_label = 0
            else:
                raise ValueError('Data error!')
            video_name = prefix[2]
            frame_id = int(prefix[3])
            
            if frame_id - self._time_step >= 0 and  frame_id + self._num_pred <= int(all_frame_num):  # 0,1,2,  3(fram_id),5 
                frames.append(['/'.join(prefix[:-1]), frame_id, cls_label, rec_label, video_name])
        
        return frames

    def __getitem__(self, index):
        video_path = os.path.join(self.dir, self.samples[index][0])
        frame_id = self.samples[index][1]
        cls_label = self.samples[index][2]
        rec_label = self.samples[index][3]
        video_name = self.samples[index][4]

        batch = []
        for i in range(-self._time_step, self._num_pred):
            frame_name = os.path.join(video_path, str(frame_id + i).zfill(4) + '.jpg')
            image = np_load_frame(frame_name, self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))
        
        return np.concatenate(batch, axis=0), np.array(cls_label), np.array(rec_label), video_name, frame_id

    def __len__(self):
        return len(self.samples)


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        default_collate([b[2] for b in batch]),
        default_collate([b[3] for b in batch]),
        default_collate([b[4] for b in batch]),
    )

