import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import configparser
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def imshow(im):
    plt.close()
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])
 
    fig = plt.figure()
    fig.set_size_inches(width/100.0, height/100.0, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(im)

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--model_dir', default='./exp/ShanghaiTech_AUG', type=str, help='directory of model: exp/Ped1_AUG, exp/Ped2_AUG, exp/Avenue_AUG, exp/ShanghaiTech_AUG')
parser.add_argument('--dataset', default='shanghaitech', type=str, help='sub directory of model without flods num: exp/Ped1_AUG/ped1_dm_0')
parser.add_argument('--kfolds', type=str, help='0-9')
parser.add_argument('--alpha', type=float, help='0.0-1.0')
parser.add_argument('--save_pred_img', action='store_true', help='save predicted image')
parser.add_argument('--save_query', action='store_true', help='save queries from encoder')

args = parser.parse_args()
dataset_type = args.dataset
model_type = dataset_type + '_' + args.kfolds

config_file = os.path.join(args.model_dir, model_type, dataset_type+ '.ini') 
config = configparser.ConfigParser()
config.read(config_file)

h = config.getint('model', 'h')
w = config.getint('model', 'w')
c = config.getint('model', 'c')
t_length = config.getint('model', 't_length')
fdim = config.getint('model', 'fdim')
mdim = eval(config.get('model', 'mdim'))
msize = eval(config.get('model', 'msize'))
num_workers_test = config.getint('test', 'num_workers_test')


torch.manual_seed(2020)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = os.path.join('dataset', args.model_dir.split('/')[-1])

# Loading dataset
test_dataset = DataLoader(test_folder, 'test_' + args.kfolds, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=h, resize_width=w, time_step=t_length-1)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model_path = os.path.join(args.model_dir, model_type)
pred_img_save_path = os.path.join('viz/pred', model_type)
query_save_path = os.path.join('viz/query', model_type)

checkpoint_path = os.path.join(model_path, 'checkpoint_latest.pth')
m_items_path = os.path.join(model_path, 'keys_latest.pt')

checkpoint = torch.load(checkpoint_path)
model = convAE(c, t_length, msize, fdim, mdim)
model.load_state_dict(checkpoint["model_state_dict"])
model.cuda()

m_items = torch.load(m_items_path)

cls_list = {}
rec_list = {}
psnr_list = {}
feature_distance_list = {}

video_num = 0
m_items_test = m_items.copy()
m_items_f = m_items_test['mem_f']
m_items_b = m_items_test['mem_b']
m_items_f_batch = m_items_f.expand(args.test_batch_size, -1, -1) 
m_items_b_batch = m_items_b.expand(args.test_batch_size, -1, -1)

model.eval()

with torch.no_grad():
    for k, (imgs, cls_label_tensor, rec_label_tensor, video_name_tuple, frame_id_tensor) in enumerate(test_batch):
        tprint('%d of %d ' % (k, len(test_batch)))

        imgs = Variable(imgs).cuda()
        cls_label = int(cls_label_tensor[0])   # norm 0 abnorm 1
        rec_label = int(cls_label_tensor[0])  
        video_name_org = video_name_tuple[0]
        video_name = '_'.join(video_name_org.split('_')[:-1])
        frame_id = int(frame_id_tensor[0])

        outputs, fea_f, fea_b, _, _, _, _, _, _, loss_tensor_f, loss_tensor_b = model.forward(imgs[:,0:12], m_items_f_batch, m_items_b_batch, state_flag='anchor', mem_update_flag=False, train=False)

        compactness_loss = loss_tensor_b[0][0]

        if cls_label == 0:
            flag = 'norm2norm'
        else:
            flag = 'abnorm2abnorm'
        
        if args.save_query:
            query_video_path = os.path.join(query_save_path, flag, video_name_org)
            os.makedirs(query_video_path, exist_ok=True)
            query_img_name = os.path.join(query_video_path, str(frame_id).zfill(4) + '.npz')
            np.savez(query_img_name, fea_f=fea_f[0].cpu().numpy(), fea_b=fea_b[0].cpu().numpy())

        if args.save_pred_img:
            pred_img = ((outputs[0].permute(1,2,0).cpu().numpy() + 1.0) * 127.5).astype(int)
            pred_video_path = os.path.join(pred_img_save_path, flag, video_name_org)
            os.makedirs(pred_video_path, exist_ok=True)
            pred_img_name = os.path.join(pred_video_path, str(frame_id).zfill(4) + '.png')
            cv2.imwrite(pred_img_name, pred_img)

            img_diff = torch.abs(imgs[0,3*4:] - outputs[0]).permute(1,2,0).cpu().numpy() / 2
            imshow(img_diff)
            diff_img_name = os.path.join(pred_video_path, str(frame_id).zfill(4) + '_diff.png')
            plt.savefig(diff_img_name)

        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()
    
        if video_name not in psnr_list:
            psnr_list[video_name] = []
        if video_name not in feature_distance_list:
            feature_distance_list[video_name] = []
        if video_name not in cls_list:
            cls_list[video_name] = []
        if video_name not in rec_list:
            rec_list[video_name] = []
        
        psnr_list[video_name].append(psnr(mse_imgs))
        feature_distance_list[video_name].append(mse_feas)
        cls_list[video_name].append(cls_label)
        rec_list[video_name].append(rec_label)
        

# Measuring the abnormality score and the AUC  # norm 0 abnorm 1
anomaly_score_total_list = []
cls_labels = []
rec_labels = []
for video_name in psnr_list.keys():
    print(video_name)
    anomaly_score_total_list += score_sum(anomaly_score_list_inv(psnr_list[video_name]), 
                                    anomaly_score_list(feature_distance_list[video_name]), args.alpha)
     
    cls_labels += cls_list[video_name]
    rec_labels += rec_list[video_name]
   
anomaly_score_total_list = np.asarray(anomaly_score_total_list)
cls_labels = np.asarray(cls_labels)
rec_labels = np.asarray(rec_labels)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(rec_labels, 0))

os.makedirs('viz/final_result', exist_ok=True)
save_name = 'viz/final_result/' + model_type + '.npz'

np.savez(save_name, anomaly_score=anomaly_score_total_list, rec_label=rec_labels, auc=accuracy) 
print()
print('The result of ', model_type)
print('AUC: ', accuracy*100, '%')

