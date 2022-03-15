import argparse
from collections import OrderedDict
import configparser
import copy
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as v_utils

from model.utils_tri import DataLoader, collate
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from trainer import Trainer
from utils import *



parser = argparse.ArgumentParser(description="GDAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=4, help='batch size for test')
parser.add_argument('--config_file', type=str, help='configs/ped1.ini')
parser.add_argument('--kfolds', type=str, help='0-9')

args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config_file)

epochs = config.getint('train', 'epochs')
lr = config.getfloat('train', 'lr')
w_list = eval(config.get('train', 'w_list'))
validation_interval = config.getint('train', 'validation_interval')
margin = config.getfloat('train', 'margin')
batch_multiplier = config.getint('train', 'batch_multiplier')
num_workers = config.getint('train', 'num_workers')

num_workers_test = config.getint('test', 'num_workers_test')

h = config.getint('model', 'h')
w = config.getint('model', 'w')
c = config.getint('model', 'c')
t_length = config.getint('model', 't_length')

fdim = config.getint('model', 'fdim')
mdim = eval(config.get('model', 'mdim'))
msize = eval(config.get('model', 'msize'))

dataset_type =  config.get('data', 'dataset_type')
dataset_path =  config.get('data', 'dataset_path')
exp_dir = args.config_file.split('/')[-1].split('.')[0] + '_' + args.kfolds


seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
np.random.seed(seed) # Numpy module.
random.seed(seed) # Python random module.

# torch.backends.cudnn.benchmark = False  # default is True
torch.backends.cudnn.deterministic = True  # default is False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")

base_folder = os.path.join(dataset_path, dataset_type)

# Loading dataset
train_dataset = DataLoader(base_folder, 'train_' + args.kfolds, transforms.Compose([transforms.ToTensor(),]), 
                            resize_height=h, resize_width=w, time_step=t_length-1)
test_dataset = DataLoader(base_folder, 'test_' + args.kfolds, transforms.Compose([transforms.ToTensor(),]), 
                            resize_height=h, resize_width=w, time_step=t_length-1)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_loader = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate)
val_loader = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=num_workers_test, drop_last=True, collate_fn=collate)


# Report the training process
log_dir = os.path.join('./exp', dataset_type, exp_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

shutil.copy(args.config_file, log_dir)

# Model setting
model = convAE(c, t_length, msize, fdim, mdim)

model_path = os.path.join(log_dir, 'checkpoint_latest.pth')
mfile_path = os.path.join(log_dir, 'keys_latest.pt')
if os.path.exists(model_path) and os.path.exists(mfile_path):
    print("Resumed from pretrained model!")
    resume = True
else:
    resume = False

iteration = 0
num_image = 0
m_items_test = {'mem_f': None, 'mem_b': None}
if resume:
     # Loading the trained model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    m_items = torch.load(mfile_path)
    m_items_test = m_items.copy()
    
    # optim_state = checkpoint['optim_state_dict']
    iteration = checkpoint['iteration']
    num_image = checkpoint['num_image']

params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params_discriminater_f = list(model.discriminater_f.parameters())
params_discriminater_b = list(model.discriminater_b.parameters())
params = params_encoder + params_decoder + params_discriminater_f + params_discriminater_b
    
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)

model.cuda()

optimizer = torch.optim.Adam(params, lr=lr)
if resume:
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch_size = len(train_loader)
    start_epoch = iteration // epoch_size
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=start_epoch-1)
else:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

orig_stdout = sys.stdout
#f = open(os.path.join(log_dir, 'log.txt'),'w')
#sys.stdout= f

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    batch_multiplier=batch_multiplier,
    train_loader=train_loader,
    val_loader=val_loader,
    validation_interval=validation_interval,
    max_epoch=epochs,
    lr_decay_epoch=None,
    iteration=iteration,
    num_image=num_image,
    m_items=m_items_test,
    logdir=log_dir,
    msize=msize,
    mdim=mdim,
    w_list=w_list,
    margin=margin,
)
trainer.train()

#sys.stdout = orig_stdout
#f.close()


