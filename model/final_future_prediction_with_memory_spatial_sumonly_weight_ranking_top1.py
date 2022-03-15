import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_final_spatial_sumonly_weight_ranking_top1 import *


class Discriminater(torch.nn.Module):
    def __init__(self):
        super(Discriminater, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)

        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):

        tensorConv1 = self.conv1(x)              # 1024 x 32 x 32
        tensorBN1 = self.bn1(tensorConv1)
        tensorRL1 = self.relu(tensorBN1)
        tensorPool1 = self.pooling1(tensorRL1)   # 1024 x 16 x 16

        tensorConv2 = self.conv2(tensorPool1)
        tensorBN2 = self.bn2(tensorConv2)
        tensorRL2 = self.relu(tensorBN2)
        tensorPool2 = self.pooling2(tensorRL2)   # 512 x 8 x 8

        tensorConv3 = self.conv3(tensorPool2)
        tensorBN3 = self.bn3(tensorConv3)
        tensorRL3 = self.relu(tensorBN3)
        tensorPool3 = self.pooling3(tensorRL3)   # 256 x 4 x 4

        tensorResize = tensorPool3.reshape(-1, 256 * 4 * 4)
        tensorFC1 = self.fc1(tensorResize)
        tensorFC2 = self.fc2(tensorFC1)
        tensorFC3 = self.fc3(tensorFC2)

        return tensorFC3


class Discriminater2(torch.nn.Module):
    def __init__(self):
        super(Discriminater2, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)

        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):

        tensorConv1 = self.conv1(x)              # 512 x 32 x 32
        tensorBN1 = self.bn1(tensorConv1)
        tensorRL1 = self.relu(tensorBN1)
        tensorPool1 = self.pooling1(tensorRL1)   # 512 x 16 x 16

        tensorConv2 = self.conv2(tensorPool1)
        tensorBN2 = self.bn2(tensorConv2)
        tensorRL2 = self.relu(tensorBN2)
        tensorPool2 = self.pooling2(tensorRL2)   # 256 x 8 x 8

        tensorConv3 = self.conv3(tensorPool2)
        tensorBN3 = self.bn3(tensorConv3)
        tensorRL3 = self.relu(tensorBN3)
        tensorPool3 = self.pooling3(tensorRL3)   # 128 x 4 x 4

        tensorResize = tensorPool3.reshape(-1, 128 * 4 * 4)
        tensorFC1 = self.fc1(tensorResize)
        tensorFC2 = self.fc2(tensorFC1)
        tensorFC3 = self.fc3(tensorFC2)
        
        output = F.normalize(tensorFC3, dim=1)
        
        return output


class Discriminater3(torch.nn.Module):
    def __init__(self):
        super(Discriminater3, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        

        # self.conv2 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.avepooling = torch.nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(256)
        #self.pooling1 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        tensorConv1 = self.conv1(x)              # 256 x 32 x 32
        tensorBN1 = self.bn1(tensorConv1)
        tensorPool =  self.avepooling(tensorBN1).squeeze(3).squeeze(2)
       
        output = F.normalize(tensorPool, dim=1)
        
        return output


class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4_f = Basic_(256, 256)
        self.moduleConv4_b = Basic_(256, 256)
        # self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        # self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4_f = self.moduleConv4_f(tensorPool3)
        tensorConv4_b = self.moduleConv4_b(tensorPool3)

        return tensorConv4_f, tensorConv4_b, tensorConv1, tensorConv2, tensorConv3

    
class Decoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        
        
        
    def forward(self, x, skip1, skip2, skip3):
        
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(cat2)

                
        return output
    

class Decoder2(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder2, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(256, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(128, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(64,n_channel,64)
        
        
    def forward(self, x, skip1, skip2, skip3):
        
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        # cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        # cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        # cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(tensorUpsample2)
                
        return output
    


class convAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 5, memory_size = 20, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory_f = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)
        self.memory_b = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)
        self.discriminater_f = Discriminater3()
        self.discriminater_b = Discriminater3()

    def forward(self, x, keys_f, keys_b, state_flag, mem_update_flag=True, train=True, baseline_flag=False):
        fea_f, fea_b, skip1, skip2, skip3 = self.encoder(x)
        if (state_flag == 'pos') and mem_update_flag:
            f_mem_update_flag = True
        else:
            f_mem_update_flag = False

        if (state_flag == 'anchor' or state_flag == 'neg') and mem_update_flag:
            b_mem_update_flag = True
        else:
            b_mem_update_flag = False
            
        updated_fea_f, keys_f, softmax_score_memory_f, loss_tensor_f = self.memory_f(fea_f, keys_f, mem_update_flag=f_mem_update_flag, train=train)
        updated_fea_b, keys_b, softmax_score_memory_b, loss_tensor_b = self.memory_b(fea_b, keys_b, mem_update_flag=b_mem_update_flag, train=train)

        updated_fea = torch.cat((updated_fea_f, updated_fea_b), dim=1) 
        output = self.decoder(updated_fea, skip1, skip2, skip3)
        
        dis_fea_f = self.discriminater_f(updated_fea_f)
        dis_fea_b = self.discriminater_b(updated_fea_b)
        
        return output, fea_f, fea_b, dis_fea_f, dis_fea_b, keys_f, keys_b, softmax_score_memory_f, softmax_score_memory_b, loss_tensor_f, loss_tensor_b
        
        
