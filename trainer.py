import os
import cv2
import time
import atexit
import shutil
import signal
import os.path as osp
import threading
import subprocess
from timeit import default_timer as timer

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import io
from torch.utils.tensorboard import SummaryWriter
from utils import *


class WMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, t1, t2, w):
        if w.sum() == 0:
            return self.loss(t1, t2) * 0
        else:
            idx  = (w == 1).nonzero(as_tuple=False).squeeze(1)
            tt1 = torch.index_select(t1, 0, idx)
            tt2 = torch.index_select(t2, 0, idx)

            return self.loss(tt1, tt2)


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, batch_multiplier, train_loader, val_loader, validation_interval, 
                max_epoch, lr_decay_epoch, iteration, num_image, m_items, logdir, msize, mdim, w_list, margin):
        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.batch_multiplier = batch_multiplier

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.validation_interval = validation_interval

        self.logdir = logdir
        if not osp.exists(self.logdir):
            os.makedirs(self.logdir)

        self.tensorboard_port = 6006
        self.run_tensorboard()
        time.sleep(1)

        self.epoch = 0
        self.iteration = iteration
        self.num_image = num_image
        self.m_items_f = m_items['mem_f']
        self.m_items_b = m_items['mem_b']
        self.max_epoch = max_epoch
        self.lr_decay_epoch = lr_decay_epoch

        # self.mean_loss = self.best_mean_loss = 1e1000
        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)
        
        self.loss_func_mse = torch.nn.MSELoss(reduction='none')
        self.loss_func_wmse = WMSELoss()
        self.loss_cross_entropy = torch.nn.CrossEntropyLoss()
        self.msize = msize
        self.mdim = mdim

        self.w_list = w_list
        self.margin = margin

    def run_tensorboard(self):
        board_out = osp.join(self.logdir, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        '''
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        p = subprocess.Popen(
            ["tensorboard", f"--logdir={board_out}", f"--port={self.tensorboard_port}"]
        )

        def killme():
            os.kill(p.pid, signal.SIGTERM)

        atexit.register(killme)
        '''

    def _loss(self):
        return 0


    def _eval(self, cls_label, eval_score):
        norm_num = cls_label.sum() 
        abnorm_num = (1-cls_label).sum()
        
        mask = cls_label.unsqueeze(1).repeat(1,2)
        norm_eval_score = mask * eval_score
        abnorm_eval_score = (1 - mask) * eval_score
            
        norm2norm_score = norm_eval_score[:,0].sum()
        norm2abnorm_score = norm_eval_score[:,1].sum()

        abnorm2norm_score = abnorm_eval_score[:,0].sum()
        abnorm2abnorm_score = abnorm_eval_score[:,1].sum()
        
        return norm2norm_score, norm2abnorm_score, abnorm2norm_score, abnorm2abnorm_score, norm_num, abnorm_num
        

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        npz = osp.join(self.logdir, "npz", f"{self.num_image:09d}")
        osp.exists(npz) or os.makedirs(npz)

        triplet_loss_f = triplet_loss_b = pixel_loss_anchor = pixel_loss_neg = loss_all_f_anchor = loss_all_f_pos = loss_all_b_anchor = loss_all_b_neg = 0
        
        #rec_labels_list = []
        #psnr_list = []
        feature_distance_list = []
        loss_margin_f = torch.nn.TripletMarginLoss(margin=self.margin, reduction='none')
        loss_margin_b = torch.nn.TripletMarginLoss(margin=self.margin, reduction='none')
        with torch.no_grad():
            for k, (imgs_anchor, imgs_pos, imgs_neg) in enumerate(self.val_loader):
                test_batch_size = imgs_anchor.shape[0]
                
                imgs_anchor = Variable(imgs_anchor).cuda()
                imgs_pos = Variable(imgs_pos).cuda()
                imgs_neg = Variable(imgs_neg).cuda()
                
                m_items_f_batch = self.m_items_f.expand(test_batch_size, -1, -1)
                m_items_b_batch = self.m_items_b.expand(test_batch_size, -1, -1)
               
                # run forward
                outputs_anchor, _, _, dis_fea_f_anchor, dis_fea_b_anchor, _, _, _, _, loss_tensor_f_anchor, loss_tensor_b_anchor = self.model.forward(imgs_anchor[:,0:12], 
                            m_items_f_batch, m_items_b_batch, state_flag='anchor', mem_update_flag=False, train=False)
                
                outputs_pos, _, _, dis_fea_f_pos, dis_fea_b_pos, _, _, _, _, loss_tensor_f_pos, loss_tensor_b_pos = self.model.forward(imgs_pos[:,0:12], 
                            m_items_f_batch, m_items_b_batch, state_flag='pos', mem_update_flag=False, train=False)            
                
                outputs_neg, _, _, dis_fea_f_neg, dis_fea_b_neg, _, _, _, _, loss_tensor_f_neg, loss_tensor_b_neg = self.model.forward(imgs_neg[:,0:12], 
                            m_items_f_batch, m_items_b_batch, state_flag='neg', mem_update_flag=False, train=False)
                
                # loss for each iteration
                triplet_loss_f += loss_margin_f(dis_fea_f_anchor, dis_fea_f_neg, dis_fea_f_pos).mean(0)
                triplet_loss_b += loss_margin_b(dis_fea_b_anchor, dis_fea_b_neg, dis_fea_b_pos).mean(0)
                
                pixel_loss_anchor += torch.mean(self.loss_func_mse(outputs_anchor, imgs_anchor[:,12:]))
                pixel_loss_neg += torch.mean(self.loss_func_mse(outputs_neg, imgs_neg[:,12:]))
            
                loss_all_f_anchor += loss_tensor_f_anchor.mean(0)
                loss_all_f_pos += loss_tensor_f_pos.mean(0)
            
                loss_all_b_anchor += loss_tensor_b_anchor.mean(0)
                loss_all_b_neg += loss_tensor_b_neg.mean(0)

                #for batch_t in range(test_batch_size):
                #    rec_labels_list.append(rec_label[batch_t].item())
                #
                #    mse_imgs = torch.mean(self.loss_func_mse((outputs_anchor[batch_t]+1)/2, (imgs_anchor[batch_t,3*4:]+1)/2)).item()  # here they ask batch size = 1, the imgs are centerilized
                #    mse_feas = loss_tensor_f_anchor[:,0][batch_t].item()  # I think it does not work.
                #    
                #    psnr_list.append(psnr(mse_imgs))
                #    feature_distance_list.append(mse_feas)
            
            num_batch = len(self.val_loader)
            triplet_loss_f /= num_batch
            triplet_loss_b /= num_batch
            pixel_loss_anchor /= num_batch
            pixel_loss_neg /= num_batch
            loss_all_f_anchor /= num_batch
            loss_all_f_pos /= num_batch
            loss_all_b_anchor /= num_batch
            loss_all_b_neg /= num_batch

        #anomaly_score_total_list = score_sum(anomaly_score(psnr_list, np.max(psnr_list), np.min(psnr_list)), 
        #                            anomaly_score_inv(feature_distance_list, np.max(feature_distance_list), np.min(feature_distance_list)), 0.6)
        #anomaly_score_total_list = np.asarray(anomaly_score_total_list)

        #rec_labels_list = np.array(rec_labels_list)
        #accuracy = AUC(anomaly_score_total_list, np.expand_dims(rec_labels_list, 0))
        
        loss = pixel_loss_anchor + pixel_loss_neg +\
                (torch.FloatTensor(self.w_list[0]).cuda() * loss_all_f_anchor).sum() +\
                (torch.FloatTensor(self.w_list[0]).cuda() * loss_all_f_pos).sum() +\
                (torch.FloatTensor(self.w_list[1]).cuda() * loss_all_b_anchor).sum() +\
                (torch.FloatTensor(self.w_list[1]).cuda() * loss_all_b_neg).sum() +\
                self.w_list[2][0] * triplet_loss_f + self.w_list[2][1] * triplet_loss_b 

        loss_dict = {'Total_loss': loss, 'Pixel_loss_anchor': pixel_loss_anchor, 'Pixel_loss_neg': pixel_loss_neg,  'F_triplet_loss': triplet_loss_f, 'B_triplet_loss': triplet_loss_b,
                    'F_loss1_anchor': loss_all_f_anchor[0], 'F_loss2_anchor': loss_all_f_anchor[1], 'F_loss1_pos': loss_all_f_pos[0], 'F_loss2_pos': loss_all_f_pos[1],
                    'B_loss1_anchor': loss_all_b_anchor[0], 'B_loss2_anchor': loss_all_b_anchor[1], 'B_loss1_neg': loss_all_b_neg[0], 'B_loss2_neg': loss_all_b_neg[1]} 
        
        self._write_metrics(1, loss_dict, "validation")
        # self.mean_loss = loss / len(self.val_loader)

        torch.save(
            {
                "iteration": self.iteration,
                "num_image": self.num_image,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                #"best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.logdir, "checkpoint_latest.pth"),
        )
        torch.save({'mem_f': self.m_items_f, 'mem_b': self.m_items_b}, os.path.join(self.logdir, 'keys_latest.pt'))
        
        shutil.copy(
            osp.join(self.logdir, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        shutil.copy(
            osp.join(self.logdir, "keys_latest.pt"),
            osp.join(npz, "keys.pt"),
        )
        self.model.train()
    

    def train_epoch(self):
        self.model.train()
       
        time = timer()
        self.optim.zero_grad()
    
        loss_margin = torch.nn.TripletMarginLoss(margin=self.margin, reduction='none')

        triplet_loss_f_bm = triplet_loss_b_bm = pixel_loss_bm_anchor = pixel_loss_bm_neg = loss_all_f_bm_anchor = loss_all_f_bm_pos = loss_all_b_bm_anchor = loss_all_b_bm_neg = loss_bm = 0
        # n2n_score_bm = n2a_score_bm = a2n_score_bm = a2a_score_bm = norm_num_bm = abnorm_num_bm = 0
        for j, (imgs_anchor, imgs_pos, imgs_neg) in enumerate(self.train_loader):
            batch_size = imgs_anchor.shape[0] 
            
            imgs_anchor = Variable(imgs_anchor).cuda()
            imgs_pos = Variable(imgs_pos).cuda()
            imgs_neg = Variable(imgs_neg).cuda()
            
            m_items_f_batch = self.m_items_f.expand(batch_size, -1, -1)
            m_items_b_batch = self.m_items_b.expand(batch_size, -1, -1)
            if self.iteration % self.batch_multiplier == 0:        # and self.iteration > self.batch_multiplier * 10:  # avoid cold start
                mem_update_flag=True
            else:
                mem_update_flag=False
            
            # run forward
            outputs_anchor, _, _, dis_fea_f_anchor, dis_fea_b_anchor, m_items_f_batch_anchor, m_items_b_batch_anchor, _, _, loss_tensor_f_anchor, loss_tensor_b_anchor = self.model.forward(imgs_anchor[:,0:12], 
                        m_items_f_batch, m_items_b_batch, state_flag='anchor', mem_update_flag=mem_update_flag, train=True)
            
            outputs_pos, _, _, dis_fea_f_pos, dis_fea_b_pos, m_items_f_batch_pos, _, _, _, loss_tensor_f_pos, loss_tensor_b_pos = self.model.forward(imgs_pos[:,0:12],
                        m_items_f_batch, m_items_b_batch, state_flag='pos', mem_update_flag=mem_update_flag, train=True)            
            
            outputs_neg, _, _, dis_fea_f_neg, dis_fea_b_neg, _, m_items_b_batch_neg, _, _, loss_tensor_f_neg, loss_tensor_b_neg = self.model.forward(imgs_neg[:,0:12],
                        m_items_f_batch, m_items_b_batch, state_flag='neg', mem_update_flag=mem_update_flag, train=True)
           
            if mem_update_flag:
                self.m_items_f = F.normalize(m_items_f_batch[0] + m_items_f_batch_pos[0], dim=1)
                self.m_items_b = F.normalize(m_items_b_batch[0] + m_items_b_batch_neg[0] + m_items_b_batch_anchor[0], dim=1)
            
            # loss for each iteration
            triplet_loss_f = loss_margin(dis_fea_f_anchor, dis_fea_f_neg, dis_fea_f_pos).mean(0)
            triplet_loss_b = loss_margin(dis_fea_b_anchor, dis_fea_b_neg, dis_fea_b_pos).mean(0)
        
            pixel_loss_anchor = torch.mean(self.loss_func_mse(outputs_anchor, imgs_anchor[:,12:]))
            pixel_loss_neg = torch.mean(self.loss_func_mse(outputs_neg, imgs_neg[:,12:]))
            
            loss_all_f_anchor = loss_tensor_f_anchor.mean(0)
            loss_all_f_pos = loss_tensor_f_pos.mean(0)
            
            loss_all_b_anchor = loss_tensor_b_anchor.mean(0)
            loss_all_b_neg = loss_tensor_b_neg.mean(0)
           
            loss = (pixel_loss_anchor + pixel_loss_neg + 
                    (torch.FloatTensor(self.w_list[0]).cuda() * loss_all_f_anchor).sum() + 
                    (torch.FloatTensor(self.w_list[0]).cuda() * loss_all_f_pos).sum() + 
                    (torch.FloatTensor(self.w_list[1]).cuda() * loss_all_b_anchor).sum() + 
                    (torch.FloatTensor(self.w_list[1]).cuda() * loss_all_b_neg).sum() + 
                    self.w_list[2][0] * triplet_loss_f + self.w_list[2][1] * triplet_loss_b) / self.batch_multiplier
            
            # loss for each batch_multiplier
            triplet_loss_f_bm += triplet_loss_f / self.batch_multiplier 
            triplet_loss_b_bm += triplet_loss_b / self.batch_multiplier 

            pixel_loss_bm_anchor += pixel_loss_anchor / self.batch_multiplier
            pixel_loss_bm_neg += pixel_loss_neg / self.batch_multiplier

            loss_all_f_bm_anchor += loss_all_f_anchor / self.batch_multiplier
            loss_all_f_bm_pos += loss_all_f_pos / self.batch_multiplier
            
            loss_all_b_bm_anchor += loss_all_b_anchor / self.batch_multiplier
            loss_all_b_bm_neg += loss_all_b_neg / self.batch_multiplier
            
            loss_bm += loss
            
            loss.backward(retain_graph=True)
            # loss.backward()
            self.iteration += 1
            self.num_image += batch_size 
            
            if self.iteration % self.batch_multiplier == 0:
                lr = self.optim.param_groups[0]['lr'] 
                self.optim.step()
                self.optim.zero_grad()
            
                loss_dict = {'Total_loss': loss_bm, 'Pixel_loss_anchor': pixel_loss_bm_anchor, 'Pixel_loss_neg': pixel_loss_bm_neg, 
                                'F_triplet_loss': triplet_loss_f_bm, 'B_triplet_loss': triplet_loss_b_bm,
                                'F_loss1_anchor': loss_all_f_bm_anchor[0], 'F_loss2_anchor': loss_all_f_bm_anchor[1], 'F_loss1_pos': loss_all_f_bm_pos[0], 'F_loss2_pos': loss_all_f_bm_pos[1],
                                'B_loss1_anchor': loss_all_b_bm_anchor[0], 'B_loss2_anchor': loss_all_b_bm_anchor[1], 'B_loss1_neg': loss_all_b_bm_neg[0], 'B_loss2_neg': loss_all_b_bm_neg[1], 'Learning_rate': np.array(lr)} 
                
                triplet_loss_f_bm = triplet_loss_b_bm = pixel_loss_bm_anchor = pixel_loss_bm_neg = loss_all_f_bm_anchor = loss_all_f_bm_pos = loss_all_b_bm_anchor = loss_all_b_bm_neg = loss_bm = 0
                
                self._write_metrics(1, loss_dict, "training")

                if self.iteration % 4 == 0:
                    tprint(
                        f"{self.epoch:03}/{self.num_image // 1000:04}k| "
                        + f"| {4 * batch_size / (timer() - time):04.1f} "
                    )
                    time = timer()

                if self.iteration % self.validation_interval == 0 or self.num_image == 800:
                    self.validate()
                    time = timer()
            

    def train(self):
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size

        if self.m_items_f is None or self.m_items_b is None:
            # self.m_items = F.normalize(torch.rand((self.msize, self.mdim), dtype=torch.float), dim=1).cuda()  # Initialize the memory items
            
            mem_f = torch.rand((self.msize[0], self.mdim[0]), dtype=torch.float) 
            mem_b = torch.rand((self.msize[1], self.mdim[1]), dtype=torch.float)

            self.m_items_f = F.normalize(mem_f, dim=1).cuda()
            self.m_items_b = F.normalize(mem_b, dim=1).cuda()

        for self.epoch in range(start_epoch, self.max_epoch):
            #if self.epoch == self.lr_decay_epoch:
            #    self.optim.param_groups[0]["lr"] /= 10  # ??? maybe bugs here
            self.train_epoch()
            self.scheduler.step()
        
        self.validate()
    

    def _write_metrics(self, size, loss_dict, prefix):
        for key in loss_dict.keys():
            if key == 'AUC':
                self.writer.add_scalar(
                    f"{prefix}/" + key, loss_dict[key].item(), self.iteration
                )
            else:
                self.writer.add_scalar(
                    f"{prefix}/" + key, loss_dict[key].item() / size, self.iteration
                )


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)
