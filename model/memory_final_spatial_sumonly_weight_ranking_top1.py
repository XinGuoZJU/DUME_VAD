import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F


class MyL2Loss(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, output):
        return torch.mean(output, dim=self.dims)


def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result

def multiply(x): #to flatten matrix into a vector 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):

    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2 # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))


class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        
    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem,torch.t(self.keys_var))
        similarity[:,i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)
        
        return self.keys_var[max_idx]
    
    def random_pick_memory(self, mem, max_indices):
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices==i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)
            
        return torch.tensor(output)
    
    def forward(self, query, keys_b, mem_update_flag=True, train=True):
        # keys_b: b x m x d, the first half of memory is normal, the last half is abnormal
        keys = keys_b[0]
        batch_size, dims, h, w = query.size() # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0,2,3,1)        # b X h X w X d, this will make 'Warning: Mixed memory format inputs detected while calling the operator.'
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        
        m = keys.shape[0]
        
        # read
        updated_query = self.read(query, keys, softmax_score_memory)
       
        # losses
        loss_list = self.gather_loss(query, keys, softmax_score_query, softmax_score_memory)
            
        # update
        if train and mem_update_flag:
            updated_memory = self.update(query, keys, softmax_score_query, softmax_score_memory)
            updated_memory_b = updated_memory.expand(batch_size, -1, -1)
        else:
            # updated_memory_b = keys.expand(batch_size, -1, -1)
            updated_memory_b = None

        score_memory_b = softmax_score_memory.reshape(batch_size, h, w, m)
        
        return updated_query, updated_memory_b, score_memory_b.detach(), loss_list
    
    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)   # score of query for each memory
        score_memory = F.softmax(score,dim=1)   # score of memory for each query
       
        return score_query, score_memory
   
    def get_update_query(self, mem, max_indices, score, query):
        m, d = mem.size()
        
        query_update = torch.zeros((m,d)).cuda()
        # random_update = torch.zeros((m,d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1)==i, as_tuple=False)  # for memory i
            a, _ = idx.size()
            if a != 0:
                query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) * query[idx].squeeze(1)), dim=0)
            else:
                query_update[i] = 0
        
        return query_update

    def update(self, query, keys, softmax_score_query, softmax_score_memory):
        batch_size, h, w, dims = query.size() # b X h X w X d 
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        # _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
   
        query_update = self.get_update_query(keys, gathering_indices, softmax_score_query, query_reshape)
        # updated_memory = F.normalize(query_update + keys, dim=1)
        # return updated_memory.detach()

        return query_update.detach()

    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n,dims = query_reshape.size() # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')
        
        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
                
        return pointwise_loss

    def loss1(self, softmax_score_memory, query, keys):
        # softmax_score_memory: batch_size * h * w, m
        # query: # b, h, w, d 
        # keys: m x d
        
        batch_size, h, w, dims = query.size()                   # b x 32 x 32 x 512
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, max_ind = torch.topk(softmax_score_memory, 1, dim=1)
        loss_mse = torch.nn.MSELoss(reduction='none')

        loss = loss_mse(query_reshape, keys[max_ind].squeeze(1).detach())
        top1_loss = torch.mean(loss.view(batch_size, h, w, dims).sum(-1), dim=(1,2))   # b (1D)

        return top1_loss

    def loss2(self, softmax_score_memory, query, keys):
        # softmax_score_memory: batch_size * h * w, m
        # query: # b, h, w, d 
        # keys: m x d
        
        batch_size, h, w, dims = query.size()                   # b x 32 x 32 x 512
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)
        
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,1]]

        loss_margin = torch.nn.TripletMarginLoss(margin=1.0, reduction='none')
        loss = loss_margin(query_reshape, pos.detach(), neg.detach())

        gathering_loss = torch.mean(loss.view(batch_size, h, w), dim=(1,2))

        return gathering_loss

    '''
    def eval_loss(self, softmax_score_memory, batch_size, h, w):
        # softmax_score_memory: batch_size * h * w, m
        m = softmax_score_memory.shape[1]
        
        softmax_score_memory_re = softmax_score_memory.reshape(batch_size, h, w, m)

        # norm_mem_score = torch.mean(softmax_score_memory_re[:,:,:,:self.mem_division], dim=(1,2,3))  # we should use max here
        # abnorm_mem_score = torch.mean(softmax_score_memory_re[:,:,:,self.mem_division:], dim=(1,2,3))

        norm_mem_score, _ = torch.max(softmax_score_memory[:,:self.mem_division].reshape(batch_size, h*w*self.mem_division), dim=1)
        abnorm_mem_score, _ = torch.max(softmax_score_memory[:,self.mem_division:].reshape(batch_size, h*w*(m-self.mem_division)), dim=1)

        mem_score = torch.stack((norm_mem_score, abnorm_mem_score), 1)
        
        return mem_score
    '''

    def gather_loss(self, query, keys, softmax_score_query, softmax_score_memory):
        # softmax_score_memory: (b X h X w) X m
        batch_size, h, w, dims = query.size() # b X h X w X d  # b x 32 x 32 x 512
        
        loss1 = self.loss1(softmax_score_memory, query, keys)
        loss2 = self.loss2(softmax_score_memory, query, keys)
            
        loss = torch.stack([loss1, loss2], 1)
        # eval_score = self.eval_loss(softmax_score_memory, batch_size, h, w)
                
        return loss
    
    def read(self, query, updated_memory, softmax_score_memory):
        # updated_memory: m x d
        # softmax_score_memory: batch_size * h * w, dims
        
        batch_size, h, w, dims = query.size() # b X h X w X d
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
       
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d
        
        updated_query = torch.cat((query_reshape, concat_memory), dim = 1) # (b X h X w) X 2d
        updated_query = updated_query.contiguous().view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)

        return updated_query
    
    
