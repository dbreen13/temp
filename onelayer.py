#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:10:21 2024

@author: dbreen
"""
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim

import sys,os
import tltorch
sys.path.append("/home/Documents/tddl")
from tddl.factorizations import factorize_network
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torchvision.transforms as transforms
from time import time, perf_counter
import logging
logging.basicConfig(level = logging.INFO)
from datetime import datetime
import time as timers


logger=logging.getLogger('Layertest')
#create a fh
fh=logging.FileHandler('laytesten.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

device = torch.device('cpu')


# class SimpleNet(nn.Module):
#     def __init__(self,
#         img_channels:int,
#         kernel_size:int,
#         stride:int,
#         padding:int,
#         num_classes:int,
#         in_channels:int,
#             ):
#         super(SimpleNet,self).__init__()
#         self.in_channels=512
#         self.conv1=nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels,kernel_size=kernel_size,stride=stride,padding=padding)
#         self.conv2=nn.Conv2d(in_channels=512, out_channels=512,kernel_size=kernel_size,stride=stride,padding=padding)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu=nn.ReLU()
#         self.maxpool=nn.MaxPool2d(kernel_size,stride)
#         self.fc=nn.Linear(self.in_channels,num_classes)
    
#     def forward(self,x):
#         x=self.conv1(x)
#         x=self.conv2(x)
#         x=self.bn1(x)
#         x=self.relu(x)
#         x=self.maxpool(x)
#         x=torch.flatten(x,1)
#         x=self.fc(x)
        
# model=SimpleNet(img_channels=3,kernel_size=3,stride=1,padding=1, num_classes=10)


class SimpleNet(nn.Module):
    def __init__(self,
        out_channels:int,
        kernel_size:int,
        stride:int,
        padding:int,
        num_classes:int,
        in_channels:int,
            ):
        super(SimpleNet,self).__init__()
    
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        
    def forward(self,x):
        x=self.conv1(x)
        return x


def factorize_model(model, rank,factorization, decomposition_kwargs, fixed_rank_modes,decompose_weights):
    children=dict(model.named_children())
    for child in children.items():
        model._modules['conv1'] = tltorch.FactorizedConv.from_conv(
                    child[1], 
                    rank=rank, 
                    decompose_weights=decompose_weights, 
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation='factorized',
                )
        
def run_model(cnn_dict, fact_dict):
    #params cnn
    padding=1
    stride=1
    in_channels=cnn_dict['in_channels']
    out_channels=cnn_dict['out_channels']
    kernel_size=cnn_dict['kernel_size']
    batch_size=cnn_dict['batch_size']
    num_classes=cnn_dict['num_classes']
    n_epochs=cnn_dict['n_epochs']
    lr=cnn_dict['lr']
    img_h=cnn_dict['img_h']
    img_w=cnn_dict['img_w']
    m=cnn_dict['iterations']
    
    #params fact
    decompose_weights=False
    td_init=0.02
    return_error=False
    decompose=fact_dict['decompose']
    layers=fact_dict['layers']
    factorization=fact_dict['factorization']
    rank=fact_dict['rank']
    decomposition_kwargs = {'init': 'random'} if factorization == 'cp' else {}
    fixed_rank_modes = 'spatial' if factorization == 'tucker' else None
    ind=fact_dict['index']
    
    model=SimpleNet(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding, num_classes=num_classes)
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay = 0.005, momentum = 0.9)

    if decompose==True:
        factorize_model(model, rank,factorization, decomposition_kwargs, fixed_rank_modes, decompose_weights)
        model.to(device)
        
    model.train()
    x=torch.randn(batch_size, in_channels, img_h, img_w).to(device)
    
    output = model(x)
    batch_size, num_channels, height, width = output.size()

    output_shape=model(x).shape    
    # Reshape output for CrossEntropyLoss
    batch_size, num_channels, height, width = output.size()
    reshaped_output = output.view(batch_size, num_channels * height * width)

    # Create random target labels (assuming 10 classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    # Define CrossEntropyLoss criterion
    criterion = nn.CrossEntropyLoss()

    list_out=[]
    list_labels=[]
    
    for i in range(2):
        output_new=torch.randn(output.size(),requires_grad=True).to(device)
        reshaped_output = output_new.view(batch_size, num_channels * height * width)
        list_out.append(reshaped_output)
        list_labels.append(torch.randint(0, num_classes, (batch_size,))
    )
        
    #start_time = time()
    timers.sleep(30)
    now=datetime.now()
    sec_wait=60-now.second
    timers.sleep(sec_wait)
    logger.info(f"start-forward-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    for _ in tqdm(range(m), desc="Forward Iterations"):
        output = model(x)
    logger.info(f"end-forward-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    #end_time = time()
    #elapsed_time = end_time - start_time

    #print(f"Time taken: {elapsed_time} seconds")

    #start_time = time()
    logger.info(f"start-back-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    for i in tqdm(range(m), desc="Backward Iterations"):
        if i % 2 == 0:
            output_it=list_out[0]
        else:
            output_it=list_out[1]
            
        optimizer.zero_grad()
        # Compute the loss directly on reshaped output
        loss = criterion(output_it, labels)
        # Backward pass
        loss.backward()
        optimizer.step() 
    logger.info(f"end-back-outch{out_channels}-inch{in_channels}-fact{factorization}-ind{ind}s")
    #end_time = time()
    #elapsed_time = end_time - start_time
    #print(f"Time taken: {elapsed_time} seconds")

    return(model)
    
            
    
#%%Define all params 

#Define cnn model parameters
in_channels=[64]
out_channels=[64]
kernel_size=1
num_classes=2
padding=1
stride=1
batch_size=200
n_epochs=1
lr=1e-5
it=30000

#parameters the dataset
input_size=in_channels
img_h=10
img_w=10
#Define factorization parameters
decompose=True
layers=[0]
factorizations=['cp','tucker','tt']
rank=0.1
ind=0

cnn_dict={"in_channels": in_channels[0],
             "out_channels": out_channels[0],
             "kernel_size": kernel_size,
             "batch_size": batch_size,
             "num_classes": num_classes,
             "n_epochs": n_epochs,
             "lr":lr,
             "img_h": img_h,
             "img_w": img_w,
             "iterations": it,
             }

fact_dict={"decompose":decompose,
           "layers": layers,
           "factorization": factorizations[0],
           "rank" : rank,
           "index": ind,
           }

#model_trained=run_model(cnn_dict, fact_dict)


#%% Run model
for ind in range(2):
    fact_dict["index"]=ind
    for out_channel in out_channels:
        for in_channel in in_channels:
            cnn_dict['in_channels']=in_channel
            cnn_dict['out_channels']=out_channel
            if in_channel==out_channel or in_channel==out_channel/2:
                for factorization in factorizations:
                    cnn_dict['factorization']=factorization
                    model_trained=run_model(cnn_dict, fact_dict)
    
    
    

