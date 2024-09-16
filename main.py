import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import os
import logging
from dataset import NPZDataset
import numpy as np
from utils import Metric,beijing_time, set_logger,init_weights,set_seed
import argparse
import time
from models import SSRMamba,HSRNet,SSRNet
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parse = argparse.ArgumentParser()
parse.add_argument('--model', type=str,default='SSRMamba')
parse.add_argument('--log_out', type=int,default=0)
parse.add_argument('--dataset', type=str,default='CAVE')
parse.add_argument('--check_point', type=str,default=None)
parse.add_argument('--check_step', type=int,default=1)
parse.add_argument('--lr', type=float, default=1e-4)
parse.add_argument('--batch_size', type=int, default=32)
parse.add_argument('--epochs', type=int,default=1000)
parse.add_argument('--seed', type=int,default=3407) 
parse.add_argument('--dim', type=int,default=128)
parse.add_argument('--comments', type=str,default='')
parse.add_argument('--grid_size', type=int,default=5)
parse.add_argument('--spline_order', type=int,default=3)
parse.add_argument('--wandb', type=int,default=1)
parse.add_argument('--gpu', type=int,default=1)
args = parse.parse_args()

    
if args.log_out == 0:
    os.environ['WANDB_MODE'] = 'offline'
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = args.model
model = None
HSI_bands = None
test_dataset_path = None
train_dataset_path = None
if args.dataset == 'CAVE':
    HSI_bands = 31
    train_dataset_path = '/wxw/lbs/SR/SSR/datasets/cave_train_data.npz'
    test_dataset_path = '/wxw/lbs/SR/SSR/datasets/cave_test_data.npz'

if args.model == 'SSRMamba':
    model = SSRMamba(HSI_bands,dim=args.dim)

os_id = os.getpid()
model = model.to(device)
set_seed(args.seed)
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lr=args.lr,params=model.parameters())
scheduler = StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
train_dataset = NPZDataset(train_dataset_path)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
test_dataloader = DataLoader(NPZDataset(test_dataset_path),batch_size=128,drop_last=False,shuffle=False)
start_epoch = 0 
if args.check_point is not None:
    checkpoint = torch.load(args.check_point)  
    model.load_state_dict(checkpoint['net'],strict=False)  
    optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch']+1 
    scheduler.load_state_dict(checkpoint['scheduler'])
    log_dir,_ = os.path.split(args.check_point)
    print(f'check_point: {args.check_point}')
    

if args.check_point is  None:
    init_weights(model)
    log_dir = f'./trained_models/{beijing_time()},{model_name},dataset:{args.dataset}'
    if not os.path.exists(log_dir) and args.log_out == 1:
        os.mkdir(log_dir)
        
logger = set_logger(model_name, log_dir, args.log_out)

logger.info("".center(39, '-').center(41, '+'))
for _,arg in enumerate(vars(args)):
    logger.info(f" {arg}: {getattr(args,arg)}".ljust(39, ' ').center(41, '|'))
logger.info("".center(39, '-').center(41, '+'))

def train():
    for epoch in range(start_epoch, args.epochs):
        loss_list = []
        start_time = time.time()
        for idx,loader_data in enumerate(train_dataloader):
            RGB,GT = loader_data[0].to(device),loader_data[1].to(device)
            pre_hsi = model(RGB)
            loss = loss_func(GT,pre_hsi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        print(f'Epoch: {epoch}, Loss: {np.mean(loss_list)}, Time: {time.time()-start_time:.2f}s')

        
if __name__ == "__main__":
    train()

