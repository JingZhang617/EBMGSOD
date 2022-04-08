import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import time
import pickle
import os.path as osp
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import argparse
from data import eval_Dataset
import matplotlib.pyplot as plt


def Eval_mae(loader,cuda=True):
    #print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
    avg_mae, img_num, total = 0.0, 0.0, 0.0
    mae_list = []
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in loader:
            if cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)
            mae = torch.abs(pred - gt).mean()
            if mae == mae:  # for Nan
                avg_mae += mae
                img_num += 1.0
            mae_list.append(mae.item())
        avg_mae /= img_num
    return avg_mae, mae_list

with open('temp_dir/ours_results_DUTS', 'rb') as f:  
    mae_list_ours = pickle.loads(f.read())
with open('temp_dir/resnet_results_DUTS', 'rb') as f:  
    mae_list_resnet = pickle.loads(f.read())
plt.hist(mae_list_ours, edgecolor='k', alpha=0.2, label='Transformer')
plt.hist(mae_list_resnet, edgecolor='k', alpha=0.2, label='ResNet')
plt.legend()
plt.savefig('plot.png')
