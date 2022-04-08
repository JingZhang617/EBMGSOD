import os
import time
import argparse


parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--task', type=str, default='RGBD-SOD', choices=['SOD','RGBD-SOD'])
parser.add_argument('--model', type=str, default='swin',
                    choices=['swin'])
parser.add_argument('--training_path', type=str, default='/home/jingzhang/jing_files/RGB_Dataset/train/DUTS/')
parser.add_argument('--log_info', type=str, default='REMOVE')
parser.add_argument('--ckpt', type=str, default='SOD')
args = parser.parse_args()

# Configs
param = {}
param['task'] = args.task

# Training Config
param['epoch'] = 100           # max epoch
param['seed'] = 1234          # random seeds
param['batch_size'] = 10       # batch size
param['save_epoch'] = 10       # snap shoots
param['lr'] = 2.5e-5          # learning rate
param['lr_dis'] = 1e-5          # learning rate
param['lr_ebm'] = 1e-4          # learning rate
param['trainsize'] = 384      # training image size
param['decay_rate'] = 0.9
param['decay_epoch'] = 80
param['beta'] = [0.5, 0.999]  # Adam related parameters
param['size_rates'] = [1]     # Multi-scale  [0.75, 1, 1.25]/[1]
param['use_pretrain'] = True
param['attention_decoder'] = True
## ABP related
param['latent_dim'] = 32
param['langevin_step_num_gen'] = 5
param['sigma_gen'] = 0.3
param['langevin_s'] = 0.1
## EBM related
param['ebm_out_dim'] = 1
param['ebm_middle_dim'] = 60
param['e_init_sig'] = 1.0
param['e_l_steps'] = 5
# param['e_l_steps'] = 80
param['e_l_step_size'] = 0.4
param['e_prior_sig'] = 1.0
param['g_l_steps'] = 5
# param['g_l_steps'] = 40
param['g_llhd_sigma'] = 0.3
param['g_l_step_size'] = 0.1
param['e_energy_form'] = 'identity'

param['langevin_step_num_gen'] = 5
param['sigma_gen'] = 0.3
param['langevin_s'] = 0.1
param['iter_num'] = 10

# Backbone Config
param['model_name'] = args.model   # [VGG, ResNet, DPT]
param['backbone_name'] = "vitb_rn50_384"   # vitl16_384


# Dataset Config
if param['task'] == 'SOD':
    param['image_root'] = args.training_path + '/img/'
    param['gt_root'] = args.training_path + '/gt/'
    param['test_dataset_root'] = '/home/jingzhang/jing_files/RGB_Dataset/test/img/'
elif param['task'] == 'RGBD-SOD':
    param['image_root'] = '/home/jingzhang/jing_files/RGBD_Dataset/train/old_train/RGB/'
    param['gt_root'] = '/home/jingzhang/jing_files/RGBD_Dataset/train/old_train/GT/'
    param['depth_root'] = '/home/jingzhang/jing_files/RGBD_Dataset/train/old_train/depth/'
    param['test_dataset_root'] = '/home/jingzhang/jing_files/RGBD_Dataset/test/'


# Experiment Dir Config
log_info = args.model + '_' + args.log_info    #
param['training_info'] = param['task'] + '_' + param['backbone_name'] + '_' + str(param['lr']) + '_' + log_info
param['log_path'] = 'experiments/{}'.format(param['training_info'])   #
param['ckpt_save_path'] = param['log_path'] + '/models/'              #
print('[INFO] Experiments saved in: ', param['training_info'])


# Test Config
param['testsize'] = 384
param['checkpoint'] = args.ckpt
param['eval_save_path'] = param['log_path'] + '/save_images/'         #
if param['task'] == 'SOD':
    param['datasets'] = ['DUTS_Test', 'ECSSD', 'DUT', 'HKU-IS', 'PASCAL', 'SOD']
elif param['task'] == 'RGBD-SOD':
    param['datasets'] = ['DES', 'LFSD', 'NJU2K', 'NLPR','SIP','STERE']
