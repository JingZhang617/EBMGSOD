import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import datetime

from model.vgg_models import VGG_backbone, EBM_Prior
from data import get_loader
from utils import clip_gradient, adjust_lr, AvgMeter
from scipy import misc
from PIL import Image
import cv2
import torchvision.transforms.functional as tf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--lr_ebm', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
parser.add_argument('--modal_loss', type=float, default=0.5, help='weight of the fusion modal')
parser.add_argument('--focal_lamda', type=int, default=1, help='lamda of focal loss')
parser.add_argument('--bnn_steps', type=int, default=6, help='BNN sampling iterations')
parser.add_argument('--lvm_steps', type=int, default=6, help='LVM sampling iterations')
parser.add_argument('--pred_steps', type=int, default=6, help='Predictive sampling iterations')
parser.add_argument('--smooth_loss_weight', type=float, default=0.4, help='weight of the smooth loss')
parser.add_argument('--ebm_out_dim', type=int, default=1, help='ebm initial sigma')
parser.add_argument('--ebm_middle_dim', type=int, default=60, help='ebm initial sigma')
parser.add_argument('--latent_dim', type=int, default=32, help='ebm initial sigma')
parser.add_argument('--e_init_sig', type=float, default=1.0, help='ebm initial sigma')
parser.add_argument('--e_l_steps', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--e_l_step_size', type=float, default=0.4, help='ebm initial sigma')
parser.add_argument('--e_prior_sig', type=float, default=1.0, help='ebm initial sigma')
parser.add_argument('--g_l_steps', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='ebm initial sigma')
parser.add_argument('--g_l_step_size', type=float, default=0.1, help='ebm initial sigma')
parser.add_argument('--e_energy_form', type=str, default='identity', help='ebm initial sigma')

parser.add_argument('--langevin_step_num_gen', type=int, default=5, help='ebm initial sigma')
parser.add_argument('--sigma_gen', type=float, default=0.3, help='ebm initial sigma')
parser.add_argument('--langevin_s', type=float, default=0.1, help='ebm initial sigma')

opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr_gen))
# build models
model = VGG_backbone(channel=opt.feat_channel, latent_dim = opt.latent_dim)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr_gen)

ebm_model = EBM_Prior(opt.ebm_out_dim, opt.ebm_middle_dim, opt.latent_dim)
ebm_model.cuda()
ebm_model_params = ebm_model.parameters()
ebm_model_optimizer = torch.optim.Adam(ebm_model_params, opt.lr_ebm)

print("Model based on {} have {:.4f}Mb paramerters in total".format('Generator', sum(x.numel()/1e6 for x in model.parameters())))
print("EBM based on {} have {:.4f}Mb paramerters in total".format('EBM', sum(x.numel()/1e6 for x in ebm_model.parameters())))


image_root = '/home/jingzhang/jing_files/RGB_Dataset/train/DUTS/img/'
gt_root = '/home/jingzhang/jing_files/RGB_Dataset/train/DUTS/gt/'

train_loader, training_set_size = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
train_z = torch.FloatTensor(training_set_size, opt.latent_dim).normal_(0, 1).cuda()

CE = torch.nn.BCEWithLogitsLoss()
size_rates = [1]  # multi-scale training
mse_loss = torch.nn.MSELoss(size_average=True, reduction='sum')

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def compute_entropy_loss(pred):
    pred = torch.sigmoid(pred)
    entropy_cur = -pred.mul(torch.log(pred + 1e-7))
    mean_entropy = torch.mean(entropy_cur)
    return mean_entropy

def visualize_fused(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fused.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_init_rgb(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_init_rgb.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_init_depth(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_init_depth.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_pred_gt(pred,gt):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred2.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

    for kk in range(gt.shape[0]):
        pred_edge_kk = gt[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_ref_depth(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_ref_depth.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_pred(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_pred.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def sample_p_0(n=opt.batchsize, sig=opt.e_init_sig):
    return sig * torch.randn(*[n, opt.latent_dim, 1, 1]).to(device)

def compute_energy(score):
    if opt.e_energy_form == 'tanh':
        energy = F.tanh(score.squeeze())
    elif opt.e_energy_form == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif opt.e_energy_form == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy

for epoch in range(1, opt.epoch+1):
    # scheduler.step()
    model.train()
    ebm_model.train()
    loss_record, loss_record_ebm = AvgMeter(), AvgMeter()
    print('Generator Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
    print('EBM Learning Rate: {}'.format(ebm_model_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            ebm_model_optimizer.zero_grad()
            images, gts, index_batch = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            z_e_0 = sample_p_0(n=images.shape[0])
            z_g_0 = sample_p_0(n=images.shape[0])
            ## sample langevin prior of z
            z_e_0 = Variable(z_e_0)
            z = z_e_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.e_l_steps):
                en = ebm_model(z)
                z_grad = torch.autograd.grad(en.sum(), z)[0]
                z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                        z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.e_l_step_size * torch.randn_like(z).data
            z_e_noise = z.detach()  ## z_

            ## sample langevin post of z
            z_g_0 = Variable(z_g_0)
            z = z_g_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.g_l_steps):
                gen_res = model(images, z)
                g_log_lkhd = 1.0 / (2.0 * opt.g_llhd_sigma * opt.g_llhd_sigma) * mse_loss(
                    torch.sigmoid(gen_res), gts)
                z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                en = ebm_model(z)
                z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * opt.g_l_step_size * opt.g_l_step_size * (
                        z_grad_g + z_grad_e + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.g_l_step_size * torch.randn_like(z).data

            z_g_noise = z.detach()  ## z+

            pred = model(images, z_g_noise)
            loss = structure_loss(pred, gts)
            loss.backward()
            optimizer.step()

            ## learn the ebm
            en_neg = compute_energy(ebm_model(
                z_e_noise.detach())).mean()
            en_pos = compute_energy(ebm_model(z_g_noise.detach())).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            ebm_model_optimizer.step()

            visualize_pred(torch.sigmoid(pred))
            visualize_gt(gts)

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                loss_record_ebm.update(loss_e.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss_EBM: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss_record_ebm.show()))

    adjust_lr(optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(ebm_model_optimizer, opt.lr_ebm, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(ebm_model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_ebm.pth')
