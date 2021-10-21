import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from data import get_loader, get_loader_rgbd, get_loader_weak
from img_trans import scale_trans
from config import param as option
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import AvgMeter, set_seed, visualize_all,adjust_lr
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from img_trans import rot_trans, scale_trans
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mse_loss = torch.nn.MSELoss(reduction='sum')
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)


def energy(score):
    if option['e_energy_form'] == 'tanh':
        energy = F.tanh(score.squeeze())
    elif option['e_energy_form'] == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif option['e_energy_form'] == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy

def get_optim(option, params):
    optimizer = torch.optim.Adam(params, option['lr'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler

def get_optim_dis(option, params):
    optimizer = torch.optim.Adam(params, option['lr_dis'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler

def sample_p_0(n=option['batch_size'], sig=option['e_init_sig']):
    return sig * torch.randn(*[n, option['latent_dim'], 1, 1]).to(device)

def train_one_epoch(model, generator_optimizer, ebm_model, ebm_optimizer, train_loader, loss_fun):
    model.train()
    ebm_model.train()
    loss_record, ebm_loss_record = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    print('EBM Learning Rate: {:.2e}'.format(ebm_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            ebm_optimizer.zero_grad()
            task_name = option['task']
            if task_name == 'RGBD-SOD':
                images, gts, depths, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
                z_g_0 = sample_p_0(n=images.shape[0])
                z_e_0 = sample_p_0(n=images.shape[0])

                ## sample langevin prior of z
                z_e_0 = Variable(z_e_0)
                z = z_e_0.clone().detach()
                z.requires_grad = True
                for kk in range(option['e_l_steps']):
                    en = ebm_model(z)
                    z_grad = torch.autograd.grad(en.sum(), z)[0]
                    z.data = z.data - 0.5 * option['e_l_step_size'] * option['e_l_step_size'] * (
                                z_grad + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z.data)
                    z.data += option['e_l_step_size'] * torch.randn_like(z).data

                z_e_noise = z.detach() ## z_

                ## sample langevin post of z
                z_g_0 = Variable(z_g_0)
                z = z_g_0.clone().detach()
                z.requires_grad = True
                for i in range(option['g_l_steps']):
                    gen_res = model(images, z, depths)
                    g_log_lkhd = 1.0 / (2.0 * option['g_llhd_sigma'] * option['g_llhd_sigma']) * mse_loss(
                        torch.sigmoid(gen_res[0]), gts)
                    z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                    en = ebm_model(z)
                    z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                    z.data = z.data - 0.5 * option['g_l_step_size'] * option['g_l_step_size'] * (
                                z_grad_g + z_grad_e + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z.data)
                    z.data += option['g_l_step_size'] * torch.randn_like(z).data

                z_g_noise = z.detach() ## z+

                ## learn generator
                ref_pre = model(images, z_g_noise, depths)
                loss = cal_loss(ref_pre, gts,
                                 loss_fun)
                loss.backward()
                generator_optimizer.step()

                ## learn the ebm
                en_neg = energy(ebm_model(
                    z_e_noise.detach())).mean()
                en_pos = energy(ebm_model(z_g_noise.detach())).mean()
                loss_e = en_pos - en_neg
                loss_e.backward()
                ebm_optimizer.step()

                visualize_all(torch.sigmoid(ref_pre[0]), gts, option['log_path'])

                if rate == 1:
                    loss_record.update(loss.data, option['batch_size'])
                    ebm_loss_record.update(loss_e.data, option['batch_size'])
            else:
                images, gts, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda()
                z_g_0 = sample_p_0(n=images.shape[0])
                z_e_0 = sample_p_0(n=images.shape[0])

                ## sample langevin prior of z
                z_e_0 = Variable(z_e_0)
                z = z_e_0.clone().detach()
                z.requires_grad = True
                for kk in range(option['e_l_steps']):
                    en = ebm_model(z)
                    z_grad = torch.autograd.grad(en.sum(), z)[0]
                    z.data = z.data - 0.5 * option['e_l_step_size'] * option['e_l_step_size'] * (
                            z_grad + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z.data)
                    z.data += option['e_l_step_size'] * torch.randn_like(z).data

                z_e_noise = z.detach()  ## z_

                ## sample langevin post of z
                z_g_0 = Variable(z_g_0)
                z = z_g_0.clone().detach()
                z.requires_grad = True
                for i in range(option['g_l_steps']):
                    gen_res = model(images, z)
                    g_log_lkhd = 1.0 / (2.0 * option['g_llhd_sigma'] * option['g_llhd_sigma']) * mse_loss(
                        torch.sigmoid(gen_res[0]), gts)
                    z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                    en = ebm_model(z)
                    z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                    z.data = z.data - 0.5 * option['g_l_step_size'] * option['g_l_step_size'] * (
                            z_grad_g + z_grad_e + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z.data)
                    z.data += option['g_l_step_size'] * torch.randn_like(z).data

                z_g_noise = z.detach()  ## z+

                ## learn generator
                ref_pre = model(images, z_g_noise)
                loss = cal_loss(ref_pre, gts,
                                loss_fun)
                loss.backward()
                generator_optimizer.step()

                ## learn the ebm
                en_neg = energy(ebm_model(
                    z_e_noise.detach())).mean()
                en_pos = energy(ebm_model(z_g_noise.detach())).mean()
                loss_e = en_pos - en_neg
                loss_e.backward()
                ebm_optimizer.step()

                visualize_all(torch.sigmoid(ref_pre[0]), gts, option['log_path'])

                if rate == 1:
                    loss_record.update(loss.data, option['batch_size'])
                    ebm_loss_record.update(loss_e.data, option['batch_size'])



        progress_bar.set_postfix(gen_loss=f'{loss_record.show():.5f}',ebm_loss=f'{ebm_loss_record.show():.5f}')

    adjust_lr(generator_optimizer, option['lr'], epoch, option['decay_rate'], option['decay_epoch'])
    adjust_lr(ebm_optimizer, option['lr_dis'], epoch, option['decay_rate'], option['decay_epoch'])

    return model, loss_record, ebm_model, ebm_loss_record


if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    model, ebm_model = get_model(option)
    optimizer, scheduler = get_optim(option, model.parameters())
    optimizer_ebm, scheduler_ebm = get_optim(option, ebm_model.parameters())
    task_name = option['task']
    if task_name == 'RGBD-SOD':
        train_loader, training_set_size = get_loader_rgbd(image_root=option['image_root'], gt_root=option['gt_root'], depth_root=option['depth_root'],
                                  batchsize=option['batch_size'], trainsize=option['trainsize'])
    else:
        train_loader, training_set_size = get_loader(image_root=option['image_root'], gt_root=option['gt_root'],
                                  batchsize=option['batch_size'], trainsize=option['trainsize'])

    train_z = torch.FloatTensor(training_set_size, option['latent_dim']).normal_(0, 1).cuda()
    size_rates = option['size_rates']  # multi-scale training
    writer = SummaryWriter(option['log_path'])
    for epoch in range(1, (option['epoch']+1)):
        model, loss_record, ebm_model, ebm_loss_record = train_one_epoch(model, optimizer, ebm_model, optimizer_ebm, train_loader, loss_fun)
        writer.add_scalar('gen loss', loss_record.show(), epoch)
        writer.add_scalar('ebm loss', ebm_loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('lr', optimizer_ebm.param_groups[0]['lr'], epoch)
        scheduler.step()
        scheduler_ebm.step()

        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(model.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, loss_record.show()) + '_gen.pth')
            torch.save(ebm_model.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, ebm_loss_record.show()) + '_ebm.pth')
