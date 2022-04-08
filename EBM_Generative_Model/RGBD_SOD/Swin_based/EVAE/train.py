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
from utils import AvgMeter, set_seed, visualize_all3,adjust_lr, l2_regularisation
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from img_trans import rot_trans, scale_trans
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mse_loss = torch.nn.MSELoss(reduction='sum')
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
import numpy as np
CE = torch.nn.BCEWithLogitsLoss()

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

def get_optim_ebm(option, params):
    optimizer = torch.optim.Adam(params, option['lr_ebm'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler

def sample_p_0(n=option['batch_size'], sig=option['e_init_sig']):
    return sig * torch.randn(*[n, option['latent_dim'], 1, 1]).to(device)

def make_Dis_label(label,gts):
    D_label = np.ones(gts.shape)*label
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label

# labels for adversarial training
pred_label = 0
gt_label = 1
epcilon_ls = 1e-4

# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed

def train_one_epoch(model, generator_optimizer, ebm_model, ebm_optimizer, train_loader, loss_fun):
    model.train()
    # discriminator.train()
    ebm_model.train()
    loss_record, loss_record_ebm = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    # print('Dis Learning Rate: {:.2e}'.format(discriminator_optimizer.param_groups[0]['lr']))
    print('EBM Learning Rate: {:.2e}'.format(ebm_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            # discriminator_optimizer.zero_grad()
            ebm_optimizer.zero_grad()
            task_name = option['task']
            if task_name == 'RGBD-SOD':
                images, gts, depths, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()
                ## learn generator
                z_prior0 = sample_p_0(n=images.shape[0])
                z_post0 = sample_p_0(n=images.shape[0])

                z_e_0, z_g_0 = model(images, z_prior0, z_post0, depths, gts, prior_z_flag=True, istraining=True)
                z_e_0 = torch.unsqueeze(z_e_0, 2)
                z_e_0 = torch.unsqueeze(z_e_0, 3)

                z_g_0 = torch.unsqueeze(z_g_0, 2)
                z_g_0 = torch.unsqueeze(z_g_0, 3)

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
                    _, _, _, gen_res, _ = model(images, z_prior0, z, depths, gts, prior_z_flag=False, istraining=True)
                    g_log_lkhd = 1.0 / (2.0 * option['g_llhd_sigma'] * option['g_llhd_sigma']) * mse_loss(
                        torch.sigmoid(gen_res[0]), gts)
                    z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                    en = ebm_model(z)
                    z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                    z.data = z.data - 0.5 * option['g_l_step_size'] * option['g_l_step_size'] * (
                            z_grad_g + z_grad_e + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z.data)
                    z.data += option['g_l_step_size'] * torch.randn_like(z).data

                z_g_noise = z.detach()  ## z+

                _, _, pred_prior, pred_post, latent_loss = model(images, z_e_noise, z_g_noise, depths, gts, prior_z_flag=False,
                                                                 istraining=True)
                reg_loss = l2_regularisation(model.enc_x) + \
                           l2_regularisation(model.enc_xy) + l2_regularisation(model.prior_dec) + l2_regularisation(
                    model.post_dec)
                reg_loss = option['reg_weight'] * reg_loss
                anneal_reg = linear_annealing(0, 1, epoch, option['epoch'])
                loss_latent = option['lat_weight'] * anneal_reg * latent_loss
                gen_loss_cvae = option['vae_loss_weight'] * (cal_loss(pred_post, gts, loss_fun) + loss_latent)
                gen_loss_gsnn = (1 - option['vae_loss_weight']) * cal_loss(pred_prior, gts, loss_fun)
                loss_all = gen_loss_cvae + gen_loss_gsnn + reg_loss

                loss_all.backward()
                generator_optimizer.step()

                # # ## train discriminator
                # dis_pred = torch.sigmoid(pred).detach()
                # Dis_output = discriminator(images, dis_pred)
                # Dis_output = F.upsample(Dis_output, size=(images.shape[2], images.shape[3]), mode='bilinear',
                #                         align_corners=True)
                # Dis_target = discriminator(images, gts)
                # Dis_target = F.upsample(Dis_target, size=(images.shape[2], images.shape[3]), mode='bilinear',
                #                         align_corners=True)
                #
                # loss_dis_output = CE(Dis_output, make_Dis_label(pred_label, gts))
                # loss_dis_target = CE(Dis_target, make_Dis_label(gt_label, gts))
                # dis_loss = 0.5 * (loss_dis_output + loss_dis_target)
                # dis_loss.backward()
                # discriminator_optimizer.step()

                ## learn the ebm
                en_neg = energy(ebm_model(
                    z_e_noise.detach())).mean()
                en_pos = energy(ebm_model(z_g_noise.detach())).mean()
                loss_e = en_pos - en_neg
                loss_e.backward()
                ebm_optimizer.step()

                visualize_all3(torch.sigmoid(pred_prior), torch.sigmoid(pred_post), gts, option['log_path'])

                if rate == 1:
                    loss_record.update(loss_all.data, option['batch_size'])
                    # loss_record_dis.update(dis_loss.data, option['batch_size'])
                    loss_record_ebm.update(loss_e.data, option['batch_size'])
            else:
                images, gts, index_batch = pack[0].cuda(), pack[1].cuda(), pack[2].cuda()

                z_prior0 = sample_p_0(n=images.shape[0])
                z_post0 = sample_p_0(n=images.shape[0])

                z_e_0, z_g_0 = model(images, z_prior0, z_post0, gts, prior_z_flag=True, istraining=True)
                z_e_0 = torch.unsqueeze(z_e_0, 2)
                z_e_0 = torch.unsqueeze(z_e_0, 3)

                z_g_0 = torch.unsqueeze(z_g_0, 2)
                z_g_0 = torch.unsqueeze(z_g_0, 3)

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
                    _, _, _, gen_res, _ = model(images, z_prior0, z, gts, prior_z_flag=False, istraining=True)
                    g_log_lkhd = 1.0 / (2.0 * option['g_llhd_sigma'] * option['g_llhd_sigma']) * mse_loss(
                        torch.sigmoid(gen_res[0]), gts)
                    z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                    en = ebm_model(z)
                    z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                    z.data = z.data - 0.5 * option['g_l_step_size'] * option['g_l_step_size'] * (
                            z_grad_g + z_grad_e + 1.0 / (option['e_prior_sig'] * option['e_prior_sig']) * z.data)
                    z.data += option['g_l_step_size'] * torch.randn_like(z).data

                z_g_noise = z.detach()  ## z+

                _, _, pred_prior, pred_post, latent_loss = model(images, z_e_noise, z_g_noise, gts, prior_z_flag=False,
                                                                 istraining=True)
                reg_loss = l2_regularisation(model.enc_x) + \
                           l2_regularisation(model.enc_xy) + l2_regularisation(model.prior_dec) + l2_regularisation(
                    model.post_dec)
                reg_loss = option['reg_weight'] * reg_loss
                anneal_reg = linear_annealing(0, 1, epoch, option['epoch'])
                loss_latent = option['lat_weight'] * anneal_reg * latent_loss
                gen_loss_cvae = option['vae_loss_weight'] * (cal_loss(pred_post, gts,loss_fun) + loss_latent)
                gen_loss_gsnn = (1 - option['vae_loss_weight']) * cal_loss(pred_prior, gts,loss_fun)
                loss_all = gen_loss_cvae + gen_loss_gsnn + reg_loss

                loss_all.backward()
                generator_optimizer.step()

                # # ## train discriminator
                # dis_pred = torch.sigmoid(pred).detach()
                # Dis_output = discriminator(images, dis_pred)
                # Dis_output = F.upsample(Dis_output, size=(images.shape[2], images.shape[3]), mode='bilinear',
                #                         align_corners=True)
                # Dis_target = discriminator(images, gts)
                # Dis_target = F.upsample(Dis_target, size=(images.shape[2], images.shape[3]), mode='bilinear',
                #                         align_corners=True)
                #
                # loss_dis_output = CE(Dis_output, make_Dis_label(pred_label, gts))
                # loss_dis_target = CE(Dis_target, make_Dis_label(gt_label, gts))
                # dis_loss = 0.5 * (loss_dis_output + loss_dis_target)
                # dis_loss.backward()
                # discriminator_optimizer.step()

                ## learn the ebm
                en_neg = energy(ebm_model(
                    z_e_noise.detach())).mean()
                en_pos = energy(ebm_model(z_g_noise.detach())).mean()
                loss_e = en_pos - en_neg
                loss_e.backward()
                ebm_optimizer.step()


                visualize_all3(torch.sigmoid(pred_prior),torch.sigmoid(pred_post), gts, option['log_path'])

                if rate == 1:
                    loss_record.update(loss_all.data, option['batch_size'])
                    # loss_record_dis.update(dis_loss.data, option['batch_size'])
                    loss_record_ebm.update(loss_e.data, option['batch_size'])

        progress_bar.set_postfix(gen_loss=f'{loss_record.show():.5f}',ebm_loss=f'{loss_record_ebm.show():.5f}')

    # adjust_lr(generator_optimizer, option['lr'], epoch, option['decay_rate'], option['decay_epoch'])
    # adjust_lr(discriminator_optimizer, option['lr'], epoch, option['decay_rate'], option['decay_epoch'])
    # adjust_lr(ebm_optimizer, option['lr'], epoch, option['decay_rate'], option['decay_epoch'])

    return model,ebm_model, loss_record, loss_record_ebm


if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    model, ebm_model = get_model(option)
    optimizer,scheduler = get_optim(option, model.parameters())
    # optimizer_dis,scheduler_dis = get_optim_dis(option, discriminator.parameters())
    optimizer_ebm,scheduler_ebm = get_optim_ebm(option, ebm_model.parameters())
    task_name = option['task']
    if task_name == 'RGBD-SOD':
        train_loader, training_set_size = get_loader_rgbd(image_root=option['image_root'], gt_root=option['gt_root'], depth_root=option['depth_root'],
                                  batchsize=option['batch_size'], trainsize=option['trainsize'])
    else:
        train_loader, training_set_size = get_loader(image_root=option['image_root'], gt_root=option['gt_root'],
                                  batchsize=option['batch_size'], trainsize=option['trainsize'])

    size_rates = option['size_rates']  # multi-scale training
    writer = SummaryWriter(option['log_path'])
    for epoch in range(1, (option['epoch']+1)):
        model, ebm_model, loss_record, loss_record_ebm = train_one_epoch(model, optimizer, ebm_model, optimizer_ebm, train_loader, loss_fun)
        writer.add_scalar('gen loss', loss_record.show(), epoch)
        # writer.add_scalar('dis loss', loss_record_dis.show(), epoch)
        writer.add_scalar('ebm loss', loss_record_ebm.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        # writer.add_scalar('lr', optimizer_dis.param_groups[0]['lr'], epoch)
        writer.add_scalar('lr', optimizer_ebm.param_groups[0]['lr'], epoch)

        scheduler.step()
        # scheduler_dis.step()
        scheduler_ebm.step()

        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(model.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, loss_record.show()) + '_gen.pth')
            # torch.save(discriminator.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, loss_record_dis.show()) + '_dis.pth')
            torch.save(ebm_model.state_dict(),
                       save_path + '/{:d}_{:.4f}'.format(epoch, loss_record_ebm.show()) + '_ebm.pth')
