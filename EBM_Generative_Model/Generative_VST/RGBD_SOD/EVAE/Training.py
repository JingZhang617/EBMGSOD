import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from dataset import get_loader
import math
from Models.ImageDepthNet import ImageDepthNet, EBM_Prior
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
import numpy as np

def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

def sample_p_0(n, sig, latent_dim):
    return sig * torch.randn(*[n, 1, latent_dim]).to(device)

def compute_energy(score, e_energy_form):
    if e_energy_form == 'tanh':
        energy = F.tanh(score.squeeze())
    elif e_energy_form == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif e_energy_form == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

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

def main(local_rank, num_gpus, args):

    cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = ImageDepthNet(args)
    net.train()
    net.cuda()

    ebm_model = EBM_Prior(args.ebm_out_dim, args.ebm_middle_dim, args.latent_dim)
    ebm_model.cuda()
    ebm_model_params = ebm_model.parameters()
    ebm_model_optimizer = torch.optim.Adam(ebm_model_params, args.lr_ebm)


    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(depths.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True)),  \
                                        Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                                                      Variable(contour_28.cuda()), \
                                                      Variable(contour_56.cuda()), Variable(contour_112.cuda())

            z_prior0 = sample_p_0(n=images.shape[0], sig=args.e_init_sig, latent_dim=args.latent_dim)
            z_post0 = sample_p_0(n=images.shape[0], sig=args.e_init_sig, latent_dim=args.latent_dim)
            z_e_0, z_g_0 = net(images, depths, z_prior0, z_post0, label_224, prior_z_flag=True, istraining=True)
            z_e_0 = torch.unsqueeze(z_e_0, 2)
            z_e_0 = z_e_0.permute(0,2,1)
            z_g_0 = torch.unsqueeze(z_g_0, 2)
            z_g_0 = z_g_0.permute(0, 2, 1)
            ## sample langevin prior of z
            z_e_0 = Variable(z_e_0)
            z = z_e_0.clone().detach()
            z.requires_grad = True
            for kk in range(args.e_l_steps):
                en = ebm_model(z)
                z_grad = torch.autograd.grad(en.sum(), z)[0]
                z.data = z.data - 0.5 * args.e_l_step_size * args.e_l_step_size * (
                        z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
                z.data += args.e_l_step_size * torch.randn_like(z).data
            z_e_noise = z.detach()  ## z_

            ## sample langevin post of z
            z_g_0 = Variable(z_g_0)
            z = z_g_0.clone().detach()
            z.requires_grad = True
            for kk in range(args.g_l_steps):
                _, _, _, gen_res, _ = net(images, depths, z_prior0, z, label_224, prior_z_flag=False, istraining=True)
                outputs_saliency, _ = gen_res
                mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
                g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse_loss(
                    torch.sigmoid(mask_1_1), label_224)
                z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                en = ebm_model(z)
                z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * args.g_l_step_size * args.g_l_step_size * (
                        z_grad_g + z_grad_e + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
                z.data += args.g_l_step_size * torch.randn_like(z).data

            z_g_noise = z.detach()  ## z+

            _, _, pred_prior, pred_post, latent_loss = net(images, depths, z_e_noise, z_g_noise, label_224,
                                                                 prior_z_flag=False,
                                                                 istraining=True)
            # reg_loss = l2_regularisation(net.enc_x) + \
            #            l2_regularisation(net.enc_xy) + l2_regularisation(net.prior_dec) + l2_regularisation(
            #     net.post_dec)
            # reg_loss = args.reg_weight * reg_loss
            anneal_reg = linear_annealing(0, 1, epoch, args.train_steps)
            loss_latent = args.lat_weight * anneal_reg * latent_loss

            outputs_saliency_prior, outputs_contour_prior = pred_prior


            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency_prior
            cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour_prior
            # loss
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)

            # contour loss
            c_loss5 = criterion(cont_1_16, contour_14)
            c_loss4 = criterion(cont_1_8, contour_28)
            c_loss3 = criterion(cont_1_4, contour_56)
            c_loss1 = criterion(cont_1_1, contour_224)

            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5
            contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + loss_weights[4] * c_loss5

            total_loss = img_total_loss + contour_total_loss

            gen_loss_gsnn = (1 - args.vae_loss_weight) * total_loss

            outputs_saliency_post, outputs_contour_post = pred_post
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency_post
            cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour_post
            # loss
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)

            # contour loss
            c_loss5 = criterion(cont_1_16, contour_14)
            c_loss4 = criterion(cont_1_8, contour_28)
            c_loss3 = criterion(cont_1_4, contour_56)
            c_loss1 = criterion(cont_1_1, contour_224)

            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[
                4] * loss5
            contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + \
                                 loss_weights[4] * c_loss5

            total_loss = img_total_loss + contour_total_loss
            gen_loss_cvae = args.vae_loss_weight * (total_loss + loss_latent)

            total_loss = gen_loss_gsnn + gen_loss_cvae


            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f}'.format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item()))

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()


            ## learn the ebm
            en_neg = compute_energy(ebm_model(
                z_e_noise.detach()), args.e_energy_form).mean()
            en_pos = compute_energy(ebm_model(z_g_noise.detach()), args.e_energy_form).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            ebm_model_optimizer.step()

            whole_iter_num += 1

            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(),
                           args.save_model_dir + 'RGBD_VST.pth')
                torch.save(ebm_model.state_dict(), args.save_model_dir + 'RGB_VST_ebm.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                ebm_model_optimizer = adjust_learning_rate(ebm_model_optimizer, decay_rate=args.lr_decay_gamma)

                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                save_lr(save_dir, ebm_model_optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)






