import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet, EBM_Prior
from torch.utils import data
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2

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

def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir + 'RGBD_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)

    print('Model loaded from {}'.format(model_path))

    ebm_model = EBM_Prior(args.ebm_out_dim, args.ebm_middle_dim, args.latent_dim)
    ebm_model.load_state_dict(torch.load(args.save_model_dir + 'RGB_VST_ebm.pth'))
    ebm_model.cuda()
    ebm_model.eval()

    # load model
    # net.load_state_dict(torch.load(args.test_model_dir))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(args.test_model_dir))

    test_paths = args.test_paths.split('+')
    for test_dir_img in test_paths:

        dataset = test_dir_img.split('/')[0]

        save_path_mean = './results_mean/' + dataset + '/'
        if not os.path.exists(save_path_mean):
            os.makedirs(save_path_mean)

        save_path_aleatoric = './results_aleatoric/' + dataset + '/'
        if not os.path.exists(save_path_aleatoric):
            os.makedirs(save_path_aleatoric)

        save_path_epistemic = './results_epistemic/' + dataset + '/'
        if not os.path.exists(save_path_epistemic):
            os.makedirs(save_path_epistemic)

        save_path_predictive = './results_predictive/' + dataset + '/'
        if not os.path.exists(save_path_predictive):
            os.makedirs(save_path_predictive)

        test_dataset = get_loader(test_dir_img, args.test_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

        time_list = []
        for i, data_batch in enumerate(test_loader):
            print(i)
            mean_pred = 0
            alea_uncertainty = 0
            images, depths, image_w, image_h, image_path = data_batch
            images, depths = Variable(images.cuda()), Variable(depths.cuda())

            starts = time.time()

            for iter in range(args.iter_num):
                z_e_0 = sample_p_0(n=images.shape[0], sig=args.e_init_sig, latent_dim=args.latent_dim)
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
                    # z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

                z_e_noise = z.detach()  ## z_
                outputs_saliency, outputs_contour = net(images, depths, z_e_noise)
                mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
                res = mask_1_1
                mean_pred = mean_pred + torch.sigmoid(res)
                preds = torch.sigmoid(res)
                cur_alea = -1 * preds * torch.log(preds + 1e-8)
                # cur_alea = compute_entropy(preds)
                alea_uncertainty = alea_uncertainty + cur_alea

            # outputs_saliency, outputs_contour = net(images)
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)

            HH, WW = int(image_w[0]), int(image_h[0])
            filename = image_path[0].split('/')[-1].split('.')[0]
            name = filename + '.png'

            mean_prediction = mean_pred / args.iter_num
            alea_uncertainty = alea_uncertainty / args.iter_num
            predictive_uncertainty = -1 * mean_prediction * torch.log(mean_prediction + 1e-8)
            # predictive_uncertainty = compute_entropy(mean_prediction)
            epistemic_uncertainty = predictive_uncertainty - alea_uncertainty

            res = F.upsample(mean_prediction, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path_mean + name, res)

            res = F.upsample(alea_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res.astype(np.uint8)
            res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
            cv2.imwrite(save_path_aleatoric + name, res)

            res = F.upsample(epistemic_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res.astype(np.uint8)
            res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
            cv2.imwrite(save_path_epistemic + name, res)

            res = F.upsample(predictive_uncertainty, size=[WW, HH], mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res.astype(np.uint8)
            res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
            cv2.imwrite(save_path_predictive + name, res)

            # outputs_saliency, outputs_contour = net(images, depths)
            # ends = time.time()
            # time_use = ends - starts
            # time_list.append(time_use)
            #
            # mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            #
            # image_w, image_h = int(image_w[0]), int(image_h[0])
            #
            # output_s = F.sigmoid(mask_1_1)
            #
            # output_s = output_s.data.cpu().squeeze(0)
            #
            # transform = trans.Compose([
            #     transforms.ToPILImage(),
            #     trans.Scale((image_w, image_h))
            # ])
            # output_s = transform(output_s)
            #
            # dataset = test_dir_img.split('/')[0]
            # filename = image_path[0].split('/')[-1].split('.')[0]
            #
            # # save saliency maps
            # save_test_path = args.save_test_path_root + dataset + '/'
            # if not os.path.exists(save_test_path):
            #     os.makedirs(save_test_path)
            # output_s.save(os.path.join(save_test_path, filename + '.png'))

        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list)*1000))






