import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from scipy import misc
from torch.autograd import Variable
from model.vgg_models import VGG_backbone, EBM_Prior
from data import test_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import time


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
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
parser.add_argument('--batchsize', type=int, default=5, help='training batch size')
parser.add_argument('--iter_num', type=int, default=7, help='training batch size')

opt = parser.parse_args()

# dataset_path = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/img/'
dataset_path = '/home/jingzhang/jing_files/RGB_Dataset/test/img/'
# gt_path = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/gt/'

model = VGG_backbone(channel=opt.feat_channel, latent_dim = opt.latent_dim)
model.load_state_dict(torch.load('./models/VGG/Model_50_gen.pth'))
model.cuda()
model.eval()

discriminator = EBM_Prior(opt.ebm_out_dim, opt.ebm_middle_dim, opt.latent_dim)
discriminator.load_state_dict(torch.load('./models/VGG/Model_50_ebm.pth'))

discriminator.cuda()
discriminator.eval()

test_datasets = ['DUTS_Test', 'ECSSD','DUT', 'HKU-IS', 'PASCAL', 'SOD']
#test_datasets =  ['MSRABTest']

def sample_p_0(n=opt.batchsize, sig=opt.e_init_sig):
    return sig * torch.randn(*[n, opt.latent_dim, 1, 1]).to(device)

time_list = []

for dataset in test_datasets:
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
    # save_path_base = './results/' + dataset + '/'
    # # save_path = './results/ResNet50/holo/train/left/'
    # if not os.path.exists(save_path_base):
    #     os.makedirs(save_path_base)

    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()

        mean_pred = 0
        alea_uncertainty = 0

        torch.cuda.synchronize()
        start = time.time()
        for iter in range(opt.iter_num):
            z_e_0 = sample_p_0(n=image.shape[0])
            ## sample langevin prior of z
            z_e_0 = Variable(z_e_0)
            z = z_e_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.e_l_steps):
                en = discriminator(z)
                z_grad = torch.autograd.grad(en.sum(), z)[0]
                z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                        z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.e_l_step_size * torch.randn_like(z).data
                # z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

            z_e_noise = z.detach()  ## z_
            generator_pred = model(image, z_e_noise)
            res = generator_pred
            mean_pred = mean_pred + torch.sigmoid(res)
            preds = torch.sigmoid(res)
            cur_alea = -1 * preds * torch.log(preds + 1e-8)
            # cur_alea = compute_entropy(preds)
            alea_uncertainty = alea_uncertainty + cur_alea

        mean_prediction = mean_pred / opt.iter_num
        alea_uncertainty = alea_uncertainty / opt.iter_num
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

        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
    print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))
