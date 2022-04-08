import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import Training
import Testing
from Evaluation import main
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33116', type=str, help='init_method')
    parser.add_argument('--data_root', default='/home/jingzhang/jing_files/TPAMI_Jing_Nips/Applying_to_existing_sod/VST-main/RGBD_VST/Data/', type=str, help='data path')
    parser.add_argument('--test_root', default='/home/jingzhang/jing_files/RGBD_Dataset/test/', type=str, help='data path')

    parser.add_argument('--train_steps', default=40000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='/home/jingzhang/jing_files/TPAMI_Jing_Nips/Applying_to_existing_sod/VST-main/RGBD_VST/pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=2.5e-5, type=int, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--batch_size', default=6, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=20000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=30000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='NJUD+NLPR+DUTLF-Depth', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='NJU2K+NLPR+LFSD+STERE+SIP+DES')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGBD_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')
    parser.add_argument('--iter_num', default=5, type=int, help='the step 1 for adjusting lr')

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
    parser.add_argument('--lr_ebm', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--lr_dis', type=float, default=1e-5, help='learning rate for generator')

    parser.add_argument('--lat_weight', type=float, default=5.0, help='weight for latent loss')
    parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')


    args = parser.parse_args()



    num_gpus = torch.cuda.device_count()
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        main.evaluate(args)