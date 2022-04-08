import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder
import torch.nn.utils.spectral_norm as sn
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable

class Encode_x(nn.Module):
    def __init__(self, input_channels, latent_size, channels=32):
        super(Encode_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 7 * 7, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels * 8 * 7 * 7, latent_size)  # adjust according to input size

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        # print(output.size())
        output = output.view(-1, self.channel * 8 * 7 * 7)  # adjust according to input size
        # print(output.size())
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist

class Encode_xy(nn.Module):
    def __init__(self, input_channels, latent_size, channels=32):
        super(Encode_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 7 * 7, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels * 8 * 7 * 7, latent_size)  # adjust according to input size

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        # print(output.size())
        output = output.view(-1, self.channel * 8 * 7 * 7)  # adjust according to input size
        # print(output.size())
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist

class FCDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super(FCDiscriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(7, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(ndf)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x, pred):
        x = torch.cat((x, pred), 1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class EBM_Prior(nn.Module):
    def __init__(self, ebm_out_dim, ebm_middle_dim, latent_dim):
        super().__init__()

        e_sn = False

        apply_sn = sn if e_sn else lambda x: x

        f = nn.GELU()

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(latent_dim, ebm_middle_dim)),
            f,

            apply_sn(nn.Linear(ebm_middle_dim, ebm_middle_dim)),
            f,

            apply_sn(nn.Linear(ebm_middle_dim, ebm_out_dim))
        )
        self.ebm_out_dim = ebm_out_dim

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, self.ebm_out_dim, 1, 1)



class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.depth_backbone = T2t_vit_t_14(pretrained=True, args=args)

        self.prior_dec = Decoder_SOD(args)
        self.post_dec = Decoder_SOD(args)

        self.enc_x = Encode_x(6, args.latent_dim)
        self.enc_xy = Encode_xy(7, args.latent_dim)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, image_Input, depth_Input, z_prior0, z_post0, y=None, prior_z_flag=True, istraining = True):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        depth_fea_1_16, _, _ = self.depth_backbone(depth_Input)

        mu_prior, logvar_prior, dist_prior = self.enc_x(torch.cat((image_Input, depth_Input), 1))
        z_prior = self.reparametrize(mu_prior, logvar_prior)

        if istraining == False:
            if prior_z_flag == True:
                return z_prior
            else:
                pred = self.prior_dec(rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, depth_fea_1_16, z_prior0)
                return pred
        else:
            mu_post, logvar_post, dist_post = self.enc_xy(torch.cat((image_Input, depth_Input, y), 1))
            kld = torch.mean(self.kl_divergence(dist_post, dist_prior))
            if prior_z_flag == True:
                z_post = self.reparametrize(mu_post, logvar_post)
                return z_prior, z_post
            else:
                pred_prior = self.prior_dec(rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, depth_fea_1_16, z_prior0)
                pred_post = self.post_dec(rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, depth_fea_1_16, z_post0)
                return z_prior0, z_post0, pred_prior, pred_post, kld


class Decoder_SOD(nn.Module):
    def __init__(self, args):
        super(Decoder_SOD, self).__init__()


        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)
        self.noise_mlp = nn.Sequential(
            nn.Linear(384 + args.latent_dim, 384),
            nn.GELU(),
            nn.Linear(384, 384),
        )

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, depth_fea_1_16, z_noise):



        z_noise = self.tile(z_noise, 1, rgb_fea_1_16.shape[1])
        rgb_fea_1_16 = torch.cat((rgb_fea_1_16, z_noise), 2)
        rgb_fea_1_16 = self.noise_mlp(rgb_fea_1_16)

        # VST Convertor
        rgb_fea_1_16, depth_fea_1_16 = self.transformer(rgb_fea_1_16, depth_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]   depth_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16,
                                                                                                          depth_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens,
                               rgb_fea_1_8, rgb_fea_1_4)

        return outputs
