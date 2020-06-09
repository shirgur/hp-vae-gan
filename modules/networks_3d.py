from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import utils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(act):
    activations = {
        "relu": nn.ReLU(inplace=True),
        "lrelu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "prelu": nn.PReLU(num_parameters=1, init=0.25),
        "selu": nn.SELU(inplace=True)
    }
    return activations[act]


def reparameterize(mu, logvar, training):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return torch.zeros_like(mu).normal_()


def reparameterize_bern(x, training):
    if training:
        eps = torch.zeros_like(x).uniform_()
        return torch.log(x + 1e-20) - torch.log(-torch.log(eps + 1e-20) + 1e-20)
    else:
        return torch.zeros_like(x).bernoulli_()


# Basic blocks

class ConvBlock3D(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3D, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                          stride=stride, padding=padding))
        if bn:
            self.add_module('norm', nn.BatchNorm3d(out_channel))
        if act is not None:
            self.add_module(act, get_activation(act))


class ConvBlock3DSN(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, bn=True, act='lrelu'):
        super(ConvBlock3DSN, self).__init__()
        if bn:
            self.add_module('conv', nn.utils.spectral_norm(nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                                                     stride=stride, padding=padding)))
        else:
            self.add_module('conv',
                            nn.Conv3d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padding,
                                      padding_mode='reflect'))
        if act is not None:
            self.add_module(act, get_activation(act))


class FeatureExtractor(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padding, stride, num_blocks=2, return_linear=False):
        super(FeatureExtractor, self).__init__()
        self.add_module('conv_block_0', ConvBlock3DSN(in_channel, out_channel, ker_size, padding, stride)),
        for i in range(num_blocks - 1):
            self.add_module('conv_block_{}'.format(i + 1),
                            ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride))
        if return_linear:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride, bn=False, act=None))
        else:
            self.add_module('conv_block_{}'.format(num_blocks),
                            ConvBlock3DSN(out_channel, out_channel, ker_size, padding, stride))


class Encode3DVAE(nn.Module):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode3DVAE, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)
        self.logvar = ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class Encode3DVAE_nb(nn.Module):
    def __init__(self, opt, out_dim=None, num_blocks=2):
        super(Encode3DVAE_nb, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, opt.ker_size, opt.ker_size // 2, 1, num_blocks=num_blocks)
        self.mu = nn.Sequential(
            ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None),
            nn.AdaptiveAvgPool3d(1)
        )
        self.logvar = nn.Sequential(
            ConvBlock3D(opt.nfc, output_dim, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None),
            nn.AdaptiveAvgPool3d(1)
        )
        self.bern = ConvBlock3D(opt.nfc, 1, opt.ker_size, opt.ker_size // 2, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        bern = torch.sigmoid(self.bern(features))
        features = bern * features
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar, bern


class Encode3DVAE1x1(nn.Module):
    def __init__(self, opt, out_dim=None):
        super(Encode3DVAE1x1, self).__init__()

        if out_dim is None:
            output_dim = opt.nfc
        else:
            assert type(out_dim) is int
            output_dim = out_dim

        self.features = FeatureExtractor(opt.nc_im, opt.nfc, 1, 0, 1, num_blocks=2)
        self.mu = ConvBlock3D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)
        self.logvar = ConvBlock3D(opt.nfc, output_dim, 1, 0, 1, bn=False, act=None)

    def forward(self, x):
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)

        return mu, logvar


class WDiscriminator3D(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator3D, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.head = ConvBlock3DSN(opt.nc_im, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock3DSN(N, N, opt.ker_size, opt.ker_size // 2, stride=1, bn=True, act='lrelu')
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=1, stride=1)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class WDiscriminatorBaselines(nn.Module):
    def __init__(self, opt):
        super(WDiscriminatorBaselines, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.p3d = (self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2)

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, opt.padd_size, stride=1, bn=False, act='lrelu')
        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1, bn=True, act='lrelu')
            self.body.add_module('block%d' % (i), block)

        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size, stride=1)

        self.apply(weights_init)

    def forward(self, x):
        x = F.pad(x, self.p3d)

        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class GeneratorCSG(nn.Module):
    def __init__(self, opt):
        super(GeneratorCSG, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.p3d_once = (1, 1,
                         1, 1,
                         1, 1)
        self.p3d = (self.opt.num_layer + 0, self.opt.num_layer + 0,
                    self.opt.num_layer + 0, self.opt.num_layer + 0,
                    self.opt.num_layer + 0, self.opt.num_layer + 0)

        self.head = ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1)

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)
            _first_stage.add_module('block%d' % (i), block)
        self.body.append(_first_stage)

        self.tail = nn.Sequential(
            nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=0, stride=1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise_init, noise_amp, mode='rand'):
        x = self.head(F.pad(noise_init, self.p3d_once))

        x_prev_out = self.body[0](F.pad(x, self.p3d))

        for idx, block in enumerate(self.body[1:], 1):
            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                x_prev_out_up_2 = utils.interpolate_3D(x_prev_out, size=[
                    x_prev_out_up.shape[-3] + (self.opt.num_layer + 0) * 2,
                    x_prev_out_up.shape[-2] + (self.opt.num_layer + 0) * 2,
                    x_prev_out_up.shape[-1] + (self.opt.num_layer + 0) * 2
                ])
                noise = utils.generate_noise(ref=x_prev_out_up_2)
                x_prev = block(x_prev_out_up_2 + noise * noise_amp[idx])
            else:
                x_prev = block(F.pad(x_prev_out_up, self.p3d))
            x_prev_out = x_prev + x_prev_out_up

        out = self.tail(F.pad(x_prev_out, self.p3d_once))
        return out


class GeneratorSG(nn.Module):
    def __init__(self, opt):
        super(GeneratorSG, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.p3d = (self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2,
                    self.opt.num_layer + 2, self.opt.num_layer + 2)

        self.body = nn.ModuleList([])

        _first_stage = nn.Sequential()
        _first_stage.add_module('head', ConvBlock3D(opt.nc_im, N, opt.ker_size, padding=0, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, padding=0, stride=1)
            _first_stage.add_module('block%d' % (i), block)
        _first_stage.add_module('tail', nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=0, stride=1))
        self.body.append(_first_stage)

        self.apply(weights_init)

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise_init, noise_amp, mode='rand'):

        x_prev_out = self.body[0](F.pad(noise_init, self.p3d))

        for idx, block in enumerate(self.body[1:], 1):
            x_prev_out = torch.tanh(x_prev_out)

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                x_prev_out_up_2 = utils.interpolate_3D(x_prev_out, size=[
                    x_prev_out_up.shape[-3] + (self.opt.num_layer + 2) * 2,
                    x_prev_out_up.shape[-2] + (self.opt.num_layer + 2) * 2,
                    x_prev_out_up.shape[-1] + (self.opt.num_layer + 2) * 2
                ])
                noise = utils.generate_noise(ref=x_prev_out_up_2)
                x_prev = block(x_prev_out_up_2 + noise * noise_amp[idx])
            else:
                x_prev = block(F.pad(x_prev_out_up, self.p3d))
            x_prev_out = x_prev + x_prev_out_up

        out = torch.tanh(x_prev_out)
        return out


class GeneratorHPVAEGAN(nn.Module):
    def __init__(self, opt):
        super(GeneratorHPVAEGAN, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode3DVAE(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.Sequential()

        # Normal Decoder
        self.decoder.add_module('head', ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.add_module('block%d' % (i), block)
        self.decoder.add_module('tail', nn.Conv3d(N, opt.nc_im, opt.ker_size, 1, opt.ker_size // 2))

        # 1x1 Decoder
        # self.decoder.add_module('head', ConvBlock3D(opt.latent_dim, N, 1, 0, stride=1))
        # for i in range(opt.num_layer):
        #     block = ConvBlock3D(N, N, 1, 0, stride=1)
        #     self.decoder.add_module('block%d' % (i), block)
        # self.decoder.add_module('tail', nn.Conv3d(N, opt.nc_im, 1, 1, 0))

        self.body = torch.nn.ModuleList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.Sequential()
            _first_stage.add_module('head',
                                    ConvBlock3D(self.opt.nc_im, self.N, self.opt.ker_size, self.opt.padd_size,
                                                stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock3D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.add_module('block%d' % (i), block)
            _first_stage.add_module('tail',
                                    nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size, 1, self.opt.ker_size // 2))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, video, noise_amp, noise_init=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        if noise_init is None:
            mu, logvar = self.encode(video)
            z_vae = reparameterize(mu, logvar, self.training)
        else:
            z_vae = noise_init

        vae_out = torch.tanh(self.decoder(z_vae))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, mode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, mode)

        if noise_init is None:
            return x_prev_out, vae_out, (mu, logvar)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1 and not self.opt.train_all:
                x_prev_out.detach_()

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand' and self.opt.vae_levels <= idx + 1:
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                x_prev = block(x_prev_out_up)

            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

        return x_prev_out


class GeneratorVAE_nb(nn.Module):
    def __init__(self, opt):
        super(GeneratorVAE_nb, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.N = N

        self.encode = Encode3DVAE_nb(opt, out_dim=opt.latent_dim, num_blocks=opt.enc_blocks)
        self.decoder = nn.Sequential()

        # Normal Decoder
        self.decoder.add_module('head', ConvBlock3D(opt.latent_dim, N, opt.ker_size, opt.padd_size, stride=1))
        for i in range(opt.num_layer):
            block = ConvBlock3D(N, N, opt.ker_size, opt.padd_size, stride=1)
            self.decoder.add_module('block%d' % (i), block)
        self.decoder.add_module('tail', nn.Conv3d(N, opt.nc_im, opt.ker_size, 1, opt.ker_size // 2))

        self.body = torch.nn.ModuleList([])

    def init_next_stage(self):
        if len(self.body) == 0:
            _first_stage = nn.Sequential()
            _first_stage.add_module('head',
                                    ConvBlock3D(self.opt.nc_im, self.N, self.opt.ker_size, self.opt.padd_size,
                                                stride=1))
            for i in range(self.opt.num_layer):
                block = ConvBlock3D(self.N, self.N, self.opt.ker_size, self.opt.padd_size, stride=1)
                _first_stage.add_module('block%d' % (i), block)
            _first_stage.add_module('tail',
                                    nn.Conv3d(self.N, self.opt.nc_im, self.opt.ker_size, 1, self.opt.ker_size // 2))
            self.body.append(_first_stage)
        else:
            self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, video, noise_amp, noise_init_norm=None, noise_init_bern=None, sample_init=None, mode='rand'):
        if sample_init is not None:
            assert len(self.body) > sample_init[0], "Strating index must be lower than # of body blocks"

        if noise_init_norm is None:
            mu, logvar, bern = self.encode(video)
            z_vae_norm = reparameterize(mu, logvar, self.training)
            z_vae_bern = reparameterize_bern(bern, self.training)
        else:
            z_vae_norm = noise_init_norm
            z_vae_bern = noise_init_bern

        vae_out = torch.tanh(self.decoder(z_vae_norm * z_vae_bern))

        if sample_init is not None:
            x_prev_out = self.refinement_layers(sample_init[0], sample_init[1], noise_amp, mode)
        else:
            x_prev_out = self.refinement_layers(0, vae_out, noise_amp, mode)

        if noise_init_norm is None:
            return x_prev_out, vae_out, (mu, logvar, bern)
        else:
            return x_prev_out, vae_out

    def refinement_layers(self, start_idx, x_prev_out, noise_amp, mode):
        for idx, block in enumerate(self.body[start_idx:], start_idx):
            if self.opt.vae_levels == idx + 1:
                x_prev_out.detach_()

            # Upscale
            x_prev_out_up = utils.upscale(x_prev_out, idx + 1, self.opt)

            # Add noise if "random" sampling, else, add no noise is "reconstruction" mode
            if mode == 'rand':
                noise = utils.generate_noise(ref=x_prev_out_up)
                x_prev = block(x_prev_out_up + noise * noise_amp[idx + 1])
            else:
                x_prev = block(x_prev_out_up)

            x_prev_out = torch.tanh(x_prev + x_prev_out_up)

        return x_prev_out
