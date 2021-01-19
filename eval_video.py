import argparse
import utils
import os
from glob import glob
import ast

from utils import logger, tools
import logging
import colorama

import torch
from torch.utils.data import DataLoader

from modules import networks_3d
from datasets import SingleVideoDataset

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


@torch.no_grad()
def eval(opt, netG):
    # Re-generate dataset frames

    fps, td, fps_index = utils.get_fps_td_by_index(opt.scale_idx, opt)
    opt.fps = fps
    opt.td = td
    opt.fps_index = fps_index
    # opt.tds.append(opt.td)
    opt.dataset.generate_frames(opt.scale_idx)

    torch.save(opt.dataset.frames, os.path.join(opt.saver.eval_dir, "real_full_scale.pth"))

    if not hasattr(opt, 'Z_init_size'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init_size = [opt.batch_size, opt.latent_dim, opt.td, *initial_size]

    # Parallel
    if opt.device == 'cuda':
        G_curr = torch.nn.DataParallel(netG)
    else:
        G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter),
        "desc": "Generation scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = tools.create_progressbar(**progressbar_args)

    iterator = iter(data_loader)

    random_samples = []

    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, real_zero = data
            real = real.to(opt.device)
        else:
            real = data.to(opt.device)

        noise_init = utils.generate_noise(size=opt.Z_init_size, device=opt.device)

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        with torch.no_grad():
            fake_var = []
            fake_vae_var = []
            for _ in range(opt.num_samples):
                noise_init = utils.generate_noise(ref=noise_init)
                fake, fake_vae = G_curr(noise_init, opt.Noise_Amps, noise_init=noise_init, mode="rand")
                fake_var.append(fake)
                fake_vae_var.append(fake_vae)
            fake_var = torch.cat(fake_var, dim=0)
            fake_vae_var = torch.cat(fake_vae_var, dim=0)

        opt.summary.visualize_video(opt, iteration, real, 'Real')
        opt.summary.visualize_video(opt, iteration, fake_var, 'Fake var')
        opt.summary.visualize_video(opt, iteration, fake_vae_var, 'Fake VAE var')

        random_samples.append(fake_var)

    random_samples = torch.cat(random_samples, dim=0)
    torch.save(random_samples, os.path.join(opt.saver.eval_dir, "random_samples.pth"))
    epoch_iterator.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-dir', required=True, help="Experiment directory")
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to generate')
    parser.add_argument('--netG', default='netG.pth', help="path to netG (to continue training)")
    parser.add_argument('--niter', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    exceptions = ['no-cuda', 'niter', 'data_rep', 'batch_size', 'netG']
    all_dirs = glob(opt.exp_dir)

    progressbar_args = {
        "iterable": all_dirs,
        "desc": "Experiments",
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    exp_iterator = tools.create_progressbar(**progressbar_args)

    for idx, exp_dir in enumerate(exp_iterator):
        opt.experiment_dir = exp_dir
        keys = vars(opt).keys()
        with open(os.path.join(exp_dir, 'args.txt'), 'r') as f:
            for line in f.readlines():
                log_arg = line.replace(' ', '').replace('\n', '').split(':')
                assert len(log_arg) == 2
                if log_arg[0] in exceptions:
                    continue
                try:
                    setattr(opt, log_arg[0], ast.literal_eval(log_arg[1]))
                except Exception:
                    setattr(opt, log_arg[0], log_arg[1])

        opt.netG = os.path.join(exp_dir, opt.netG)
        if not os.path.exists(opt.netG):
            logging.info('Skipping {}, file not exists!'.format(opt.netG))
            continue

        # Define Saver
        opt.saver = utils.VideoSaver(opt)

        # Define Tensorboard Summary
        opt.summary = utils.TensorboardSummary(opt.saver.eval_dir)

        # Logger
        logger.configure_logging(os.path.abspath(os.path.join(opt.experiment_dir, 'logbook.txt')))

        # CUDA
        device = 'cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu'
        opt.device = device
        if torch.cuda.is_available() and device == 'cpu':
            logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

        # Adjust scales
        utils.adjust_scales2image(opt.img_size, opt)

        # Initial parameters
        opt.scale_idx = 0
        opt.nfc_prev = 0
        opt.Noise_Amps = []

        # Date
        dataset = SingleVideoDataset(opt)
        data_loader = DataLoader(dataset,
                                 shuffle=True,
                                 drop_last=True,
                                 batch_size=opt.batch_size,
                                 num_workers=2)

        opt.dataset = dataset
        opt.data_loader = data_loader

        # Current networks
        assert hasattr(networks_3d, opt.generator)
        netG = getattr(networks_3d, opt.generator)(opt).to(opt.device)

        if not os.path.isfile(opt.netG):
            raise RuntimeError("=> no <G> checkpoint found at '{}'".format(opt.netG))
        checkpoint = torch.load(opt.netG)
        opt.scale_idx = checkpoint['scale']
        opt.resumed_idx = checkpoint['scale']
        opt.resume_dir = '/'.join(opt.netG.split('/')[:-1])
        for _ in range(opt.scale_idx):
            netG.init_next_stage()
        netG.load_state_dict(checkpoint['state_dict'])

        # NoiseAmp
        opt.Noise_Amps = torch.load(os.path.join(opt.resume_dir, 'Noise_Amps.pth'))['data']

        eval(opt, netG)
