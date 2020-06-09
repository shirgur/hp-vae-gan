import argparse
import utils
import random
import os

from utils import logger, tools
import logging
import colorama

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from modules import networks_3d
from modules.utils import *
from datasets import SingleVideoDataset

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT


def train(opt, netG):
    # Re-generate dataset frames
    fps, td, fps_index = utils.get_fps_td_by_index(opt.scale_idx, opt)
    opt.fps = fps
    opt.td = td
    opt.fps_index = fps_index

    with logger.LoggingBlock("Updating dataset", emph=True):
        logging.info("{}FPS :{} {}{}".format(green, clear, opt.fps, clear))
        logging.info("{}Time-Depth :{} {}{}".format(green, clear, opt.td, clear))
        logging.info("{}Sampling-Ratio :{} {}{}".format(green, clear, opt.sampling_rates[opt.fps_index], clear))
        opt.dataset.generate_frames(opt.scale_idx)

    # Initialize noise
    if not hasattr(opt, 'Z_init'):
        initial_size = utils.get_scales_by_index(0, opt.scale_factor, opt.stop_scale, opt.img_size)
        initial_size = [int(initial_size * opt.ar), initial_size]
        opt.Z_init = utils.generate_noise(size=[opt.batch_size, 3, opt.td, *initial_size]).to(device)

        opt.saver.save_checkpoint({'data': opt.Z_init}, 'Z_init.pth')

    D_curr = getattr(networks_3d, opt.discriminator)(opt).to(opt.device)
    if opt.scale_idx > 0:
        D_curr.load_state_dict(
            torch.load('{}/netD_{}.pth'.format(opt.saver.experiment_dir, opt.scale_idx - 1))['state_dict'])

    # Current optimizers
    optimizerD = optim.Adam(D_curr.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [
        {"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
        for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if hasattr(netG, 'head'):
        if opt.scale_idx - opt.train_depth < 0:
            parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** opt.scale_idx)}]
    if hasattr(netG, 'tail'):
        parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # Parallel
    if opt.device == 'cuda':
        G_curr = torch.nn.DataParallel(netG)
        D_curr = torch.nn.DataParallel(D_curr)
    else:
        G_curr = netG

    progressbar_args = {
        "iterable": range(opt.niter),
        "desc": "Training scale [{}/{}]".format(opt.scale_idx + 1, opt.stop_scale + 1),
        "train": True,
        "offset": 0,
        "logging_on_update": False,
        "logging_on_close": True,
        "postfix": True
    }
    epoch_iterator = tools.create_progressbar(**progressbar_args)

    iterator = iter(data_loader)

    # idx = 0
    for iteration in epoch_iterator:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(opt.data_loader)
            data = next(iterator)

        if opt.scale_idx > 0:
            real, _ = data
            real = real.to(opt.device)
        else:
            real = data.to(opt.device)

        noise_init = utils.generate_noise(ref=opt.Z_init)

        ############################
        # calculate noise_amp
        ###########################
        if iteration == 0:
            if opt.scale_idx == 0:
                opt.noise_amp = 1
                opt.Noise_Amps.append(opt.noise_amp)
            else:
                opt.Noise_Amps.append(0)
                z_reconstruction = G_curr(opt.Z_init, opt.Noise_Amps, mode="rec")

                RMSE = torch.sqrt(F.mse_loss(real, z_reconstruction))
                opt.noise_amp = opt.noise_amp_init * RMSE.item() / opt.batch_size
                opt.Noise_Amps[-1] = opt.noise_amp

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            #################

            # Train 3D Discriminator
            D_curr.zero_grad()
            output = D_curr(real)
            errD_real = -output.mean()

            # train with fake
            #################
            if j == opt.Dsteps - 1:
                fake = G_curr(noise_init, opt.Noise_Amps, mode="rand")
            else:
                with torch.no_grad():
                    fake = G_curr(noise_init, opt.Noise_Amps, mode="rand")

            # Train 3D Discriminator
            output = D_curr(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = calc_gradient_penalty(D_curr, real, fake, opt.lambda_grad, opt.device)
            errD_total = errD_real + errD_fake + gradient_penalty
            errD_total.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        errG_total = 0

        # Train with 3D Discriminator
        output = D_curr(fake)
        errG = -output.mean() * opt.disc_loss_weight
        errG_total += errG

        # Train reconstruction
        generated = None
        if opt.alpha > 0:
            generated = G_curr(opt.Z_init, opt.Noise_Amps, mode="rec")
            rec_loss = opt.alpha * opt.rec_loss(generated, real)
            errG_total += rec_loss

        G_curr.zero_grad()
        errG_total.backward()

        for _ in range(opt.Gsteps):
            optimizerG.step()

        # Update progress bar
        epoch_iterator.set_description('Scale [{}/{}], Iteration [{}/{}]'.format(
            opt.scale_idx + 1, opt.stop_scale + 1,
            iteration + 1, opt.niter,
        ))

        if opt.visualize:
            # Tensorboard
            opt.summary.add_scalar('Video/Scale {}/errG'.format(opt.scale_idx), errG.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/errD_fake'.format(opt.scale_idx), errD_fake.item(), iteration)
            opt.summary.add_scalar('Video/Scale {}/errD_real'.format(opt.scale_idx), errD_real.item(), iteration)
            if opt.alpha > 0:
                opt.summary.add_scalar('Video/Scale {}/rec_loss'.format(opt.scale_idx), rec_loss.item(), iteration)
                opt.summary.add_scalar('Video/Scale {}/noise_amp'.format(opt.scale_idx), opt.noise_amp, iteration)

            if iteration % opt.print_interval == 0:
                opt.summary.visualize_video(opt, iteration, real, 'Real')

                if generated is not None:
                    opt.summary.visualize_video(opt, iteration, generated, 'Generated')

                opt.summary.visualize_video(opt, iteration, fake, 'Fake')

    epoch_iterator.close()

    # Save data
    opt.saver.save_checkpoint({'data': opt.Z_init}, 'Z_init.pth')
    opt.saver.save_checkpoint({'data': opt.Noise_Amps}, 'Noise_Amps.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': netG.state_dict(),
        'optimizer': optimizerG.state_dict(),
        'noise_amps': opt.Noise_Amps,
    }, 'netG.pth')
    opt.saver.save_checkpoint({
        'scale': opt.scale_idx,
        'state_dict': D_curr.module.state_dict() if opt.device == 'cuda' else D_curr.state_dict(),
        'optimizer': optimizerD.state_dict(),
    }, 'netD_{}.pth'.format(opt.scale_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # load, input, save configurations:
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--netD', default='', help='path to netD (to continue training)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    # networks hyper parameters:
    parser.add_argument('--nc-z', type=int, default=3, help='noise # channels')
    parser.add_argument('--nc-im', type=int, help='image # channels', default=3)
    parser.add_argument('--nfc', type=int, default=64, help='model basic # channels')
    parser.add_argument('--ker-size', type=int, default=3, help='kernel size')
    parser.add_argument('--num-layer', type=int, default=5, help='number of layers')
    parser.add_argument('--stride', default=1, help='stride')
    parser.add_argument('--padd-size', type=int, default=1, help='net pad size')
    parser.add_argument('--generator', type=str, help='Generator model', default='GeneratorCSG')
    parser.add_argument('--discriminator', type=str, help='Discriminator model', default='WDiscriminator3D')

    # pyramid parameters:
    parser.add_argument('--scale-factor', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--noise_amp', type=float, default=0.1, help='addative noise cont weight')
    parser.add_argument('--min-size', type=int, default=32, help='image minimal size at the coarser scale')
    parser.add_argument('--max-size', type=int, default=256, help='image minimal size at the coarser scale')

    # optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=50000, help='number of iterations to train per scale')
    parser.add_argument('--lr-g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr-d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--disc-loss-weight', type=float, default=1.0, help='3D disc weight')
    parser.add_argument('--Gsteps', type=int, default=1, help='generator inner steps')
    parser.add_argument('--Dsteps', type=int, default=1, help='discriminator inner steps')
    parser.add_argument('--lambda-grad', type=float, default=0.1, help='gradient penelty weight')
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10.)
    parser.add_argument('--lr-scale', type=float, default=0.2, help='scaling of learning rate for lower stages')
    parser.add_argument('--train-depth', type=int, default=1, help='how many layers are trained if growing')

    # Dataset
    parser.add_argument('--video-path', required=True, help='video path')
    parser.add_argument('--start-frame', default=0, type=int, help='start frame number')
    parser.add_argument('--max-frames', default=1000, type=int, help='# frames to save')
    parser.add_argument('--hflip', action='store_true', default=False, help='horizontal flip')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--sampling-rates', type=int, nargs='+', default=[4, 3, 2, 1], help='sampling rates')
    parser.add_argument('--stop-scale-time', type=int, default=-1)
    parser.add_argument('--data-rep', type=int, default=1, help='data repetition')

    # main arguments
    parser.add_argument('--checkname', type=str, default='DEBUG', help='check name')
    parser.add_argument('--mode', default='train', help='task to be done')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--print-interval', type=int, default=100, help='print interva')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize using tensorboard')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda')

    parser.set_defaults(hflip=False)
    opt = parser.parse_args()

    assert opt.disc_loss_weight > 0

    # Define Saver
    opt.saver = utils.VideoSaver(opt)

    # Define Tensorboard Summary
    opt.summary = utils.TensorboardSummary(opt.saver.experiment_dir)
    logger.configure_logging(os.path.abspath(os.path.join(opt.saver.experiment_dir, 'logbook.txt')))

    # CUDA
    device = 'cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu'
    opt.device = device
    if torch.cuda.is_available() and device == 'cpu':
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Initial config
    opt.noise_amp_init = opt.noise_amp
    opt.scale_factor_init = opt.scale_factor

    # Adjust scales
    utils.adjust_scales2image(opt.img_size, opt)

    # Manual seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Reconstruction loss
    opt.rec_loss = torch.nn.MSELoss()

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
                             num_workers=4)

    if opt.stop_scale_time == -1:
        opt.stop_scale_time = opt.stop_scale

    opt.dataset = dataset
    opt.data_loader = data_loader

    with logger.LoggingBlock("Commandline Arguments", emph=True):
        for argument, value in sorted(vars(opt).items()):
            if type(value) in (str, int, float, tuple, list):
                logging.info('{}: {}'.format(argument, value))

    with logger.LoggingBlock("Experiment Summary", emph=True):
        video_file_name, checkname, experiment = opt.saver.experiment_dir.split('/')[-3:]
        logging.info("{}Video file :{} {}{}".format(magenta, clear, video_file_name, clear))
        logging.info("{}Checkname  :{} {}{}".format(magenta, clear, checkname, clear))
        logging.info("{}Experiment :{} {}{}".format(magenta, clear, experiment, clear))

        with logger.LoggingBlock("Commandline Summary", emph=True):
            logging.info("{}Start frame    :{} {}{}".format(blue, clear, opt.start_frame, clear))
            logging.info("{}Max frames     :{} {}{}".format(blue, clear, opt.max_frames, clear))
            logging.info("{}Generator      :{} {}{}".format(blue, clear, opt.generator, clear))
            logging.info("{}Iterations     :{} {}{}".format(blue, clear, opt.niter, clear))
            logging.info("{}Alpha          :{} {}{}".format(blue, clear, opt.alpha, clear))
            logging.info("{}Sampling rates :{} {}{}".format(blue, clear, opt.sampling_rates, clear))

    # Current networks
    assert hasattr(networks_3d, opt.generator)
    netG = getattr(networks_3d, opt.generator)(opt).to(opt.device)

    if opt.netG != '':
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
    else:
        opt.resumed_idx = -1

    while opt.scale_idx < opt.stop_scale + 1:
        if (opt.scale_idx > 0) and (opt.resumed_idx != opt.scale_idx):
            netG.init_next_stage()
        train(opt, netG)

        # Increase scale
        opt.scale_idx += 1
