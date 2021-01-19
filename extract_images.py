import torch
import numpy as np
import imageio
from glob import glob
import os
import argparse


def generate_images(opt):
    for exp_dir in opt.experiments:
        fakes_path = os.path.join(exp_dir, 'random_samples.pth')
        os.makedirs(os.path.join(exp_dir, opt.save_path), exist_ok=True)
        print('Generating dir {}'.format(os.path.join(exp_dir, opt.save_path)))

        random_samples = torch.load(fakes_path).permute(0, 2, 3, 1)[:opt.max_samples]
        random_samples = (random_samples + 1) / 2
        random_samples = random_samples[:20] * 255
        random_samples = (random_samples.data.cpu().numpy()).astype(np.uint8)
        for i, sample in enumerate(random_samples):
            imageio.imwrite(os.path.join(exp_dir, opt.save_path, 'fake_{}.png'.format(i)), sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-dir', required=True, help="Experiment directory (glob format)")
    parser.add_argument('--max-samples', type=int, default=4, help="Maximum number of samples")
    parser.add_argument('--save-path', default='images', help="New directory to be created for outputs")

    opt = parser.parse_args()

    opt.experiments = sorted(glob(opt.exp_dir))
    generate_images(opt)
