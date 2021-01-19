import argparse
import torch
import numpy as np
import moviepy.editor as mpy
import imageio
from glob import glob
import os
from torchvision.utils import make_grid


def make_video(tensor, fps, filename):
    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)


def generate_gifs(opt):
    for exp_dir in opt.experiments:
        reals_path = os.path.join(exp_dir, 'real_full_scale.pth')
        fakes_path = os.path.join(exp_dir, 'random_samples.pth')
        os.makedirs(os.path.join(exp_dir, opt.save_path), exist_ok=True)
        print('Generating dir {}'.format(os.path.join(exp_dir, opt.save_path)))

        real_sample = torch.load(reals_path)
        make_video(real_sample, 4, os.path.join(exp_dir, opt.save_path, 'real.gif'))

        random_samples = torch.load(fakes_path).permute(0, 2, 3, 4, 1)[:opt.max_samples]

        # Make grid
        real_transpose = torch.tensor(real_sample).permute(0, 3, 1, 2)[::2]  # TxCxHxW
        grid_image = make_grid(real_transpose, real_transpose.shape[0]).permute(1, 2, 0)
        imageio.imwrite(os.path.join(exp_dir, opt.save_path, 'real_unfold.png'), grid_image.data.numpy())

        fake = (random_samples.data.cpu().numpy() * 255).astype(np.uint8)
        fake_transpose = torch.tensor(fake).permute(0, 1, 4, 2, 3)[:, ::2]  # BxTxCxHxW
        fake_reshaped = fake_transpose.flatten(0, 1)  # (B+T)xCxHxW
        grid_image = make_grid(fake_reshaped[:10 * fake_transpose.shape[1], :, :, :], fake_transpose.shape[1]).permute(
            1, 2, 0)
        imageio.imwrite(os.path.join(exp_dir, opt.save_path, 'fake_unfold.png'), grid_image.data.numpy())

        white_space = torch.ones_like(random_samples)[:, :, :, :10] * 255

        random_samples = random_samples.data.cpu().numpy()
        random_samples = (random_samples * 255).astype(np.uint8)
        white_space = white_space.data.cpu().numpy()
        white_space = (white_space * 255).astype(np.uint8)

        concat_gif = []
        for i, (vid, ws) in enumerate(zip(random_samples, white_space)):
            if i < len(random_samples) - 1:
                concat_gif.append(np.concatenate((vid, ws), axis=2))
            else:
                concat_gif.append(vid)
        concat_gif = np.concatenate(concat_gif, axis=2)
        make_video(concat_gif, 4, os.path.join(exp_dir, opt.save_path, 'fakes.gif'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-dir', required=True, help="Experiment directory (glob format)")
    parser.add_argument('--max-samples', type=int, default=4, help="Maximum number of samples")
    parser.add_argument('--save-path', default='gifs', help="New directory to be created for outputs")

    opt = parser.parse_args()

    opt.experiments = sorted(glob(opt.exp_dir))
    generate_gifs(opt)
