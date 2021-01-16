import os
import random
from torch.utils.data import Dataset
import numpy as np
import cv2
import kornia as K
from . import video_to_frames
import utils
import logging


class SingleVideoDataset(Dataset):
    def __init__(self, opt, transforms=None):
        super(SingleVideoDataset, self).__init__()

        self.zero_scale_frames = None
        self.frames = None

        self.transforms = transforms

        self.video_path = opt.video_path
        if not os.path.exists(self.video_path):
            logging.error("invalid path")
            exit(0)

        # Get original frame size and aspect-ratio
        capture = cv2.VideoCapture(opt.video_path)
        opt.org_fps = capture.get(cv2.CAP_PROP_FPS)
        h, w = capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        capture.release()
        self.org_frame_size = [h, w]
        opt.ar = self.org_frame_size[0] / self.org_frame_size[1]  # H2W

        opt.fps_lcm = np.lcm.reduce(opt.sampling_rates)

        self.opt = opt

        logging.info("Saving zero-level frames...")
        self.zero_scale_frames = self._generate_frames(0)

    def __len__(self):
        return (len(self.zero_scale_frames) - self.opt.fps_lcm) * self.opt.data_rep

    def __getitem__(self, idx):
        idx = idx % (len(self.zero_scale_frames) - self.opt.fps_lcm)

        # Horizontal flip (Until Kornia will handle videos
        hflip = random.random() < 0.5 if self.opt.hflip else False

        every = self.opt.sampling_rates[self.opt.fps_index]
        frames = self.frames[idx:idx + self.opt.fps_lcm + 1:every]
        frames = K.image_to_tensor(frames).float()
        frames = frames / 255  # Set range [0, 1]
        frames_transformed = self._get_transformed_frames(frames, hflip)

        # Extract o-level index
        if self.opt.scale_idx > 0:
            every_zero_scale = self.opt.sampling_rates[0]
            frames_zero_scale = self.zero_scale_frames[idx:idx + self.opt.fps_lcm + 1:every_zero_scale]
            frames_zero_scale = K.image_to_tensor(frames_zero_scale).float()
            frames_zero_scale = frames_zero_scale / 255
            frames_zero_scale_transformed = self._get_transformed_frames(frames_zero_scale, hflip)

            return [frames_transformed, frames_zero_scale_transformed]

        return frames_transformed

    @staticmethod
    def _get_transformed_frames(frames, hflip):

        frames_transformed = frames

        if hflip:
            frames_transformed = K.hflip(frames_transformed)

        # Normalize
        frames_transformed = K.normalize(frames_transformed, 0.5, 0.5)

        # Permute CTHW
        frames_transformed = frames_transformed.permute(1, 0, 2, 3)

        return frames_transformed

    def _generate_frames(self, scale_idx):
        base_size = utils.get_scales_by_index(scale_idx, self.opt.scale_factor, self.opt.stop_scale, self.opt.img_size)
        scaled_size = [int(base_size * self.opt.ar), base_size]
        self.opt.scaled_size = scaled_size

        return video_to_frames(self.opt)

    def generate_frames(self, scale_idx):
        self.frames = self._generate_frames(scale_idx)
