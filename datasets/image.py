import random
from torch.utils.data import Dataset
import imageio
import kornia as K
import utils
import cv2
import logging
import os


class SingleImageDataset(Dataset):
    def __init__(self, opt, transforms=None):
        super(SingleImageDataset, self).__init__()

        self.zero_scale_frames = None
        self.frames = None

        self.transforms = transforms

        self.image_path = opt.image_path
        if not os.path.exists(opt.image_path):
            logging.error("invalid path")
            exit(0)

        # Get original frame size and aspect-ratio
        self.image_full_scale = imageio.imread(self.image_path)[:, :, :3]
        self.org_size = [self.image_full_scale.shape[0], self.image_full_scale.shape[1]]
        h, w = self.image_full_scale.shape[:2]
        opt.ar = h / w  # H2W

        self.opt = opt

    def __len__(self):
        return self.opt.data_rep

    def __getitem__(self, idx):

        # Horizontal flip (Until Kornia will handle videos
        hflip = random.random() < 0.5 if self.opt.hflip else False

        images = self.generate_image(self.opt.scale_idx)
        images = K.image_to_tensor(images).float()
        images = images / 255  # Set range [0, 1]
        images_transformed = self._get_transformed_images(images, hflip)

        # Extract o-level index
        if self.opt.scale_idx > 0:
            images_zero_scale = self.generate_image(0)
            images_zero_scale = K.image_to_tensor(images_zero_scale).float()
            images_zero_scale = images_zero_scale / 255
            images_zero_scale_transformed = self._get_transformed_images(images_zero_scale, hflip)

            return [images_transformed, images_zero_scale_transformed]

        return images_transformed

    @staticmethod
    def _get_transformed_images(images, hflip):

        images_transformed = images

        if hflip:
            images_transformed = K.hflip(images_transformed)

        # Normalize
        images_transformed = K.normalize(images_transformed, 0.5, 0.5)

        return images_transformed

    def generate_image(self, scale_idx):
        base_size = utils.get_scales_by_index(scale_idx, self.opt.scale_factor, self.opt.stop_scale, self.opt.img_size)
        scaled_size = [int(base_size * self.opt.ar), base_size]
        self.opt.scaled_size = scaled_size
        img = cv2.resize(self.image_full_scale, tuple(scaled_size[::-1]))
        return img
