import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t, range=None):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def add_scalar(self, *args):
        self.writer.add_scalar(*args)

    def visualize_video(self, opt, global_step, video, name):

        video_transpose = video.permute(0, 2, 1, 3, 4)  # BxTxCxHxW
        video_reshaped = video_transpose.flatten(0, 1)  # (B+T)xCxHxW

        # image_range = opt.td #+ opt.num_targets
        image_range = video.shape[2]

        grid_image = make_grid(video_reshaped[:3 * image_range, :, :, :].clone().cpu().data, image_range,
                               normalize=True)
        self.writer.add_image('Video/Scale {}/{}_unfold'.format(opt.scale_idx, name), grid_image, global_step)
        norm_range(video_transpose)
        self.writer.add_video('Video/Scale {}/{}'.format(opt.scale_idx, name), video_transpose[:3], global_step)

    def visualize_image(self, opt, global_step, ןimages, name):
        grid_image = make_grid(ןimages[:3, :, :, :].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Image/Scale {}/{}'.format(opt.scale_idx, name), grid_image, global_step)
