import torch
import torch.nn.functional as F
import math

__all__ = ['interpolate', 'interpolate_3D', 'adjust_scales2image', 'generate_noise', 'get_scales_by_index',
           'get_fps_td_by_index', 'get_fps_by_index', 'upscale', 'upscale_2d']


def interpolate(input, size=None, scale_factor=None, interpolation='bilinear'):
    if input.dim() == 5:
        b, c, t, h0, w0 = input.shape
        img = input.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B+T)CHW
        scaled = F.interpolate(img, size=size, scale_factor=scale_factor, mode=interpolation, align_corners=True)
        _, _, h1, w1 = scaled.shape
        scaled = scaled.reshape(b, t, c, h1, w1).permute(0, 2, 1, 3, 4)
    else:
        scaled = F.interpolate(input, size=size, scale_factor=scale_factor, mode=interpolation, align_corners=True)

    return scaled


def interpolate_3D(input, size=None, scale_factor=None, interpolation='trilinear'):
    assert input.dim() == 5, "input must be 5D"
    scaled = F.interpolate(input, size=size, scale_factor=scale_factor, mode=interpolation, align_corners=True)

    return scaled


def adjust_scales2image(size, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / size, 1)
    opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop


def generate_noise(ref=None, size=None, type='normal', emb_size=None, device=None):
    # Initiate noise without batch size
    if ref is not None:
        noise = torch.zeros_like(ref)
    elif size is not None:
        noise = torch.zeros(*size).to(device)
    else:
        raise Exception("ref or size must be applied")

    if type == 'normal':
        return noise.normal_(0, 1)
    elif type == 'benoulli':
        return noise.bernoulli_(0.5)

    if type == 'int':
        assert (emb_size is not None) and (size is not None) and (device is not None)
        return torch.randint(0, emb_size, size=size, device=device)

    return noise.uniform_(0, 1)  # Default == Uniform


def get_scales_by_index(index, scale_factor, stop_scale, img_size):
    scale = math.pow(scale_factor, stop_scale - index)
    s_size = math.ceil(scale * img_size)

    return s_size


def get_fps_by_index(index, opt):
    # Linear fps interpolation by divisors
    fps_index = int((index / opt.stop_scale_time) * (len(opt.sampling_rates) - 1))

    return opt.org_fps / opt.sampling_rates[fps_index], fps_index


def get_fps_td_by_index(index, opt):
    fps, fps_index = get_fps_by_index(index, opt)

    every = opt.sampling_rates[fps_index]
    time_depth = opt.fps_lcm // every + 1

    return fps, time_depth, fps_index


def upscale(video, index, opt):
    assert index > 0

    next_shape = get_scales_by_index(index, opt.scale_factor, opt.stop_scale, opt.img_size)
    next_fps, next_td, _ = get_fps_td_by_index(index, opt)
    next_shape = [next_td, int(next_shape * opt.ar), next_shape]

    # Video interpolation
    vid_up = interpolate_3D(video, size=next_shape)

    return vid_up


def upscale_2d(image, index, opt):
    assert index > 0

    next_shape = get_scales_by_index(index, opt.scale_factor, opt.stop_scale, opt.img_size)
    next_shape = [int(next_shape * opt.ar), next_shape]

    # Video interpolation
    img_up = interpolate(image, size=next_shape)

    return img_up
