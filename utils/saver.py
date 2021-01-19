import os
import torch
import glob
import numpy as np
import cv2


def write_video(array, filename, opt):

    _, num_frames, height, width = array.shape

    FPS = opt.fps
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), float(FPS), (width, height))

    for i in range(num_frames):
        frame = (array[:, i, :, :] + 1)*127.5
        frame = frame.transpose(1,2, 0)
        video.write(np.uint8(frame))
    video.release()


class VideoSaver(object):
    def __init__(self, opt, run_id=None):
        self.opt = opt
        if not hasattr(opt, 'experiment_dir') or not os.path.exists(opt.experiment_dir):
            clip_name = '.'.join(opt.video_path.split('/')[-1].split('.')[:-1])
            self.directory = os.path.join('run', clip_name, opt.checkname)
            if run_id is None:
                self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
                run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        else:
            self.experiment_dir = opt.experiment_dir

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.eval_dir = os.path.join(self.experiment_dir, "eval")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        filename = os.path.join(self.experiment_dir, filename)
        return torch.load(filename)

    def save_video(self, array, filename):
        filename = os.path.join(self.eval_dir, filename)
        write_video(array, filename, self.opt)


class ImageSaver(object):
    def __init__(self, opt, run_id=None):
        self.opt = opt
        if not hasattr(opt, 'experiment_dir') or not os.path.exists(opt.experiment_dir):
            clip_name = '.'.join(opt.image_path.split('/')[-1].split('.')[:-1])
            self.directory = os.path.join('run', clip_name, opt.checkname)
            if run_id is None:
                self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
                run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        else:
            self.experiment_dir = opt.experiment_dir

        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.eval_dir = os.path.join(self.experiment_dir, "eval")
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        filename = os.path.join(self.experiment_dir, filename)
        return torch.load(filename)
