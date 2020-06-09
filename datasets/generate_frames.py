import cv2
import os
import numpy as np
import logging


def video_to_frames(opt):
    assert os.path.exists(opt.video_path)

    video_dir, video_filename = os.path.split(opt.video_path)

    logging.info(
        "Extracting frames from {}, at size ({}, {})".format(video_filename, opt.scaled_size[0], opt.scaled_size[1]))

    capture = cv2.VideoCapture(opt.video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    assert total_frames > opt.start_frame >= 0, "Start-Frame out of range"

    trimmed_total_frames = total_frames - opt.start_frame
    required_frames = opt.max_frames
    end = required_frames if trimmed_total_frames > required_frames else trimmed_total_frames

    capture.set(1, opt.start_frame)  # Set starting frame
    frame = 0
    while_safety = 0
    frames_mem = []

    while frame < end:

        if while_safety > 500:
            break

        _, image = capture.read()

        if image is None:
            while_safety += 1
            continue

        while_safety = 0

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if hasattr(opt, 'scaled_size') and opt.scaled_size is not None:
            rgb_frame = cv2.resize(rgb_frame, (opt.scaled_size[1], opt.scaled_size[0]),
                                   interpolation=cv2.INTER_LINEAR)
        frames_mem.append(rgb_frame)

        frame += 1

    capture.release()

    frames_mem = np.stack(frames_mem)

    return frames_mem
