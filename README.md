# Hierarchical Patch VAE-GAN
Official repository of the paper "Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample" (NeurIPS 2020)

**[Project](https://shirgur.github.io/hp-vae-gan/) | [arXiv](https://arxiv.org/abs/2006.12226) | [Code](https://github.com/shirgur/hp-vae-gan)**

Real Videos

<img src='visuals/wingsuit_real.gif' align="left" width=256>

<br><br><br><br><br><br>

Fake Videos

<img src='visuals/wingsuit_fake.gif' align="left" width='100%'>

<br><br><br><br><br><br>

 ## Environment setting
Use commands in ```env.sh``` to setup the correct conda environment

## Colab
An example for training and extracting samples for image generation.
The same can be easily modified for video generation using `````*_video(s).py````` files.
https://colab.research.google.com/drive/1SmxFVqUvEkU7pHIwyLUz4VM1AxoVU-ER?usp=sharing

## Training Video
For training a single video, use the following command for example:

```CUDA_VISIBLE_DEVICES=0 python train_video.py --video-path data/vids/air_balloons.mp4 --vae-levels 3 --checkname myvideotest --visualize```

Common training options:
```
# Networks Hyper Parameters
--nfc                model basic # channels
--latent-dim         Latent dim size
--vae-levels         # VAE levels
--generator          generator mode

# Optimization hyper parameters
--niter              number of iterations to train per scale
--rec-weight         reconstruction loss weight
--train-all          train all levels w.r.t. train-depth

# Dataset
--video-path         video path (required)
--start-frame        start frame number
--max-frames         # frames to save
--sampling-rates     sampling rates

# Misc
--visualize     visualize using tensorboard
```

## Training Image
For training a single video, use the following command for example:

```CUDA_VISIBLE_DEVICES=0 python train_image.py --image-path data/imgs/air_balloons.jpg --vae-levels 3 --checkname myimagetest --visualize```

## Training baselines for video
For training a single video using SinGan re-implementation, use the following command:

```CUDA_VISIBLE_DEVICES=0 python train_video_baselines.py --video-path data/vids/air_balloons.mp4 --checkname myimagetest --visualize --generator GeneratorSG --train-depth 1```

## Generating Samples
Use ```eval_*.py``` to generate samples from an "experiment" folder created during training.
The code uses Glob package for multiple experiments evaluation, for example, the following line will generate 100 video samples for all trained movies:
```shell
python eval_video.py --num-samples 100 --exp-dir run/**/*/experiment_0
```
results are saved under ```run/**/*/experiment_0/eval```

In order to extract gifs and images, use the ```extract_*.py``` files similarly:
```shell
python eval_video.py --max-samples 4 --exp-dir run/**/*/experiment_0/eval
```
results are saved under ```run/**/*/experiment_0/eval/gifs(images)```.