conda create --name hpvaegan python=3.6 numpy Cython scipy h5py matplotlib tqdm ipython scikit-learn tensorboard ipython imageio scikit-image future colorama
conda activate hpvaegan
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
pip install kornia==0.4.0 moviepy pypng opencv-python