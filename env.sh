conda create --name torch python=3.6 numpy Cython scipy h5py matplotlib tqdm ipython scikit-learn tensorboard ipython imageio scikit-image future colorama
conda activate torch
conda install pytorch==1.4.0 torchvision==0.2.2 cudatoolkit=10.1 -c pytorch
conda install -c menpo opencv
pip install kornia==0.2.0 moviepy pypng