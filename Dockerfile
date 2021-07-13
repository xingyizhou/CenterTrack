# Dockerfile is based on https://github.com/xingyizhou/CenterTrack/pull/176
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
# Add sudo
RUN apt-get -y install sudo
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y --no-install-recommends python3.6

# Default to python3
RUN cd /usr/bin && ln -s python3.6 python
RUN apt-get install -y --no-install-recommends \
  bash \
  unzip \
  vim \
  build-essential \
  make \
  cmake \
  wget \
  curl \
  lv \
  less \
  git \
  imagemagick \
  ffmpeg \
  libfreetype6-dev \
  libpng-dev \
  libsm6 \
  libxext6 \
  libx11-xcb-dev \
  libglu1-mesa-dev \
  libxrender-dev \
  libzmq3-dev \
  python3.6-dev \
  python3-pip \
  python3.6-tk \
  pkg-config \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  llvm \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libffi-dev \
  liblzma-dev \
  protobuf-compiler

# pip3
RUN python3.6 -m pip install --upgrade \
  pip \
  setuptools

# python lib
RUN python3.6 -m pip install  --use-feature=2020-resolver \
  opencv-python \
  matplotlib \
  scipy \
  numba \
  imgaug \
  numpy \
  torch==1.4.0 \
  torchvision==0.5.0 \
  Pillow==6.2.1 \
  tqdm \
  motmetrics==1.1.3 \
  Cython \
  progress \
  easydict \
  pyquaternion \
  nuscenes-devkit \
  pyyaml \
  # for motmetrics
  pandas==0.25.3 \
  scikit-learn==0.22.2

RUN python3.6 -m pip install --use-feature=2020-resolver -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

COPY . /CenterTrack

# check cuda
RUN bash -c "/usr/bin/nvidia-smi"
RUN bash -c "python -c 'import torch; assert torch.cuda.is_available(), \"Cuda is not available.\"'"
WORKDIR /CenterTrack/src/lib/model/networks
RUN git clone --recursive https://github.com/CharlesShang/DCNv2
RUN cd DCNv2 && bash ./make.sh

# Install other dependencies
RUN cd /tmp && wget --quiet https://github.com/BurntSushi/ripgrep/releases/download/13.0.0/ripgrep_13.0.0_amd64.deb && dpkg -i ripgrep*.deb
RUN apt install dumb-init

WORKDIR /CenterTrack

# clean up
RUN sudo apt-get clean and; \
  sudo rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["echo","ready"]
