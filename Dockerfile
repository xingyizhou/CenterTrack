FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Set proxies if required
# ENV https_proxy= 

RUN apt-get updateRUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
# Add sudo# RUN apt-get -y install sudo
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y --no-install-recommends python3.6

# Default to python3RUN cd /usr/bin && ln -s python3.6 python
RUN apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        imagemagick \
        ffmpeg \
        libfreetype6-dev \
        libpng-dev \
        libsm6 \
        libxext6 \
        libx11-xcb-dev \
        libglu1-mesa-dev \
        libxrender-dev \
        libzmq3-dev \
        python3.6-dev \
        python3-pip \
        python3.6-tk \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        wget

# pip3
RUN python3.6 -m pip install --upgrade \
    pip \
    setuptools

# python lib
RUN python3.6 -m pip install  --use-feature=2020-resolver \
    opencv-python \
    matplotlib \
    scipy \
    numba \
    imgaug \
    numpy \
    torch==1.4.0 \
    torchvision==0.5.0 \
    Pillow==6.2.1 \
    tqdm \
    motmetrics==1.1.3 \
    Cython \
    progress \
    easydict \
    pyquaternion \
    nuscenes-devkit \
    pyyaml \
    scikit-learn==0.22.2

RUN python3.6 -m pip install --use-feature=2020-resolver -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 

COPY . /centertrack

WORKDIR /centertrack/src/lib/model/networks/DCNv2RUN ./make.sh

WORKDIR /centertrack

# clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
