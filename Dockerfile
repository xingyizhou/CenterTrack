# Dockerfile is based on:
# - https://github.com/xingyizhou/CenterTrack/pull/176
# - https://github.com/fcwu/docker-ubuntu-vnc-desktop/blob/e4922ce92f945fc482994b7a0fd95ca5de7295b3/Dockerfile.amd64

FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 as system

################################################################################
# vnc desktop
################################################################################

# built-in packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common curl apache2-utils \
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        supervisor nginx sudo net-tools zenity xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# install debs error if combine together
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        xvfb x11vnc \
        vim-tiny firefox ttf-ubuntu-font-family ttf-wqy-zenhei  \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y gpg-agent \
    && curl -LO https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && (dpkg -i ./google-chrome-stable_current_amd64.deb || apt-get install -fy) \
    && curl -sSL https://dl.google.com/linux/linux_signing_key.pub | apt-key add \
    && rm google-chrome-stable_current_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        lxde gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*


# Additional packages require ~600MB
# libreoffice  pinta language-pack-zh-hant language-pack-gnome-zh-hant firefox-locale-zh-hant libreoffice-l10n-zh-tw

# tini to fix subreap
ARG TINI_VERSION=v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# ffmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /usr/local/ffmpeg \
    && ln -s /usr/bin/ffmpeg /usr/local/ffmpeg/ffmpeg

# python library
COPY docker-vnc/rootfs/usr/local/lib/web/backend/requirements.txt /tmp/
RUN apt-get update \
    && dpkg-query -W -f='${Package}\n' > /tmp/a.txt \
    && apt-get install -y python3-pip python3-dev build-essential \
	&& pip3 install setuptools wheel && pip3 install -r /tmp/requirements.txt \
    && dpkg-query -W -f='${Package}\n' > /tmp/b.txt \
    && apt-get remove -y `diff --changed-group-format='%>' --unchanged-group-format='' /tmp/a.txt /tmp/b.txt | xargs` \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/* /tmp/a.txt /tmp/b.txt


################################################################################
# CenterTrack
################################################################################

ENV apt-get update && apt-get install -y tzdata sudo apt-transport-https
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends python3.6

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

# python lib
RUN python3.6 -m pip install --upgrade \
  pip \
  setuptools
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
# cuda 11.1
#RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# cuda 10.2
#RUN pip3 install torch torchvision torchaudio

RUN python3.6 -m pip install --use-feature=2020-resolver -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# check cuda
RUN bash -c "/usr/bin/nvidia-smi"
RUN bash -c "python -c 'import torch; assert torch.cuda.is_available(), \"Cuda is not available.\"'"

RUN apt-get update && apt-get install -y ninja-build
RUN mkdir -p /models \
    && cd /models \
    && git clone --recursive https://github.com/CharlesShang/DCNv2 \
    && cd DCNv2 \
    && bash ./make.sh

# Install other dependencies
RUN cd /tmp && wget --quiet https://github.com/BurntSushi/ripgrep/releases/download/13.0.0/ripgrep_13.0.0_amd64.deb && dpkg -i ripgrep*.deb
RUN wget -q -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 && chmod +x /usr/local/bin/dumb-init
RUN python3.6 -m pip install gdown

################################################################################
# vnc desktop: builder
################################################################################
FROM ubuntu:20.04 as builder

RUN sed -i 's#http://archive.ubuntu.com/ubuntu/#mirror://mirrors.ubuntu.com/mirrors.txt#' /etc/apt/sources.list;

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg patch

# nodejs
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt-get install -y nodejs

# yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
    && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y yarn

# build frontend
COPY docker-vnc/web /src/web
RUN cd /src/web \
    && yarn \
    && yarn build
RUN sed -i 's#app/locale/#novnc/app/locale/#' /src/web/dist/static/novnc/app/ui.js

################################################################################
# vnc desktop: merge
################################################################################
FROM system

RUN apt-get install -y nomacs vlc

ENV CODE_SERVER_VERSION 3.11.0
RUN cd /tmp \
    && curl -fOL https://github.com/cdr/code-server/releases/download/v${CODE_SERVER_VERSION}/code-server_${CODE_SERVER_VERSION}_amd64.deb \
    && dpkg -i code-server_${CODE_SERVER_VERSION}_amd64.deb

RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && apt-get install -y nodejs
RUN apt-get update && apt-get install -y neovim

WORKDIR /root

RUN npm install argon2-cli

COPY --from=builder /src/web/dist/ /usr/local/lib/web/frontend/
COPY docker-vnc/rootfs /
COPY docker/rootfs /
RUN ln -sf /usr/local/lib/web/frontend/static/websockify /usr/local/lib/web/frontend/static/novnc/utils/websockify && \
    chmod +x /usr/local/lib/web/frontend/static/websockify/run

EXPOSE 80
ENV SHELL=/bin/bash
HEALTHCHECK --interval=30s --timeout=5s CMD curl --fail http://127.0.0.1:6079/api/health
ENTRYPOINT ["/init.sh"]

