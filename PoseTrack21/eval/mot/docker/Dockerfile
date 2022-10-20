FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN apt-get update && apt-get install -y apt-file build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    gfortran \
    libjpeg8-dev libpng-dev \
    libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
    libxine2-dev libv4l-dev \
    liblmdb-dev libleveldb-dev libsnappy-dev \
    mesa-utils libgl1-mesa-glx x11-apps eog \
    vim tmux curl wget

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh
RUN echo "ASD"
RUN bash Miniconda3-4.7.12-Linux-x86_64.sh -b -p /opt/miniconda3
COPY environment.yml /root/environment.yml

RUN conda env update -f /root/environment.yml

