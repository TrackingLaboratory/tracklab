FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include
RUN echo "hey"
RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-get update
RUN apt-get install -y build-essential \
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
    mesa-utils and libgl1-mesa-glx x11-apps eog \
    vim tmux curl

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
RUN conda install -c anaconda h5py==3.6.0
RUN conda install conda==4.11.0
RUN conda install -c conda-forge opencv==4.5.5

RUN pip install cython==0.29.27 munkres==1.1.4 seaborn==0.11.2 networkx==2.6.3
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
RUN pip install scikit-learn==1.0.2 scipy==1.8.0 matplotlib==3.5.1
RUN pip install tensorboard==2.8.0
RUN pip install sacred gdown==3.12.2
RUN pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip
RUN pip install kornia==0.1.4
RUN echo "DONE"


