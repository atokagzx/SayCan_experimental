FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y    apt-utils \
                                    lsb-release \
                                    mesa-utils \
                                    gnupg2 \
                                    net-tools \
                                    iputils-ping \
                                    build-essential \
                                    wget \
                                    unzip \
                                    curl \
                                    git \
                                    mc \
                                    vim \
                                    nano \
                                    ffmpeg \
                                    x11-apps

## GUI tunnel from docker
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"


RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*
# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt update && apt install -y --allow-change-held-packages libcudnn8

RUN apt update &&  apt install -y make \
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
                                libgdbm-dev \
                                libnss3-dev \
                                libedit-dev \
                                libc6-dev \
                                cmake \
                                unzip \
                                autoconf \
                                autogen \
                                automake \
                                libtool \
                                mlocate

RUN apt update && apt install -y python3-pip python3-dev && pip3 install --upgrade setuptools pip
RUN apt update && apt install -y python3-tk
ENV DEBIAN_FRONTEND=noninteractive


# Update, install
ENV PY_VERSION='3.9'
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt update && apt install -y build-essential python${PY_VERSION} git python3-pip python3-distutils 
RUN python${PY_VERSION} -m pip install --upgrade pip setuptools wheel
RUN python${PY_VERSION} -m pip install --upgrade numpy \
                                                scipy \
                                                matplotlib \
                                                pandas \
                                                scikit-learn \
                                                seaborn \
                                                bitsandbytes

RUN python${PY_VERSION} -m pip install --upgrade datasets \
                                                loralib \
                                                sentencepiece \
                                                git+https://github.com/zphang/transformers@c3dc391 \
                                                git+https://github.com/huggingface/peft.git
                                                # simple_ai_server

RUN python${PY_VERSION} -m pip install  git+https://github.com/atokagzx/simpleAI_fixed_response.git
WORKDIR /workspace
ENTRYPOINT python${PY_VERSION} -m simple_ai serve --port 8080
