FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            libsm6 \
            libxext6 \
            libxrender-dev \
            python3 \
            python3-pip \
            vim \
            git &&\
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /workspace

COPY . /workspace/mmdetection/

WORKDIR /workspace/mmdetection

RUN pip3 install -r requirements.txt

# RUN pip3 install numpy && \
#       pip3 install mmcv && \
#       rm -rf mmcv

RUN git clone https://github.com/jsong0327/mmcv.git && \
      cd mmcv && \
      pip3 install . && \
      cd .. && \
      rm -rf mmcv

RUN cd /workspace/mmdetection && \
    PYTHON=python3 bash ./compile.sh && \
    pip3 install -e . && \
    cd ..

RUN mkdir /workspace/mmdetection/checkpoints && \
      cd /workspace/mmdetection/checkpoints && \
      wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth

RUN mkdir /.cache && chmod -R a+rw /.cache/

# WORKDIR /workspace
