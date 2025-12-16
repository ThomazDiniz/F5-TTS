FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ARG DEBIAN_FRONTEND=noninteractive

ENV SHELL=/bin/bash
ENV TRANSFORMERS_NO_TORCHCODEC=1

RUN set -x \
  && apt-get update \
  && apt-get install -y \
    wget curl man git less openssl libssl-dev unzip unar \
    build-essential aria2 tmux vim \
    openssh-server sox libsox-fmt-all libsox-fmt-mp3 \
    libsndfile1-dev ffmpeg \
    librdmacm1 libibumad3 librdmacm-dev \
    libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/firstpixel/F5-TTS.git \
  && cd F5-TTS \
  && git submodule update --init --recursive \
  && pip install -U pip \
  && pip install -U "gradio==4.44.1" \
  && pip install -e . --no-cache-dir \
  && pip uninstall -y torchcodec || true

EXPOSE 7860
WORKDIR /workspace/F5-TTS
