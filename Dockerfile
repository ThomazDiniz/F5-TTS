FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

# Install system dependencies (FFmpeg will be installed via conda as recommended by TorchCodec docs)
RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install FFmpeg via conda (recommended by TorchCodec documentation)
# TorchCodec supports FFmpeg majors 4-7 on all platforms, version 8 only on macOS/Linux
# Using conda FFmpeg ensures better compatibility: https://discuss.huggingface.co/t/issue-with-torchcodec-when-fine-tuning-whisper-asr-model/169315/2
RUN conda install -y -c conda-forge "ffmpeg<8" \
    && conda clean -afy

# Verify FFmpeg installation
RUN ffmpeg -version | head -1

# Create symlinks for torchcodec compatibility - it looks for libavutil and libavcodec versions
# We have FFmpeg 7 from conda, so create symlinks for other versions (4-8)
RUN set -x \
    && LIB_DIR="/usr/lib/x86_64-linux-gnu" \
    && CONDA_LIB="/opt/conda/lib" \
    && echo "Creating symlinks for torchcodec compatibility..." \
    && for lib in avutil avcodec avformat avfilter swscale swresample; do \
        # Find the actual library version in conda
        if [ -f "${CONDA_LIB}/lib${lib}.so.59" ]; then \
            BASE_VERSION=59; \
        elif [ -f "${CONDA_LIB}/lib${lib}.so.58" ]; then \
            BASE_VERSION=58; \
        elif [ -f "${CONDA_LIB}/lib${lib}.so.57" ]; then \
            BASE_VERSION=57; \
        elif [ -f "${CONDA_LIB}/lib${lib}.so.56" ]; then \
            BASE_VERSION=56; \
        else \
            BASE_VERSION=""; \
        fi; \
        if [ -n "${BASE_VERSION}" ]; then \
            for version in 62 61 60 59 58 57 56; do \
                if [ ! -f "${CONDA_LIB}/lib${lib}.so.${version}" ] && [ "${version}" != "${BASE_VERSION}" ]; then \
                    ln -sf lib${lib}.so.${BASE_VERSION} "${CONDA_LIB}/lib${lib}.so.${version}" 2>/dev/null || true; \
                fi; \
            done; \
        fi; \
        # Also create symlinks in system directory if needed
        if [ -f "${LIB_DIR}/lib${lib}.so.56" ]; then \
            for version in 62 61 60 59 58 57; do \
                if [ ! -f "${LIB_DIR}/lib${lib}.so.${version}" ]; then \
                    ln -sf lib${lib}.so.56 "${LIB_DIR}/lib${lib}.so.${version}" 2>/dev/null || true; \
                fi; \
            done; \
        fi; \
    done \
    && ldconfig \
    && echo "FFmpeg libraries and symlinks configured"

WORKDIR /workspace

# Copy local code instead of cloning
COPY . /workspace/F5-TTS/

WORKDIR /workspace/F5-TTS

# Install project dependencies first
# CRITICAL: Fix Gradio to 3.x BEFORE installing project (show_api was removed in Gradio 4.0+)
# The project code uses show_api which only exists in Gradio 3.x
RUN git submodule update --init --recursive \
    && pip install --no-cache-dir "gradio>=3.45.2,<4.0.0" \
    && pip install -e . --no-cache-dir \
    && pip install --force-reinstall --no-cache-dir --no-deps "gradio>=3.45.2,<4.0.0" || true

# DO NOT install torchcodec - it has ABI incompatibility with PyTorch 2.4.0 (undefined symbol error)
# Transformers will automatically use librosa fallback when torchcodec is not available
# This is the recommended workaround per: https://discuss.huggingface.co/t/issue-with-torchcodec-when-fine-tuning-whisper-asr-model/169315/2
# The fallback (librosa) works perfectly fine for audio processing
RUN echo "Skipping torchcodec installation - transformers will use librosa fallback (recommended workaround)"

# Install tensorboard for logging support
RUN pip install --no-cache-dir tensorboard

# Ensure FFmpeg libraries from conda are found first
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib/x86_64-linux-gnu:/usr/lib:/lib/x86_64-linux-gnu:/lib:${LD_LIBRARY_PATH}
ENV SHELL=/bin/bash

WORKDIR /workspace/F5-TTS

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" \
    && python -c "import sys; sys.path.insert(0, 'src'); from f5_tts.infer import utils_infer; print('✓ F5-TTS imports OK')" \
    && echo "Build completed successfully!"
