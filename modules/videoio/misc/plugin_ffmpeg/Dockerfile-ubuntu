ARG VER
FROM ubuntu:$VER

RUN apt-get update && apt-get install -y \
        libavcodec-dev \
        libavfilter-dev \
        libavformat-dev \
        libavresample-dev \
        libavutil-dev \
        pkg-config \
        cmake \
        g++ \
        ninja-build \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
