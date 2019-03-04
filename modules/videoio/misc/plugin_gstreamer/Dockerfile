FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer1.0-dev \
        cmake \
        g++ \
        ninja-build \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
