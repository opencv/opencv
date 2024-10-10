#!/bin/bash
# This file contains documentation snippets for Linux installation tutorial
if [ "$1" = "--check" ] ; then
sudo()
{
    command $@
}
fi

# [body]
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip ninja-build

# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/5.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/5.x.zip
unzip opencv.zip
unzip opencv_contrib.zip

# Create build directory and switch into it
mkdir -p build && cd build

# Configure
cmake -GNinja -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-5.x/modules ../opencv-5.x

# Build
cmake --build .
# [body]
