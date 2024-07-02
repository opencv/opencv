#!/bin/bash

NAME=build
rm CMakeCache.txt # Remove prior program.
mkdir $NAME # Create a binary folder to store output
cd $NAME # Move to said output
clear
cmake .. -DOPENCV_EXTRA_MODULES_PATH=/home/comp3000/libopencv-vas-stuff/opencv_contrib/modules -DCMAKE_BUILD_TYPE=RELEASE -DWITH_OPENMP=ON -DCPU_BASELINE=AVX2
make
