#!/bin/sh

mkdir -p build_carma
cd build_carma

cmake -DSOFTFP=ON -DCARMA=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DUSE_NEON=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/arm-linux-gnueabi/cuda/ \
-DCUDA_ARCH_BIN="2.1(2.0)" -DCUDA_ARCH_PTX="" -DCMAKE_SKIP_RPATH=ON -DWITH_CUDA=ON -DWITH_CUBLAS=ON \
-DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake $@ ../../..
