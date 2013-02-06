#!/bin/sh
#
# Copyright (c) 2012, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

mkdir -p build_carma
cd build_carma

cmake -DCARMA=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DUSE_NEON=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/arm-linux-gnueabi/cuda/ \
-DCUDA_ARCH_BIN="2.1(2.0)" -DCUDA_ARCH_PTX="" -DCMAKE_SKIP_RPATH=ON -DWITH_CUDA=ON -DWITH_CUBLAS=ON \
-DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi-softfp.toolchain.cmake $@ ../../..
