#!/bin/sh
cd `dirname $0`/..

mkdir -p build_linux_arm_softfp
cd build_linux_arm_softfp

cmake -DSOFTFP=ON -DCMAKE_TOOLCHAIN_FILE=../linux/arm-gnueabi.toolchain.cmake $@ ../..
