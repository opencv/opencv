#!/bin/sh
cd `dirname $0`/..

mkdir -p build_linux_arm_hardfp
cd build_linux_arm_hardfp

cmake -DCMAKE_TOOLCHAIN_FILE=../linux/arm-gnueabi.toolchain.cmake $@ ../..
