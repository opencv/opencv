#!/bin/sh
cd `dirname $0`/..

mkdir -p build_hardfp
cd build_hardfp

cmake -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake $@ ../../..

