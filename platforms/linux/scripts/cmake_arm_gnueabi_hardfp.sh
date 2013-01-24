#!/bin/sh
cd `dirname $0`/..

mkdir -p build_hardfp
cd build_hardfp

cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi-hardfp.toolchain.cmake $@ ../../..

