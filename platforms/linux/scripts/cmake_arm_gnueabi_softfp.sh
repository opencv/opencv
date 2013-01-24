#!/bin/sh
cd `dirname $0`/..

mkdir -p build_softfp
cd build_softfp

cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi-softfp.toolchain.cmake $@ ../../..

