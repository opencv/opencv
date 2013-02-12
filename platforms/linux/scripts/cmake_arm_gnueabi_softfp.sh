#!/bin/sh
cd `dirname $0`/..

mkdir -p build_softfp
cd build_softfp

cmake -DSOFTFP=ON -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake $@ ../../..
