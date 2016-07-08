#!/bin/sh
cd `dirname $0`/..

mkdir -p build_linux_aarch64
cd build_linux_aarch64

cmake -DCMAKE_TOOLCHAIN_FILE=../linux/aarch64-gnu.toolchain.cmake $@ ../..
