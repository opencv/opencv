#!/bin/sh
cd `dirname $0`/..

mkdir -p build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake $@ ../..

