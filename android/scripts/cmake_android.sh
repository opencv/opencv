#!/bin/sh
cd `dirname $0`/..

mkdir -p build
cd build

cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake $@ ../..

