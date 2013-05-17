#!/bin/sh
cd `dirname $0`/..

mkdir -p build_debug
cd build_debug

cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake $@ ../../..

