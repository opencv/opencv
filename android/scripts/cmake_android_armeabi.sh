#!/bin/sh
cd `dirname $0`/..

mkdir -p build_armeabi
cd build_armeabi

cmake -C ../CMakeCache.android.initial.cmake -DARM_TARGET=armeabi -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake $@ ../..

