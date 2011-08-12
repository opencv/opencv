#!/bin/sh
cd `dirname $0`/..

mkdir -p build_neon
cd build_neon

cmake -C ../CMakeCache.android.initial.cmake -DARM_TARGET="armeabi-v7a with NEON" -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake $@ ../..

