#!/bin/sh

cd `dirname $0`/..

mkdir -p build_android_x86
cd build_android_x86

cmake -DANDROID_ABI=x86 -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
