#!/bin/sh
cd `dirname $0`/..

mkdir -p build_android_mips
cd build_android_mips

cmake -DANDROID_ABI=mips -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
