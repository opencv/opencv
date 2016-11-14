#!/bin/sh
cd `dirname $0`/..

mkdir -p build_android_arm
cd build_android_arm

cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DANDROID_NATIVE_API_LEVEL=9 -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
