#!/bin/sh
cd `dirname $0`/..

mkdir -p build_android_arm_hard
cd build_android_arm_hard

cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DARMEABI_HARD=ON -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
