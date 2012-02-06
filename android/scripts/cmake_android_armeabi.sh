#!/bin/sh
cd `dirname $0`/..

mkdir -p build_armeabi
cd build_armeabi

cmake -DANDROID_ABI=armeabi -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake $@ ../..

