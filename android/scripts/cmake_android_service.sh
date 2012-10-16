#!/bin/sh
cd `dirname $0`/..

mkdir -p build_service
cd build_service

cmake -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_USE_STLPORT=ON -DBUILD_ANDROID_SERVICE=ON -DANDROID_SOURCE_TREE=~/Projects/AndroidSource/2.2.2/ $@ ../..

