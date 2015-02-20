#!/bin/sh
cd `dirname $0`/..

mkdir -p build_android_service
cd build_android_service

cmake -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake -DANDROID_TOOLCHAIN_NAME="arm-linux-androideabi-4.6" -DANDROID_STL=stlport_static -DANDROID_STL_FORCE_FEATURES=OFF -DBUILD_ANDROID_SERVICE=ON -DANDROID_SOURCE_TREE=~/Projects/AndroidSource/ServiceStub/ $@ ../..
