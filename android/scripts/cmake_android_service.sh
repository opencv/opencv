#!/bin/sh
cd `dirname $0`/..

mkdir -p build_service
cd build_service

cmake -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_TOOLCHAIN_NAME="arm-linux-androideabi-4.4.3" -DANDROID_STL=stlport_static -DANDROID_STL_FORCE_FEATURES=OFF -DBUILD_ANDROID_SERVICE=ON -DANDROID_SOURCE_TREE=~/Projects/AndroidSource/ServiceStub/ $@ ../..

