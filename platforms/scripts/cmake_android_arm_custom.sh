#!/bin/sh
cd `dirname $0`/..

if [ -d build_android_arm_neon ]; then
        rm -rf build_android_arm_neon
fi
mkdir -p build_android_arm_neon
pushd build_android_arm_neon
cmake -DANDROID_ABI="armeabi-v7a-hard with NEON" -DINSTALL_ANDROID_EXAMPLES=OFF -DENABLE_NEON=ON -DENABLE_VFPV3=ON \
        -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_OPENCL=ON -DWITH_CUDA=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
                -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
popd

if [ -d build_android_arm64 ]; then
        rm -rf build_android_arm64
fi
mkdir -p build_android_arm64
pushd build_android_arm64
cmake -DANDROID_ABI="arm64-v8a" -DINSTALL_ANDROID_EXAMPLES=OFF -DENABLE_NEON=ON -DENABLE_VFPV3=ON -DWITH_TBB=ON \
        -DBUILD_TBB=ON -DWITH_OPENCL=ON -DWITH_CUDA=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
                -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
popd
