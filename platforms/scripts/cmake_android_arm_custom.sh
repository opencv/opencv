#!/bin/sh
cd `dirname $0`/..

[ -d build_product_debug ] && rm -rf build_product_debug
[ -d build_product_release ] && rm -rf build_product_release
mkdir -p build_product/opencv/src/main/jniLibs

if [ -d build_android_arm_neon_debug ]; then
        rm -rf build_android_arm_neon_debug
fi
mkdir -p build_android_arm_neon_debug
pushd build_android_arm_neon_debug
cmake -DANDROID_ABI="armeabi-v7a-hard with NEON" -DENABLE_NEON=ON -DENABLE_VFPV3=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_OPENCL=OFF \
  -DWITH_CUDA=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_ANDROID_EXAMPLES=OFF -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
  -DANDROID_STL=gnustl_static -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 \
  -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
make -j8
make install
cp -av install/sdk/native/3rdparty/libs/* ../build_product_debug/opencv/src/main/jniLibs
cp -av install/sdk/native/libs/* ../build_product_debug/opencv/src/main/jniLibs
cp -av install/sdk/native/jni ../build_product_debug/opencv/src/main
popd

if [ -d build_android_arm64_debug ]; then
        rm -rf build_android_arm64_debug
fi
mkdir -p build_android_arm64_debug
pushd build_android_arm64_debug
cmake -DANDROID_ABI="arm64-v8a" -DENABLE_NEON=ON -DENABLE_VFPV3=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_OPENCL=OFF \
  -DWITH_CUDA=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_ANDROID_EXAMPLES=OFF -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
  -DANDROID_STL=gnustl_static -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 \
  -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
make -j8
make install
cp -av install/sdk/native/3rdparty/libs/* ../build_product_debug/opencv/src/main/jniLibs
cp -av install/sdk/native/libs/* ../build_product_debug/opencv/src/main/jniLibs
cp -av install/sdk/native/jni ../build_product_debug/opencv/src/main
popd

if [ -d build_android_arm_neon_release ]; then
        rm -rf build_android_arm_neon_release
fi
mkdir -p build_android_arm_neon_release
pushd build_android_arm_neon_release
cmake -DANDROID_ABI="armeabi-v7a-hard with NEON" -DENABLE_NEON=ON -DENABLE_VFPV3=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_OPENCL=ON \
  -DWITH_CUDA=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ANDROID_EXAMPLES=OFF -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
  -DANDROID_STL=gnustl_static -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 \
  -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
make -j8
make install/strip
cp -av install/sdk/native/3rdparty/libs/* ../build_product_release/opencv/src/main/jniLibs
cp -av install/sdk/native/libs/* ../build_product_release/opencv/src/main/jniLibs
cp -av install/sdk/native/jni ../build_product_release/opencv/src/main
popd

if [ -d build_android_arm64_release ]; then
        rm -rf build_android_arm64_release
fi
mkdir -p build_android_arm64_release
pushd build_android_arm64_release
cmake -DANDROID_ABI="arm64-v8a" -DENABLE_NEON=ON -DENABLE_VFPV3=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_OPENCL=ON \
  -DWITH_CUDA=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ANDROID_EXAMPLES=OFF -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF \
  -DANDROID_STL=gnustl_static -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 \
  -DCMAKE_TOOLCHAIN_FILE=../android/android.toolchain.cmake $@ ../..
make -j8
make install/strip
cp -av install/sdk/native/3rdparty/libs/* ../build_product_release/opencv/src/main/jniLibs
cp -av install/sdk/native/libs/* ../build_product_release/opencv/src/main/jniLibs
cp -av install/sdk/native/jni ../build_product_release/opencv/src/main
popd
