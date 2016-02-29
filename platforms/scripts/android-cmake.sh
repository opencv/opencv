#!/bin/bash -ex
cd `dirname $0`/..

for i in "$@"
do
case $i in
    clean)
    rm -rf build
    shift
    ;;
    -t=*|--targets=*)
    BUILD_ABIS="${i#*=}"
    shift
    ;;
    -l=*|--lib=*)
    LIBPATH="${i#*=}"
    shift
    ;;
    -h|--help|*)
    echo "`basename $0` - [clean] remake cmake files [-a=,--abi=] x68,arm7,arm7h,arm8,mips"
    exit 0
    ;;
esac
done

function copy_android_library ()
# $1 = target build directory
# $2 = debug/release variant
{
  pwd
  mkdir -p ../opencv/src/main/jniLibs
  cp -av install/sdk/native/3rdparty/libs/* ../opencv/src/main/jniLibs
  cp -av install/sdk/native/libs/* ../opencv/src/main/jniLibs
  cp -av install/sdk/native/jni ../opencv/src/main
  cp -av install/sdk/java ../opencv/src/main
  
  mkdir -p ../../opencv/src/main/$2/jniLibs
  cp -av ../opencv/src/main/jniLibs/* ../../opencv/src/main/$2/jniLibs
}

function build_target ()
# $1 = target cmake directory
# $2 = cmake build type
# $3 = target ABI
# $4 = target platform
{
  local TARGET_DIR=${1}
  local TARGET_CMAKE_TYPE=${2}
  local TARGET_ABI=${3}
  local TARGET_PLATFORM=${4}
  local REBUILD_CMAKE=
  [ "$TARGET_ABI" == "arm7" ] && TARGET_ABI="armeabi-v7a-hard with NEON"
  [ "$TARGET_ABI" == "arm8" ] && TARGET_ABI="arm64-v8a"
  [ "$TARGET_PLATFORM" == "osx" ] && TARGET_ABI="x86_64"
  [ ! -d "$TARGET_DIR" ] && mkdir -p "$TARGET_DIR" && REBUILD_CMAKE=true
  pushd "$TARGET_DIR"
  if [ -n "$REBUILD_CMAKE" ] ; then
    cmake '-GUnix Makefiles' -DCMAKE_BUILD_TYPE=$2 -DANDROID_ABI="$TARGET_ABI" $EXTRA_OPTIONS $COMMON_OPTIONS ../../../../..
  fi
  make -j8
  #cmake -DCOMPONENT=libs -P cmake_install.cmake
  #cmake -DCOMPONENT=dev -P cmake_install.cmake
  #cmake -DCOMPONENT=java -P cmake_install.cmake
  #cmake -DCOMPONENT=samples -P cmake_install.cmake
  if [ "$2" == "Debug" ] ; then
    make install
    [ "$TARGET_PLATFORM" == "android" ] && copy_android_library $TARGET_DIR debug
  else
    make install/strip
    [ "$TARGET_PLATFORM" == "android" ] && copy_android_library $TARGET_DIR release
  fi
  popd
}

# valid ABIs = arm7,arm8
function build_platform ()
{
  for i in "$@"
  do
    local TARGET_ABI=${i%-*}
    local TARGET_PLATFORM=${i#*-}
    echo "Building $1"
  case $i in
    arm7-android)
    COMMON_OPTIONS="-DENABLE_NEON=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_CUDA=OFF\
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_ANDROID_EXAMPLES=OFF\
     -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF\
     -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 -DNDK_CCACHE=ccache -DANDROID_STL=gnustl_static\
     -DCMAKE_TOOLCHAIN_FILE=../../../../android/android.toolchain.cmake"
    EXTRA_OPTIONS="-DWITH_OPENCL=OFF -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9"
    build_target "build/android/debug/arm7" Debug $TARGET_ABI $TARGET_PLATFORM
    EXTRA_OPTIONS="-DWITH_OPENCL=OFF -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9"
    build_target "build/android/release/arm7" Release $TARGET_ABI $TARGET_PLATFORM
    shift
    ;;
    arm8-android)
    COMMON_OPTIONS="-DENABLE_NEON=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_CUDA=OFF\
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_ANDROID_EXAMPLES=OFF\
     -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF\
     -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 -DNDK_CCACHE=ccache -DANDROID_STL=gnustl_static\
     -DCMAKE_TOOLCHAIN_FILE=../../../../android/android.toolchain.cmake"
    EXTRA_OPTIONS="-DWITH_OPENCL=ON -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9"
    build_target "build/android/debug/arm8" Debug $TARGET_ABI $TARGET_PLATFORM
    EXTRA_OPTIONS="-DWITH_OPENCL=ON -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9"
    build_target "build/android/release/arm8" Release $TARGET_ABI $TARGET_PLATFORM
    shift
    ;;
    x86_64-osx)
    COMMON_OPTIONS="-DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_CUDA=OFF\
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF\
     -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF"
    EXTRA_OPTIONS="-DWITH_OPENCL=ON"
    build_target "build/osx/debug/x86_64/opencv" Debug $TARGET_ABI $TARGET_PLATFORM
    EXTRA_OPTIONS="-DWITH_OPENCL=ON"
    build_target "build/osx/release/x86_64/opencv" Release $TARGET_ABI $TARGET_PLATFORM
    shift
    ;;
  esac
done
}

build_platform arm7-android arm8-android