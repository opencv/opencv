#!/bin/bash -ex
cd `dirname $0`/..

for i in "$@"
do
case $i in
    clean)
    rm -rf build
    shift
    ;;
    -a=*|--abi=*)
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

function build_target ()
# $1 = target cmake directory
# $2 = cmake build type
{
  local REBUILD_CMAKE=
  [ "$3" == "arm7" ] && TARGET_ABI="armeabi-v7a-hard with NEON"
  [ "$3" == "arm8" ] && TARGET_ABI="arm64-v8a"
  [ ! -d "$1" ] && mkdir -p "$1" && REBUILD_CMAKE=true
  pushd "$1"
  if [ -n "$REBUILD_CMAKE" ] ; then
    cmake '-GUnix Makefiles' -DCMAKE_BUILD_TYPE=$2 -DANDROID_ABI="$TARGET_ABI" $EXTRA_OPTIONS $COMMON_OPTIONS ../../../../../..
  fi
  make -j8
  #cmake -DCOMPONENT=libs -P cmake_install.cmake
  #cmake -DCOMPONENT=dev -P cmake_install.cmake
  #cmake -DCOMPONENT=java -P cmake_install.cmake
  #cmake -DCOMPONENT=samples -P cmake_install.cmake
  if [ "$2" == "Debug" ] ; then
    make install
  else
    make install/strip
  fi
  popd
  #cp -av install/sdk/native/3rdparty/libs/* ../build_product_release/opencv/src/main/jniLibs
  #cp -av install/sdk/native/libs/* ../build_product_release/opencv/src/main/jniLibs
  #cp -av install/sdk/native/jni ../build_product_release/opencv/src/main
}

# valid ABIs = arm7,arm8
function build_abi ()
{
  for i in "$@"
  do
  case $i in
    arm7)
    EXTRA_OPTIONS="-DWITH_OPENCL=OFF -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9"
    build_target "build/android/debug/arm7/opencv-library" Debug $i

    EXTRA_OPTIONS="-DWITH_OPENCL=OFF -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9"
    build_target "build/android/release/arm7/opencv-library" Release $i
    shift
    ;;
    arm8)
    EXTRA_OPTIONS="-DWITH_OPENCL=ON -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9"
    build_target "build/android/debug/arm8/opencv-library" Debug $i

    EXTRA_OPTIONS="-DWITH_OPENCL=ON -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9"
    build_target "build/android/release/arm8/opencv-library" Release $i
    shift
    ;;
  esac
done
}

#    ABI("1", "armeabi",     "arm-linux-androideabi-4.8"),
#    ABI("2",  "armeabi-v7a", "arm-linux-androideabi-4.8", cmake_name="armeabi-v7a with NEON"),
#    ABI("3",  "arm64-v8a",   "aarch64-linux-android-4.9")
#    ABI("5", "x86_64",      "x86_64-4.9"),
#    ABI("4", "x86",         "x86-4.8"),
#    ABI("7", "mips64",      "mips64el-linux-android-4.9"),
#    ABI("6", "mips",        "mipsel-linux-android-4.8")

COMMON_OPTIONS="-DENABLE_NEON=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_CUDA=OFF\
 -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_ANDROID_EXAMPLES=OFF\
 -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF\
 -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 -DNDK_CCACHE=ccache -DANDROID_STL=gnustl_static\
 -DCMAKE_TOOLCHAIN_FILE=../../../../../android/android.toolchain.cmake"

build_abi arm7 arm8