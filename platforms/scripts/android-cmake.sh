#!/bin/bash -ex

function main ()
{
  SCRIPT_FILEPATH="$(cd "$(dirname "$0")"; pwd)/$(basename "$0")"
  SCRIPT_PATH=`dirname $SCRIPT_FILEPATH`
  OPENCV_PATH=${SCRIPT_PATH%%\/platforms\/scripts}
  # defaults
  BUILD_ABIS="arm7-android arm8-android"
  BUILD_ROOT=$OPENCV_PATH
  INSTALL_PATH=$OPENCV_PATH/install

  # number of parallel jobs
  if [ "${TRAVIS}" == "true" -a "${CI}" == "true" ] ; then
    export BUILD_NUM_CORES=1
  else
    if [[ "$OSTYPE" == *darwin* ]] ; then
      export BUILD_NUM_CORES=`sysctl -n hw.ncpu`
    elif [[ "$OSTYPE" == *linux* ]] ; then
      export BUILD_NUM_CORES=`nproc`
    else
      export BUILD_NUM_CORES=1
    fi
  fi

  pushd $BUILD_ROOT
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
      -i=*)
      INSTALL_PATH="${i#*=}"
      shift
      ;;
      -h|--help|*)
      echo "`basename $0` [clean] remake cmake files [-t=,--targets=] x64-osx,arm7-android,arm8-android [-i=] install path"
      exit 0
      ;;
  esac
  done

  [ ! -d ${INSTALL_PATH} ] && mkdir -p ${INSTALL_PATH}
  [[ -n "${BUILD_ABIS}" ]] && build_platform ${BUILD_ABIS}
  popd
}

function install_android_library ()
# $1 install path
# $2 debug or release
{
  set +e
  pwd
  if [[ "$2" == "Debug" ]] ; then
    BUILD_TYPE_EXT=debug
  else
    BUILD_TYPE_EXT=release
  fi
  cp -av $BUILD_ROOT/platforms/android/template/opencv-lib/* $1
  cp -av lint.xml $1
  cp -av bin/aidl $1/src/main
  cp -av bin/AndroidManifest.xml $1/src/main
  #cp -av install/sdk/native/3rdparty/libs/* $1/src/main/jniLibs
  #mkdir -p $1/src/main/${BUILD_TYPE_EXT}/jnilibs
  #cp -av install/sdk/native/libs/ $1/src/main/${BUILD_TYPE_EXT}/jnilibs
  cp -av install/sdk/native/libs/ $1/src/main/jnilibs
  #mkdir -p $1/src/main/${BUILD_TYPE_EXT}/jni
  #cp -av install/sdk/native/jni/include $1/src/main/${BUILD_TYPE_EXT}/jni
  cp -av install/sdk/native/jni/include $1/src/main/jni
  #mkdir -p $1/src/main/${BUILD_TYPE_EXT}/java
  #cp -av install/sdk/java/src/ $1/src/main/${BUILD_TYPE_EXT}/java
  cp -av install/sdk/java/src/ $1/src/main/java
  cp -av install/sdk/java/res $1/src/main
  cp -av install/sdk/java/AndroidManifest.xml $1/src/main
  cp -av install/sdk/java/lint.xml $1
  set -e
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
    cmake '-GUnix Makefiles' -DCMAKE_BUILD_TYPE=$2 -DANDROID_ABI="$TARGET_ABI" $EXTRA_OPTIONS $COMMON_OPTIONS ${BUILD_ROOT}
  fi
  make -j${BUILD_NUM_CORES}
  #cmake -DCOMPONENT=libs -P cmake_install.cmake
  #cmake -DCOMPONENT=dev -P cmake_install.cmake
  #cmake -DCOMPONENT=java -P cmake_install.cmake
  #cmake -DCOMPONENT=samples -P cmake_install.cmake
  if [ "$2" == "Debug" ] ; then
    make install
  else
    make install/strip
  fi
  [ "$TARGET_PLATFORM" == "android" ] && install_android_library $INSTALL_PATH $2
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
     -DCMAKE_TOOLCHAIN_FILE=${BUILD_ROOT}/platforms/android/android.toolchain.cmake"
    EXTRA_OPTIONS="-DWITH_OPENCL=ON -DANDROID_TOOLCHAIN_NAME=arm-linux-androideabi-4.9"
    build_target "build/android/debug/arm7" Debug $TARGET_ABI $TARGET_PLATFORM
    build_target "build/android/release/arm7" Release $TARGET_ABI $TARGET_PLATFORM
    shift
    ;;
    arm8-android)
    COMMON_OPTIONS="-DENABLE_NEON=ON -DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_CUDA=OFF\
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_ANDROID_EXAMPLES=OFF\
     -DINSTALL_ANDROID_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF\
     -DANDROID_NATIVE_API_LEVEL=21 -DANDROID_SDK_TARGET=21 -DNDK_CCACHE=ccache -DANDROID_STL=gnustl_static\
     -DCMAKE_TOOLCHAIN_FILE=${BUILD_ROOT}/platforms/android/android.toolchain.cmake"
    EXTRA_OPTIONS="-DWITH_OPENCL=ON -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9"
    build_target "build/android/debug/arm8" Debug $TARGET_ABI $TARGET_PLATFORM
    build_target "build/android/release/arm8" Release $TARGET_ABI $TARGET_PLATFORM
    shift
    ;;
    x86_64-osx)
    COMMON_OPTIONS="-DWITH_TBB=ON -DBUILD_TBB=ON -DWITH_CUDA=OFF\
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DENABLE_PRECOMPILED_HEADERS=OFF\
     -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF"
    EXTRA_OPTIONS="-DWITH_OPENCL=OFF"
    build_target "build/osx/debug/x86_64/opencv" Debug $TARGET_ABI $TARGET_PLATFORM
    build_target "build/osx/release/x86_64/opencv" Release $TARGET_ABI $TARGET_PLATFORM
    shift
    ;;
  esac
  done
}

main "$@"
