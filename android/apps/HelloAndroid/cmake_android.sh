#!/bin/sh
cd `dirname $0`

BUILD_DIR=build_armeabi
opencv_android=`pwd`/../..
opencv_build_dir=$opencv_android/$BUILD_DIR

mkdir -p $BUILD_DIR
cd $BUILD_DIR

RUN_CMAKE="cmake -DOpenCV_DIR=$opencv_build_dir -DARM_TARGET=armeabi -DCMAKE_TOOLCHAIN_FILE=$opencv_android/android.toolchain.cmake .."
echo $RUN_CMAKE
$RUN_CMAKE
