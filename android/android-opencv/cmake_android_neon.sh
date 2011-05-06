#!/bin/sh
cd `dirname $0`

opencv_build_dir=`pwd`/../build_neon
mkdir -p build_neon
cd build_neon

cmake -DOpenCVDIR=$opencv_build_dir -DARM_TARGET="armeabi-v7a with NEON" -DCMAKE_TOOLCHAIN_FILE=../../android.toolchain.cmake ..
