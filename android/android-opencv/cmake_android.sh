#!/bin/sh
cd `dirname $0`

opencv_build_dir=`pwd`/../build
mkdir -p build
cd build

cmake -DOpenCVDIR=$opencv_build_dir -DCMAKE_TOOLCHAIN_FILE=../../android.toolchain.cmake ..

