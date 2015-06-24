#!/bin/bash -ex
# This file is well tested on Ubuntu 14.04
# List of required packages for build see in docker/Dockerfile
# (preferred way is to use Docker to produce similar environment)
#
# This script may not work in MINGW installation on Windows
#
# Usage ${0} <build_dir>
# build directory should be created via download_src.sh script:
#
CURRENT_DIR=`pwd`
BUILD_DIR=${1:-.}

openh264_DIR=${BUILD_DIR}/openh264
if [ ! -d ${openh264_DIR} ]; then
  echo "OpenH264 source tree is not found"
  exit 1
fi
(
 cd ${openh264_DIR}
 make PREFIX=`pwd`/install install-headers
 mkdir -p ${openh264_DIR}/install/lib/pkgconfig
 cat >${openh264_DIR}/install/lib/pkgconfig/openh264.pc << EOF
prefix=${openh264_DIR}/install
includedir=\${prefix}/include

Name: OpenH264
Description: OpenH264 wrapper with dynamic loading
Version: 1.4
Libs:
Libs.private:
Cflags: -I${CURRENT_DIR}/openh264_wrapper -I${openh264_DIR}/install -I\${includedir}
EOF
)

FFMPEG_DIR=${BUILD_DIR}/ffmpeg
if [ ! -d ${FFMPEG_DIR} ]; then
  echo "FFMPEG source tree is not found"
  exit 1
fi

FFMPEG_x86_DIR=${BUILD_DIR}/ffmpeg_x86
FFMPEG_x86_64_DIR=${BUILD_DIR}/ffmpeg_x86_64
#[ -d ${FFMPEG_x86_DIR} ] ||
(
 cd ${FFMPEG_DIR}
 mkdir -p ${FFMPEG_x86_DIR}
 rsync -a ./ ${FFMPEG_x86_DIR} --exclude .git
 cd ${FFMPEG_x86_DIR}
 PKG_CONFIG_PATH=${openh264_DIR}/install/lib/pkgconfig ./configure --enable-cross-compile --arch=x86 --target-os=mingw32 --cross-prefix=i686-w64-mingw32- --pkg-config=pkg-config --enable-static --enable-w32threads --enable-libopenh264 --disable-filters --disable-bsfs --disable-hwaccels --disable-programs --disable-debug --prefix=`pwd`/install
 make -j5 install
)
#[ -d ${FFMPEG_x86_64_DIR} ] ||
(
 cd ${FFMPEG_DIR}
 mkdir -p ${FFMPEG_x86_64_DIR}
 rsync -a ./ ${FFMPEG_x86_64_DIR} --exclude .git
 cd ${FFMPEG_x86_64_DIR}
 PKG_CONFIG_PATH=${openh264_DIR}/install/lib/pkgconfig ./configure --enable-cross-compile --arch=x86_64 --target-os=mingw32 --cross-prefix=x86_64-w64-mingw32- --pkg-config=pkg-config --enable-static --enable-w32threads --enable-libopenh264 --disable-filters --disable-bsfs --disable-hwaccels --disable-programs --disable-debug --prefix=`pwd`/install
 make -j5 install
)

# Build OpenCV DLL
OPENCV_LOCATION=${BUILD_DIR}/opencv
if [ ! -d ${OPENCV_LOCATION} ]; then
  OPENCV_LOCATION=${CURRENT_DIR}/../..
fi
i686-w64-mingw32-gcc \
 -m32 -s -Wall -shared -o opencv_ffmpeg.dll -O2 -x c++ -I${FFMPEG_x86_DIR}/install/include -I${OPENCV_LOCATION}/modules/videoio/src ffopencv.c \
 -L${FFMPEG_x86_DIR}/install/lib -lavformat -lavcodec -lavdevice -lswscale -lavutil -lws2_32 -lswresample -static -static-libgcc -static-libstdc++ -Wl,-Bstatic
x86_64-w64-mingw32-gcc \
 -m64 -s -Wall -shared -o opencv_ffmpeg_64.dll -O2 -x c++ -I${FFMPEG_x86_64_DIR}/install/include -I${OPENCV_LOCATION}/modules/videoio/src ffopencv.c \
 -L${FFMPEG_x86_64_DIR}/install/lib -lavformat -lavcodec -lavdevice -lswscale -lavutil -lws2_32 -lswresample -static -static-libgcc -static-libstdc++ -Wl,-Bstatic
