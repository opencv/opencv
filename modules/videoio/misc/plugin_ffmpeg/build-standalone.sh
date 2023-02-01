#!/bin/bash

set -e

mkdir -p build_shared && pushd build_shared
PKG_CONFIG_PATH=/ffmpeg-shared/lib/pkgconfig \
cmake -GNinja \
    -DOPENCV_PLUGIN_NAME=opencv_videoio_ffmpeg_shared_$2 \
    -DOPENCV_PLUGIN_DESTINATION=$1 \
    -DCMAKE_BUILD_TYPE=$3 \
    /opencv/modules/videoio/misc/plugin_ffmpeg
ninja
popd

mkdir -p build_static && pushd build_static
PKG_CONFIG_PATH=/ffmpeg-static/lib/pkgconfig \
cmake -GNinja \
    -DOPENCV_PLUGIN_NAME=opencv_videoio_ffmpeg_static_$2 \
    -DOPENCV_PLUGIN_DESTINATION=$1 \
    -DCMAKE_MODULE_LINKER_FLAGS=-Wl,-Bsymbolic \
    -DCMAKE_BUILD_TYPE=$3 \
    /opencv/modules/videoio/misc/plugin_ffmpeg
ninja
popd
