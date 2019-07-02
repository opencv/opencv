#!/bin/bash

set -e

cmake -GNinja \
    -DOPENCV_PLUGIN_NAME=opencv_videoio_ffmpeg_ubuntu_$2 \
    -DOPENCV_PLUGIN_DESTINATION=$1 \
    -DCMAKE_BUILD_TYPE=$3 \
    /opencv/modules/videoio/misc/plugin_ffmpeg
ninja
