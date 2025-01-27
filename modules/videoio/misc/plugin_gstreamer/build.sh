#!/bin/bash

set -e

cmake -GNinja \
    -DOPENCV_PLUGIN_NAME=opencv_videoio_gstreamer \
    -DOPENCV_PLUGIN_DESTINATION=$1 \
    -DCMAKE_BUILD_TYPE=$2 \
    /opencv/modules/videoio/misc/plugin_gstreamer

ninja
