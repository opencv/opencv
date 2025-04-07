#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

OPENCV_PLUGIN_DESTINATION=$1
OPENCV_PLUGIN_NAME=opencv_highgui_$2
CMAKE_BUILD_TYPE=${3:-Release}

shift 3 || true

set -x
cmake -GNinja \
    -DOPENCV_PLUGIN_NAME=${OPENCV_PLUGIN_NAME} \
    -DOPENCV_PLUGIN_DESTINATION=${OPENCV_PLUGIN_DESTINATION} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    "$@" \
    $DIR

ninja -v
