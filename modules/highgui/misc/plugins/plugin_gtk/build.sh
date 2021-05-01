#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cmake -GNinja \
    -DOPENCV_PLUGIN_NAME=opencv_highgui_$2 \
    -DOPENCV_PLUGIN_DESTINATION=$1 \
    -DCMAKE_BUILD_TYPE=$3 \
    $DIR

ninja -v
