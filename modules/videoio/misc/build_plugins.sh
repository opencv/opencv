#!/bin/bash

set -e

if [ -z $1 ] ; then
    echo "$0 <destination directory>"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OCV="$( cd "${DIR}/../../.." >/dev/null 2>&1 && pwd )"
mkdir -p "${1}"  # Docker creates non-existed mounts with 'root' owner, lets ensure that dir exists under the current user to avoid "Permission denied" problem
DST="$( cd "$1" >/dev/null 2>&1 && pwd )"
CFG=$2

do_build()
{
TAG=$1
D=$2
F=$3
shift 3
docker build \
    --build-arg http_proxy \
    --build-arg https_proxy \
    $@ \
    -t $TAG \
    -f "${D}/${F}" \
    "${D}"
}

do_run()
{
TAG=$1
shift 1
docker run \
    -it \
    --rm \
    -v "${OCV}":/opencv:ro \
    -v "${DST}":/dst \
    -e CFG=$CFG \
    --user $(id -u):$(id -g) \
    $TAG \
    $@
}

build_gstreamer()
{
TAG=opencv_gstreamer_builder
do_build $TAG "${DIR}/plugin_gstreamer" Dockerfile
do_run $TAG /opencv/modules/videoio/misc/plugin_gstreamer/build.sh /dst $CFG
}

build_ffmpeg_ubuntu()
{
VER=$1
TAG=opencv_ffmpeg_ubuntu_builder:${VER}
do_build $TAG "${DIR}/plugin_ffmpeg" Dockerfile-ubuntu --build-arg VER=${VER}
do_run $TAG /opencv/modules/videoio/misc/plugin_ffmpeg/build-ubuntu.sh /dst ${VER} ${CFG}
}

build_ffmpeg()
{
VER=$1
TAG=opencv_ffmpeg_builder:${VER}
ARCHIVE="${DIR}/plugin_ffmpeg/ffmpeg-${VER}.tar.xz"
if [ ! -f "${ARCHIVE}" ] ; then
    wget https://www.ffmpeg.org/releases/ffmpeg-${VER}.tar.xz -O "${ARCHIVE}"
fi
do_build $TAG "${DIR}/plugin_ffmpeg" Dockerfile-ffmpeg --build-arg VER=${VER}
do_run $TAG /opencv/modules/videoio/misc/plugin_ffmpeg/build-standalone.sh /dst ${VER} ${CFG}
}

echo "OpenCV: ${OCV}"
echo "Destination: ${DST}"

build_gstreamer
build_ffmpeg_ubuntu 18.04
build_ffmpeg_ubuntu 16.04
build_ffmpeg 4.1
build_ffmpeg 3.4.5
build_ffmpeg 2.8.15
