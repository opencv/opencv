#!/bin/bash

set -e

if [ -z $1 ] ; then
    echo "$0 <destination directory>"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OCV="$( cd "${DIR}/../../../.." >/dev/null 2>&1 && pwd )"
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

build_gtk2_ubuntu()
{
VER=$1
TAG=opencv_highgui_ubuntu_gtk2_builder:${VER}
do_build $TAG "${DIR}/plugin_gtk" Dockerfile-ubuntu-gtk2 --build-arg VER=${VER}
do_run $TAG /opencv/modules/highgui/misc/plugins/plugin_gtk/build.sh /dst gtk2_ubuntu${VER} ${CFG}

}

build_gtk3_ubuntu()
{
VER=$1
TAG=opencv_highgui_ubuntu_gtk3_builder:${VER}
do_build $TAG "${DIR}/plugin_gtk" Dockerfile-ubuntu-gtk3 --build-arg VER=${VER}
do_run $TAG /opencv/modules/highgui/misc/plugins/plugin_gtk/build.sh /dst gtk3_ubuntu${VER} ${CFG}
}

echo "OpenCV: ${OCV}"
echo "Destination: ${DST}"

build_gtk2_ubuntu 16.04
build_gtk2_ubuntu 18.04
build_gtk3_ubuntu 18.04
build_gtk3_ubuntu 20.04
