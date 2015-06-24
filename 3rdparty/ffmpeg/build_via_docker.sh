#!/bin/bash -e
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Build Docker image
docker build -t opencv_ffmpeg_mingw_build docker

if [ ! -d build ]; then
  echo "Build directory not found. Need to download 3rdparty sources..."
  ./download_src.sh
fi

echo "Running docker container:"
docker run --rm=true -it --name opencv_ffmpeg_mingw_build \
-e "APP_UID=$UID" -e APP_GID=$GROUPS \
-v $(pwd):/app -v $(pwd)/build:/build -v $(pwd)/../..:/build/opencv opencv_ffmpeg_mingw_build
