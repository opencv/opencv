#!/bin/bash
# This file contains documentation snippets for Linux installation tutorial
if [ "$1" = "--check" ] ; then
sudo()
{
    command $@
}
fi

sudo apt update

# [gcc]
sudo apt install -y g++
# [gcc]

# [make]
sudo apt install -y make
# [make]

# [cmake]
sudo apt install -y cmake
# [cmake]

# [wget]
sudo apt install -y wget unzip
# [wget]

# [download]
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mv opencv-4.x opencv
# [download]

# [prepare]
mkdir -p build && cd build
# [prepare]

# [configure]
cmake ../opencv
# [configure]

# [build]
make -j4
# [build]

# [check]
ls bin
ls lib
# [check]

# [check cmake]
ls OpenCVConfig*.cmake
ls OpenCVModules.cmake
# [check cmake]

# [install]
sudo make install
# [install]

# [python-quick-check]
echo "Python quick check:"
python3 - <<'PY'
import cv2
print("OpenCV import OK, version:", cv2.__version__)
PY
# [python-quick-check]
