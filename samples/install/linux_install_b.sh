#!/bin/bash
# This file contains documentation snippets for Linux installation tutorial
if [ "$1" = "--check" ] ; then
sudo()
{
    command $@
}
fi

sudo apt update

# [clang]
sudo apt install -y clang
# [clang]

# [ninja]
sudo apt install -y ninja-build
# [ninja]

# [cmake]
sudo apt install -y cmake
# [cmake]

# [git]
sudo apt install -y git
# [git]

# [download]
git clone https://github.com/opencv/opencv.git
git -C opencv checkout 4.x
# [download]

# [prepare]
mkdir -p build && cd build
# [prepare]

# [configure]
cmake -GNinja ../opencv
# [configure]

# [build]
ninja
# [build]

# [install]
sudo ninja install
# [install]
