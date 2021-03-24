#!/usr/bin/env bash

# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html

RAW_PIXELS_SIZE=$1

echo "/*********************************"
echo "This file has been produced with the following invocation:"
echo ""
echo "    ./fill_random_image.sh <number> > raw_pixels.hpp"
echo "*********************************/"
echo ""
echo "#include <cstdint>"
echo "#ifndef RAW_PIXELS_HPP"
echo "#define RAW_PIXELS_HPP"
echo "#define RAW_PIXELS_SIZE $RAW_PIXELS_SIZE"

# Set the seed to guarantee always the same sequence is created for a
# given size combination. See
# https://stackoverflow.com/questions/42004870/seed-for-random-environment-variable-in-bash
RANDOM=314

echo "static std::uint32_t raw_pixels[$RAW_PIXELS_SIZE] = {"
for i in $(seq 1 $RAW_PIXELS_SIZE);
do
    printf "$RANDOM, "
done
echo "};"
echo "#endif /* RAW_PIXELS_HPP */"
