#!/usr/bin/env bash

if [ -d build ]; then
	echo "build directory exists"
else
	echo "Creating the build directory"
	mkdir build
fi

cd build

cmake -G "Unix Makefiles" ..

make -j8
