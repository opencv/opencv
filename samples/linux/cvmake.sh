#!/bin/bash

app=`echo $1 | awk -F "." '{ print $1 }'`;

if [ -f CMakeLists.txt ]
then
    mv CMakeLists.txt CMakeLists.txt.old
    echo Old CMakeLists.txt has renamed as CMakeLists.txt.old
    rm CMakeLists.txt
fi
    touch CMakeLists.txt
    echo -e "cmake_minimum_required(VERSION 2.6)" >> CMakeLists.txt
    echo -e "project( $app )" >> CMakeLists.txt
    echo -e "find_package( OpenCV REQUIRED )" >> CMakeLists.txt
    echo -e "add_executable( $app $app.cpp )" >> CMakeLists.txt
    echo -e "target_link_libraries( $app \${OpenCV_LIBS} )" >> CMakeLists.txt
echo CMakeLists.txt has been created!
mkdir build
cd build
cmake ..
make
cp $app ..
cd ..

