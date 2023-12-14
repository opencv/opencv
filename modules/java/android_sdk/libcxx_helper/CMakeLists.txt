cmake_minimum_required(VERSION 3.6)

project(opencv_java4)

# dummy target to bring libc++_shared.so into packages
add_library(opencv_java4 STATIC dummy.cpp)
