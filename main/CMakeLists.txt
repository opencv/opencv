cmake_minimum_required(VERSION 3.10)

project(opencv_pca_umat_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "${CMAKE_CURRENT_LIST_DIR}/../build")
endif()

find_package(OpenCV REQUIRED COMPONENTS core)

add_executable(opencv_pca_umat_repro
    main.cpp
)

target_link_libraries(opencv_pca_umat_repro
    PRIVATE
    opencv_core
)
