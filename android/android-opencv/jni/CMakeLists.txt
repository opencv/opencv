#########################################################
# Find opencv and android-opencv
#########################################################

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../build 
    CACHE PATH "The path where you built opencv for android")
find_package(OpenCV REQUIRED)

#########################################################
#c flags, included, and lib dependencies
#########################################################
#notice the "recycling" of CMAKE_C_FLAGS
#this is necessary to pick up android flags
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -fPIC" )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} 
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

set( LIBRARY_DEPS ${OpenCV_LIBS} )
if(ANDROID)
  set( LIBRARY_DEPS ${LIBRARY_DEPS} log dl GLESv2)
endif(ANDROID)

#########################################################
#SWIG STUFF
#########################################################
#the java package to place swig generated java files in
set(MY_PACKAGE com.opencv.jni)

if(NOT ANDROID)
  #non android swig and jni
  #jni is available by default on android
  find_package(JNI REQUIRED)
  include_directories(${JNI_INCLUDE_DIRS})
  FIND_PACKAGE(SWIG)
endif()

INCLUDE(${SWIG_USE_FILE}) #on android this is found by the cmake toolchain

if(ANDROID)
  #this will set the output path for the java package
  #and properly create the package declarations in generated java sources
  SET_SWIG_JAVA_PACKAGE( ${MY_PACKAGE} ) #defined in the android toolchain
endif(ANDROID)

SET_SOURCE_FILES_PROPERTIES(android-cv.i PROPERTIES CPLUSPLUS ON)

SWIG_ADD_MODULE(android-opencv java 
  android-cv.i
  Calibration.cpp
  gl_code.cpp
  image_pool.cpp
  yuv420sp2rgb.c
  #yuv420rgb888c.c
  #yuv420rgb888.s
  yuv2rgb16tab.c
  )
  
target_link_libraries(android-opencv ${LIBRARY_DEPS} )

###################################################################
# Setup the configure file for other's to link against.
###################################################################
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_LIB_DIRS_CONFIGCMAKE ${LIBRARY_OUTPUT_PATH})
set(CMAKE_LIBS_CONFIGCMAKE android-opencv)
set(CMAKE_SWIG_DIR_CONFIGCMAKE ${CMAKE_CURRENT_SOURCE_DIR})
configure_file( "${CMAKE_SOURCE_DIR}/AndroidOpenCVConfig.cmake.in"
                "${CMAKE_BINARY_DIR}/AndroidOpenCVConfig.cmake"
                IMMEDIATE @ONLY)
