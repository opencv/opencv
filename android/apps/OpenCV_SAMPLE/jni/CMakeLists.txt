#########################################################
# Find opencv and android-opencv
#########################################################

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../build 
    CACHE PATH "The path where you built opencv for android")
set(AndroidOpenCV_DIR ${CMAKE_SOURCE_DIR}/../../android-opencv/build 
    CACHE PATH "The path where you built android-opencv")

find_package(OpenCV REQUIRED)
FIND_PACKAGE(AndroidOpenCV REQUIRED )

#########################################################
#c flags, included, and lib dependencies
#########################################################

#notice the "recycling" of CMAKE_C_FLAGS
#this is necessary to pick up android flags
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -fPIC" )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

set( LIBRARY_DEPS ${AndroidOpenCV_LIBS} ${OpenCV_LIBS} )
if(ANDROID)
  set( LIBRARY_DEPS ${LIBRARY_DEPS} log dl)
endif(ANDROID)

#########################################################
#SWIG STUFF
#########################################################
#the java package to place swig generated java files in
set(MY_PACKAGE com.OpenCV_SAMPLE.jni )
set(MY_MODULE OpenCV_SAMPLE )
set(MY_SWIG 
    OpenCV_SAMPLE.i #swig file
    )
set(MY_SRCS
    cvsample.cpp #cpp files can be compiled to
    )

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

#this add's the swig path for the opencv wrappers
SET(CMAKE_SWIG_FLAGS ${CMAKE_SWIG_FLAGS} "-I${AndroidOpenCV_SWIG_DIR}" )

SET_SOURCE_FILES_PROPERTIES(${MY_SWIG} PROPERTIES CPLUSPLUS ON)

#add the swig module, giving it the name, java, and then all of the source files
SWIG_ADD_MODULE(${MY_MODULE} java 
                ${MY_SWIG}
                ${MY_SRCS}
                )
#link the module like any other   
target_link_libraries(${MY_MODULE} ${LIBRARY_DEPS} )
