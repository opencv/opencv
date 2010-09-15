#summary OpenCV build instructions
#labels Featured

Using the NDK, OpenCV may be built for the android platform.

The opencv port is in svn/trunk/opencv.  It is essentially a snapshot of the opencv trunk - rev 3096.  In the future this will be made compatible with trunk, but for 
simplicity we are freezing opencv until this works consistantly.

= pre-req =

use crystax ndk 4 - http://crystax.net/android/ndk.php

the crystax ndk supports the STL, expections, RTTI. opencv will not build with the standard android ndk!


= The way =
Using {{{r4}}} of the ndk, cd to the top level of opencv, and run 

{{{ndk-build NDK_APPLICATION_MK=Application.mk}}}

or build using the build.sh, which is just that line above...

Assuming the ndk directory is in your path of course.

this has advantages because it stores the library locally in the opencv folder. This is now the preferred method for building opencv.  The libraries will all be build as static libs which may be linked to from an external ndk project(see samples).

== Using opencv in your applications ==
See the samples directory.

Two convenience makefiles have been created, one for the libraries and one for includes.  They expect OPENCV_ROOT and OPENCV_LIBS_DIR
to be defined before including them.

A sample Android.mk file for CVCamera follows, that requires opencv
{{{
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#pass in OPENCV_ROOT or define it here
#OPENCV_ROOT := ~/android-opencv/opencv
#OPENCV_LIBS_DIR := $(OPENCV_ROOT)/bin/ndk/local/armeabi

#define OPENCV_INCLUDES
include $(OPENCV_ROOT)/includes.mk
#define OPENCV_LIBS
include $(OPENCV_ROOT)/libs.mk

LOCAL_LDLIBS += $(OPENCV_LIBS) -llog
    
LOCAL_C_INCLUDES +=  $(OPENCV_INCLUDES) 

LOCAL_MODULE    := cvcamera

LOCAL_SRC_FILES := android_cv_wrap.cpp image_pool.cpp yuv420sp2rgb.c Processor.cpp

include $(BUILD_SHARED_LIBRARY)
}}}


= old way (not supported) =
in {{{<ndk-dir>/apps}}} make a link to opencv
{{{
cd android-ndk-r4-crystax/apps
ln -s ~/android-opencv/opencv/
cd ..
make APP=opencv -j4
}}}
this should make everything as a static lib.  These libs will be located in the android-ndk-r4-crystax/out/apps/opencv/armeabi
folder.

now in  you ndk project do the following:


try building the samples/hello-jni project


a sample Android.mk:
{{{
LOCAL_PATH := $(call my-dir)
OpenCV_Root := apps/opencv

OpenCVInclude := $(OpenCV_Root)/include/opencv $(OpenCV_Root)/3rdparty/include/ \
    $(OpenCV_Root)/modules/core/include/ $(OpenCV_Root)/modules/highgui/include/ \
    $(OpenCV_Root)/modules/imgproc/include $(OpenCV_Root)/modules/ml/include \
    $(OpenCV_Root)/modules/features2d/include \
    $(OpenCV_Root)/modules/legacy/include \
    $(OpenCV_Root)/modules/calib3d/include \
    $(OpenCV_Root)/modules/objdetect/include \
    $(OpenCV_Root)/modules/contrib/include \
    $(OpenCV_Root)/modules/video/include

include $(CLEAR_VARS)



LOCAL_MODULE := my-project

LOCAL_SRC_FILES := test-opencv.cpp
    
LOCAL_LDLIBS :=  -L$(NDK_APP_OUT)/opencv/armeabi -lcalib3d -lfeatures2d \
    -lobjdetect -lvideo  -limgproc   -lhighgui -lcore -llegacy -lml -lopencv_lapack -lflann \
    -lzlib  -L$(SYSROOT)/usr/lib  -lstdc++ -lgcc -lsupc++ -lc -ldl
    
LOCAL_C_INCLUDES :=  $(OpenCVInclude) 

include $(BUILD_SHARED_LIBRARY)
}}}

The LOCAL_LDLIBS are very picky.  {{{-L$(NDK_APP_OUT)/opencv/armeabi}}} is where the ndk builds opencv, usually in {{{<ndk>/out/apps/opencv/armeabi}}}. You can navigate there and see the static libraries that were built.

test edit

