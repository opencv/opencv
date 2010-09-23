Author: Ethan Rublee
email: ethan.rublee@gmail.com

  
To build with cmake:

mkdir build
cd build
cmake ..
make



Make sure to set the path in the cache for the crystax ndk available 
here:
   http://www.crystax.net/android/ndk-r4.php
   
   
to include in an android project -
just include the generated android-opencv.mk in you android ndk project 
(in an Android.mk file)
with:

include android-opencv.mk

this defines OPENCV_INCLUDES and OPENCV_LIBS - which you should add to your
makefiles like:

#define OPENCV_INCLUDES and OPENCV_LIBS
include $(PATH_TO_OPENCV_ANDROID_BUILD)/android-opencv.mk

LOCAL_LDLIBS += $(OPENCV_LIBS)
    
LOCAL_C_INCLUDES +=  $(OPENCV_INCLUDES)

for now, you also need to cd to android-jni and run make
this will create the android shared library with some useful functionality
that may be reused in android projects.

