android-jni

this is an example of an android library project that has some reusable
code that exposes part of OpenCV to android. In particular this provides a
native camera interface for loading live video frames from the android camera
into native opencv functions(as cv::Mat's)

pre-reqs:
* build the opencv/android libraries - up one directory
* you need swig in you path for android-jni
    on ubuntu - sudo apt-get install swig
    others: http://www.swig.org/
   
to build:

make

that should work...  If it doesn't make sure to edit the generated local.env.mk
to reflect your machine's setup

see the sample for how to use this in your own projects
    
    
