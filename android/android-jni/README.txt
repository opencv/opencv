android-opencv

this is an example of an android library project that has some reusable
code that exposes part of OpenCV to android. In particular this provides a
native camera interface for loading live video frames from the android camera
into native opencv functions(as cv::Mat's)

to build make sure you have swig and the crystax ndk in your path

make

that should work...  If it doesn't make sure to edit the generated local.env.mk
the key variable

more later on how to build actual project for android 
    - see the code.google.com/p/android-opencv for details on this
    
    
