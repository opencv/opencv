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

If you only support armeabi-v7a or armeabi your final apks will be much smaller.

To build the class files, either start a new Android project from existing sources
in eclipse
or from the commmand line:
sh project_create.sh
ant debug

This should be linked to in your android projects, if you would like to reuse the
code. See Calibration or CVCamera in the opencv/android/apps directory

With cdt installed in eclipse, you may also "convert to C++ project" once you have
opened this as an android project. Select makefile project->toolchain other to do this.

Eclipse tip of the day:
You may get build warnings when linking to the project, complainging about duplicate something
or other in you .svn directories.  Right click project->settings->java build path->source->excude paths->add
.svn/ and **/.svn/ should do it ;)
    
