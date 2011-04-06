Author: Ethan Rublee
email: ethan.rublee@gmail.com
########################################################
Prerequisites:
########################################################
   android-ndk-r5b http://developer.android.com/sdk/ndk/index.html
      the official ndk with standalone toolchain
   android-cmake http://code.google.com/p/android-cmake/
      this is for the cmake toolchain for android
   mercurial
    sudo apt-get install mercurial
   cmake
   opencv (you should have this if you're reading this file :)

########################################################   
Quick NDK Setup(ubuntu and bash):
########################################################
create some working directory:
  WORK=$HOME/android_dev
  cd $WORK

now get the android-cmake project with mercurial
  hg clone https://android-cmake.googlecode.com/hg/ android-cmake

there is a convenience script in there for pulling down and setting up the
android ndk as a standalone toolchain
  cd android-cmake/scripts
  ./get_ndk_toolchain_linux.sh $WORK

add the cmake toolchain location to your bashrc or otherwise export it to your env
  echo export ANDTOOLCHAIN=$WORK/android-cmake/toolchain/android.toolchain.cmake >> $HOME/.bashrc

########################################################
Quick opencv build(ubuntu and bash):
########################################################
Make sure you either source your bashrc or otherwise export the ANDTOOLCHAIN variable.

There is a script in the android folder for running cmake with the proper cache
variables set.  It is recommended that you use this to setup a smake build directory.
  cd opencv/android
  sh ./cmake_android.sh

You should now see a build directory, that is ready to be made.
  cd build
  make -j8

That will build most of the opencv modules, except for those that don't make sense
on android - gpu, etc..

To install to the toolchain:
  make install
########################################################
Using opencv in you're cmake own projects.
########################################################
Use the cmake find script for opencv:
  find_package(OpenCV REQUIRED)
  
Then when you run cmake, use:
  cmake -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ..

And direct your cmake cache for OpenCV_Dir to the path that you build opencv for android.
  something like : opencv/android/build

To avoid setting the cmake cache for OpenCV_Dir, you can just "install" opencv to your
android toolchain. Run the following from the opencv/android/build path:
  make install

########################################################
android targets
########################################################
You may wish to build android for multiple hardware targets.

Just change the cmake cache ARM_TARGETS to either:
 "armeabi" "armeab-v7a" "armeab-v7a with NEON"
 
You may install each of these to the toolchain, and they should be linked against
properly via way of the android-cmake toolchain.
