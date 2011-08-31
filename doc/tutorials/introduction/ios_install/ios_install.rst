.. _iOS-Installation:

Installation in iOS
***********************

Required packages
==================

  * GCC 4.x or later
  * CMake 2.8 or higher
  * Xcode 4.0 or higher

Getting the cutting-edge OpenCV from SourceForge SVN repository
-----------------------------------------------------------------

Launch SVN client and checkout either

a. the current OpenCV snapshot from here: https://code.ros.org/svn/opencv/trunk

#. or the latest tested OpenCV snapshot from here: http://code.ros.org/svn/opencv/tags/latest_tested_snapshot

In MacOS it can be done using the following command in Terminal:

.. code-block:: bash

   cd ~/<my_working _directory>
   svn co https://code.ros.org/svn/opencv/trunk  
 

Building OpenCV from source using CMake, using the command line
================================================================

#. Create a temporary directory, which we denote as <cmake_binary_dir>, where you want to put the generated Makefiles, project files as well the object filees and output binaries

#. Enter the <cmake_binary_dir> and type

   .. code-block:: bash
     
      cmake [<some optional parameters>] <path to the OpenCV source directory>

   For example

   .. code-block:: bash
       
      cd ~/opencv
      cd ..
      mkdir release
      cd release
      cmake -GXcode -DCMAKE_TOOLCHAIN_FILE=../opencv/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake -DCMAKE_INSTALL_PREFIX=../OpenCV_iPhoneOS -DCMAKE_BUILD_TYPE=RELEASE ../opencv


#. Enter the created temporary directory (<cmake_binary_dir>) and proceed with:

   .. code-block:: bash
      
      xcodebuild -sdk iphoneos -configuration Release -target ALL_BUILD
      xcodebuild -sdk iphoneos -configuration Release -target install install

