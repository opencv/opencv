.. _iOS-Installation:

Installation in iOS
***********************

Required packages
==================

  * CMake 2.8.8 or higher
  * Xcode 4.3 or higher

Getting the cutting-edge OpenCV from SourceForge SVN repository
-----------------------------------------------------------------

Launch SVN client and checkout the current OpenCV snapshot from here: http://code.opencv.org/svn/opencv/trunk/opencv

In MacOS it can be done using the following command in Terminal:

.. code-block:: bash

   cd ~/<my_working _directory>
   svn co http://code.opencv.org/svn/opencv/trunk/opencv  
 

Building OpenCV from source using CMake, using the command line
================================================================

#. Make symbolic link for Xcode to let OpenCV build scripts find the compiler, header files etc.

    .. code-block:: bash
    
       cd /
       sudo ln -s /Applications/Xcode.app/Contents/Developer Developer
       
#. Build OpenCV framework

    .. code-block:: bash
    
       cd ~/<my_working_directory>
       python opencv/ios/build_framework.py ios
       
If everything's fine, after a few minutes you will get ~/<my_working_directory>/ios/opencv2.framework. You can add this framework to your Xcode projects.
