.. _Linux-Installation:

Installation in Linux
***********************
These steps have been tested for Ubuntu 10.04 but should work with other distros.

Required packages
==================

  * GCC 4.x or later. This can be installed with

    .. code-block:: bash

       sudo apt-get install build-essential 
 
  * CMake 2.6 or higher
  * Subversion (SVN) client
  * GTK+2.x or higher, including headers
  * pkgconfig
  * libpng, zlib, libjpeg, libtiff, libjasper with development files (e.g. libpjeg-dev)
  * Python 2.3 or later with developer packages (e.g. python-dev)
  * SWIG 1.3.30 or later (only for versions prior to OpenCV 2.3)
  * libavcodec
  * libdc1394 2.x 

All the libraries above can be installed via Terminal or by using Synaptic Manager

Getting OpenCV source code 
============================

You can use the latest stable OpenCV version available in *sourceforge* or you can grab the latest snapshot from the `SVN repository <http://code.ros.org/svn/opencv/>`_.

Getting the latest stable OpenCV version
------------------------------------------

* Go to http://sourceforge.net/projects/opencvlibrary

* Download the source tarball and unpack it


Getting the cutting-edge OpenCV from SourceForge SVN repository
-----------------------------------------------------------------

Launch SVN client and checkout either

a. the current OpenCV snapshot from here: https://code.ros.org/svn/opencv/trunk

#. or the latest tested OpenCV snapshot from here: http://code.ros.org/svn/opencv/tags/latest_tested_snapshot

In Ubuntu it can be done using the following command, e.g.:

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
      mkdir release
      cd release
      cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX= /usr/local
       
#. Enter the created temporary directory (<cmake_binary_dir>) and proceed with:

   .. code-block:: bash
      
      make
      sudo make install

.. note::
  
   If the size of the created library is a critical issue (like in case of an Android build) you can use the ``install/strip`` command to get the smallest size as possible. The *stripped* version appears to be twice as small. However, we do not recommend using this unless those extra megabytes do really matter.

