.. _Android_Binary_Package:

Using Android binary package with Eclipse
*****************************************

.. include:: <isonum.txt>

This tutorial was tested using Ubuntu 10.04 and Windows 7 SP1 operating systems. Nevertheless, it should also work on any other **OS**\ es supported by Android SDK (including Mac OS X). If you encounter errors after following the steps described here feel free to contact us via *android-opencv* disscussion group https://groups.google.com/group/android-opencv/ and we will try to fix your problem.

.. _Android_Environment_Setup_Lite: 

Setup environment to start Android Development
==============================================

You need the following tools to be installed:

1. **Sun JDK 6**

   Visit http://www.oracle.com/technetwork/java/javase/downloads/index.html and download installer for your OS.

   Here is detailed JDK installation guide for Ubuntu and Mac OS: http://source.android.com/source/initializing.html (only JDK sections are applicable for OpenCV)

   .. note:: OpenJDK is not usable for Android development because Android SDK supports only Sun JDK.

#. **Android SDK**

   Get the latest Android SDK from http://developer.android.com/sdk/index.html

   Here is Google's install guide for SDK http://developer.android.com/sdk/installing.html

   .. note:: If you choose SDK packed into Windows installer then you should have installed 32-bit JRE. It does not needed for Android development but installer is x86 application and requires 32-bit Java runtime.

   .. note:: If you are running x64 version of Ubuntu Linux then you need ia32 shared libraries for use on amd64 and ia64 systems installed. You can install them with following command:

      .. code-block:: bash

         sudo apt-get install ia32-libs

