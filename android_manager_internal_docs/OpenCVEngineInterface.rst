*********************************
Java OpenCV OpenCVEngineInterface
*********************************

.. highlight:: java
.. module:: org.opencv.engine
    :platform: Android
    :synopsis: Defines OpenCV Manager interface for Android.
.. Class:: OpenCVEngineInterface

This class provides a Java interface to the OpenCV Manager service. It is similar to the OpenCVEngine class.

.. note:: Do not use this class directly. Use OpenCVLoader instead!

int getEngineVersion()
----------------------

.. method:: int GetEngineVersion()

    Get OpenCV Manager version

    :rtype: int
    :return: Version of the OpenCV Manager

String getLibPathByVersion()
----------------------------

.. method:: String GetLibPathByVersion(String version)

    Find already installed OpenCV library 

    :param version: OpenCV library version
    :rtype: String
    :return: Path to OpenCV native libs or empty string if OpenCV was not found

String getLibraryList()
-----------------------

.. method:: String GetLibraryList(String version)

    Get list of OpenCV native libraries in loading order separated by ";" symbol

    :param version: OpenCV library version
    :rtype: String
    :return: OpenCV library names separated by symbol ";" in loading order

boolean installVersion()
------------------------

.. method:: boolean InstallVersion(String version)

    Try to install defined version of OpenCV from Google Play (Android Market).

    :param version: OpenCV library version
    :rtype: String
    :return: True if installation was successful or package was already installed
 