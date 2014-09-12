***************************************
Native OpenCV Manager service interface
***************************************

.. highlight:: cpp
.. module:: IOpenCVEngine.h
    :platform: Android
    :synopsis: Defines OpenCV Manager interface for Android Binder component
.. Class:: OpenCVEngine

This class provides a binder interface to the OpenCV Manager service

int getEngineVersion()
----------------------

.. method:: int GetEngineVersion()

    Get OpenCV Manager version

    :rtype: int
    :return: Version of the OpenCV Manager

android::String16 getLibPathByVersion()
---------------------------------------

.. method:: android::String16 GetLibPathByVersion(android::String16 version)

    Find already installed OpenCV library 

    :param version: OpenCV Library version
    :rtype: String;
    :return: Path to OpenCV native libs or empty string if OpenCV was not found

android::String16 getLibraryList()
----------------------------------

.. method:: android::String16 GetLibraryList(android::String16 version)

    Get list of OpenCV native libraries in loading order

    :param version: OpenCV Library version
    :rtype: String;
    :return: OpenCV library names separated by symbol ";" in loading order

boolean installVersion()
------------------------

.. method:: boolean InstallVersion(android::String16 version)

    Tries to install defined version of OpenCV

    :param version: OpenCV Library version
    :rtype: String
    :return: True if installation was successful or package was already installed
