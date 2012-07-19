***************************************
Native OpenCV Manager service interface
***************************************

.. highlight:: cpp
.. module:: IOpenCVEngine.h
    :platform: Android
    :synopsis: Defines OpenCV Manager interface for Android Binder component
.. Class:: OpenCVEngine

OpenCVEngine class provides Binder interface to OpenCV Manager Service

int getEngineVersion()
----------------------

.. method:: int GetEngineVersion()

    Gets OpenCV Manager version

    :rtype: int
    :return: Returns OpenCV Manager version

android::String16 getLibPathByVersion()
---------------------------------------

.. method:: android::String16 GetLibPathByVersion(android::String16 version)

    Gets path to native OpenCV libraries 

    :param version: OpenCV Library version
    :rtype: String;
    :return: Returns path to OpenCV native libs or empty string if OpenCV was not found

android::String16 getLibraryList()
----------------------------------

.. method:: android::String16 GetLibraryList(android::String16 version)

    Gets list of OpenCV native libraries in loading order

    :param version: OpenCV Library version
    :rtype: String;
    :return: Returns OpenCV libraries names separated by semicolon symbol in loading order

boolean installVersion()
------------------------

.. method:: boolean InstallVersion(android::String16 version)

    Trys to install defined version of OpenCV

    :param version: OpenCV Library version
    :rtype: String
    :return: Returns true if installation successful or package has been already installed
