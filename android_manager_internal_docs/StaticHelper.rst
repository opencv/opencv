************************************
Java Static OpenCV Helper (internal)
************************************

.. highlight:: java
.. module:: org.opencv.android
    :platform: Android
    :synopsis: Implements Android dependent Java classes
.. Class:: StaticHelper

This helper class provides an implementation of static OpenCV initialization. All OpenCV libraries must be included into the application package.

.. note:: This is an internal class. Do not use it directly, use OpenCVLoader.initDebug() instead!

int initOpenCV()
----------------

.. method:: int initOpenCV()

    Tries to initialise OpenCV library using libraries from the application package. Method uses libopencv_info.so library for getting a
    list of libraries in loading order. Method loads libopencv_java.so, if info library is not present.

    :rtype: boolean
    :return: Return true if initialisation was successful