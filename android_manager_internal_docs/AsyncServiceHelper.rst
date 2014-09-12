*******************************************
Java Asynchronous OpenCV Helper (internal)
*******************************************

.. highlight:: java
.. module:: org.opencv.android
    :platform: Android
    :synopsis: Implements Android-dependent Java classes
.. Class:: AsyncServiceHelper

This helper class provides an implementation of asynchronous OpenCV initialization using the Android OpenCV Engine Service.

.. note:: This is an internal class. Do not use it directly, use OpenCVLoader.initAsync() instead!

int initOpenCV()
----------------

.. method:: int initOpenCV(String Version, Context AppContext, LoaderCallbackInterface Callback)

    Tries to initialise OpenCV library using OpenCV Engine Service. Callback method will be called, when initialisation finishes.

    :param Version: Version of OpenCV
    :param AppContext: Application context for service connection
    :param CallBack: Object that implements LoaderCallbackInterface. See Helper callback interface
    :rtype: boolean
    :return: True if initialisation starts successfully