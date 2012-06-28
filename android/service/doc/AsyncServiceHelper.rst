*******************************************
Java Asynchronious OpenCV Helper (internal)
*******************************************

.. highlight:: java
.. module:: org.opencv.android
    :platform: Android
    :synopsis: Implements Android dependent Java classes
.. Class:: AsyncServiceHelper

Helper class provides implementation of asynchronious OpenCV initialization with Android OpenCV Engine Service.

.. note:: This is imternal class. Does not use it directly. Use OpenCVLoader.initAsync() instead!

int initOpenCV()
----------------

.. method:: int initOpenCV(String Version, Context AppContext, LoaderCallbackInterface Callback)

    Tries to init OpenCV library using OpenCV Engine Service. Callback method will be called, when initialisation finishes

    :param Version: Version of OpenCV
    :param AppContext: Application context for service connection
    :param CallBack: Object that implements LoaderCallbackInterface. See Helper callback interface
    :rtype: boolean
    :return: Return true if initialization starts successfully