******************
Java OpenCV Loader
******************

.. highlight:: java
.. Class:: OpenCVLoader

Helper class provides common initialization methods for OpenCV library.

boolean initDebug()
-------------------

.. method:: static boolean initDebug()

    Loads and initializes OpenCV library from within current application package. Roughly it is
    analog of ``system.loadLibrary("opencv_java")``.

    :rtype: boolean;
    :return: returns true if initialization of OpenCV was successful.

.. note:: This method is deprecated for production code. It is designed for experimental and local
          development purposes only. If you want to publish your app use approach with async
          initialization.

boolean initAsync()
-------------------

.. method:: static boolean initAsync(String Version, Context AppContext, LoaderCallbackInterface Callback)

    Loads and initializes OpenCV library using OpenCV Manager.

    :param Version: OpenCV Library version.
    :param AppContext: application context for connecting to the service.
    :param Callback: object, that implements ``LoaderCallbackInterface`` for handling connection
                     status (see ``BaseLoaderCallback``).

    :rtype: boolean;
    :return: returns true if initialization of OpenCV starts successfully.

OpenCV version constants
-------------------------

.. data:: OPENCV_VERSION_2_4_2

    OpenCV Library version 2.4.2

.. data:: OPENCV_VERSION_2_4_3

    OpenCV Library version 2.4.3

.. data:: OPENCV_VERSION_2_4_4

    OpenCV Library version 2.4.4

.. data:: OPENCV_VERSION_2_4_5

    OpenCV Library version 2.4.5

.. data:: OPENCV_VERSION_2_4_6

    OpenCV Library version 2.4.6

.. data:: OPENCV_VERSION_2_4_7

    OpenCV Library version 2.4.7

.. data:: OPENCV_VERSION_2_4_8

    OpenCV Library version 2.4.8

.. data:: OPENCV_VERSION_2_4_9

    OpenCV Library version 2.4.9

.. data:: OPENCV_VERSION_2_4_10

    OpenCV Library version 2.4.10

.. data:: OPENCV_VERSION_2_4_11

    OpenCV Library version 2.4.11
