*******
HighGUI
*******

.. highlight:: cpp

Using Kinect sensor
===================

Kinect sensor is supported through ``VideoCapture`` class. Depth map, RGB image and some other formats of Kinect output can be retrieved by using familiar interface of ``VideoCapture``.

In order to use Kinect with OpenCV you should do the following preliminary steps:

#.
    Install OpenNI library and PrimeSensor Module for OpenNI from here \url{http://www.openni.org/downloadfiles}. The installation should be done to default folders listed in the instructions of these products:

    .. code-block:: text
    
        OpenNI:
            Linux & MacOSX:
                Libs into: /usr/lib
                Includes into: /usr/include/ni
            Windows:
                Libs into: c:/Program Files/OpenNI/Lib
                Includes into: c:/Program Files/OpenNI/Include
        PrimeSensor Module:
            Linux & MacOSX:
                Bins into: /usr/bin
            Windows:
                Bins into: c:/Program Files/Prime Sense/Sensor/Bin

    If one or both products were installed to the other folders, the user should change corresponding CMake variables ``OPENNI_LIB_DIR``, ``OPENNI_INCLUDE_DIR`` or/and ``OPENNI_PRIME_SENSOR_MODULE_BIN_DIR``.
    
#.
    Configure OpenCV with OpenNI support by setting \texttt{WITH\_OPENNI} flag in CMake. If OpenNI is found in default install folders OpenCV will be built with OpenNI library regardless of whether PrimeSensor Module is found or not. If PrimeSensor Module was not found you will get a warning in CMake log. Without PrimeSensor module OpenCV will be successfully compiled with OpenNI library, but ``VideoCapture`` object will not grab data from Kinect sensor.

#.
    Build OpenCV.

VideoCapture can retrieve the following Kinect data:

#.
    data given from depth generator:
      * ``OPENNI_DEPTH_MAP``          - depth values in mm (CV_16UC1)
      * ``OPENNI_POINT_CLOUD_MAP``    - XYZ in meters (CV_32FC3)
      * ``OPENNI_DISPARITY_MAP``      - disparity in pixels (CV_8UC1)
      * ``OPENNI_DISPARITY_MAP_32F``  - disparity in pixels (CV_32FC1)
      * ``OPENNI_VALID_DEPTH_MASK``   - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)
#.
    data given from RGB image generator:
      * ``OPENNI_BGR_IMAGE``          - color image (CV_8UC3)
      * ``OPENNI_GRAY_IMAGE``         - gray image (CV_8UC1)

In order to get depth map from Kinect use ``VideoCapture::operator >>``, e. g. ::

    VideoCapture capture(0); // or CV_CAP_OPENNI
    for(;;)
    {
        Mat depthMap;    
        capture >> depthMap;
    
        if( waitKey( 30 ) >= 0 )
            break;
    }

For getting several Kinect maps use ``VideoCapture::grab`` and ``VideoCapture::retrieve``, e.g. ::

    VideoCapture capture(0); // or CV_CAP_OPENNI
    for(;;)
    {
        Mat depthMap;
        Mat rgbImage
    
        capture.grab();
    
        capture.retrieve( depthMap, OPENNI_DEPTH_MAP );
        capture.retrieve( bgrImage, OPENNI_BGR_IMAGE );
    
        if( waitKey( 30 ) >= 0 )
            break;
    }

For more information please refer to a Kinect example of usage ``kinect_maps.cpp`` in ``opencv/samples/cpp`` folder.
