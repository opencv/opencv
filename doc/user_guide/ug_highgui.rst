*******
HighGUI
*******

.. highlight:: cpp

Using Kinect and other OpenNI compatible depth sensors
======================================================

Depth sensors compatible with OpenNI (Kinect, XtionPRO, ...) are supported through ``VideoCapture`` class. Depth map, RGB image and some other formats of output can be retrieved by using familiar interface of ``VideoCapture``.

In order to use depth sensor with OpenCV you should do the following preliminary steps:

#.
    Install OpenNI library (from here http://www.openni.org/downloadfiles) and PrimeSensor Module for OpenNI (from here https://github.com/avin2/SensorKinect). The installation should be done to default folders listed in the instructions of these products, e.g.:

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
    Configure OpenCV with OpenNI support by setting ``WITH_OPENNI`` flag in CMake. If OpenNI is found in install folders OpenCV will be built with OpenNI library (see a status ``OpenNI`` in CMake log) whereas PrimeSensor Modules can not be found (see a status ``OpenNI PrimeSensor Modules`` in CMake log). Without PrimeSensor module OpenCV will be successfully compiled with OpenNI library, but ``VideoCapture`` object will not grab data from Kinect sensor.

#.
    Build OpenCV.

VideoCapture can retrieve the following data:

#.
    data given from depth generator:
      * ``CV_CAP_OPENNI_DEPTH_MAP``          - depth values in mm (CV_16UC1)
      * ``CV_CAP_OPENNI_POINT_CLOUD_MAP``    - XYZ in meters (CV_32FC3)
      * ``CV_CAP_OPENNI_DISPARITY_MAP``      - disparity in pixels (CV_8UC1)
      * ``CV_CAP_OPENNI_DISPARITY_MAP_32F``  - disparity in pixels (CV_32FC1)
      * ``CV_CAP_OPENNI_VALID_DEPTH_MASK``   - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)
#.
    data given from RGB image generator:
      * ``CV_CAP_OPENNI_BGR_IMAGE``          - color image (CV_8UC3)
      * ``CV_CAP_OPENNI_GRAY_IMAGE``         - gray image (CV_8UC1)

In order to get depth map from depth sensor use ``VideoCapture::operator >>``, e. g. ::

    VideoCapture capture( CV_CAP_OPENNI );
    for(;;)
    {
        Mat depthMap;
        capture >> depthMap;

        if( waitKey( 30 ) >= 0 )
            break;
    }

For getting several data maps use ``VideoCapture::grab`` and ``VideoCapture::retrieve``, e.g. ::

    VideoCapture capture(0); // or CV_CAP_OPENNI
    for(;;)
    {
        Mat depthMap;
        Mat rgbImage

        capture.grab();

        capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP );
        capture.retrieve( bgrImage, CV_CAP_OPENNI_BGR_IMAGE );

        if( waitKey( 30 ) >= 0 )
            break;
    }

For setting and getting some property of sensor` data generators use ``VideoCapture::set`` and ``VideoCapture::get`` methods respectively, e.g. ::

    VideoCapture capture( CV_CAP_OPENNI );
    capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ );
    cout << "FPS    " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;

Since two types of sensor's data generators are supported (image generator and depth generator), there are two flags that should be used to set/get property of the needed generator:

* CV_CAP_OPENNI_IMAGE_GENERATOR -- A flag for access to the image generator properties.

* CV_CAP_OPENNI_DEPTH_GENERATOR -- A flag for access to the depth generator properties. This flag value is assumed by default if neither of the two possible values of the property is not set.

Some depth sensors (for example XtionPRO) do not have image generator. In order to check it you can get ``CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT`` property.
::

    bool isImageGeneratorPresent = capture.get( CV_CAP_PROP_OPENNI_IMAGE_GENERATOR_PRESENT ) != 0; // or == 1


Flags specifing the needed generator type must be used in combination with particular generator property. The following properties of cameras available through OpenNI interfaces are supported:

*
  For image generator:

  - ``CV_CAP_PROP_OPENNI_OUTPUT_MODE`` -- Three output modes are supported: ``CV_CAP_OPENNI_VGA_30HZ`` used by default (image generator returns images in VGA resolution with 30 FPS), ``CV_CAP_OPENNI_SXGA_15HZ`` (image generator returns images in SXGA resolution with 15 FPS) and ``CV_CAP_OPENNI_SXGA_30HZ`` (image generator returns images in SXGA resolution with 30 FPS, the mode is supported by XtionPRO Live); depth generator's maps are always in VGA resolution.


*
  For depth generator:

  - ``CV_CAP_PROP_OPENNI_REGISTRATION`` -- Flag that registers the remapping depth map to image map  by changing depth generator's view point (if the flag is ``"on"``) or sets this view point to its normal one (if the flag is ``"off"``). The registration process’s resulting images are pixel-aligned,which means that every pixel in the image is aligned to a pixel in the depth image.

    Next properties are available for getting only:

  - ``CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH`` -- A maximum supported depth of Kinect in mm.
  - ``CV_CAP_PROP_OPENNI_BASELINE`` -- Baseline value in mm.
  - ``CV_CAP_PROP_OPENNI_FOCAL_LENGTH`` -- A focal length in pixels.
  - ``CV_CAP_PROP_FRAME_WIDTH`` -- Frame width in pixels.
  - ``CV_CAP_PROP_FRAME_HEIGHT`` -- Frame height in pixels.
  - ``CV_CAP_PROP_FPS`` -- Frame rate in FPS.

*
  Some typical flags combinations "generator type + property" are defined as single flags:

    - ``CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE = CV_CAP_OPENNI_IMAGE_GENERATOR + CV_CAP_PROP_OPENNI_OUTPUT_MODE``
    - ``CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_BASELINE``
    - ``CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_FOCAL_LENGTH``
    - ``CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION = CV_CAP_OPENNI_DEPTH_GENERATOR + CV_CAP_PROP_OPENNI_REGISTRATION``

For more information please refer to the example of usage openni_capture.cpp_ in ``opencv/samples/cpp`` folder.

.. _openni_capture.cpp: https://github.com/Itseez/opencv/tree/master/samples/cpp/openni_capture.cpp
