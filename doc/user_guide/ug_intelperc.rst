*******
HighGUI
*******

.. highlight:: cpp

Using Creative Senz3D and other Intel Perceptual Computing SDK compatible depth sensors
=======================================================================================

Depth sensors compatible with Intel Perceptual Computing SDK are supported through ``VideoCapture`` class. Depth map, RGB image and some other formats of output can be retrieved by using familiar interface of ``VideoCapture``.

In order to use depth sensor with OpenCV you should do the following preliminary steps:

#.
    Install Intel Perceptual Computing SDK (from here http://www.intel.com/software/perceptual).

#.
    Configure OpenCV with Intel Perceptual Computing SDK support by setting ``WITH_INTELPERC`` flag in CMake. If Intel Perceptual Computing SDK is found in install folders OpenCV will be built with Intel Perceptual Computing SDK library (see a status ``INTELPERC`` in CMake log). If CMake process doesn't find Intel Perceptual Computing SDK installation folder automatically, the user should change corresponding CMake variables ``INTELPERC_LIB_DIR`` and ``INTELPERC_INCLUDE_DIR`` to the proper value.

#.
    Build OpenCV.

VideoCapture can retrieve the following data:

#.
    data given from depth generator:
      * ``CV_CAP_INTELPERC_DEPTH_MAP``       - each pixel is a 16-bit integer. The value indicates the distance from an object to the camera's XY plane or the Cartesian depth. (CV_16UC1)
      * ``CV_CAP_INTELPERC_UVDEPTH_MAP``     - each pixel contains two 32-bit floating point values in the range of 0-1, representing the mapping of depth coordinates to the color coordinates. (CV_32FC2)
      * ``CV_CAP_INTELPERC_IR_MAP``          - each pixel is a 16-bit integer. The value indicates the intensity of the reflected laser beam. (CV_16UC1)
#.
    data given from RGB image generator:
      * ``CV_CAP_INTELPERC_IMAGE``           - color image. (CV_8UC3)

In order to get depth map from depth sensor use ``VideoCapture::operator >>``, e. g. ::

    VideoCapture capture( CV_CAP_INTELPERC );
    for(;;)
    {
        Mat depthMap;
        capture >> depthMap;

        if( waitKey( 30 ) >= 0 )
            break;
    }

For getting several data maps use ``VideoCapture::grab`` and ``VideoCapture::retrieve``, e.g. ::

    VideoCapture capture(CV_CAP_INTELPERC);
    for(;;)
    {
        Mat depthMap;
        Mat image;
        Mat irImage;

        capture.grab();

        capture.retrieve( depthMap, CV_CAP_INTELPERC_DEPTH_MAP );
        capture.retrieve(    image, CV_CAP_INTELPERC_IMAGE );
        capture.retrieve(  irImage, CV_CAP_INTELPERC_IR_MAP);

        if( waitKey( 30 ) >= 0 )
            break;
    }

For setting and getting some property of sensor` data generators use ``VideoCapture::set`` and ``VideoCapture::get`` methods respectively, e.g. ::

    VideoCapture capture( CV_CAP_INTELPERC );
    capture.set( CV_CAP_INTELPERC_DEPTH_GENERATOR | CV_CAP_PROP_INTELPERC_PROFILE_IDX, 0 );
    cout << "FPS    " << capture.get( CV_CAP_INTELPERC_DEPTH_GENERATOR+CV_CAP_PROP_FPS ) << endl;

Since two types of sensor's data generators are supported (image generator and depth generator), there are two flags that should be used to set/get property of the needed generator:

* CV_CAP_INTELPERC_IMAGE_GENERATOR -- a flag for access to the image generator properties.

* CV_CAP_INTELPERC_DEPTH_GENERATOR -- a flag for access to the depth generator properties. This flag value is assumed by default if neither of the two possible values of the property is set.

For more information please refer to the example of usage intelperc_capture.cpp_ in ``opencv/samples/cpp`` folder.

.. _intelperc_capture.cpp: https://github.com/Itseez/opencv/tree/master/samples/cpp/intelperc_capture.cpp