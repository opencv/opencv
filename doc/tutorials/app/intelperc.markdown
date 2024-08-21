Using Creative Senz3D and other Intel RealSense SDK compatible depth sensors {#tutorial_intelperc}
=======================================================================================

@tableofcontents

@prev_tutorial{tutorial_orbbec_uvc}
@next_tutorial{tutorial_wayland_ubuntu}

![hardwares](images/realsense.jpg)

**Note**: This tutorial is partially obsolete since PerC SDK has been replaced with RealSense SDK

Depth sensors compatible with Intel® RealSense SDK are supported through VideoCapture
class. Depth map, RGB image and some other formats of output can be retrieved by using familiar
interface of VideoCapture.

In order to use depth sensor with OpenCV you should do the following preliminary steps:

-#  Install Intel RealSense SDK 2.0 (from here <https://github.com/IntelRealSense/librealsense>).

-#  Configure OpenCV with Intel RealSense SDK support by setting WITH_LIBREALSENSE flag in
    CMake. If Intel RealSense SDK is found in install folders OpenCV will be built with
    Intel Realsense SDK library (see a status LIBREALSENSE in CMake log).

-#  Build OpenCV.

VideoCapture can retrieve the following data:

-#  data given from depth generator:
    -   CAP_INTELPERC_DEPTH_MAP - each pixel is a 16-bit integer. The value indicates the
            distance from an object to the camera's XY plane or the Cartesian depth. (CV_16UC1)
    -   CAP_INTELPERC_UVDEPTH_MAP - each pixel contains two 32-bit floating point values in
        the range of 0-1, representing the mapping of depth coordinates to the color
        coordinates. (CV_32FC2)
    -   CAP_INTELPERC_IR_MAP - each pixel is a 16-bit integer. The value indicates the
        intensity of the reflected laser beam. (CV_16UC1)

-#  data given from RGB image generator:
    -   CAP_INTELPERC_IMAGE - color image. (CV_8UC3)

In order to get depth map from depth sensor use VideoCapture::operator \>\>, e. g. :
@code{.cpp}
    VideoCapture capture( CAP_REALSENSE );
    for(;;)
    {
        Mat depthMap;
        capture >> depthMap;

        if( waitKey( 30 ) >= 0 )
            break;
    }
@endcode
For getting several data maps use VideoCapture::grab and VideoCapture::retrieve, e.g. :
@code{.cpp}
    VideoCapture capture(CAP_REALSENSE);
    for(;;)
    {
        Mat depthMap;
        Mat image;
        Mat irImage;

        capture.grab();

        capture.retrieve( depthMap, CAP_INTELPERC_DEPTH_MAP );
        capture.retrieve(    image, CAP_INTELPERC_IMAGE );
        capture.retrieve(  irImage, CAP_INTELPERC_IR_MAP);

        if( waitKey( 30 ) >= 0 )
            break;
    }
@endcode
For setting and getting some property of sensor\` data generators use VideoCapture::set and
VideoCapture::get methods respectively, e.g. :
@code{.cpp}
    VideoCapture capture(CAP_REALSENSE);
    capture.set( CAP_INTELPERC_DEPTH_GENERATOR | CAP_PROP_INTELPERC_PROFILE_IDX, 0 );
    cout << "FPS    " << capture.get( CAP_INTELPERC_DEPTH_GENERATOR+CAP_PROP_FPS ) << endl;
@endcode
Since two types of sensor's data generators are supported (image generator and depth generator),
there are two flags that should be used to set/get property of the needed generator:

-   CAP_INTELPERC_IMAGE_GENERATOR -- a flag for access to the image generator properties.
-   CAP_INTELPERC_DEPTH_GENERATOR -- a flag for access to the depth generator properties. This
    flag value is assumed by default if neither of the two possible values of the property is set.

For more information please refer to the example of usage
[videocapture_realsense.cpp](https://github.com/opencv/opencv/tree/4.x/samples/cpp/videocapture_realsense.cpp)
in opencv/samples/cpp folder.
