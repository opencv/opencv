Using Orbbec Astra 3D cameras {#tutorial_orbbec_astra_openni}
======================================================

@tableofcontents

@prev_tutorial{tutorial_kinect_openni}
@next_tutorial{tutorial_intelperc}


### Introduction

This tutorial is devoted to the Astra Series of Orbbec 3D cameras (https://www.orbbec.com/products/structured-light-camera/astra-series/).
That cameras have a depth sensor in addition to a common color sensor. The depth sensors can be read using
the open source OpenNI API with @ref cv::VideoCapture class. The video stream is provided through the regular
camera interface.

### Installation Instructions

In order to use the Astra camera's depth sensor with OpenCV you should do the following steps:

-#  Download the latest version of Orbbec OpenNI SDK (from here <https://www.orbbec.com/developers/openni-sdk/>).
    Unzip the archive, choose the build according to your operating system and follow installation
    steps provided in the Readme file.

-#  For instance, if you use 64bit GNU/Linux run:
    @code{.bash}
    $ cd Linux/OpenNI-Linux-x64-2.3.0.63/
    $ sudo ./install.sh
    @endcode
    When you are done with the installation, make sure to replug your device for udev rules to take
    effect. The camera should now work as a general camera device. Note that your current user should
    belong to group `video` to have access to the camera. Also, make sure to source `OpenNIDevEnvironment` file:
    @code{.bash}
    $ source OpenNIDevEnvironment
    @endcode
    To verify that the source command works and OpenNI library and header files can be found, run the following
    command and you should see something similar in your terminal:
    @code{.bash}
    $ echo $OPENNI2_INCLUDE
    /home/user/OpenNI_2.3.0.63/Linux/OpenNI-Linux-x64-2.3.0.63/Include
    $ echo $OPENNI2_REDIST
    /home/user/OpenNI_2.3.0.63/Linux/OpenNI-Linux-x64-2.3.0.63/Redist
    @endcode
    If the above two variables are empty, then you need to source `OpenNIDevEnvironment` again.

    @note Orbbec OpenNI SDK version 2.3.0.86 and newer does not provide `install.sh` any more.
    You can use the following script to initialize environment:
    @code{.text}
    # Check if user is root/running with sudo
    if [ `whoami` != root ]; then
        echo Please run this script with sudo
        exit
    fi

    ORIG_PATH=`pwd`
    cd `dirname $0`
    SCRIPT_PATH=`pwd`
    cd $ORIG_PATH

    if [ "`uname -s`" != "Darwin" ]; then
        # Install UDEV rules for USB device
        cp ${SCRIPT_PATH}/orbbec-usb.rules /etc/udev/rules.d/558-orbbec-usb.rules
        echo "usb rules file install at /etc/udev/rules.d/558-orbbec-usb.rules"
    fi

    OUT_FILE="$SCRIPT_PATH/OpenNIDevEnvironment"
    echo "export OPENNI2_INCLUDE=$SCRIPT_PATH/../sdk/Include" > $OUT_FILE
    echo "export OPENNI2_REDIST=$SCRIPT_PATH/../sdk/libs" >> $OUT_FILE
    chmod a+r $OUT_FILE
    echo "exit"
    @endcode

    @note The last tried version `2.3.0.86_202210111154_4c8f5aa4_beta6` does not work correctly with
    modern Linux, even after libusb rebuild as recommended by the instruction. The last know good
    configuration is version 2.3.0.63 (tested with Ubuntu 18.04 amd64). It's not provided officialy
    with the downloading page, but published by Orbbec technical suport on Orbbec community forum
    [here](https://3dclub.orbbec3d.com/t/universal-download-thread-for-astra-series-cameras/622).

-#  Now you can configure OpenCV with OpenNI support enabled by setting the `WITH_OPENNI2` flag in CMake.
    You may also like to enable the `BUILD_EXAMPLES` flag to get a code sample working with your Astra camera.
    Run the following commands in the directory containing OpenCV source code to enable OpenNI support:
    @code{.bash}
    $ mkdir build
    $ cd build
    $ cmake -DWITH_OPENNI2=ON ..
    @endcode
    If the OpenNI library is found, OpenCV will be built with OpenNI2 support. You can see the status of OpenNI2
    support in the CMake log:
    @code{.text}
    --   Video I/O:
    --     DC1394:                      YES (2.2.6)
    --     FFMPEG:                      YES
    --       avcodec:                   YES (58.91.100)
    --       avformat:                  YES (58.45.100)
    --       avutil:                    YES (56.51.100)
    --       swscale:                   YES (5.7.100)
    --       avresample:                NO
    --     GStreamer:                   YES (1.18.1)
    --     OpenNI2:                     YES (2.3.0)
    --     v4l/v4l2:                    YES (linux/videodev2.h)
    @endcode

-#  Build OpenCV:
    @code{.bash}
    $ make
    @endcode

### Code

The Astra Pro camera has two sensors -- a depth sensor and a color sensor. The depth sensor
can be read using the OpenNI interface with @ref cv::VideoCapture class. The video stream is
not available through OpenNI API and is only provided via the regular camera interface.
So, to get both depth and color frames, two @ref cv::VideoCapture objects should be created:

@snippetlineno samples/cpp/tutorial_code/videoio/openni_orbbec_astra/openni_orbbec_astra.cpp Open streams

The first object will use the OpenNI2 API to retrieve depth data. The second one uses the
Video4Linux2 interface to access the color sensor. Note that the example above assumes that
the Astra camera is the first camera in the system. If you have more than one camera connected,
you may need to explicitly set the proper camera number.

Before using the created VideoCapture objects you may want to set up stream parameters by setting
objects' properties. The most important parameters are frame width, frame height and fps.
For this example, we’ll configure width and height of both streams to VGA resolution, which is
the maximum resolution available for both sensors, and we’d like both stream parameters to be the
same for easier color-to-depth data registration:

@snippetlineno samples/cpp/tutorial_code/videoio/openni_orbbec_astra/openni_orbbec_astra.cpp Setup streams

For setting and retrieving some property of sensor data generators use @ref cv::VideoCapture::set and
@ref cv::VideoCapture::get methods respectively, e.g. :

@snippetlineno samples/cpp/tutorial_code/videoio/openni_orbbec_astra/openni_orbbec_astra.cpp Get properties

The following properties of cameras available through OpenNI interface are supported for the depth
generator:

-   @ref cv::CAP_PROP_FRAME_WIDTH -- Frame width in pixels.
-   @ref cv::CAP_PROP_FRAME_HEIGHT -- Frame height in pixels.
-   @ref cv::CAP_PROP_FPS -- Frame rate in FPS.
-   @ref cv::CAP_PROP_OPENNI_REGISTRATION -- Flag that registers the remapping depth map to image map
    by changing the depth generator's viewpoint (if the flag is "on") or sets this view point to
    its normal one (if the flag is "off"). The registration process’ resulting images are
    pixel-aligned, which means that every pixel in the image is aligned to a pixel in the depth
    image.
-   @ref cv::CAP_PROP_OPENNI2_MIRROR -- Flag to enable or disable mirroring for this stream. Set to 0
    to disable mirroring

    Next properties are available for getting only:

-   @ref cv::CAP_PROP_OPENNI_FRAME_MAX_DEPTH -- A maximum supported depth of the camera in mm.
-   @ref cv::CAP_PROP_OPENNI_BASELINE -- Baseline value in mm.

After the VideoCapture objects have been set up, you can start reading frames from them.

@note
    OpenCV's VideoCapture provides synchronous API, so you have to grab frames in a new thread
    to avoid one stream blocking while another stream is being read. VideoCapture is not a
    thread-safe class, so you need to be careful to avoid any possible deadlocks or data races.

As there are two video sources that should be read simultaneously, it’s necessary to create two
threads to avoid blocking. Example implementation that gets frames from each sensor in a new thread
and stores them in a list along with their timestamps:

@snippetlineno samples/cpp/tutorial_code/videoio/openni_orbbec_astra/openni_orbbec_astra.cpp Read streams

VideoCapture can retrieve the following data:

-#  data given from the depth generator:
    -   @ref cv::CAP_OPENNI_DEPTH_MAP - depth values in mm (CV_16UC1)
    -   @ref cv::CAP_OPENNI_POINT_CLOUD_MAP - XYZ in meters (CV_32FC3)
    -   @ref cv::CAP_OPENNI_DISPARITY_MAP - disparity in pixels (CV_8UC1)
    -   @ref cv::CAP_OPENNI_DISPARITY_MAP_32F - disparity in pixels (CV_32FC1)
    -   @ref cv::CAP_OPENNI_VALID_DEPTH_MASK - mask of valid pixels (not occluded, not shaded, etc.)
        (CV_8UC1)

-#  data given from the color sensor is a regular BGR image (CV_8UC3).

When new data are available, each reading thread notifies the main thread using a condition variable.
A frame is stored in the ordered list -- the first frame in the list is the earliest captured,
the last frame is the latest captured. As depth and color frames are read from independent sources
two video streams may become out of sync even when both streams are set up for the same frame rate.
A post-synchronization procedure can be applied to the streams to combine depth and color frames into
pairs. The sample code below demonstrates this procedure:

@snippetlineno samples/cpp/tutorial_code/videoio/openni_orbbec_astra/openni_orbbec_astra.cpp Pair frames

In the code snippet above the execution is blocked until there are some frames in both frame lists.
When there are new frames, their timestamps are being checked -- if they differ more than a half of
the frame period then one of the frames is dropped. If timestamps are close enough, then two frames
are paired. Now, we have two frames: one containing color information and another one -- depth information.
In the example above retrieved frames are simply shown with cv::imshow function, but you can insert
any other processing code here.

In the sample images below you can see the color frame and the depth frame representing the same scene.
Looking at the color frame it's hard to distinguish plant leaves from leaves painted on a wall,
but the depth data makes it easy.

![Color frame](images/astra_color.jpg)
![Depth frame](images/astra_depth.png)

The complete implementation can be found in
[openni_orbbec_astra.cpp](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/videoio/openni_orbbec_astra/openni_orbbec_astra.cpp)
in `samples/cpp/tutorial_code/videoio` directory.
