Reading and Writing Images and Video
====================================

.. highlight:: cpp

.. index:: imdecode

.. _imdecode:

imdecode
------------
.. c:function:: Mat imdecode( const Mat\& buf,  int flags )

    Reads an image from a buffer in memory.

    :param buf: Input array of vector of bytes.

    :param flags: The same flags as in  :ref:`imread` .
    
The function reads an image from the specified buffer in memory.
If the buffer is too short or contains invalid data, the empty matrix is returned.

See
:ref:`imread` for the list of supported formats and flags description.

.. index:: imencode

.. _imencode:

imencode
------------
.. c:function:: bool imencode( const string\& ext,               const Mat\& img,               vector<uchar>\& buf,               const vector<int>\& params=vector<int>())

    Encode an image into a memory buffer.

    :param ext: File extension that defines the output format.

    :param img: Image to be written.

    :param buf: Output buffer resized to fit the compressed image.

    :param params: Format-specific parameters. See  :ref:`imwrite` .

The function compresses the image and stores it in the memory buffer that is resized to fit the result.
See
:ref:`imwrite` for the list of supported formats and flags description.

.. index:: imread

.. _imread:

imread
----------
.. c:function:: Mat imread( const string\& filename,  int flags=1 )

    Loads an image from a file.

    :param filename: Name of file to be loaded.

    :param flags: Flags specifying the color type of a loaded image:

        * **>0**  a 3-channel color image

        * **=0** a grayscale image

        * **<0** The image is loaded as is. Note that in the current implementation the alpha channel, if any, is stripped from the output image. For example, a 4-channel RGBA image is loaded as RGB if  :math:`flags\ge0` .

The function ``imread`` loads an image from the specified file and returns it. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format), the function returns an empty matrix ( ``Mat::data==NULL`` ). Currently, the following file formats are supported:

 * Windows bitmaps - ``*.bmp, *.dib`` (always supported)

 * JPEG files - ``*.jpeg, *.jpg, *.jpe`` (see the *Notes* section)

 * JPEG 2000 files - ``*.jp2`` (see the *Notes* section)

 * Portable Network Graphics - ``*.png`` (see the *Notes* section)

 * Portable image format - ``*.pbm, *.pgm, *.ppm``     (always supported)

 * Sun rasters - ``*.sr, *.ras``     (always supported)

 * TIFF files - ``*.tiff, *.tif`` (see the *Notes* section)

**Notes**:

* The function determines the type of an image by the content, not by the file extension.

* On Microsoft Windows* OS and MacOSX*, the codecs shipped with an OpenCV image (libjpeg, libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs, and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware that currently these native image loaders give images with different pixel values because of the color management embedded into MacOSX.

* On Linux*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for codecs supplied with an OS image. Install the relevant packages (do not forget the development files, for example, "libjpeg-dev", in Debian* and Ubuntu*) to get the codec support or turn on the ``OPENCV_BUILD_3RDPARTY_LIBS`` flag in CMake.

.. index:: imwrite

.. _imwrite:

imwrite
-----------
.. c:function:: bool imwrite( const string\& filename,  const Mat\& img,              const vector<int>\& params=vector<int>())

    Saves an image to a specified file.

    :param filename: Name of the file.

    :param img: Image to be saved.

    :param params: Format-specific save parameters encoded as pairs  ``paramId_1, paramValue_1, paramId_2, paramValue_2, ...`` . The following parameters are currently supported:

        *  For JPEG, it can be a quality ( ``CV_IMWRITE_JPEG_QUALITY`` ) from 0 to 100 (the higher is the better). Default value is 95.

        *  For PNG, it can be the compression level ( ``CV_IMWRITE_PNG_COMPRESSION`` ) from 0 to 9. A higher value means a smaller size and longer compression time. Default value is 3.

        *  For PPM, PGM, or PBM, it can be a binary format flag ( ``CV_IMWRITE_PXM_BINARY`` ), 0 or 1. Default value is 1.

The function ``imwrite`` saves the image to the specified file. The image format is chosen based on the ``filename`` extension (see
:ref:`imread` for the list of extensions). Only 8-bit (or 16-bit in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images can be saved using this function. If the format, depth or channel order is different, use
:ref:`Mat::convertTo` , and
:ref:`cvtColor` to convert it before saving. Or, use the universal XML I/O functions to save the image to XML or YAML format.

.. index:: VideoCapture

.. _VideoCapture:

VideoCapture
------------
.. c:type:: VideoCapture

Class for video capturing from video files or cameras ::

    class VideoCapture
    {
    public:
        // the default constructor
        VideoCapture();
        // the constructor that opens video file
        VideoCapture(const string& filename);
        // the constructor that starts streaming from the camera
        VideoCapture(int device);

        // the destructor
        virtual ~VideoCapture();

        // opens the specified video file
        virtual bool open(const string& filename);

        // starts streaming from the specified camera by its id
        virtual bool open(int device);

        // returns true if the file was open successfully or if the camera
        // has been initialized succesfully
        virtual bool isOpened() const;

        // closes the camera stream or the video file
        // (automatically called by the destructor)
        virtual void release();

        // grab the next frame or a set of frames from a multi-head camera;
        // returns false if there are no more frames
        virtual bool grab();
        // reads the frame from the specified video stream
        // (non-zero channel is only valid for multi-head camera live streams)
        virtual bool retrieve(Mat& image, int channel=0);
        // equivalent to grab() + retrieve(image, 0);
        virtual VideoCapture& operator >> (Mat& image);

        // sets the specified property propId to the specified value
        virtual bool set(int propId, double value);
        // retrieves value of the specified property
        virtual double get(int propId);

    protected:
        ...
    };


The class provides C++ video capturing API. Here is how the class can be used: ::

    #include "cv.h"
    #include "highgui.h"

    using namespace cv;

    int main(int, char**)
    {
        VideoCapture cap(0); // open the default camera
        if(!cap.isOpened())  // check if we succeeded
            return -1;

        Mat edges;
        namedWindow("edges",1);
        for(;;)
        {
            Mat frame;
            cap >> frame; // get a new frame from camera
            cvtColor(frame, edges, CV_BGR2GRAY);
            GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
            Canny(edges, edges, 0, 30, 3);
            imshow("edges", edges);
            if(waitKey(30) >= 0) break;
        }
        // the camera will be deinitialized automatically in VideoCapture destructor
        return 0;
    }


.. index:: VideoCapture::VideoCapture

.. _VideoCapture::VideoCapture:

VideoCapture::VideoCapture
------------------------------
.. c:function:: VideoCapture::VideoCapture()

.. c:function:: VideoCapture::VideoCapture(const string& filename)

.. c:function:: VideoCapture::VideoCapture(int device)

VideoCapture constructors.

    :param filename: name of the opened video file

    :param device: id of the opened video capturing device (i.e. a camera index).

.. index:: VideoCapture::get

.. _VideoCapture::get:

VideoCapture::get
---------------------
.. c:function:: double VideoCapture::get(int property_id)

    :param property_id: Property identifier. It can be one of the following:

        * **CV_CAP_PROP_POS_MSEC** Film current position in milliseconds or video capture timestamp.

        * **CV_CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next.

        * **CV_CAP_PROP_POS_AVI_RATIO** Relative position of the video file: 0 - start of the film, 1 - end of the film.

        * **CV_CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream.

        * **CV_CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream.

        * **CV_CAP_PROP_FPS** Frame rate.

        * **CV_CAP_PROP_FOURCC** 4-character code of codec.

        * **CV_CAP_PROP_FRAME_COUNT** Number of frames in the video file.

        * **CV_CAP_PROP_FORMAT** Format of the Mat objects returned by ``retrieve()`` .

        * **CV_CAP_PROP_MODE** Backend-specific value indicating the current capture mode.

        * **CV_CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras).

        * **CV_CAP_PROP_CONTRAST** Contrast of the image (only for cameras).

        * **CV_CAP_PROP_SATURATION** Saturation of the image (only for cameras).

        * **CV_CAP_PROP_HUE** Hue of the image (only for cameras).

        * **CV_CAP_PROP_GAIN** Gain of the image (only for cameras).

        * **CV_CAP_PROP_EXPOSURE** Exposure (only for cameras).

        * **CV_CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted to RGB.

        * **CV_CAP_PROP_WHITE_BALANCE** Currently unsupported

        * **CV_CAP_PROP_RECTIFICATION** TOWRITE (note: only supported by DC1394 v 2.x backend currently)


**Note**: When querying a property that is not supported by the backend used by the ``VideoCapture`` class, value 0 is returned.

.. index:: VideoCapture::set

.. _VideoCapture::set:

VideoCapture::set
---------------------
.. c:function:: bool VideoCapture::set(int property_id, double value)

    Sets a property in the VideoCapture backend.

    :param property_id: Property identifier. It can be one of the following:

        * **CV_CAP_PROP_POS_MSEC** Film current position in milliseconds or video capture timestamp.

        * **CV_CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next.

        * **CV_CAP_PROP_POS_AVI_RATIO** Relative position of the video file: 0 - start of the film, 1 - end of the film.

        * **CV_CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream.

        * **CV_CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream.

        * **CV_CAP_PROP_FPS** Frame rate.

        * **CV_CAP_PROP_FOURCC** 4-character code of codec.

        * **CV_CAP_PROP_FRAME_COUNT** Number of frames in the video file.

        * **CV_CAP_PROP_FORMAT** Format of the Mat objects returned by ``retrieve()`` .

        * **CV_CAP_PROP_MODE** Backend-specific value indicating the current capture mode.

        * **CV_CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras).

        * **CV_CAP_PROP_CONTRAST** Contrast of the image (only for cameras).

        * **CV_CAP_PROP_SATURATION** Saturation of the image (only for cameras).

        * **CV_CAP_PROP_HUE** Hue of the image (only for cameras).

        * **CV_CAP_PROP_GAIN** Gain of the image (only for cameras).

        * **CV_CAP_PROP_EXPOSURE** Exposure (only for cameras).

        * **CV_CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted to RGB.

        * **CV_CAP_PROP_WHITE_BALANCE** Currently unsupported

        * **CV_CAP_PROP_RECTIFICATION** TOWRITE (note: only supported by DC1394 v 2.x backend currently)

    :param value: Value of the property.



.. index:: VideoWriter

.. _VideoWriter:

VideoWriter
-----------
.. c:type:: VideoWriter

Video writer class ::

    class VideoWriter
    {
    public:
        // default constructor
        VideoWriter();
        // constructor that calls open
        VideoWriter(const string& filename, int fourcc,
                    double fps, Size frameSize, bool isColor=true);

        // the destructor
        virtual ~VideoWriter();

        // opens the file and initializes the video writer.
        // filename - the output file name.
        // fourcc - the codec
        // fps - the number of frames per second
        // frameSize - the video frame size
        // isColor - specifies whether the video stream is color or grayscale
        virtual bool open(const string& filename, int fourcc,
                          double fps, Size frameSize, bool isColor=true);

        // returns true if the writer has been initialized successfully
        virtual bool isOpened() const;

        // writes the next video frame to the stream
        virtual VideoWriter& operator << (const Mat& image);

    protected:
        ...
    };

..

