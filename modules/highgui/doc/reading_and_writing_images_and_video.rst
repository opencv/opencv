Reading and Writing Images and Video
====================================

.. highlight:: cpp

imdecode
------------
Reads an image from a buffer in memory.

.. ocv:function:: Mat imdecode( InputArray buf,  int flags )

.. ocv:pyfunction:: cv2.imdecode(buf, flags) -> retval

    :param buf: Input array of vector of bytes.

    :param flags: The same flags as in  :ocv:func:`imread` .
    
The function reads an image from the specified buffer in the memory.
If the buffer is too short or contains invalid data, the empty matrix is returned.

See
:ocv:func:`imread` for the list of supported formats and flags description.

imencode
------------
Encodes an image into a memory buffer.

.. ocv:function:: bool imencode( const string& ext, InputArray img, vector<uchar>& buf, const vector<int>& params=vector<int>())

.. ocv:pyfunction:: cv2.imencode(ext, img, buf[, params]) -> retval

    :param ext: File extension that defines the output format.

    :param img: Image to be written.

    :param buf: Output buffer resized to fit the compressed image.

    :param params: Format-specific parameters. See  :ocv:func:`imwrite` .

The function compresses the image and stores it in the memory buffer that is resized to fit the result.
See
:ocv:func:`imwrite` for the list of supported formats and flags description.

imread
----------
Loads an image from a file.

.. ocv:function:: Mat imread( const string& filename, int flags=1 )

.. ocv:pyfunction:: cv2.imread(filename[, flags]) -> retval

.. ocv:cfunction:: IplImage* cvLoadImage( const char* filename, int flags=CV_LOAD_IMAGE_COLOR )

.. ocv:cfunction:: CvMat* cvLoadImageM( const char* filename, int flags=CV_LOAD_IMAGE_COLOR )

.. ocv:pyoldfunction:: cv.LoadImage(filename, flags=CV_LOAD_IMAGE_COLOR)->None

.. ocv:pyoldfunction:: cv.LoadImageM(filename, flags=CV_LOAD_IMAGE_COLOR)->None

    :param filename: Name of file to be loaded.

    :param flags: Flags specifying the color type of a loaded image:

        * **>0**  Return a 3-channel color image

        * **=0**  Return a grayscale image

        * **<0**  Return the loaded image as is. Note that in the current implementation the alpha channel, if any, is stripped from the output image. For example, a 4-channel RGBA image is loaded as RGB if  :math:`flags\ge0` .

The function ``imread`` loads an image from the specified file and returns it. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format), the function returns an empty matrix ( ``Mat::data==NULL`` ). Currently, the following file formats are supported:

 * Windows bitmaps - ``*.bmp, *.dib`` (always supported)

 * JPEG files - ``*.jpeg, *.jpg, *.jpe`` (see the *Notes* section)

 * JPEG 2000 files - ``*.jp2`` (see the *Notes* section)

 * Portable Network Graphics - ``*.png`` (see the *Notes* section)

 * Portable image format - ``*.pbm, *.pgm, *.ppm``     (always supported)

 * Sun rasters - ``*.sr, *.ras``     (always supported)

 * TIFF files - ``*.tiff, *.tif`` (see the *Notes* section)

.. note::

    * The function determines the type of an image by the content, not by the file extension.

    * On Microsoft Windows* OS and MacOSX*, the codecs shipped with an OpenCV image (libjpeg, libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs, and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware that currently these native image loaders give images with different pixel values because of the color management embedded into MacOSX.

    * On Linux*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for codecs supplied with an OS image. Install the relevant packages (do not forget the development files, for example, "libjpeg-dev", in Debian* and Ubuntu*) to get the codec support or turn on the ``OPENCV_BUILD_3RDPARTY_LIBS`` flag in CMake.

imwrite
-----------
Saves an image to a specified file.

.. ocv:function:: bool imwrite( const string& filename, InputArray image, const vector<int>& params=vector<int>())

.. ocv:pyfunction:: cv2.imwrite(filename, image[, params]) -> retval

.. ocv:cfunction:: int cvSaveImage( const char* filename, const CvArr* image )

.. ocv:pyoldfunction:: cv.SaveImage(filename, image)-> None

    :param filename: Name of the file.

    :param image: Image to be saved.

    :param params: Format-specific save parameters encoded as pairs  ``paramId_1, paramValue_1, paramId_2, paramValue_2, ...`` . The following parameters are currently supported:

        *  For JPEG, it can be a quality ( ``CV_IMWRITE_JPEG_QUALITY`` ) from 0 to 100 (the higher is the better). Default value is 95.

        *  For PNG, it can be the compression level ( ``CV_IMWRITE_PNG_COMPRESSION`` ) from 0 to 9. A higher value means a smaller size and longer compression time. Default value is 3.

        *  For PPM, PGM, or PBM, it can be a binary format flag ( ``CV_IMWRITE_PXM_BINARY`` ), 0 or 1. Default value is 1.

The function ``imwrite`` saves the image to the specified file. The image format is chosen based on the ``filename`` extension (see
:ocv:func:`imread` for the list of extensions). Only 8-bit (or 16-bit in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images can be saved using this function. If the format, depth or channel order is different, use
:ocv:func:`Mat::convertTo` , and
:ocv:func:`cvtColor` to convert it before saving. Or, use the universal XML I/O functions to save the image to XML or YAML format.

VideoCapture
------------
.. ocv:class:: VideoCapture

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


VideoCapture::VideoCapture
------------------------------
VideoCapture constructors.

.. ocv:function:: VideoCapture::VideoCapture()

.. ocv:function:: VideoCapture::VideoCapture(const string& filename)

.. ocv:function:: VideoCapture::VideoCapture(int device)

    :param filename: name of the opened video file

    :param device: id of the opened video capturing device (i.e. a camera index).


VideoCapture::get
---------------------
Returns the specified ``VideoCapture`` property 

.. ocv:function:: double VideoCapture::get(int property_id)

.. ocv:pyfunction:: cv2.VideoCapture.get(propId) -> retval

    :param property_id: Property identifier. It can be one of the following:

        * **CV_CAP_PROP_POS_MSEC** Current position of the video file in milliseconds or video capture timestamp.

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

        * **CV_CAP_PROP_WHITE_BALANCE** Currently not supported

        * **CV_CAP_PROP_RECTIFICATION** Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)


**Note**: When querying a property that is not supported by the backend used by the ``VideoCapture`` class, value 0 is returned.

VideoCapture::set
---------------------
Sets a property in the ``VideoCapture``.

.. ocv:function:: bool VideoCapture::set(int property_id, double value)

.. ocv:pyfunction:: cv2.VideoCapture.set(propId, value) -> retval

    :param property_id: Property identifier. It can be one of the following:

        * **CV_CAP_PROP_POS_MSEC** Current position of the video file in milliseconds.

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

        * **CV_CAP_PROP_RECTIFICATION** Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

    :param value: Value of the property.

VideoWriter
-----------
.. ocv:class:: VideoWriter

Video writer class. ::

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

For more detailed description see http://opencv.willowgarage.com/wiki/documentation/cpp/highgui/VideoWriter
..

