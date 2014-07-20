Reading and Writing Video
=========================

.. highlight:: cpp


VideoCapture
------------
.. ocv:class:: VideoCapture

Class for video capturing from video files, image sequences or cameras.
The class provides C++ API for capturing video from cameras or for reading video files and image sequences. Here is how the class can be used: ::

    #include "opencv2/opencv.hpp"

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
            cvtColor(frame, edges, COLOR_BGR2GRAY);
            GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
            Canny(edges, edges, 0, 30, 3);
            imshow("edges", edges);
            if(waitKey(30) >= 0) break;
        }
        // the camera will be deinitialized automatically in VideoCapture destructor
        return 0;
    }


.. note:: In C API the black-box structure ``CvCapture`` is used instead of ``VideoCapture``.

.. note::

   * A basic sample on using the VideoCapture interface can be found at opencv_source_code/samples/cpp/starter_video.cpp
   * Another basic video processing sample can be found at opencv_source_code/samples/cpp/video_dmtx.cpp

   * (Python) A basic sample on using the VideoCapture interface can be found at opencv_source_code/samples/python2/video.py
   * (Python) Another basic video processing sample can be found at opencv_source_code/samples/python2/video_dmtx.py
   * (Python) A multi threaded video processing sample can be found at opencv_source_code/samples/python2/video_threaded.py


VideoCapture::VideoCapture
------------------------------
VideoCapture constructors.

.. ocv:function:: VideoCapture::VideoCapture()

.. ocv:function:: VideoCapture::VideoCapture(const String& filename)

.. ocv:function:: VideoCapture::VideoCapture(int device)

.. ocv:pyfunction:: cv2.VideoCapture() -> <VideoCapture object>
.. ocv:pyfunction:: cv2.VideoCapture(filename) -> <VideoCapture object>
.. ocv:pyfunction:: cv2.VideoCapture(device) -> <VideoCapture object>

.. ocv:cfunction:: CvCapture* cvCaptureFromCAM( int device )
.. ocv:cfunction:: CvCapture* cvCaptureFromFile( const char* filename )

    :param filename: name of the opened video file (eg. video.avi) or image sequence (eg. img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)

    :param device: id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0.

.. note:: In C API, when you finished working with video, release ``CvCapture`` structure with ``cvReleaseCapture()``, or use ``Ptr<CvCapture>`` that calls ``cvReleaseCapture()`` automatically in the destructor.


VideoCapture::open
---------------------
Open video file or a capturing device for video capturing

.. ocv:function:: bool VideoCapture::open(const String& filename)
.. ocv:function:: bool VideoCapture::open(int device)

.. ocv:pyfunction:: cv2.VideoCapture.open(filename) -> retval
.. ocv:pyfunction:: cv2.VideoCapture.open(device) -> retval

    :param filename: name of the opened video file (eg. video.avi) or image sequence (eg. img_%02d.jpg, which will read samples like img_00.jpg, img_01.jpg, img_02.jpg, ...)

    :param device: id of the opened video capturing device (i.e. a camera index).

The methods first call :ocv:func:`VideoCapture::release` to close the already opened file or camera.


VideoCapture::isOpened
----------------------
Returns true if video capturing has been initialized already.

.. ocv:function:: bool VideoCapture::isOpened()

.. ocv:pyfunction:: cv2.VideoCapture.isOpened() -> retval

If the previous call to ``VideoCapture`` constructor or ``VideoCapture::open`` succeeded, the method returns true.

VideoCapture::release
---------------------
Closes video file or capturing device.

.. ocv:function:: void VideoCapture::release()

.. ocv:pyfunction:: cv2.VideoCapture.release() -> None

.. ocv:cfunction:: void cvReleaseCapture(CvCapture** capture)

The methods are automatically called by subsequent :ocv:func:`VideoCapture::open` and by ``VideoCapture`` destructor.

The C function also deallocates memory and clears ``*capture`` pointer.


VideoCapture::grab
---------------------
Grabs the next frame from video file or capturing device.

.. ocv:function:: bool VideoCapture::grab()

.. ocv:pyfunction:: cv2.VideoCapture.grab() -> retval

.. ocv:cfunction:: int cvGrabFrame(CvCapture* capture)

The methods/functions grab the next frame from video file or camera and return true (non-zero) in the case of success.

The primary use of the function is in multi-camera environments, especially when the cameras do not have hardware synchronization. That is, you call ``VideoCapture::grab()`` for each camera and after that call the slower method ``VideoCapture::retrieve()`` to decode and get frame from each camera. This way the overhead on demosaicing or motion jpeg decompression etc. is eliminated and the retrieved frames from different cameras will be closer in time.

Also, when a connected camera is multi-head (for example, a stereo camera or a Kinect device), the correct way of retrieving data from it is to call `VideoCapture::grab` first and then call :ocv:func:`VideoCapture::retrieve` one or more times with different values of the ``channel`` parameter. See https://github.com/Itseez/opencv/tree/master/samples/cpp/openni_capture.cpp


VideoCapture::retrieve
----------------------
Decodes and returns the grabbed video frame.

.. ocv:function:: bool VideoCapture::retrieve( OutputArray image, int flag=0 )

.. ocv:pyfunction:: cv2.VideoCapture.retrieve([image[, flag]]) -> retval, image

.. ocv:cfunction:: IplImage* cvRetrieveFrame( CvCapture* capture, int streamIdx=0 )

The methods/functions decode and return the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the methods return false and the functions return NULL pointer.

.. note:: OpenCV 1.x functions ``cvRetrieveFrame`` and ``cv.RetrieveFrame`` return image stored inside the video capturing structure. It is not allowed to modify or release the image! You can copy the frame using :ocv:cfunc:`cvCloneImage` and then do whatever you want with the copy.


VideoCapture::read
----------------------
Grabs, decodes and returns the next video frame.

.. ocv:function:: VideoCapture& VideoCapture::operator >> (Mat& image)

.. ocv:function:: VideoCapture& VideoCapture::operator >> (UMat& image)

.. ocv:function:: bool VideoCapture::read(OutputArray image)

.. ocv:pyfunction:: cv2.VideoCapture.read([image]) -> retval, image

.. ocv:cfunction:: IplImage* cvQueryFrame(CvCapture* capture)

The methods/functions combine :ocv:func:`VideoCapture::grab` and :ocv:func:`VideoCapture::retrieve` in one call. This is the most convenient method for reading video files or capturing data from decode and return the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the methods return false and the functions return NULL pointer.

.. note:: OpenCV 1.x functions ``cvRetrieveFrame`` and ``cv.RetrieveFrame`` return image stored inside the video capturing structure. It is not allowed to modify or release the image! You can copy the frame using :ocv:cfunc:`cvCloneImage` and then do whatever you want with the copy.


VideoCapture::get
---------------------
Returns the specified ``VideoCapture`` property

.. ocv:function:: double VideoCapture::get(int propId)

.. ocv:pyfunction:: cv2.VideoCapture.get(propId) -> retval

.. ocv:cfunction:: double cvGetCaptureProperty( CvCapture* capture, int property_id )

    :param propId: Property identifier. It can be one of the following:

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

.. ocv:function:: bool VideoCapture::set( int propId, double value )

.. ocv:pyfunction:: cv2.VideoCapture.set(propId, value) -> retval

.. ocv:cfunction:: int cvSetCaptureProperty( CvCapture* capture, int property_id, double value )

    :param propId: Property identifier. It can be one of the following:

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

Video writer class.



VideoWriter::VideoWriter
------------------------
VideoWriter constructors

.. ocv:function:: VideoWriter::VideoWriter()

.. ocv:function:: VideoWriter::VideoWriter(const String& filename, int fourcc, double fps, Size frameSize, bool isColor=true)

.. ocv:pyfunction:: cv2.VideoWriter([filename, fourcc, fps, frameSize[, isColor]]) -> <VideoWriter object>

.. ocv:cfunction:: CvVideoWriter* cvCreateVideoWriter( const char* filename, int fourcc, double fps, CvSize frame_size, int is_color=1 )

.. ocv:pyfunction:: cv2.VideoWriter.isOpened() -> retval
.. ocv:pyfunction:: cv2.VideoWriter.open(filename, fourcc, fps, frameSize[, isColor]) -> retval
.. ocv:pyfunction:: cv2.VideoWriter.write(image) -> None

    :param filename: Name of the output video file.

    :param fourcc: 4-character code of codec used to compress the frames. For example, ``CV_FOURCC('P','I','M','1')``  is a MPEG-1 codec, ``CV_FOURCC('M','J','P','G')``  is a motion-jpeg codec etc. List of codes can be obtained at `Video Codecs by FOURCC <http://www.fourcc.org/codecs.php>`_ page.

    :param fps: Framerate of the created video stream.

    :param frameSize: Size of the  video frames.

    :param isColor: If it is not zero, the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only).

The constructors/functions initialize video writers. On Linux FFMPEG is used to write videos; on Windows FFMPEG or VFW is used; on MacOSX QTKit is used.



ReleaseVideoWriter
------------------
Releases the AVI writer.

.. ocv:cfunction:: void cvReleaseVideoWriter( CvVideoWriter** writer )

The function should be called after you finished using ``CvVideoWriter`` opened with :ocv:cfunc:`CreateVideoWriter`.


VideoWriter::open
-----------------
Initializes or reinitializes video writer.

.. ocv:function:: bool VideoWriter::open(const String& filename, int fourcc, double fps, Size frameSize, bool isColor=true)

.. ocv:pyfunction:: cv2.VideoWriter.open(filename, fourcc, fps, frameSize[, isColor]) -> retval

The method opens video writer. Parameters are the same as in the constructor :ocv:func:`VideoWriter::VideoWriter`.


VideoWriter::isOpened
---------------------
Returns true if video writer has been successfully initialized.

.. ocv:function:: bool VideoWriter::isOpened()

.. ocv:pyfunction:: cv2.VideoWriter.isOpened() -> retval


VideoWriter::write
------------------
Writes the next video frame

.. ocv:function:: VideoWriter& VideoWriter::operator << (const Mat& image)

.. ocv:function:: void VideoWriter::write(const Mat& image)

.. ocv:pyfunction:: cv2.VideoWriter.write(image) -> None

.. ocv:cfunction:: int cvWriteFrame( CvVideoWriter* writer, const IplImage* image )

    :param writer: Video writer structure (OpenCV 1.x API)

    :param image: The written frame

The functions/methods write the specified image to video file. It must have the same size as has been specified when opening the video writer.
