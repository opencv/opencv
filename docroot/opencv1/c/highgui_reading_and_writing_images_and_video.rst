Reading and Writing Images and Video
====================================

.. highlight:: c



.. index:: LoadImage

.. _LoadImage:

LoadImage
---------

`id=0.469255746245 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/LoadImage>`__




.. cfunction:: IplImage* cvLoadImage(  const char* filename,  int iscolor=CV_LOAD_IMAGE_COLOR )

    Loads an image from a file as an IplImage.





    
    :param filename: Name of file to be loaded. 
    
    
    :param iscolor: Specific color type of the loaded image: 
         
            * **CV_LOAD_IMAGE_COLOR** the loaded image is forced to be a 3-channel color image 
            
            * **CV_LOAD_IMAGE_GRAYSCALE** the loaded image is forced to be grayscale 
            
            * **CV_LOAD_IMAGE_UNCHANGED** the loaded image will be loaded as is. 
            
            
    
    
    
The function 
``cvLoadImage``
loads an image from the specified file and returns the pointer to the loaded image. Currently the following file formats are supported:


    

*
    Windows bitmaps - BMP, DIB
    

*
    JPEG files - JPEG, JPG, JPE
    

*
    Portable Network Graphics - PNG
    

*
    Portable image format - PBM, PGM, PPM
    

*
    Sun rasters - SR, RAS
    

*
    TIFF files - TIFF, TIF
    
    
Note that in the current implementation the alpha channel, if any, is stripped from the output image, e.g. 4-channel RGBA image will be loaded as RGB.


.. index:: LoadImageM

.. _LoadImageM:

LoadImageM
----------

`id=0.563485365507 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/LoadImageM>`__




.. cfunction:: CvMat* cvLoadImageM(  const char* filename,  int iscolor=CV_LOAD_IMAGE_COLOR )

    Loads an image from a file as a CvMat.





    
    :param filename: Name of file to be loaded. 
    
    
    :param iscolor: Specific color type of the loaded image: 
         
            * **CV_LOAD_IMAGE_COLOR** the loaded image is forced to be a 3-channel color image 
            
            * **CV_LOAD_IMAGE_GRAYSCALE** the loaded image is forced to be grayscale 
            
            * **CV_LOAD_IMAGE_UNCHANGED** the loaded image will be loaded as is. 
            
            
    
    
    
The function 
``cvLoadImageM``
loads an image from the specified file and returns the pointer to the loaded image.
urrently the following file formats are supported:


    

*
    Windows bitmaps - BMP, DIB
    

*
    JPEG files - JPEG, JPG, JPE
    

*
    Portable Network Graphics - PNG
    

*
    Portable image format - PBM, PGM, PPM
    

*
    Sun rasters - SR, RAS
    

*
    TIFF files - TIFF, TIF
    
    
Note that in the current implementation the alpha channel, if any, is stripped from the output image, e.g. 4-channel RGBA image will be loaded as RGB.


.. index:: SaveImage

.. _SaveImage:

SaveImage
---------

`id=0.495970549198 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/SaveImage>`__




.. cfunction:: int cvSaveImage( const char* filename, const CvArr* image )

    Saves an image to a specified file.





    
    :param filename: Name of the file. 
    
    
    :param image: Image to be saved. 
    
    
    
The function 
``cvSaveImage``
saves the image to the specified file. The image format is chosen based on the 
``filename``
extension, see 
:ref:`LoadImage`
. Only 8-bit single-channel or 3-channel (with 'BGR' channel order) images can be saved using this function. If the format, depth or channel order is different, use 
``cvCvtScale``
and 
``cvCvtColor``
to convert it before saving, or use universal 
``cvSave``
to save the image to XML or YAML format.



.. index:: CvCapture

.. _CvCapture:

CvCapture
---------

`id=0.279260095238 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/CvCapture>`__

.. ctype:: CvCapture



Video capturing structure.



.. cfunction:: typedef struct CvCapture CvCapture

    


The structure 
``CvCapture``
does not have a public interface and is used only as a parameter for video capturing functions.


.. index:: CaptureFromCAM

.. _CaptureFromCAM:

CaptureFromCAM
--------------

`id=0.051648241367 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/CaptureFromCAM>`__




.. cfunction:: CvCapture* cvCaptureFromCAM( int index )

    Initializes capturing a video from a camera.





    
    :param index: Index of the camera to be used. If there is only one camera or it does not matter what camera is used -1 may be passed. 
    
    
    
The function 
``cvCaptureFromCAM``
allocates and initializes the CvCapture structure for reading a video stream from the camera. Currently two camera interfaces can be used on Windows: Video for Windows (VFW) and Matrox Imaging Library (MIL); and two on Linux: V4L and FireWire (IEEE1394).

To release the structure, use 
:ref:`ReleaseCapture`
.



.. index:: CaptureFromFile

.. _CaptureFromFile:

CaptureFromFile
---------------

`id=0.832457799312 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/CaptureFromFile>`__




.. cfunction:: CvCapture* cvCaptureFromFile( const char* filename )

    Initializes capturing a video from a file.





    
    :param filename: Name of the video file. 
    
    
    
The function 
``cvCaptureFromFile``
allocates and initializes the CvCapture structure for reading the video stream from the specified file. Which codecs and file formats are supported depends on the back end library. On Windows HighGui uses Video for Windows (VfW), on Linux ffmpeg is used and on Mac OS X the back end is QuickTime. See VideoCodecs for some discussion on what to expect and how to prepare your video files.

After the allocated structure is not used any more it should be released by the 
:ref:`ReleaseCapture`
function.


.. index:: GetCaptureProperty

.. _GetCaptureProperty:

GetCaptureProperty
------------------

`id=0.315272026867 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/GetCaptureProperty>`__




.. cfunction:: double cvGetCaptureProperty( CvCapture* capture, int property_id )

    Gets video capturing properties.





    
    :param capture: video capturing structure. 
    
    
    :param property_id: Property identifier. Can be one of the following: 
    
    
    
        
        * **CV_CAP_PROP_POS_MSEC** Film current position in milliseconds or video capture timestamp 
        
        
        * **CV_CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next 
        
        
        * **CV_CAP_PROP_POS_AVI_RATIO** Relative position of the video file (0 - start of the film, 1 - end of the film) 
        
        
        * **CV_CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream 
        
        
        * **CV_CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream 
        
        
        * **CV_CAP_PROP_FPS** Frame rate 
        
        
        * **CV_CAP_PROP_FOURCC** 4-character code of codec 
        
        
        * **CV_CAP_PROP_FRAME_COUNT** Number of frames in the video file 
        
        
        * **CV_CAP_PROP_FORMAT** The format of the Mat objects returned by retrieve() 
        
        
        * **CV_CAP_PROP_MODE** A backend-specific value indicating the current capture mode 
        
        
        * **CV_CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_CONTRAST** Contrast of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_SATURATION** Saturation of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_HUE** Hue of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_GAIN** Gain of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_EXPOSURE** Exposure (only for cameras) 
        
        
        * **CV_CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted to RGB 
        
        
        * **CV_CAP_PROP_WHITE_BALANCE** Currently unsupported 
        
        
        * **CV_CAP_PROP_RECTIFICATION** TOWRITE (note: only supported by DC1394 v 2.x backend currently) 
        
        
        
    
    
The function 
``cvGetCaptureProperty``
retrieves the specified property of the camera or video file.


.. index:: GrabFrame

.. _GrabFrame:

GrabFrame
---------

`id=0.423832304356 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/GrabFrame>`__




.. cfunction:: int cvGrabFrame( CvCapture* capture )

    Grabs the frame from a camera or file.





    
    :param capture: video capturing structure. 
    
    
    
The function 
``cvGrabFrame``
grabs the frame from a camera or file. The grabbed frame is stored internally. The purpose of this function is to grab the frame 
*quickly*
so that syncronization can occur if it has to read from several cameras simultaneously. The grabbed frames are not exposed because they may be stored in a compressed format (as defined by the camera/driver). To retrieve the grabbed frame, 
:ref:`RetrieveFrame`
should be used.



.. index:: QueryFrame

.. _QueryFrame:

QueryFrame
----------

`id=0.155007724473 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/QueryFrame>`__




.. cfunction:: IplImage* cvQueryFrame( CvCapture* capture )

    Grabs and returns a frame from a camera or file.





    
    :param capture: video capturing structure. 
    
    
    
The function 
``cvQueryFrame``
grabs a frame from a camera or video file, decompresses it and returns it. This function is just a combination of 
:ref:`GrabFrame`
and 
:ref:`RetrieveFrame`
, but in one call. The returned image should not be released or modified by the user.  In the event of an error, the return value may be NULL.


.. index:: ReleaseCapture

.. _ReleaseCapture:

ReleaseCapture
--------------

`id=0.412581622343 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/ReleaseCapture>`__




.. cfunction:: void cvReleaseCapture( CvCapture** capture )

    Releases the CvCapture structure.





    
    :param capture: Pointer to video the capturing structure. 
    
    
    
The function 
``cvReleaseCapture``
releases the CvCapture structure allocated by 
:ref:`CaptureFromFile`
or 
:ref:`CaptureFromCAM`
.

.. index:: RetrieveFrame

.. _RetrieveFrame:

RetrieveFrame
-------------

`id=0.780832955331 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/RetrieveFrame>`__




.. cfunction:: IplImage* cvRetrieveFrame( CvCapture* capture )

    Gets the image grabbed with cvGrabFrame.





    
    :param capture: video capturing structure. 
    
    
    
The function 
``cvRetrieveFrame``
returns the pointer to the image grabbed with the 
:ref:`GrabFrame`
function. The returned image should not be released or modified by the user.  In the event of an error, the return value may be NULL.



.. index:: SetCaptureProperty

.. _SetCaptureProperty:

SetCaptureProperty
------------------

`id=0.0459451505183 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/SetCaptureProperty>`__




.. cfunction:: int cvSetCaptureProperty(  CvCapture* capture,  int property_id,  double value )

    Sets video capturing properties.





    
    :param capture: video capturing structure. 
    
    
    :param property_id: property identifier. Can be one of the following: 
    
    
    
        
        * **CV_CAP_PROP_POS_MSEC** Film current position in milliseconds or video capture timestamp 
        
        
        * **CV_CAP_PROP_POS_FRAMES** 0-based index of the frame to be decoded/captured next 
        
        
        * **CV_CAP_PROP_POS_AVI_RATIO** Relative position of the video file (0 - start of the film, 1 - end of the film) 
        
        
        * **CV_CAP_PROP_FRAME_WIDTH** Width of the frames in the video stream 
        
        
        * **CV_CAP_PROP_FRAME_HEIGHT** Height of the frames in the video stream 
        
        
        * **CV_CAP_PROP_FPS** Frame rate 
        
        
        * **CV_CAP_PROP_FOURCC** 4-character code of codec 
        
        
        * **CV_CAP_PROP_FRAME_COUNT** Number of frames in the video file 
        
        
        * **CV_CAP_PROP_FORMAT** The format of the Mat objects returned by retrieve() 
        
        
        * **CV_CAP_PROP_MODE** A backend-specific value indicating the current capture mode 
        
        
        * **CV_CAP_PROP_BRIGHTNESS** Brightness of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_CONTRAST** Contrast of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_SATURATION** Saturation of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_HUE** Hue of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_GAIN** Gain of the image (only for cameras) 
        
        
        * **CV_CAP_PROP_EXPOSURE** Exposure (only for cameras) 
        
        
        * **CV_CAP_PROP_CONVERT_RGB** Boolean flags indicating whether images should be converted to RGB 
        
        
        * **CV_CAP_PROP_WHITE_BALANCE** Currently unsupported 
        
        
        * **CV_CAP_PROP_RECTIFICATION** TOWRITE (note: only supported by DC1394 v 2.x backend currently) 
        
        
        
    
    :param value: value of the property. 
    
    
    
The function 
``cvSetCaptureProperty``
sets the specified property of video capturing. Currently the function supports only video files: 
``CV_CAP_PROP_POS_MSEC, CV_CAP_PROP_POS_FRAMES, CV_CAP_PROP_POS_AVI_RATIO``
.

NB This function currently does nothing when using the latest CVS download on linux with FFMPEG (the function contents are hidden if 0 is used and returned).



.. index:: CreateVideoWriter

.. _CreateVideoWriter:

CreateVideoWriter
-----------------

`id=0.960560559623 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/CreateVideoWriter>`__




.. cfunction:: typedef struct CvVideoWriter CvVideoWriter CvVideoWriter* cvCreateVideoWriter(  const char* filename,  int fourcc,  double fps,  CvSize frame_size,  int is_color=1 )

    Creates the video file writer.





    
    :param filename: Name of the output video file. 
    
    
    :param fourcc: 4-character code of codec used to compress the frames. For example, ``CV_FOURCC('P','I','M,'1')``  is a MPEG-1 codec, ``CV_FOURCC('M','J','P','G')``  is a motion-jpeg codec etc.
        Under Win32 it is possible to pass -1 in order to choose compression method and additional compression parameters from dialog. Under Win32 if 0 is passed while using an avi filename it will create a video writer that creates an uncompressed avi file. 
    
    
    :param fps: Framerate of the created video stream. 
    
    
    :param frame_size: Size of the  video frames. 
    
    
    :param is_color: If it is not zero, the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only). 
    
    
    
The function 
``cvCreateVideoWriter``
creates the video writer structure.

Which codecs and file formats are supported depends on the back end library. On Windows HighGui uses Video for Windows (VfW), on Linux ffmpeg is used and on Mac OS X the back end is QuickTime. See VideoCodecs for some discussion on what to expect.



.. index:: ReleaseVideoWriter

.. _ReleaseVideoWriter:

ReleaseVideoWriter
------------------

`id=0.271528060303 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/ReleaseVideoWriter>`__




.. cfunction:: void cvReleaseVideoWriter( CvVideoWriter** writer )

    Releases the AVI writer.





    
    :param writer: Pointer to the video file writer structure. 
    
    
    
The function 
``cvReleaseVideoWriter``
finishes writing to the video file and releases the structure.

.. index:: WriteFrame

.. _WriteFrame:

WriteFrame
----------

`id=0.0551918795805 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/highgui/WriteFrame>`__




.. cfunction:: int cvWriteFrame( CvVideoWriter* writer, const IplImage* image )

    Writes a frame to a video file.





    
    :param writer: Video writer structure 
    
    
    :param image: The written frame 
    
    
    
The function 
``cvWriteFrame``
writes/appends one frame to a video file.

