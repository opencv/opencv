Reading and Writing Images and Video
====================================

.. highlight:: python



.. index:: LoadImage

.. _LoadImage:

LoadImage
---------




.. function:: LoadImage(filename, iscolor=CV_LOAD_IMAGE_COLOR)->None

    Loads an image from a file as an IplImage.





    
    :param filename: Name of file to be loaded. 
    
    :type filename: str
    
    
    :param iscolor: Specific color type of the loaded image: 
         
            * **CV_LOAD_IMAGE_COLOR** the loaded image is forced to be a 3-channel color image 
            
            * **CV_LOAD_IMAGE_GRAYSCALE** the loaded image is forced to be grayscale 
            
            * **CV_LOAD_IMAGE_UNCHANGED** the loaded image will be loaded as is. 
            
            
    
    :type iscolor: int
    
    
    
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




.. function:: LoadImageM(filename, iscolor=CV_LOAD_IMAGE_COLOR)->None

    Loads an image from a file as a CvMat.





    
    :param filename: Name of file to be loaded. 
    
    :type filename: str
    
    
    :param iscolor: Specific color type of the loaded image: 
         
            * **CV_LOAD_IMAGE_COLOR** the loaded image is forced to be a 3-channel color image 
            
            * **CV_LOAD_IMAGE_GRAYSCALE** the loaded image is forced to be grayscale 
            
            * **CV_LOAD_IMAGE_UNCHANGED** the loaded image will be loaded as is. 
            
            
    
    :type iscolor: int
    
    
    
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




.. function:: SaveImage(filename,image)-> None

    Saves an image to a specified file.





    
    :param filename: Name of the file. 
    
    :type filename: str
    
    
    :param image: Image to be saved. 
    
    :type image: :class:`CvArr`
    
    
    
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



.. class:: CvCapture



Video capturing structure.

The structure 
``CvCapture``
does not have a public interface and is used only as a parameter for video capturing functions.


.. index:: CaptureFromCAM

.. _CaptureFromCAM:

CaptureFromCAM
--------------




.. function:: CaptureFromCAM(index) -> CvCapture

    Initializes capturing a video from a camera.





    
    :param index: Index of the camera to be used. If there is only one camera or it does not matter what camera is used -1 may be passed. 
    
    :type index: int
    
    
    
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




.. function:: CaptureFromFile(filename) -> CvCapture

    Initializes capturing a video from a file.





    
    :param filename: Name of the video file. 
    
    :type filename: str
    
    
    
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




.. function:: GetCaptureProperty(capture, property_id)->double

    Gets video capturing properties.





    
    :param capture: video capturing structure. 
    
    :type capture: :class:`CvCapture`
    
    
    :param property_id: Property identifier. Can be one of the following: 
    
    :type property_id: int
    
    
    
        
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




.. function:: GrabFrame(capture) -> int

    Grabs the frame from a camera or file.





    
    :param capture: video capturing structure. 
    
    :type capture: :class:`CvCapture`
    
    
    
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




.. function:: QueryFrame(capture) -> iplimage

    Grabs and returns a frame from a camera or file.





    
    :param capture: video capturing structure. 
    
    :type capture: :class:`CvCapture`
    
    
    
The function 
``cvQueryFrame``
grabs a frame from a camera or video file, decompresses it and returns it. This function is just a combination of 
:ref:`GrabFrame`
and 
:ref:`RetrieveFrame`
, but in one call. The returned image should not be released or modified by the user.  In the event of an error, the return value may be NULL.


.. index:: RetrieveFrame

.. _RetrieveFrame:

RetrieveFrame
-------------




.. function:: RetrieveFrame(capture) -> iplimage

    Gets the image grabbed with cvGrabFrame.





    
    :param capture: video capturing structure. 
    
    :type capture: :class:`CvCapture`
    
    
    
The function 
``cvRetrieveFrame``
returns the pointer to the image grabbed with the 
:ref:`GrabFrame`
function. The returned image should not be released or modified by the user.  In the event of an error, the return value may be NULL.



.. index:: SetCaptureProperty

.. _SetCaptureProperty:

SetCaptureProperty
------------------




.. function:: SetCaptureProperty(capture, property_id,value)->None

    Sets video capturing properties.





    
    :param capture: video capturing structure. 
    
    :type capture: :class:`CvCapture`
    
    
    :param property_id: property identifier. Can be one of the following: 
    
    :type property_id: int
    
    
    
        
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
    
    :type value: float
    
    
    
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




.. function:: CreateVideoWriter(filename, fourcc, fps, frame_size, is_color) -> CvVideoWriter

    Creates the video file writer.





    
    :param filename: Name of the output video file. 
    
    :type filename: str
    
    
    :param fourcc: 4-character code of codec used to compress the frames. For example, ``CV_FOURCC('P','I','M,'1')``  is a MPEG-1 codec, ``CV_FOURCC('M','J','P','G')``  is a motion-jpeg codec etc.
        Under Win32 it is possible to pass -1 in order to choose compression method and additional compression parameters from dialog. Under Win32 if 0 is passed while using an avi filename it will create a video writer that creates an uncompressed avi file. 
    
    :type fourcc: int
    
    
    :param fps: Framerate of the created video stream. 
    
    :type fps: float
    
    
    :param frame_size: Size of the  video frames. 
    
    :type frame_size: :class:`CvSize`
    
    
    :param is_color: If it is not zero, the encoder will expect and encode color frames, otherwise it will work with grayscale frames (the flag is currently supported on Windows only). 
    
    :type is_color: int
    
    
    
The function 
``cvCreateVideoWriter``
creates the video writer structure.

Which codecs and file formats are supported depends on the back end library. On Windows HighGui uses Video for Windows (VfW), on Linux ffmpeg is used and on Mac OS X the back end is QuickTime. See VideoCodecs for some discussion on what to expect.



.. index:: WriteFrame

.. _WriteFrame:

WriteFrame
----------




.. function:: WriteFrame(writer, image)->int

    Writes a frame to a video file.





    
    :param writer: Video writer structure 
    
    :type writer: :class:`CvVideoWriter`
    
    
    :param image: The written frame 
    
    :type image: :class:`IplImage`
    
    
    
The function 
``cvWriteFrame``
writes/appends one frame to a video file.

