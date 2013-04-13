Video Decoding
==============

.. highlight:: cpp



gpu::VideoReader_GPU
--------------------
Video reader class.

.. ocv:class:: gpu::VideoReader_GPU



gpu::VideoReader_GPU::Codec
---------------------------

Video codecs supported by :ocv:class:`gpu::VideoReader_GPU` .

.. ocv:enum:: gpu::VideoReader_GPU::Codec

  .. ocv:emember:: MPEG1 = 0
  .. ocv:emember:: MPEG2
  .. ocv:emember:: MPEG4
  .. ocv:emember:: VC1
  .. ocv:emember:: H264
  .. ocv:emember:: JPEG
  .. ocv:emember:: H264_SVC
  .. ocv:emember:: H264_MVC

  .. ocv:emember:: Uncompressed_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V'))

        Y,U,V (4:2:0)

  .. ocv:emember:: Uncompressed_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2'))

        Y,V,U (4:2:0)

  .. ocv:emember:: Uncompressed_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2'))

        Y,UV  (4:2:0)

  .. ocv:emember:: Uncompressed_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V'))

        YUYV/YUY2 (4:2:2)

  .. ocv:emember:: Uncompressed_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y'))

        UYVY (4:2:2)


gpu::VideoReader_GPU::ChromaFormat
----------------------------------

Chroma formats supported by :ocv:class:`gpu::VideoReader_GPU` .

.. ocv:enum:: gpu::VideoReader_GPU::ChromaFormat

  .. ocv:emember:: Monochrome = 0
  .. ocv:emember:: YUV420
  .. ocv:emember:: YUV422
  .. ocv:emember:: YUV444


gpu::VideoReader_GPU::FormatInfo
--------------------------------
.. ocv:struct:: gpu::VideoReader_GPU::FormatInfo

Struct providing information about video file format. ::

    struct FormatInfo
    {
        Codec codec;
        ChromaFormat chromaFormat;
        int width;
        int height;
    };


gpu::VideoReader_GPU::VideoReader_GPU
-------------------------------------
Constructors.

.. ocv:function:: gpu::VideoReader_GPU::VideoReader_GPU()
.. ocv:function:: gpu::VideoReader_GPU::VideoReader_GPU(const String& filename)
.. ocv:function:: gpu::VideoReader_GPU::VideoReader_GPU(const cv::Ptr<VideoSource>& source)

    :param filename: Name of the input video file.

    :param source: Video file parser implemented by user.

The constructors initialize video reader. FFMPEG is used to read videos. User can implement own demultiplexing with :ocv:class:`gpu::VideoReader_GPU::VideoSource` .



gpu::VideoReader_GPU::open
--------------------------
Initializes or reinitializes video reader.

.. ocv:function:: void gpu::VideoReader_GPU::open(const String& filename)
.. ocv:function:: void gpu::VideoReader_GPU::open(const cv::Ptr<VideoSource>& source)

The method opens video reader. Parameters are the same as in the constructor :ocv:func:`gpu::VideoReader_GPU::VideoReader_GPU` . The method throws :ocv:class:`Exception` if error occurs.



gpu::VideoReader_GPU::isOpened
------------------------------
Returns true if video reader has been successfully initialized.

.. ocv:function:: bool gpu::VideoReader_GPU::isOpened() const



gpu::VideoReader_GPU::close
---------------------------
Releases the video reader.

.. ocv:function:: void gpu::VideoReader_GPU::close()



gpu::VideoReader_GPU::read
--------------------------
Grabs, decodes and returns the next video frame.

.. ocv:function:: bool gpu::VideoReader_GPU::read(GpuMat& image)

If no frames has been grabbed (there are no more frames in video file), the methods return ``false`` . The method throws :ocv:class:`Exception` if error occurs.



gpu::VideoReader_GPU::format
----------------------------
Returns information about video file format.

.. ocv:function:: FormatInfo gpu::VideoReader_GPU::format() const

The method throws :ocv:class:`Exception` if video reader wasn't initialized.



gpu::VideoReader_GPU::dumpFormat
--------------------------------
Dump information about video file format to specified stream.

.. ocv:function:: void gpu::VideoReader_GPU::dumpFormat(std::ostream& st)

    :param st: Output stream.

The method throws :ocv:class:`Exception` if video reader wasn't initialized.



gpu::VideoReader_GPU::VideoSource
-----------------------------------
.. ocv:class:: gpu::VideoReader_GPU::VideoSource

Interface for video demultiplexing. ::

    class VideoSource
    {
    public:
        VideoSource();
        virtual ~VideoSource() {}

        virtual FormatInfo format() const = 0;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool isStarted() const = 0;
        virtual bool hasError() const = 0;

    protected:
        bool parseVideoData(const unsigned char* data, size_t size, bool endOfStream = false);
    };

User can implement own demultiplexing by implementing this interface.



gpu::VideoReader_GPU::VideoSource::format
-----------------------------------------
Returns information about video file format.

.. ocv:function:: virtual FormatInfo gpu::VideoReader_GPU::VideoSource::format() const = 0



gpu::VideoReader_GPU::VideoSource::start
----------------------------------------
Starts processing.

.. ocv:function:: virtual void gpu::VideoReader_GPU::VideoSource::start() = 0

Implementation must create own thread with video processing and call periodic :ocv:func:`gpu::VideoReader_GPU::VideoSource::parseVideoData` .



gpu::VideoReader_GPU::VideoSource::stop
---------------------------------------
Stops processing.

.. ocv:function:: virtual void gpu::VideoReader_GPU::VideoSource::stop() = 0



gpu::VideoReader_GPU::VideoSource::isStarted
--------------------------------------------
Returns ``true`` if processing was successfully started.

.. ocv:function:: virtual bool gpu::VideoReader_GPU::VideoSource::isStarted() const = 0



gpu::VideoReader_GPU::VideoSource::hasError
-------------------------------------------
Returns ``true`` if error occured during processing.

.. ocv:function:: virtual bool gpu::VideoReader_GPU::VideoSource::hasError() const = 0



gpu::VideoReader_GPU::VideoSource::parseVideoData
-------------------------------------------------
Parse next video frame. Implementation must call this method after new frame was grabbed.

.. ocv:function:: bool gpu::VideoReader_GPU::VideoSource::parseVideoData(const uchar* data, size_t size, bool endOfStream = false)

    :param data: Pointer to frame data. Can be ``NULL`` if ``endOfStream`` if ``true`` .

    :param size: Size in bytes of current frame.

    :param endOfStream: Indicates that it is end of stream.
