Video Decoding
==============

.. highlight:: cpp



cudacodec::VideoReader
----------------------
Video reader interface.

.. ocv:class:: cudacodec::VideoReader

.. note::

   * An example on how to use the videoReader class can be found at opencv_source_code/samples/gpu/video_reader.cpp


cudacodec::VideoReader::nextFrame
---------------------------------
Grabs, decodes and returns the next video frame.

.. ocv:function:: bool cudacodec::VideoReader::nextFrame(OutputArray frame)

If no frames has been grabbed (there are no more frames in video file), the methods return ``false`` . The method throws :ocv:class:`Exception` if error occurs.



cudacodec::VideoReader::format
------------------------------
Returns information about video file format.

.. ocv:function:: FormatInfo cudacodec::VideoReader::format() const



cudacodec::Codec
----------------
Video codecs supported by :ocv:class:`cudacodec::VideoReader` .

.. ocv:enum:: cudacodec::Codec

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



cudacodec::ChromaFormat
-----------------------
Chroma formats supported by :ocv:class:`cudacodec::VideoReader` .

.. ocv:enum:: cudacodec::ChromaFormat

  .. ocv:emember:: Monochrome = 0
  .. ocv:emember:: YUV420
  .. ocv:emember:: YUV422
  .. ocv:emember:: YUV444



cudacodec::FormatInfo
---------------------
.. ocv:struct:: cudacodec::FormatInfo

Struct providing information about video file format. ::

    struct FormatInfo
    {
        Codec codec;
        ChromaFormat chromaFormat;
        int width;
        int height;
    };



cudacodec::createVideoReader
----------------------------
Creates video reader.

.. ocv:function:: Ptr<VideoReader> cudacodec::createVideoReader(const String& filename)
.. ocv:function:: Ptr<VideoReader> cudacodec::createVideoReader(const Ptr<RawVideoSource>& source)

    :param filename: Name of the input video file.

    :param source: RAW video source implemented by user.

FFMPEG is used to read videos. User can implement own demultiplexing with :ocv:class:`cudacodec::RawVideoSource` .



cudacodec::RawVideoSource
-------------------------
.. ocv:class:: cudacodec::RawVideoSource

Interface for video demultiplexing. ::

    class RawVideoSource
    {
    public:
        virtual ~RawVideoSource() {}

        virtual bool getNextPacket(unsigned char** data, int* size, bool* endOfFile) = 0;

        virtual FormatInfo format() const = 0;
    };

User can implement own demultiplexing by implementing this interface.



cudacodec::RawVideoSource::getNextPacket
----------------------------------------
Returns next packet with RAW video frame.

.. ocv:function:: bool cudacodec::VideoSource::getNextPacket(unsigned char** data, int* size, bool* endOfFile) = 0

    :param data: Pointer to frame data.

    :param size: Size in bytes of current frame.

    :param endOfStream: Indicates that it is end of stream.



cudacodec::RawVideoSource::format
---------------------------------
Returns information about video file format.

.. ocv:function:: virtual FormatInfo cudacodec::RawVideoSource::format() const = 0
