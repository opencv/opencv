Video Analysis
==============

.. highlight:: cpp



gpu::BroxOpticalFlow
--------------------
.. ocv:class:: gpu::BroxOpticalFlow

Class computing the optical flow for two images using Brox et al Optical Flow algorithm ([Brox2004]_). ::

    class BroxOpticalFlow
    {
    public:
        BroxOpticalFlow(float alpha_, float gamma_, float scale_factor_, int inner_iterations_, int outer_iterations_, int solver_iterations_);

        //! Compute optical flow
        //! frame0 - source frame (supports only CV_32FC1 type)
        //! frame1 - frame to track (with the same size and type as frame0)
        //! u      - flow horizontal component (along x axis)
        //! v      - flow vertical component (along y axis)
        void operator ()(const GpuMat& frame0, const GpuMat& frame1, GpuMat& u, GpuMat& v, Stream& stream = Stream::Null());

        //! flow smoothness
        float alpha;

        //! gradient constancy importance
        float gamma;

        //! pyramid scale factor
        float scale_factor;

        //! number of lagged non-linearity iterations (inner loop)
        int inner_iterations;

        //! number of warping iterations (number of pyramid levels)
        int outer_iterations;

        //! number of linear system solver iterations
        int solver_iterations;

        GpuMat buf;
    };



gpu::GoodFeaturesToTrackDetector_GPU
------------------------------------
.. ocv:class:: gpu::GoodFeaturesToTrackDetector_GPU

Class used for strong corners detection on an image. ::

    class GoodFeaturesToTrackDetector_GPU
    {
    public:
        explicit GoodFeaturesToTrackDetector_GPU(int maxCorners_ = 1000, double qualityLevel_ = 0.01, double minDistance_ = 0.0,
            int blockSize_ = 3, bool useHarrisDetector_ = false, double harrisK_ = 0.04);

        void operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat());

        int maxCorners;
        double qualityLevel;
        double minDistance;

        int blockSize;
        bool useHarrisDetector;
        double harrisK;

        void releaseMemory();
    };

The class finds the most prominent corners in the image.

.. seealso:: :ocv:func:`goodFeaturesToTrack`



gpu::GoodFeaturesToTrackDetector_GPU::GoodFeaturesToTrackDetector_GPU
---------------------------------------------------------------------
Constructor.

.. ocv:function:: gpu::GoodFeaturesToTrackDetector_GPU::GoodFeaturesToTrackDetector_GPU(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04)

    :param maxCorners: Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.

    :param qualityLevel: Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see  :ocv:func:`gpu::cornerMinEigenVal` ) or the Harris function response (see  :ocv:func:`gpu::cornerHarris` ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the  ``qualityLevel=0.01`` , then all the corners with the quality measure less than 15 are rejected.

    :param minDistance: Minimum possible Euclidean distance between the returned corners.

    :param blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See  :ocv:func:`cornerEigenValsAndVecs` .

    :param useHarrisDetector: Parameter indicating whether to use a Harris detector (see :ocv:func:`gpu::cornerHarris`) or :ocv:func:`gpu::cornerMinEigenVal`.

    :param harrisK: Free parameter of the Harris detector.



gpu::GoodFeaturesToTrackDetector_GPU::operator ()
-------------------------------------------------
Finds the most prominent corners in the image.

.. ocv:function:: void gpu::GoodFeaturesToTrackDetector_GPU::operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat())

    :param image: Input 8-bit, single-channel image.

    :param corners: Output vector of detected corners (it will be one row matrix with CV_32FC2 type).

    :param mask: Optional region of interest. If the image is not empty (it needs to have the type  ``CV_8UC1``  and the same size as  ``image`` ), it  specifies the region in which the corners are detected.

.. seealso:: :ocv:func:`goodFeaturesToTrack`



gpu::GoodFeaturesToTrackDetector_GPU::releaseMemory
---------------------------------------------------
Releases inner buffers memory.

.. ocv:function:: void gpu::GoodFeaturesToTrackDetector_GPU::releaseMemory()



gpu::FarnebackOpticalFlow
-------------------------
.. ocv:class:: gpu::FarnebackOpticalFlow

Class computing a dense optical flow using the Gunnar Farneback’s algorithm. ::

    class CV_EXPORTS FarnebackOpticalFlow
    {
    public:
        FarnebackOpticalFlow()
        {
            numLevels = 5;
            pyrScale = 0.5;
            fastPyramids = false;
            winSize = 13;
            numIters = 10;
            polyN = 5;
            polySigma = 1.1;
            flags = 0;
        }

        int numLevels;
        double pyrScale;
        bool fastPyramids;
        int winSize;
        int numIters;
        int polyN;
        double polySigma;
        int flags;

        void operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null());

        void releaseMemory();

    private:
        /* hidden */
    };



gpu::FarnebackOpticalFlow::operator ()
--------------------------------------
Computes a dense optical flow using the Gunnar Farneback’s algorithm.

.. ocv:function:: void gpu::FarnebackOpticalFlow::operator ()(const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s = Stream::Null())

    :param frame0: First 8-bit gray-scale input image
    :param frame1: Second 8-bit gray-scale input image
    :param flowx: Flow horizontal component
    :param flowy: Flow vertical component
    :param s: Stream

.. seealso:: :ocv:func:`calcOpticalFlowFarneback`



gpu::FarnebackOpticalFlow::releaseMemory
----------------------------------------
Releases unused auxiliary memory buffers.

.. ocv:function:: void gpu::FarnebackOpticalFlow::releaseMemory()



gpu::PyrLKOpticalFlow
---------------------
.. ocv:class:: gpu::PyrLKOpticalFlow

Class used for calculating an optical flow. ::

    class PyrLKOpticalFlow
    {
    public:
        PyrLKOpticalFlow();

        void sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts,
            GpuMat& status, GpuMat* err = 0);

        void dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0);

        Size winSize;
        int maxLevel;
        int iters;
        double derivLambda;
        bool useInitialFlow;
        float minEigThreshold;
        bool getMinEigenVals;

        void releaseMemory();
    };

The class can calculate an optical flow for a sparse feature set or dense optical flow using the iterative Lucas-Kanade method with pyramids.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`



gpu::PyrLKOpticalFlow::sparse
-----------------------------
Calculate an optical flow for a sparse feature set.

.. ocv:function:: void gpu::PyrLKOpticalFlow::sparse(const GpuMat& prevImg, const GpuMat& nextImg, const GpuMat& prevPts, GpuMat& nextPts, GpuMat& status, GpuMat* err = 0)

    :param prevImg: First 8-bit input image (supports both grayscale and color images).

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param prevPts: Vector of 2D points for which the flow needs to be found. It must be one row matrix with CV_32FC2 type.

    :param nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When ``useInitialFlow`` is true, the vector must have the same size as in the input.

    :param status: Output status vector (CV_8UC1 type). Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`



gpu::PyrLKOpticalFlow::dense
-----------------------------
Calculate dense optical flow.

.. ocv:function:: void gpu::PyrLKOpticalFlow::dense(const GpuMat& prevImg, const GpuMat& nextImg, GpuMat& u, GpuMat& v, GpuMat* err = 0)

    :param prevImg: First 8-bit grayscale input image.

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param u: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param v: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.



gpu::PyrLKOpticalFlow::releaseMemory
------------------------------------
Releases inner buffers memory.

.. ocv:function:: void gpu::PyrLKOpticalFlow::releaseMemory()



gpu::interpolateFrames
----------------------
Interpolates frames (images) using provided optical flow (displacement field).

.. ocv:function:: void gpu::interpolateFrames(const GpuMat& frame0, const GpuMat& frame1, const GpuMat& fu, const GpuMat& fv, const GpuMat& bu, const GpuMat& bv, float pos, GpuMat& newFrame, GpuMat& buf, Stream& stream = Stream::Null())

    :param frame0: First frame (32-bit floating point images, single channel).

    :param frame1: Second frame. Must have the same type and size as ``frame0`` .

    :param fu: Forward horizontal displacement.

    :param fv: Forward vertical displacement.

    :param bu: Backward horizontal displacement.

    :param bv: Backward vertical displacement.

    :param pos: New frame position.

    :param newFrame: Output image.

    :param buf: Temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 GpuMat: occlusion masks for first frame, occlusion masks for second, interpolated forward horizontal flow, interpolated forward vertical flow, interpolated backward horizontal flow, interpolated backward vertical flow.

    :param stream: Stream for the asynchronous version.



gpu::FGDStatModel
-----------------
.. ocv:class:: gpu::FGDStatModel

Class used for background/foreground segmentation. ::

    class FGDStatModel
    {
    public:
        struct Params
        {
            ...
        };

        explicit FGDStatModel(int out_cn = 3);
        explicit FGDStatModel(const cv::gpu::GpuMat& firstFrame, const Params& params = Params(), int out_cn = 3);

        ~FGDStatModel();

        void create(const cv::gpu::GpuMat& firstFrame, const Params& params = Params());
        void release();

        int update(const cv::gpu::GpuMat& curFrame);

        //8UC3 or 8UC4 reference background image
        cv::gpu::GpuMat background;

        //8UC1 foreground image
        cv::gpu::GpuMat foreground;

        std::vector< std::vector<cv::Point> > foreground_regions;
    };

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [FGD2003]_.

The results are available through the class fields:

    .. ocv:member:: cv::gpu::GpuMat background

        The output background image.

    .. ocv:member:: cv::gpu::GpuMat foreground

        The output foreground mask as an 8-bit binary image.

    .. ocv:member:: cv::gpu::GpuMat foreground_regions

        The output foreground regions calculated by :ocv:func:`findContours`.



gpu::FGDStatModel::FGDStatModel
-------------------------------
Constructors.

.. ocv:function:: gpu::FGDStatModel::FGDStatModel(int out_cn = 3)
.. ocv:function:: gpu::FGDStatModel::FGDStatModel(const cv::gpu::GpuMat& firstFrame, const Params& params = Params(), int out_cn = 3)

    :param firstFrame: First frame from video stream. Supports 3- and 4-channels input ( ``CV_8UC3`` and ``CV_8UC4`` ).

    :param params: Algorithm's parameters. See [FGD2003]_ for explanation.

    :param out_cn: Channels count in output result and inner buffers. Can be 3 or 4. 4-channels version requires more memory, but works a bit faster.

.. seealso:: :ocv:func:`gpu::FGDStatModel::create`



gpu::FGDStatModel::create
-------------------------
Initializes background model.

.. ocv:function:: void gpu::FGDStatModel::create(const cv::gpu::GpuMat& firstFrame, const Params& params = Params())

    :param firstFrame: First frame from video stream. Supports 3- and 4-channels input ( ``CV_8UC3`` and ``CV_8UC4`` ).

    :param params: Algorithm's parameters. See [FGD2003]_ for explanation.



gpu::FGDStatModel::release
--------------------------
Releases all inner buffer's memory.

.. ocv:function:: void gpu::FGDStatModel::release()



gpu::FGDStatModel::update
--------------------------
Updates the background model and returns foreground regions count.

.. ocv:function:: int gpu::FGDStatModel::update(const cv::gpu::GpuMat& curFrame);

    :param curFrame: Next video frame.



gpu::VideoWriter_GPU
---------------------
Video writer class.

.. ocv:class:: gpu::VideoWriter_GPU

The class uses H264 video codec.

.. note:: Currently only Windows platform is supported.



gpu::VideoWriter_GPU::VideoWriter_GPU
-------------------------------------
Constructors.

.. ocv:function:: gpu::VideoWriter_GPU::VideoWriter_GPU()
.. ocv:function:: gpu::VideoWriter_GPU::VideoWriter_GPU(const std::string& fileName, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR)
.. ocv:function:: gpu::VideoWriter_GPU::VideoWriter_GPU(const std::string& fileName, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR)
.. ocv:function:: gpu::VideoWriter_GPU::VideoWriter_GPU(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR)
.. ocv:function:: gpu::VideoWriter_GPU::VideoWriter_GPU(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR)

    :param fileName: Name of the output video file. Only AVI file format is supported.

    :param frameSize: Size of the input video frames.

    :param fps: Framerate of the created video stream.

    :param params: Encoder parameters. See :ocv:class:`gpu::VideoWriter_GPU::EncoderParams` .

    :param format: Surface format of input frames ( ``SF_UYVY`` , ``SF_YUY2`` , ``SF_YV12`` , ``SF_NV12`` , ``SF_IYUV`` , ``SF_BGR`` or ``SF_GRAY``). BGR or gray frames will be converted to YV12 format before encoding, frames with other formats will be used as is.

    :param encoderCallback: Callbacks for video encoder. See :ocv:class:`gpu::VideoWriter_GPU::EncoderCallBack` . Use it if you want to work with raw video stream.

The constructors initialize video writer. FFMPEG is used to write videos. User can implement own multiplexing with :ocv:class:`gpu::VideoWriter_GPU::EncoderCallBack` .



gpu::VideoWriter_GPU::open
--------------------------
Initializes or reinitializes video writer.

.. ocv:function:: void gpu::VideoWriter_GPU::open(const std::string& fileName, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR)
.. ocv:function:: void gpu::VideoWriter_GPU::open(const std::string& fileName, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR)
.. ocv:function:: void gpu::VideoWriter_GPU::open(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, SurfaceFormat format = SF_BGR)
.. ocv:function:: void gpu::VideoWriter_GPU::open(const cv::Ptr<EncoderCallBack>& encoderCallback, cv::Size frameSize, double fps, const EncoderParams& params, SurfaceFormat format = SF_BGR)

The method opens video writer. Parameters are the same as in the constructor :ocv:func:`gpu::VideoWriter_GPU::VideoWriter_GPU` . The method throws :ocv:class:`Exception` if error occurs.



gpu::VideoWriter_GPU::isOpened
------------------------------
Returns true if video writer has been successfully initialized.

.. ocv:function:: bool gpu::VideoWriter_GPU::isOpened() const



gpu::VideoWriter_GPU::close
---------------------------
Releases the video writer.

.. ocv:function:: void gpu::VideoWriter_GPU::close()



gpu::VideoWriter_GPU::write
---------------------------
Writes the next video frame.

.. ocv:function:: void gpu::VideoWriter_GPU::write(const cv::gpu::GpuMat& image, bool lastFrame = false)

    :param image: The written frame.

    :param lastFrame: Indicates that it is end of stream. The parameter can be ignored.

The method write the specified image to video file. The image must have the same size and the same surface format as has been specified when opening the video writer.



gpu::VideoWriter_GPU::EncoderParams
-----------------------------------
.. ocv:struct:: gpu::VideoWriter_GPU::EncoderParams

Different parameters for CUDA video encoder. ::

    struct EncoderParams
    {
        int       P_Interval;      //    NVVE_P_INTERVAL,
        int       IDR_Period;      //    NVVE_IDR_PERIOD,
        int       DynamicGOP;      //    NVVE_DYNAMIC_GOP,
        int       RCType;          //    NVVE_RC_TYPE,
        int       AvgBitrate;      //    NVVE_AVG_BITRATE,
        int       PeakBitrate;     //    NVVE_PEAK_BITRATE,
        int       QP_Level_Intra;  //    NVVE_QP_LEVEL_INTRA,
        int       QP_Level_InterP; //    NVVE_QP_LEVEL_INTER_P,
        int       QP_Level_InterB; //    NVVE_QP_LEVEL_INTER_B,
        int       DeblockMode;     //    NVVE_DEBLOCK_MODE,
        int       ProfileLevel;    //    NVVE_PROFILE_LEVEL,
        int       ForceIntra;      //    NVVE_FORCE_INTRA,
        int       ForceIDR;        //    NVVE_FORCE_IDR,
        int       ClearStat;       //    NVVE_CLEAR_STAT,
        int       DIMode;          //    NVVE_SET_DEINTERLACE,
        int       Presets;         //    NVVE_PRESETS,
        int       DisableCabac;    //    NVVE_DISABLE_CABAC,
        int       NaluFramingType; //    NVVE_CONFIGURE_NALU_FRAMING_TYPE
        int       DisableSPSPPS;   //    NVVE_DISABLE_SPS_PPS

        EncoderParams();
        explicit EncoderParams(const std::string& configFile);

        void load(const std::string& configFile);
        void save(const std::string& configFile) const;
    };



gpu::VideoWriter_GPU::EncoderParams::EncoderParams
--------------------------------------------------
Constructors.

.. ocv:function:: gpu::VideoWriter_GPU::EncoderParams::EncoderParams()
.. ocv:function:: gpu::VideoWriter_GPU::EncoderParams::EncoderParams(const std::string& configFile)

    :param configFile: Config file name.

Creates default parameters or reads parameters from config file.



gpu::VideoWriter_GPU::EncoderParams::load
-----------------------------------------
Reads parameters from config file.

.. ocv:function:: void gpu::VideoWriter_GPU::EncoderParams::load(const std::string& configFile)

    :param configFile: Config file name.



gpu::VideoWriter_GPU::EncoderParams::save
-----------------------------------------
Saves parameters to config file.

.. ocv:function:: void gpu::VideoWriter_GPU::EncoderParams::save(const std::string& configFile) const

    :param configFile: Config file name.



gpu::VideoWriter_GPU::EncoderCallBack
-------------------------------------
.. ocv:class:: gpu::VideoWriter_GPU::EncoderCallBack

Callbacks for CUDA video encoder. ::

    class EncoderCallBack
    {
    public:
        enum PicType
        {
            IFRAME = 1,
            PFRAME = 2,
            BFRAME = 3
        };

        virtual ~EncoderCallBack() {}

        virtual unsigned char* acquireBitStream(int* bufferSize) = 0;
        virtual void releaseBitStream(unsigned char* data, int size) = 0;
        virtual void onBeginFrame(int frameNumber, PicType picType) = 0;
        virtual void onEndFrame(int frameNumber, PicType picType) = 0;
    };



gpu::VideoWriter_GPU::EncoderCallBack::acquireBitStream
-------------------------------------------------------
Callback function to signal the start of bitstream that is to be encoded.

.. ocv:function:: virtual uchar* gpu::VideoWriter_GPU::EncoderCallBack::acquireBitStream(int* bufferSize) = 0

Callback must allocate buffer for CUDA encoder and return pointer to it and it's size.



gpu::VideoWriter_GPU::EncoderCallBack::releaseBitStream
-------------------------------------------------------
Callback function to signal that the encoded bitstream is ready to be written to file.

.. ocv:function:: virtual void gpu::VideoWriter_GPU::EncoderCallBack::releaseBitStream(unsigned char* data, int size) = 0



gpu::VideoWriter_GPU::EncoderCallBack::onBeginFrame
---------------------------------------------------
Callback function to signal that the encoding operation on the frame has started.

.. ocv:function:: virtual void gpu::VideoWriter_GPU::EncoderCallBack::onBeginFrame(int frameNumber, PicType picType) = 0

    :param picType: Specify frame type (I-Frame, P-Frame or B-Frame).



gpu::VideoWriter_GPU::EncoderCallBack::onEndFrame
-------------------------------------------------
Callback function signals that the encoding operation on the frame has finished.

.. ocv:function:: virtual void gpu::VideoWriter_GPU::EncoderCallBack::onEndFrame(int frameNumber, PicType picType) = 0

    :param picType: Specify frame type (I-Frame, P-Frame or B-Frame).



gpu::VideoReader_GPU
--------------------
Class for reading video from files.

.. ocv:class:: gpu::VideoReader_GPU



gpu::VideoReader_GPU::Codec
---------------------------

Video codecs supported by ocv:class:`gpu::VideoReader_GPU` . ::

    enum Codec
    {
        MPEG1 = 0,
        MPEG2,
        MPEG4,
        VC1,
        H264,
        JPEG,
        H264_SVC,
        H264_MVC,

        Uncompressed_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V')),   // Y,U,V (4:2:0)
        Uncompressed_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,V,U (4:2:0)
        Uncompressed_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2')),   // Y,UV  (4:2:0)
        Uncompressed_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V')),   // YUYV/YUY2 (4:2:2)
        Uncompressed_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y')),   // UYVY (4:2:2)
    };



gpu::VideoReader_GPU::ChromaFormat
----------------------------------

Chroma formats supported by ocv:class:`gpu::VideoReader_GPU` . ::

    enum ChromaFormat
    {
        Monochrome=0,
        YUV420,
        YUV422,
        YUV444,
    };



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
.. ocv:function:: gpu::VideoReader_GPU::VideoReader_GPU(const std::string& filename)
.. ocv:function:: gpu::VideoReader_GPU::VideoReader_GPU(const cv::Ptr<VideoSource>& source)

    :param filename: Name of the input video file.

    :param source: Video file parser implemented by user.

The constructors initialize video reader. FFMPEG is used to read videos. User can implement own demultiplexing with :ocv:class:`gpu::VideoReader_GPU::VideoSource` .



gpu::VideoReader_GPU::open
--------------------------
Initializes or reinitializes video reader.

.. ocv:function:: void gpu::VideoReader_GPU::open(const std::string& filename)
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

Implementation must create own thread with video processing and call periodic ocv:func:`gpu::VideoReader_GPU::VideoSource::parseVideoData` .



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



.. [Brox2004] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. *High accuracy optical flow estimation based on a theory for warping*. ECCV 2004.
.. [FGD2003] Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian. *Foreground Object Detection from Videos Containing Complex Background*. ACM MM2003 9p, 2003.
