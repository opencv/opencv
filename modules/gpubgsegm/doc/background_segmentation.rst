Background Segmentation
=======================

.. highlight:: cpp



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

.. ocv:function:: int gpu::FGDStatModel::update(const cv::gpu::GpuMat& curFrame)

    :param curFrame: Next video frame.



gpu::MOG_GPU
------------
.. ocv:class:: gpu::MOG_GPU

Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm. ::

    class MOG_GPU
    {
    public:
        MOG_GPU(int nmixtures = -1);

        void initialize(Size frameSize, int frameType);

        void operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = 0.0f, Stream& stream = Stream::Null());

        void getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const;

        void release();

        int history;
        float varThreshold;
        float backgroundRatio;
        float noiseSigma;
    };

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [MOG2001]_.

.. seealso:: :ocv:class:`BackgroundSubtractorMOG`



gpu::MOG_GPU::MOG_GPU
---------------------
The constructor.

.. ocv:function:: gpu::MOG_GPU::MOG_GPU(int nmixtures = -1)

    :param nmixtures: Number of Gaussian mixtures.

Default constructor sets all parameters to default values.



gpu::MOG_GPU::operator()
------------------------
Updates the background model and returns the foreground mask.

.. ocv:function:: void gpu::MOG_GPU::operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = 0.0f, Stream& stream = Stream::Null())

    :param frame: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.

    :param stream: Stream for the asynchronous version.



gpu::MOG_GPU::getBackgroundImage
--------------------------------
Computes a background image.

.. ocv:function:: void gpu::MOG_GPU::getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const

    :param backgroundImage: The output background image.

    :param stream: Stream for the asynchronous version.



gpu::MOG_GPU::release
---------------------
Releases all inner buffer's memory.

.. ocv:function:: void gpu::MOG_GPU::release()



gpu::MOG2_GPU
-------------
.. ocv:class:: gpu::MOG2_GPU

Gaussian Mixture-based Background/Foreground Segmentation Algorithm. ::

    class MOG2_GPU
    {
    public:
        MOG2_GPU(int nmixtures = -1);

        void initialize(Size frameSize, int frameType);

        void operator()(const GpuMat& frame, GpuMat& fgmask, float learningRate = 0.0f, Stream& stream = Stream::Null());

        void getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const;

        void release();

        // parameters
        ...
    };

  The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [MOG2004]_.

  Here are important members of the class that control the algorithm, which you can set after constructing the class instance:

    .. ocv:member:: float backgroundRatio

        Threshold defining whether the component is significant enough to be included into the background model ( corresponds to ``TB=1-cf`` from the paper??which paper??). ``cf=0.1 => TB=0.9`` is default. For ``alpha=0.001``, it means that the mode should exist for approximately 105 frames before it is considered foreground.

    .. ocv:member:: float varThreshold

        Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the existing components (corresponds to ``Tg``). If it is not close to any component, a new component is generated. ``3 sigma => Tg=3*3=9`` is default. A smaller ``Tg`` value generates more components. A higher ``Tg`` value may result in a small number of components but they can grow too large.

    .. ocv:member:: float fVarInit

        Initial variance for the newly generated components. It affects the speed of adaptation. The parameter value is based on your estimate of the typical standard deviation from the images. OpenCV uses 15 as a reasonable value.

    .. ocv:member:: float fVarMin

        Parameter used to further control the variance.

    .. ocv:member:: float fVarMax

        Parameter used to further control the variance.

    .. ocv:member:: float fCT

        Complexity reduction parameter. This parameter defines the number of samples needed to accept to prove the component exists. ``CT=0.05`` is a default value for all the samples. By setting ``CT=0`` you get an algorithm very similar to the standard Stauffer&Grimson algorithm.

    .. ocv:member:: uchar nShadowDetection

        The value for marking shadow pixels in the output foreground mask. Default value is 127.

    .. ocv:member:: float fTau

        Shadow threshold. The shadow is detected if the pixel is a darker version of the background. ``Tau`` is a threshold defining how much darker the shadow can be. ``Tau= 0.5`` means that if a pixel is more than twice darker then it is not shadow. See [ShadowDetect2003]_.

    .. ocv:member:: bool bShadowDetection

        Parameter defining whether shadow detection should be enabled.

.. seealso:: :ocv:class:`BackgroundSubtractorMOG2`



gpu::MOG2_GPU::MOG2_GPU
-----------------------
The constructor.

.. ocv:function:: gpu::MOG2_GPU::MOG2_GPU(int nmixtures = -1)

    :param nmixtures: Number of Gaussian mixtures.

Default constructor sets all parameters to default values.



gpu::MOG2_GPU::operator()
-------------------------
Updates the background model and returns the foreground mask.

.. ocv:function:: void gpu::MOG2_GPU::operator()( const GpuMat& frame, GpuMat& fgmask, float learningRate=-1.0f, Stream& stream=Stream::Null() )

    :param frame: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.

    :param stream: Stream for the asynchronous version.



gpu::MOG2_GPU::getBackgroundImage
---------------------------------
Computes a background image.

.. ocv:function:: void gpu::MOG2_GPU::getBackgroundImage(GpuMat& backgroundImage, Stream& stream = Stream::Null()) const

    :param backgroundImage: The output background image.

    :param stream: Stream for the asynchronous version.



gpu::MOG2_GPU::release
----------------------
Releases all inner buffer's memory.

.. ocv:function:: void gpu::MOG2_GPU::release()



gpu::GMG_GPU
------------
.. ocv:class:: gpu::GMG_GPU

  Class used for background/foreground segmentation. ::

    class GMG_GPU_GPU
    {
    public:
        GMG_GPU();

        void initialize(Size frameSize, float min = 0.0f, float max = 255.0f);

        void operator ()(const GpuMat& frame, GpuMat& fgmask, float learningRate = -1.0f, Stream& stream = Stream::Null());

        void release();

        int    maxFeatures;
        float  learningRate;
        int    numInitializationFrames;
        int    quantizationLevels;
        float  backgroundPrior;
        float  decisionThreshold;
        int    smoothingRadius;

        ...
    };

  The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [GMG2012]_.

  Here are important members of the class that control the algorithm, which you can set after constructing the class instance:

    .. ocv:member:: int maxFeatures

        Total number of distinct colors to maintain in histogram.

    .. ocv:member:: float learningRate

        Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.

    .. ocv:member:: int numInitializationFrames

        Number of frames of video to use to initialize histograms.

    .. ocv:member:: int quantizationLevels

        Number of discrete levels in each channel to be used in histograms.

    .. ocv:member:: float backgroundPrior

        Prior probability that any given pixel is a background pixel. A sensitivity parameter.

    .. ocv:member:: float decisionThreshold

        Value above which pixel is determined to be FG.

    .. ocv:member:: float smoothingRadius

        Smoothing radius, in pixels, for cleaning up FG image.



gpu::GMG_GPU::GMG_GPU
---------------------
The default constructor.

.. ocv:function:: gpu::GMG_GPU::GMG_GPU()

Default constructor sets all parameters to default values.



gpu::GMG_GPU::initialize
------------------------
Initialize background model and allocates all inner buffers.

.. ocv:function:: void gpu::GMG_GPU::initialize(Size frameSize, float min = 0.0f, float max = 255.0f)

    :param frameSize: Input frame size.

    :param min: Minimum value taken on by pixels in image sequence. Usually 0.

    :param max: Maximum value taken on by pixels in image sequence, e.g. 1.0 or 255.



gpu::GMG_GPU::operator()
------------------------
Updates the background model and returns the foreground mask

.. ocv:function:: void gpu::GMG_GPU::operator ()( const GpuMat& frame, GpuMat& fgmask, float learningRate=-1.0f, Stream& stream=Stream::Null() )

    :param frame: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.

    :param stream: Stream for the asynchronous version.



gpu::GMG_GPU::release
---------------------
Releases all inner buffer's memory.

.. ocv:function:: void gpu::GMG_GPU::release()



.. [FGD2003] Liyuan Li, Weimin Huang, Irene Y.H. Gu, and Qi Tian. *Foreground Object Detection from Videos Containing Complex Background*. ACM MM2003 9p, 2003.
.. [MOG2001] P. KadewTraKuPong and R. Bowden. *An improved adaptive background mixture model for real-time tracking with shadow detection*. Proc. 2nd European Workshop on Advanced Video-Based Surveillance Systems, 2001
.. [MOG2004] Z. Zivkovic. *Improved adaptive Gausian mixture model for background subtraction*. International Conference Pattern Recognition, UK, August, 2004
.. [ShadowDetect2003] Prati, Mikic, Trivedi and Cucchiarra. *Detecting Moving Shadows...*. IEEE PAMI, 2003
.. [GMG2012] A. Godbehere, A. Matsukawa and K. Goldberg. *Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation*. American Control Conference, Montreal, June 2012
