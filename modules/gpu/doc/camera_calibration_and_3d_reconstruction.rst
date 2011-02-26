Camera Calibration and 3d Reconstruction
========================================

.. highlight:: cpp

.. index:: gpu::StereoBM_GPU

.. _gpu::StereoBM_GPU:

gpu::StereoBM_GPU
-----------------
.. ctype:: gpu::StereoBM_GPU

The class for computing stereo correspondence using block matching algorithm. ::

    class StereoBM_GPU
    {
    public:
        enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

        enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

        StereoBM_GPU();
        StereoBM_GPU(int preset, int ndisparities = DEFAULT_NDISP,
                     int winSize = DEFAULT_WINSZ);

        void operator() (const GpuMat& left, const GpuMat& right,
                         GpuMat& disparity);
        void operator() (const GpuMat& left, const GpuMat& right,
                         GpuMat& disparity, const Stream & stream);

        static bool checkIfGpuCallReasonable();

        int preset;
        int ndisp;
        int winSize;

        float avergeTexThreshold;

        ...
    };
..

This class computes the disparity map using block matching algorithm. The class also performs pre- and post- filtering steps: sobel prefiltering (if PREFILTER_XSOBEL flag is set) and low textureness filtering (if averageTexThreshols
:math:`>` 0). If ``avergeTexThreshold = 0`` low textureness filtering is disabled, otherwise disparity is set to 0 in each point ``(x, y)`` where for left image
:math:`\sum HorizontalGradiensInWindow(x, y, winSize) < (winSize \cdot winSize) \cdot avergeTexThreshold` i.e. input left image is low textured.

.. index:: cv::gpu::StereoBM_GPU::StereoBM_GPU

.. _cv::gpu::StereoBM_GPU::StereoBM_GPU:

cv::gpu::StereoBM_GPU::StereoBM_GPU
-----------------------------------_
.. cfunction:: StereoBM_GPU::StereoBM_GPU()

.. cfunction:: StereoBM_GPU::StereoBM_GPU(int preset,  int ndisparities = DEFAULT_NDISP,  int winSize = DEFAULT_WINSZ)

    StereoBMGPU constructors.

    :param preset: Preset:

        * **BASIC_PRESET** Without preprocessing.

        * **PREFILTER_XSOBEL** Sobel prefilter.

    :param ndisparities: Number of disparities. Must be a multiple of 8 and less or equal then 256.

    :param winSize: Block size.

.. index:: cv::gpu::StereoBM_GPU::operator ()

.. _cv::gpu::StereoBM_GPU::operator ():

cv::gpu::StereoBM_GPU::operator ()
----------------------------------
.. cfunction:: void StereoBM_GPU::operator() (const GpuMat\& left, const GpuMat\& right,  GpuMat\& disparity)

.. cfunction:: void StereoBM_GPU::operator() (const GpuMat\& left, const GpuMat\& right,  GpuMat\& disparity, const Stream\& stream)

    The stereo correspondence operator. Finds the disparity for the specified rectified stereo pair.

    :param left: Left image; supports only  ``CV_8UC1``  type.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. It will be  ``CV_8UC1``  image with the same size as the input images.

    :param stream: Stream for the asynchronous version.

.. index:: cv::gpu::StereoBM_GPU::checkIfGpuCallReasonable

.. _cv::gpu::StereoBM_GPU::checkIfGpuCallReasonable:

cv::gpu::StereoBM_GPU::checkIfGpuCallReasonable
-----------------------------------------------
.. cfunction:: bool StereoBM_GPU::checkIfGpuCallReasonable()

    Some heuristics that tries to estmate if the current GPU will be faster then CPU in this algorithm. It queries current active device.

.. index:: gpu::StereoBeliefPropagation

.. _gpu::StereoBeliefPropagation:

gpu::StereoBeliefPropagation
----------------------------
.. ctype:: gpu::StereoBeliefPropagation

The class for computing stereo correspondence using belief propagation algorithm. ::

    class StereoBeliefPropagation
    {
    public:
        enum { DEFAULT_NDISP  = 64 };
        enum { DEFAULT_ITERS  = 5  };
        enum { DEFAULT_LEVELS = 5  };

        static void estimateRecommendedParams(int width, int height,
            int& ndisp, int& iters, int& levels);

        explicit StereoBeliefPropagation(int ndisp = DEFAULT_NDISP,
            int iters  = DEFAULT_ITERS,
            int levels = DEFAULT_LEVELS,
            int msg_type = CV_32F);
        StereoBeliefPropagation(int ndisp, int iters, int levels,
            float max_data_term, float data_weight,
            float max_disc_term, float disc_single_jump,
            int msg_type = CV_32F);

        void operator()(const GpuMat& left, const GpuMat& right,
                        GpuMat& disparity);
        void operator()(const GpuMat& left, const GpuMat& right,
                        GpuMat& disparity, Stream& stream);
        void operator()(const GpuMat& data, GpuMat& disparity);
        void operator()(const GpuMat& data, GpuMat& disparity, Stream& stream);

        int ndisp;

        int iters;
        int levels;

        float max_data_term;
        float data_weight;
        float max_disc_term;
        float disc_single_jump;

        int msg_type;

        ...
    };
..

The class implements Pedro F. Felzenszwalb algorithm
felzenszwalb_bp
. It can compute own data cost (using truncated linear model) or use user-provided data cost.

**Please note:** ``StereoBeliefPropagation`` requires a lot of memory:

.. math::

    width \_ step  \cdot height  \cdot ndisp  \cdot 4  \cdot (1 + 0.25)

for message storage and

.. math::

    width \_ step  \cdot height  \cdot ndisp  \cdot (1 + 0.25 + 0.0625 +  \dotsm +  \frac{1}{4^{levels}}

for data cost storage. ``width_step`` is the number of bytes in a line including the padding.

.. index:: gpu::StereoBeliefPropagation::StereoBeliefPropagation

cv::gpu::StereoBeliefPropagation::StereoBeliefPropagation
---------------------------------------------------------
.. cfunction:: StereoBeliefPropagation::StereoBeliefPropagation( int ndisp = DEFAULT_NDISP, int iters = DEFAULT_ITERS,  int levels = DEFAULT_LEVELS, int msg_type = CV_32F)

.. cfunction:: StereoBeliefPropagation::StereoBeliefPropagation( int ndisp, int iters, int levels,  float max_data_term, float data_weight,  float max_disc_term, float disc_single_jump,  int msg_type = CV_32F)

    StereoBeliefPropagation constructors.

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param max_data_term: Threshold for data cost truncation.

    :param data_weight: Data weight.

    :param max_disc_term: Threshold for discontinuity truncation.

    :param disc_single_jump: Discontinuity single jump.

    :param msg_type: Type for messages. Supports  ``CV_16SC1``  and  ``CV_32FC1`` .
 ``StereoBeliefPropagation`` uses truncated linear model for the data cost and discontinuity term:

.. math::

    DataCost = data \_ weight  \cdot \min ( \lvert I_2-I_1  \rvert , max \_ data \_ term)

.. math::

    DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)

For more details please see
felzenszwalb_bp
.

By default ``StereoBeliefPropagation`` uses floating-point arithmetics and ``CV_32FC1`` type for messages. But also it can use fixed-point arithmetics and ``CV_16SC1`` type for messages for better perfomance. To avoid overflow in this case, the parameters must satisfy

.. math::

    10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX

.. index:: gpu::StereoBeliefPropagation::estimateRecommendedParams

cv::gpu::StereoBeliefPropagation::estimateRecommendedParams
----------------------------------------------------------- ```` ```` ```` ```` ````
.. cfunction:: void StereoBeliefPropagation::estimateRecommendedParams( int width, int height, int\& ndisp, int\& iters, int\& levels)

    Some heuristics that tries to compute recommended parameters (ndisp, itersand levels) for specified image size (widthand height).

.. index:: gpu::StereoBeliefPropagation::operator ()

cv::gpu::StereoBeliefPropagation::operator ()
---------------------------------------------
.. cfunction:: void StereoBeliefPropagation::operator()( const GpuMat\& left, const GpuMat\& right,  GpuMat\& disparity)

.. cfunction:: void StereoBeliefPropagation::operator()( const GpuMat\& left, const GpuMat\& right,  GpuMat\& disparity, Stream\& stream)

    The stereo correspondence operator. Finds the disparity for the specified rectified stereo pair or data cost.

    :param left: Left image; supports  ``CV_8UC1`` ,  ``CV_8UC3``  and  ``CV_8UC4``  types.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. If  ``disparity``  is empty output type will be  ``CV_16SC1`` , otherwise output type will be  ``disparity.type()`` .

    :param stream: Stream for the asynchronous version.

.. cfunction:: void StereoBeliefPropagation::operator()( const GpuMat\& data, GpuMat\& disparity)

.. cfunction:: void StereoBeliefPropagation::operator()( const GpuMat\& data, GpuMat\& disparity, Stream\& stream)

    * **data** The user specified data cost. It must have  ``msg_type``  type and  :math:`\texttt{imgRows} \cdot \texttt{ndisp} \times \texttt{imgCols}`  size.

    * **disparity** Output disparity map. If  ``disparity``  is empty output type will be  ``CV_16SC1`` , otherwise output type will be  ``disparity.type()`` .

    * **stream** Stream for the asynchronous version.

.. index:: gpu::StereoConstantSpaceBP

.. _gpu::StereoConstantSpaceBP:

gpu::StereoConstantSpaceBP
--------------------------
.. ctype:: gpu::StereoConstantSpaceBP

The class for computing stereo correspondence using constant space belief propagation algorithm. ::

    class StereoConstantSpaceBP
    {
    public:
        enum { DEFAULT_NDISP    = 128 };
        enum { DEFAULT_ITERS    = 8   };
        enum { DEFAULT_LEVELS   = 4   };
        enum { DEFAULT_NR_PLANE = 4   };

        static void estimateRecommendedParams(int width, int height,
            int& ndisp, int& iters, int& levels, int& nr_plane);

        explicit StereoConstantSpaceBP(int ndisp = DEFAULT_NDISP,
            int iters    = DEFAULT_ITERS,
            int levels   = DEFAULT_LEVELS,
            int nr_plane = DEFAULT_NR_PLANE,
            int msg_type = CV_32F);
        StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane,
            float max_data_term, float data_weight,
            float max_disc_term, float disc_single_jump,
            int min_disp_th = 0,
            int msg_type = CV_32F);

        void operator()(const GpuMat& left, const GpuMat& right,
                        GpuMat& disparity);
        void operator()(const GpuMat& left, const GpuMat& right,
                        GpuMat& disparity, Stream& stream);

        int ndisp;

        int iters;
        int levels;

        int nr_plane;

        float max_data_term;
        float data_weight;
        float max_disc_term;
        float disc_single_jump;

        int min_disp_th;

        int msg_type;

        bool use_local_init_data_cost;

        ...
    };
..

The class implements Q. Yang algorithm
qx_csbp
. ``StereoConstantSpaceBP`` supports both local minimum and global minimum data cost initialization algortihms. For more details please see the paper. By default local algorithm is used, and to enable global algorithm set ``use_local_init_data_cost`` to false.

.. index:: gpu::StereoConstantSpaceBP::StereoConstantSpaceBP

cv::gpu::StereoConstantSpaceBP::StereoConstantSpaceBP
-----------------------------------------------------
.. cfunction:: StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp = DEFAULT_NDISP,  int iters = DEFAULT_ITERS, int levels = DEFAULT_LEVELS,  int nr_plane = DEFAULT_NR_PLANE, int msg_type = CV_32F)

.. cfunction:: StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp, int iters,  int levels, int nr_plane,  float max_data_term, float data_weight,  float max_disc_term, float disc_single_jump,  int min_disp_th = 0, int msg_type = CV_32F)

    StereoConstantSpaceBP constructors.

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param nr_plane: Number of disparity levels on the first level

    :param max_data_term: Truncation of data cost.

    :param data_weight: Data weight.

    :param max_disc_term: Truncation of discontinuity.

    :param disc_single_jump: Discontinuity single jump.

    :param min_disp_th: Minimal disparity threshold.

    :param msg_type: Type for messages. Supports  ``CV_16SC1``  and  ``CV_32FC1`` .
 ``StereoConstantSpaceBP`` uses truncated linear model for the data cost and discontinuity term:

.. math::

    DataCost = data \_ weight  \cdot \min ( \lvert I_2-I_1  \rvert , max \_ data \_ term)

.. math::

    DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)

For more details please see
qx_csbp
.

By default ``StereoConstantSpaceBP`` uses floating-point arithmetics and ``CV_32FC1`` type for messages. But also it can use fixed-point arithmetics and ``CV_16SC1`` type for messages for better perfomance. To avoid overflow in this case, the parameters must satisfy

.. math::

    10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX

.. index:: gpu::StereoConstantSpaceBP::estimateRecommendedParams

cv::gpu::StereoConstantSpaceBP::estimateRecommendedParams
--------------------------------------------------------- ```` ```` ```` ``_`` ```` ````
.. cfunction:: void StereoConstantSpaceBP::estimateRecommendedParams( int width, int height,  int\& ndisp, int\& iters, int\& levels, int\& nr_plane)

    Some heuristics that tries to compute parameters (ndisp, iters, levelsand nrplane) for specified image size (widthand height).

.. index:: gpu::StereoConstantSpaceBP::operator ()

cv::gpu::StereoConstantSpaceBP::operator ()
-------------------------------------------
.. cfunction:: void StereoConstantSpaceBP::operator()( const GpuMat\& left, const GpuMat\& right,  GpuMat\& disparity)

.. cfunction:: void StereoConstantSpaceBP::operator()( const GpuMat\& left, const GpuMat\& right,  GpuMat\& disparity, Stream\& stream)

    The stereo correspondence operator. Finds the disparity for the specified rectified stereo pair.

    :param left: Left image; supports  ``CV_8UC1`` ,  ``CV_8UC3``  and  ``CV_8UC4``  types.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. If  ``disparity``  is empty output type will be  ``CV_16SC1`` , otherwise output type will be  ``disparity.type()`` .

    :param stream: Stream for the asynchronous version.

.. index:: gpu::DisparityBilateralFilter

.. _gpu::DisparityBilateralFilter:

gpu::DisparityBilateralFilter
-----------------------------
.. ctype:: gpu::DisparityBilateralFilter

The class for disparity map refinement using joint bilateral filtering. ::

    class CV_EXPORTS DisparityBilateralFilter
    {
    public:
        enum { DEFAULT_NDISP  = 64 };
        enum { DEFAULT_RADIUS = 3 };
        enum { DEFAULT_ITERS  = 1 };

        explicit DisparityBilateralFilter(int ndisp = DEFAULT_NDISP,
            int radius = DEFAULT_RADIUS, int iters = DEFAULT_ITERS);

        DisparityBilateralFilter(int ndisp, int radius, int iters,
            float edge_threshold, float max_disc_threshold,
            float sigma_range);

        void operator()(const GpuMat& disparity, const GpuMat& image,
                        GpuMat& dst);
        void operator()(const GpuMat& disparity, const GpuMat& image,
                        GpuMat& dst, Stream& stream);

        ...
    };
..

The class implements Q. Yang algorithm
qx_csbp
.

.. index:: gpu::DisparityBilateralFilter::DisparityBilateralFilter

cv::gpu::DisparityBilateralFilter::DisparityBilateralFilter
-----------------------------------------------------------
.. cfunction:: DisparityBilateralFilter::DisparityBilateralFilter( int ndisp = DEFAULT_NDISP, int radius = DEFAULT_RADIUS,  int iters = DEFAULT_ITERS)

.. cfunction:: DisparityBilateralFilter::DisparityBilateralFilter( int ndisp, int radius, int iters,  float edge_threshold, float max_disc_threshold,  float sigma_range)

    DisparityBilateralFilter constructors.

    :param ndisp: Number of disparities.

    :param radius: Filter radius.

    :param iters: Number of iterations.

    :param edge_threshold: Threshold for edges.

    :param max_disc_threshold: Constant to reject outliers.

    :param sigma_range: Filter range.

.. index:: gpu::DisparityBilateralFilter::operator ()

cv::gpu::DisparityBilateralFilter::operator ()
----------------------------------------------
.. cfunction:: void DisparityBilateralFilter::operator()( const GpuMat\& disparity, const GpuMat\& image, GpuMat\& dst)

.. cfunction:: void DisparityBilateralFilter::operator()( const GpuMat\& disparity, const GpuMat\& image, GpuMat\& dst,  Stream\& stream)

    Refines disparity map using joint bilateral filtering.

    :param disparity: Input disparity map; supports  ``CV_8UC1``  and  ``CV_16SC1``  types.

    :param image: Input image; supports  ``CV_8UC1``  and  ``CV_8UC3``  types.

    :param dst: Destination disparity map; will have the same size and type as  ``disparity`` .

    :param stream: Stream for the asynchronous version.

.. index:: gpu::drawColorDisp

cv::gpu::drawColorDisp
----------------------
.. cfunction:: void drawColorDisp(const GpuMat\& src_disp, GpuMat\& dst_disp, int ndisp)

.. cfunction:: void drawColorDisp(const GpuMat\& src_disp, GpuMat\& dst_disp, int ndisp,  const Stream\& stream)

    Does coloring of disparity image.

    :param src_disp: Source disparity image. Supports  ``CV_8UC1``  and  ``CV_16SC1``  types.

    :param dst_disp: Output disparity image. Will have the same size as  ``src_disp``  and  ``CV_8UC4``  type in  ``BGRA``  format (alpha = 255).

    :param ndisp: Number of disparities.

    :param stream: Stream for the asynchronous version.

This function converts
:math:`[0..ndisp)` interval to
:math:`[0..240, 1, 1]` in ``HSV`` color space, than convert ``HSV`` color space to ``RGB`` .

.. index:: gpu::reprojectImageTo3D

cv::gpu::reprojectImageTo3D
---------------------------
.. cfunction:: void reprojectImageTo3D(const GpuMat\& disp, GpuMat\& xyzw,  const Mat\& Q)

.. cfunction:: void reprojectImageTo3D(const GpuMat\& disp, GpuMat\& xyzw,  const Mat\& Q, const Stream\& stream)

    Reprojects disparity image to 3D space.

    :param disp: Input disparity image; supports  ``CV_8U``  and  ``CV_16S``  types.

    :param xyzw: Output 4-channel floating-point image of the same size as  ``disp`` . Each element of  ``xyzw(x,y)``  will contain the 3D coordinates  ``(x,y,z,1)``  of the point  ``(x,y)`` , computed from the disparity map.

    :param Q: :math:`4 \times 4`  perspective transformation matrix that can be obtained via  :ref:`StereoRectify` .

    :param stream: Stream for the asynchronous version.

See also:
:func:`reprojectImageTo3D` .

