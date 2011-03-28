Camera Calibration and 3d Reconstruction
========================================

.. highlight:: cpp



.. index:: gpu::StereoBM_GPU

gpu::StereoBM_GPU
-----------------
.. cpp:class:: gpu::StereoBM_GPU

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

This class computes the disparity map using block matching algorithm. The class also performs pre- and post- filtering steps: sobel prefiltering (if ``PREFILTER_XSOBEL`` flag is set) and low textureness filtering (if ``averageTexThreshols`` :math:`>` 0). If ``avergeTexThreshold = 0`` low textureness filtering is disabled, otherwise disparity is set to 0 in each point ``(x, y)`` where for left image

.. math::
    \sum HorizontalGradiensInWindow(x, y, winSize) < (winSize \cdot winSize) \cdot avergeTexThreshold 

i.e. input left image is low textured.



.. index:: gpu::StereoBM_GPU::StereoBM_GPU

gpu::StereoBM_GPU::StereoBM_GPU
-----------------------------------
.. cpp:function:: gpu::StereoBM_GPU::StereoBM_GPU()

.. cpp:function:: gpu::StereoBM_GPU::StereoBM_GPU(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ)

    ``StereoBM_GPU`` constructors.

    :param preset: Preset:

        * **BASIC_PRESET** Without preprocessing.

        * **PREFILTER_XSOBEL** Sobel prefilter.

    :param ndisparities: Number of disparities. Must be a multiple of 8 and less or equal then 256.

    :param winSize: Block size.



.. index:: gpu::StereoBM_GPU::operator ()

gpu::StereoBM_GPU::operator ()
----------------------------------
.. cpp:function:: void gpu::StereoBM_GPU::operator() (const GpuMat& left, const GpuMat& right, GpuMat& disparity)

.. cpp:function:: void gpu::StereoBM_GPU::operator() (const GpuMat& left, const GpuMat& right, GpuMat& disparity, const Stream& stream)

    The stereo correspondence operator. Finds the disparity for the specified rectified stereo pair.

    :param left: Left image; supports only ``CV_8UC1`` type.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. It will be ``CV_8UC1`` image with the same size as the input images.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::StereoBM_GPU::checkIfGpuCallReasonable

gpu::StereoBM_GPU::checkIfGpuCallReasonable
-----------------------------------------------
.. cpp:function:: bool gpu::StereoBM_GPU::checkIfGpuCallReasonable()

    Some heuristics that tries to estmate if the current GPU will be faster then CPU in this algorithm. It queries current active device.



.. index:: gpu::StereoBeliefPropagation

gpu::StereoBeliefPropagation
----------------------------
.. cpp:class:: gpu::StereoBeliefPropagation

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

The class implements Pedro F. Felzenszwalb algorithm [Pedro F. Felzenszwalb and Daniel P. Huttenlocher. Efficient belief propagation for early vision. International Journal of Computer Vision, 70(1), October 2006]. It can compute own data cost (using truncated linear model) or use user-provided data cost.

**Please note:** ``StereoBeliefPropagation`` requires a lot of memory:

.. math::

    width\_step \cdot height \cdot ndisp \cdot 4 \cdot (1 + 0.25)

for message storage and

.. math::

    width\_step \cdot height \cdot ndisp \cdot (1 + 0.25 + 0.0625 +  \dotsm + \frac{1}{4^{levels}})

for data cost storage. ``width_step`` is the number of bytes in a line including the padding.



.. index:: gpu::StereoBeliefPropagation::StereoBeliefPropagation

gpu::StereoBeliefPropagation::StereoBeliefPropagation
---------------------------------------------------------
.. cpp:function:: gpu::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp = DEFAULT_NDISP, int iters = DEFAULT_ITERS, int levels = DEFAULT_LEVELS, int msg_type = CV_32F)

.. cpp:function:: gpu::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp, int iters, int levels, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, int msg_type = CV_32F)

    ``StereoBeliefPropagation`` constructors.

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param max_data_term: Threshold for data cost truncation.

    :param data_weight: Data weight.

    :param max_disc_term: Threshold for discontinuity truncation.

    :param disc_single_jump: Discontinuity single jump.

    :param msg_type: Type for messages. Supports ``CV_16SC1`` and ``CV_32FC1``.
    
:cpp:class:`StereoBeliefPropagation` uses truncated linear model for the data cost and discontinuity term:

.. math::

    DataCost = data\_weight \cdot \min(\lvert I_2-I_1 \rvert, max\_data\_term)

.. math::

    DiscTerm =  \min(disc\_single\_jump \cdot \lvert f_1-f_2 \rvert, max\_disc\_term)

For more details please see [Pedro F. Felzenszwalb and Daniel P. Huttenlocher. Efficient belief propagation for early vision. International Journal of Computer Vision, 70(1), October 2006].

By default :cpp:class:`StereoBeliefPropagation` uses floating-point arithmetics and ``CV_32FC1`` type for messages. But also it can use fixed-point arithmetics and ``CV_16SC1`` type for messages for better perfomance. To avoid overflow in this case, the parameters must satisfy

.. math::

    10 \cdot 2^{levels-1} \cdot max\_data\_term < SHRT\_MAX



.. index:: gpu::StereoBeliefPropagation::estimateRecommendedParams

gpu::StereoBeliefPropagation::estimateRecommendedParams
-----------------------------------------------------------

.. cpp:function:: void gpu::StereoBeliefPropagation::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels)

    Some heuristics that tries to compute recommended parameters (``ndisp``, ``iters`` and ``levels``) for specified image size (``width`` and ``height``).



.. index:: gpu::StereoBeliefPropagation::operator ()

gpu::StereoBeliefPropagation::operator ()
---------------------------------------------
.. cpp:function:: void gpu::StereoBeliefPropagation::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity)

.. cpp:function:: void gpu::StereoBeliefPropagation::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream)

    The stereo correspondence operator. Finds the disparity for the specified rectified stereo pair or data cost.

    :param left: Left image; supports ``CV_8UC1``, ``CV_8UC3`` and ``CV_8UC4`` types.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. If ``disparity`` is empty output type will be ``CV_16SC1``, otherwise output type will be ``disparity.type()``.

    :param stream: Stream for the asynchronous version.

.. cpp:function:: void StereoBeliefPropagation::operator()(const GpuMat& data, GpuMat& disparity)

.. cpp:function:: void StereoBeliefPropagation::operator()(const GpuMat& data, GpuMat& disparity, Stream& stream)

    :param data: The user specified data cost. It must have ``msg_type`` type and :math:`\texttt{imgRows} \cdot \texttt{ndisp} \times \texttt{imgCols}` size.

    :param disparity: Output disparity map. If ``disparity`` is empty output type will be ``CV_16SC1``, otherwise output type will be ``disparity.type()``.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::StereoConstantSpaceBP

gpu::StereoConstantSpaceBP
--------------------------
.. cpp:class:: gpu::StereoConstantSpaceBP

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


The class implements Q. Yang algorithm [Q. Yang, L. Wang, and N. Ahuja. A constant-space belief propagation algorithm for stereo matching. In CVPR, 2010]. ``StereoConstantSpaceBP`` supports both local minimum and global minimum data cost initialization algortihms. For more details please see the paper. By default local algorithm is used, and to enable global algorithm set ``use_local_init_data_cost`` to false.



.. index:: gpu::StereoConstantSpaceBP::StereoConstantSpaceBP

gpu::StereoConstantSpaceBP::StereoConstantSpaceBP
-----------------------------------------------------
.. cpp:function:: gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp = DEFAULT_NDISP, int iters = DEFAULT_ITERS, int levels = DEFAULT_LEVELS, int nr_plane = DEFAULT_NR_PLANE, int msg_type = CV_32F)

.. cpp:function:: gpu::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, int min_disp_th = 0, int msg_type = CV_32F)

    ``StereoConstantSpaceBP`` constructors.

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param nr_plane: Number of disparity levels on the first level

    :param max_data_term: Truncation of data cost.

    :param data_weight: Data weight.

    :param max_disc_term: Truncation of discontinuity.

    :param disc_single_jump: Discontinuity single jump.

    :param min_disp_th: Minimal disparity threshold.

    :param msg_type: Type for messages. Supports ``CV_16SC1`` and ``CV_32FC1``.
    
:cpp:class:`StereoConstantSpaceBP` uses truncated linear model for the data cost and discontinuity term:

.. math::

    DataCost = data\_weight \cdot \min(\lvert I_2-I_1 \rvert, max\_data\_term)

.. math::

    DiscTerm =  \min(disc\_single\_jump \cdot \lvert f_1-f_2 \rvert, max\_disc\_term)

For more details please see [Q. Yang, L. Wang, and N. Ahuja. A constant-space belief propagation algorithm for stereo matching. In CVPR, 2010].

By default :cpp:class:`StereoConstantSpaceBP` uses floating-point arithmetics and ``CV_32FC1`` type for messages. But also it can use fixed-point arithmetics and ``CV_16SC1`` type for messages for better perfomance. To avoid overflow in this case, the parameters must satisfy

.. math::

    10 \cdot 2^{levels-1} \cdot max\_data\_term < SHRT\_MAX



.. index:: gpu::StereoConstantSpaceBP::estimateRecommendedParams

gpu::StereoConstantSpaceBP::estimateRecommendedParams
---------------------------------------------------------

.. cpp:function:: void gpu::StereoConstantSpaceBP::estimateRecommendedParams( int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane)

    Some heuristics that tries to compute parameters (``ndisp``, ``iters``, ``levels`` and ``nr_plane``) for specified image size (``width`` and ``height``).


.. index:: gpu::StereoConstantSpaceBP::operator ()

gpu::StereoConstantSpaceBP::operator ()
-------------------------------------------
.. cpp:function:: void gpu::StereoConstantSpaceBP::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity)

.. cpp:function:: void gpu::StereoConstantSpaceBP::operator()(const GpuMat& left, const GpuMat& right, GpuMat& disparity, Stream& stream)

    The stereo correspondence operator. Finds the disparity for the specified rectified stereo pair.

    :param left: Left image; supports ``CV_8UC1``, ``CV_8UC3`` and ``CV_8UC4`` types.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. If ``disparity`` is empty output type will be ``CV_16SC1``, otherwise output type will be ``disparity.type()``.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::DisparityBilateralFilter

gpu::DisparityBilateralFilter
-----------------------------
.. cpp:class:: gpu::DisparityBilateralFilter

The class for disparity map refinement using joint bilateral filtering. ::

    class DisparityBilateralFilter
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


The class implements Q. Yang algorithm [Q. Yang, L. Wang, and N. Ahuja. A constant-space belief propagation algorithm for stereo matching. In CVPR, 2010].



.. index:: gpu::DisparityBilateralFilter::DisparityBilateralFilter

gpu::DisparityBilateralFilter::DisparityBilateralFilter
-----------------------------------------------------------
.. cpp:function:: gpu::DisparityBilateralFilter::DisparityBilateralFilter(int ndisp = DEFAULT_NDISP, int radius = DEFAULT_RADIUS, int iters = DEFAULT_ITERS)

.. cpp:function:: gpu::DisparityBilateralFilter::DisparityBilateralFilter(int ndisp, int radius, int iters, float edge_threshold, float max_disc_threshold, float sigma_range)

    ``DisparityBilateralFilter`` constructors.

    :param ndisp: Number of disparities.

    :param radius: Filter radius.

    :param iters: Number of iterations.

    :param edge_threshold: Threshold for edges.

    :param max_disc_threshold: Constant to reject outliers.

    :param sigma_range: Filter range.



.. index:: gpu::DisparityBilateralFilter::operator ()

gpu::DisparityBilateralFilter::operator ()
----------------------------------------------
.. cpp:function:: void gpu::DisparityBilateralFilter::operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst)

.. cpp:function:: void gpu::DisparityBilateralFilter::operator()(const GpuMat& disparity, const GpuMat& image, GpuMat& dst, Stream& stream)

    Refines disparity map using joint bilateral filtering.

    :param disparity: Input disparity map; supports ``CV_8UC1`` and ``CV_16SC1`` types.

    :param image: Input image; supports ``CV_8UC1`` and ``CV_8UC3`` types.

    :param dst: Destination disparity map; will have the same size and type as ``disparity``.

    :param stream: Stream for the asynchronous version.



.. index:: gpu::drawColorDisp

gpu::drawColorDisp
----------------------
.. cpp:function:: void gpu::drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp)

.. cpp:function:: void gpu::drawColorDisp(const GpuMat& src_disp, GpuMat& dst_disp, int ndisp, const Stream& stream)

    Does coloring of disparity image.

    :param src_disp: Source disparity image. Supports ``CV_8UC1`` and ``CV_16SC1`` types.

    :param dst_disp: Output disparity image. Will have the same size as ``src_disp`` and ``CV_8UC4`` type in ``BGRA`` format (alpha = 255).

    :param ndisp: Number of disparities.

    :param stream: Stream for the asynchronous version.

This function converts :math:`[0..ndisp)` interval to :math:`[0..240, 1, 1]` in ``HSV`` color space, than convert ``HSV`` color space to ``RGB``.



.. index:: gpu::reprojectImageTo3D

gpu::reprojectImageTo3D
---------------------------
.. cpp:function:: void gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q)

.. cpp:function:: void gpu::reprojectImageTo3D(const GpuMat& disp, GpuMat& xyzw, const Mat& Q, const Stream& stream)

    Reprojects disparity image to 3D space.

    :param disp: Input disparity image; supports ``CV_8U`` and ``CV_16S`` types.

    :param xyzw: Output 4-channel floating-point image of the same size as ``disp``. Each element of ``xyzw(x,y)`` will contain the 3D coordinates ``(x,y,z,1)`` of the point ``(x,y)``, computed from the disparity map.

    :param Q: :math:`4 \times 4` perspective transformation matrix that can be obtained via :c:func:`stereoRectify`.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`reprojectImageTo3D`.



.. index:: gpu::solvePnPRansac

gpu::solvePnPRansac
-------------------

.. cpp:function:: void gpu::solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat, const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false, int num_iters=100, float max_dist=8.0, int min_inlier_count=100, vector<int>* inliers=NULL)

    Finds the object pose from the 3D-2D point correspondences.
    
    :param object: Single-row matrix of object points.
    
    :param image: Single-row matrix of image points.
    
    :param camera_mat: 3x3 matrix of intrinsic camera parameters.
    
    :param dist_coef: Distortion coefficients. See :c:func:`undistortPoints` for details.
    
    :param rvec: Output 3D rotation vector.
    
    :param tvec: Output 3D translation vector.
    
    :param use_extrinsic_guess: Indicates the function must use ``rvec`` and ``tvec`` as initial transformation guess. It isn't supported for now.
    
    :param num_iters: Maximum number of RANSAC iterations.
    
    :param max_dist: Euclidean distance threshold to detect whether point is inlier or not.
    
    :param min_inlier_count: Indicates the function must stop if greater or equal number of inliers is achieved. It isn't supported for now.
    
    :param inliers: Output vector of inlier indices.   

See also :c:func:`solvePnPRansac`.
