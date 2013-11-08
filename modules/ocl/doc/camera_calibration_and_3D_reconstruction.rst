Camera Calibration and 3D Reconstruction
========================================

.. highlight:: cpp



ocl::StereoBM_OCL
---------------------
.. ocv:class:: ocl::StereoBM_OCL

Class computing stereo correspondence (disparity map) using the block matching algorithm. ::

    class CV_EXPORTS StereoBM_OCL
    {
    public:
        enum { BASIC_PRESET = 0, PREFILTER_XSOBEL = 1 };

        enum { DEFAULT_NDISP = 64, DEFAULT_WINSZ = 19 };

        //! the default constructor
        StereoBM_OCL();
        //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size. ndisparities must be multiple of 8.
        StereoBM_OCL(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ);

        //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
        //! Output disparity has CV_8U type.
        void operator() ( const oclMat &left, const oclMat &right, oclMat &disparity);

        //! Some heuristics that tries to estmate
        // if current GPU will be faster then CPU in this algorithm.
        // It queries current active device.
        static bool checkIfGpuCallReasonable();

        int preset;
        int ndisp;
        int winSize;

        // If avergeTexThreshold  == 0 => post procesing is disabled
        // If avergeTexThreshold != 0 then disparity is set 0 in each point (x,y) where for left image
        // SumOfHorizontalGradiensInWindow(x, y, winSize) < (winSize * winSize) * avergeTexThreshold
        // i.e. input left image is low textured.
        float avergeTexThreshold;
    private:
        /* hidden */
    };


The class also performs pre- and post-filtering steps: Sobel pre-filtering (if ``PREFILTER_XSOBEL`` flag is set) and low textureness filtering (if ``averageTexThreshols > 0`` ). If ``avergeTexThreshold = 0`` , low textureness filtering is disabled. Otherwise, the disparity is set to 0 in each point ``(x, y)`` , where for the left image

.. math::
    \sum HorizontalGradiensInWindow(x, y, winSize) < (winSize \cdot winSize) \cdot avergeTexThreshold

This means that the input left image is low textured.


ocl::StereoBM_OCL::StereoBM_OCL
-----------------------------------
Enables :ocv:class:`ocl::StereoBM_OCL` constructors.

.. ocv:function:: ocl::StereoBM_OCL::StereoBM_OCL()

.. ocv:function:: ocl::StereoBM_OCL::StereoBM_OCL(int preset, int ndisparities = DEFAULT_NDISP, int winSize = DEFAULT_WINSZ)

    :param preset: Parameter presetting:

        * **BASIC_PRESET** Basic mode without pre-processing.

        * **PREFILTER_XSOBEL** Sobel pre-filtering mode.

    :param ndisparities: Number of disparities. It must be a multiple of 8 and less or equal to 256.

    :param winSize: Block size.



ocl::StereoBM_OCL::operator ()
----------------------------------
Enables the stereo correspondence operator that finds the disparity for the specified rectified stereo pair.

.. ocv:function:: void ocl::StereoBM_OCL::operator ()(const oclMat& left, const oclMat& right, oclMat& disparity)

    :param left: Left image. Only  ``CV_8UC1``  type is supported.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. It is a  ``CV_8UC1``  image with the same size as the input images.


ocl::StereoBM_OCL::checkIfGpuCallReasonable
-----------------------------------------------
Uses a heuristic method to estimate whether the current GPU is faster than the CPU in this algorithm. It queries the currently active device.

.. ocv:function:: bool ocl::StereoBM_OCL::checkIfGpuCallReasonable()

ocl::StereoBeliefPropagation
--------------------------------
.. ocv:class:: ocl::StereoBeliefPropagation

Class computing stereo correspondence using the belief propagation algorithm. ::

    class CV_EXPORTS StereoBeliefPropagation
    {
    public:
        enum { DEFAULT_NDISP  = 64 };
        enum { DEFAULT_ITERS  = 5  };
        enum { DEFAULT_LEVELS = 5  };
        static void estimateRecommendedParams(int width, int height, int &ndisp, int &iters, int &levels);
        explicit StereoBeliefPropagation(int ndisp  = DEFAULT_NDISP,
                                         int iters  = DEFAULT_ITERS,
                                         int levels = DEFAULT_LEVELS,
                                         int msg_type = CV_16S);
        StereoBeliefPropagation(int ndisp, int iters, int levels,
                                float max_data_term, float data_weight,
                                float max_disc_term, float disc_single_jump,
                                int msg_type = CV_32F);
        void operator()(const oclMat &left, const oclMat &right, oclMat &disparity);
        void operator()(const oclMat &data, oclMat &disparity);
        int ndisp;
        int iters;
        int levels;
        float max_data_term;
        float data_weight;
        float max_disc_term;
        float disc_single_jump;
        int msg_type;
    private:
        /* hidden */
    };

The class implements algorithm described in [Felzenszwalb2006]_ . It can compute own data cost (using a truncated linear model) or use a user-provided data cost.

.. note::

    ``StereoBeliefPropagation`` requires a lot of memory for message storage:

    .. math::

        width \_ step  \cdot height  \cdot ndisp  \cdot 4  \cdot (1 + 0.25)

    and for data cost storage:

    .. math::

        width\_step \cdot height \cdot ndisp \cdot (1 + 0.25 + 0.0625 +  \dotsm + \frac{1}{4^{levels}})

    ``width_step`` is the number of bytes in a line including padding.



ocl::StereoBeliefPropagation::StereoBeliefPropagation
---------------------------------------------------------
Enables the :ocv:class:`ocl::StereoBeliefPropagation` constructors.

.. ocv:function:: ocl::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp = DEFAULT_NDISP, int iters = DEFAULT_ITERS, int levels = DEFAULT_LEVELS, int msg_type = CV_16S)

.. ocv:function:: ocl::StereoBeliefPropagation::StereoBeliefPropagation(int ndisp, int iters, int levels, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, int msg_type = CV_32F)

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param max_data_term: Threshold for data cost truncation.

    :param data_weight: Data weight.

    :param max_disc_term: Threshold for discontinuity truncation.

    :param disc_single_jump: Discontinuity single jump.

    :param msg_type: Type for messages.  ``CV_16SC1``  and  ``CV_32FC1`` types are supported.

``StereoBeliefPropagation`` uses a truncated linear model for the data cost and discontinuity terms:

.. math::

    DataCost = data \_ weight  \cdot \min ( \lvert Img_Left(x,y)-Img_Right(x-d,y)  \rvert , max \_ data \_ term)

.. math::

    DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)

For more details, see [Felzenszwalb2006]_.

By default, :ocv:class:`ocl::StereoBeliefPropagation` uses floating-point arithmetics and the ``CV_32FC1`` type for messages. But it can also use fixed-point arithmetics and the ``CV_16SC1`` message type for better performance. To avoid an overflow in this case, the parameters must satisfy the following requirement:

.. math::

    10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX



ocl::StereoBeliefPropagation::estimateRecommendedParams
-----------------------------------------------------------
Uses a heuristic method to compute the recommended parameters ( ``ndisp``, ``iters`` and ``levels`` ) for the specified image size ( ``width`` and ``height`` ).

.. ocv:function:: void ocl::StereoBeliefPropagation::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels)



ocl::StereoBeliefPropagation::operator ()
---------------------------------------------
Enables the stereo correspondence operator that finds the disparity for the specified rectified stereo pair or data cost.

.. ocv:function:: void ocl::StereoBeliefPropagation::operator ()(const oclMat& left, const oclMat& right, oclMat& disparity)

.. ocv:function:: void ocl::StereoBeliefPropagation::operator ()(const oclMat& data, oclMat& disparity)

    :param left: Left image. ``CV_8UC1`` , ``CV_8UC3``  and  ``CV_8UC4``  types are supported.

    :param right: Right image with the same size and the same type as the left one.

    :param data: User-specified data cost, a matrix of ``msg_type`` type and ``Size(<image columns>*ndisp, <image rows>)`` size.

    :param disparity: Output disparity map. If  ``disparity``  is empty, the output type is  ``CV_16SC1`` . Otherwise, the type is retained.

ocl::StereoConstantSpaceBP
------------------------------
.. ocv:class:: ocl::StereoConstantSpaceBP

Class computing stereo correspondence using the constant space belief propagation algorithm. ::

    class CV_EXPORTS StereoConstantSpaceBP
    {
    public:
        enum { DEFAULT_NDISP    = 128 };
        enum { DEFAULT_ITERS    = 8   };
        enum { DEFAULT_LEVELS   = 4   };
        enum { DEFAULT_NR_PLANE = 4   };
        static void estimateRecommendedParams(int width, int height, int &ndisp, int &iters, int &levels, int &nr_plane);
        explicit StereoConstantSpaceBP(
            int ndisp    = DEFAULT_NDISP,
            int iters    = DEFAULT_ITERS,
            int levels   = DEFAULT_LEVELS,
            int nr_plane = DEFAULT_NR_PLANE,
            int msg_type = CV_32F);
        StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane,
            float max_data_term, float data_weight, float max_disc_term, float disc_single_jump,
            int min_disp_th = 0,
            int msg_type = CV_32F);
        void operator()(const oclMat &left, const oclMat &right, oclMat &disparity);
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
    private:
        /* hidden */
    };

The class implements algorithm described in [Yang2010]_. ``StereoConstantSpaceBP`` supports both local minimum and global minimum data cost initialization algorithms. For more details, see the paper mentioned above. By default, a local algorithm is used. To enable a global algorithm, set ``use_local_init_data_cost`` to ``false`` .


ocl::StereoConstantSpaceBP::StereoConstantSpaceBP
-----------------------------------------------------
Enables the :ocv:class:`ocl::StereoConstantSpaceBP` constructors.

.. ocv:function:: ocl::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp = DEFAULT_NDISP, int iters = DEFAULT_ITERS, int levels = DEFAULT_LEVELS, int nr_plane = DEFAULT_NR_PLANE, int msg_type = CV_32F)

.. ocv:function:: ocl::StereoConstantSpaceBP::StereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane, float max_data_term, float data_weight, float max_disc_term, float disc_single_jump, int min_disp_th = 0, int msg_type = CV_32F)

    :param ndisp: Number of disparities.

    :param iters: Number of BP iterations on each level.

    :param levels: Number of levels.

    :param nr_plane: Number of disparity levels on the first level.

    :param max_data_term: Truncation of data cost.

    :param data_weight: Data weight.

    :param max_disc_term: Truncation of discontinuity.

    :param disc_single_jump: Discontinuity single jump.

    :param min_disp_th: Minimal disparity threshold.

    :param msg_type: Type for messages.  ``CV_16SC1``  and  ``CV_32FC1`` types are supported.

``StereoConstantSpaceBP`` uses a truncated linear model for the data cost and discontinuity terms:

.. math::

    DataCost = data \_ weight  \cdot \min ( \lvert I_2-I_1  \rvert , max \_ data \_ term)

.. math::

    DiscTerm =  \min (disc \_ single \_ jump  \cdot \lvert f_1-f_2  \rvert , max \_ disc \_ term)

For more details, see [Yang2010]_.

By default, ``StereoConstantSpaceBP`` uses floating-point arithmetics and the ``CV_32FC1`` type for messages. But it can also use fixed-point arithmetics and the ``CV_16SC1`` message type for better performance. To avoid an overflow in this case, the parameters must satisfy the following requirement:

.. math::

    10  \cdot 2^{levels-1}  \cdot max \_ data \_ term < SHRT \_ MAX



ocl::StereoConstantSpaceBP::estimateRecommendedParams
---------------------------------------------------------
Uses a heuristic method to compute parameters (ndisp, iters, levelsand nrplane) for the specified image size (widthand height).

.. ocv:function:: void ocl::StereoConstantSpaceBP::estimateRecommendedParams(int width, int height, int& ndisp, int& iters, int& levels, int& nr_plane)



ocl::StereoConstantSpaceBP::operator ()
-------------------------------------------
Enables the stereo correspondence operator that finds the disparity for the specified rectified stereo pair.

.. ocv:function:: void ocl::StereoConstantSpaceBP::operator ()(const oclMat& left, const oclMat& right, oclMat& disparity)

    :param left: Left image. ``CV_8UC1`` , ``CV_8UC3``  and  ``CV_8UC4``  types are supported.

    :param right: Right image with the same size and the same type as the left one.

    :param disparity: Output disparity map. If  ``disparity``  is empty, the output type is  ``CV_16SC1`` . Otherwise, the output type is  ``disparity.type()`` .
