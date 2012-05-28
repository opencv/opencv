Stereo Correspondence
========================================

.. highlight:: cpp

StereoVar
----------

.. ocv:class:: StereoVar

Class for computing stereo correspondence using the variational matching algorithm ::

    class StereoVar
    {
        StereoVar();
        StereoVar(    int levels, double pyrScale,
                                        int nIt, int minDisp, int maxDisp,
                                        int poly_n, double poly_sigma, float fi,
                                        float lambda, int penalization, int cycle,
                                        int flags);
        virtual ~StereoVar();

        virtual void operator()(InputArray left, InputArray right, OutputArray disp);

        int        levels;
        double    pyrScale;
        int        nIt;
        int        minDisp;
        int        maxDisp;
        int        poly_n;
        double    poly_sigma;
        float    fi;
        float    lambda;
        int        penalization;
        int        cycle;
        int        flags;

        ...
    };

The class implements the modified S. G. Kosov algorithm [Publication] that differs from the original one as follows:

 * The automatic initialization of method's parameters is added.

 * The method of Smart Iteration Distribution (SID) is implemented.

 * The support of Multi-Level Adaptation Technique (MLAT) is not included.

 * The method of dynamic adaptation of method's parameters is not included.

StereoVar::StereoVar
--------------------------

.. ocv:function:: StereoVar::StereoVar()

.. ocv:function:: StereoVar::StereoVar( int levels, double pyrScale, int nIt, int minDisp, int maxDisp, int poly_n, double poly_sigma, float fi, float lambda, int penalization, int cycle, int flags )

    The constructor

    :param levels: The number of pyramid layers, including the initial image. levels=1 means that no extra layers are created and only the original images are used. This parameter is ignored if flag USE_AUTO_PARAMS is set.

    :param pyrScale: Specifies the image scale (<1) to build the pyramids for each image. pyrScale=0.5 means the classical pyramid, where each next layer is twice smaller than the previous. (This parameter is ignored if flag USE_AUTO_PARAMS is set).

    :param nIt: The number of iterations the algorithm does at each pyramid level. (If the flag USE_SMART_ID is set, the number of iterations will be redistributed in such a way, that more iterations will be done on more coarser levels.)

    :param minDisp: Minimum possible disparity value. Could be negative in case the left and right input images change places.

    :param maxDisp: Maximum possible disparity value.

    :param poly_n: Size of the pixel neighbourhood used to find polynomial expansion in each pixel. The larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field. Typically, poly_n = 3, 5 or 7

    :param poly_sigma: Standard deviation of the Gaussian that is used to smooth derivatives that are used as a basis for the polynomial expansion. For poly_n=5 you can set poly_sigma=1.1 , for poly_n=7 a good value would be poly_sigma=1.5

    :param fi: The smoothness parameter, ot the weight coefficient for the smoothness term.

    :param lambda: The threshold parameter for edge-preserving smoothness. (This parameter is ignored if PENALIZATION_CHARBONNIER or PENALIZATION_PERONA_MALIK is used.)

    :param penalization: Possible values: PENALIZATION_TICHONOV - linear smoothness; PENALIZATION_CHARBONNIER - non-linear edge preserving smoothness; PENALIZATION_PERONA_MALIK - non-linear edge-enhancing smoothness. (This parameter is ignored if flag USE_AUTO_PARAMS is set).

    :param cycle: Type of the multigrid cycle. Possible values: CYCLE_O and CYCLE_V for null- and v-cycles respectively. (This parameter is ignored if flag USE_AUTO_PARAMS is set).

    :param flags: The operation flags; can be a combination of the following:

        * USE_INITIAL_DISPARITY: Use the input flow as the initial flow approximation.

        * USE_EQUALIZE_HIST: Use the histogram equalization in the pre-processing phase.

        * USE_SMART_ID: Use the smart iteration distribution (SID).

        * USE_AUTO_PARAMS: Allow the method to initialize the main parameters.

        * USE_MEDIAN_FILTERING: Use the median filer of the solution in the post processing phase.

The first constructor initializes ``StereoVar`` with all the default parameters. So, you only have to set ``StereoVar::maxDisp`` and / or ``StereoVar::minDisp`` at minimum. The second constructor enables you to set each parameter to a custom value.



StereoVar::operator ()
-----------------------

.. ocv:function:: void StereoVar::operator()( const Mat& left, const Mat& right, Mat& disp )

    Computes disparity using the variational algorithm for a rectified stereo pair.

    :param left: Left 8-bit single-channel or 3-channel image.

    :param right: Right image of the same size and the same type as the left one.

    :param disp: Output disparity map. It is a 8-bit signed single-channel image of the same size as the input image.

The method executes the variational algorithm on a rectified stereo pair. See ``stereo_match.cpp`` OpenCV sample on how to prepare images and call the method.

**Note**:

The method is not constant, so you should not use the same ``StereoVar`` instance from different threads simultaneously.


