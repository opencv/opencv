Image Processing
================

.. highlight:: cpp



gpu::meanShiftFiltering
---------------------------
Performs mean-shift filtering for each point of the source image.

.. ocv:function:: void gpu::meanShiftFiltering( const GpuMat& src, GpuMat& dst, int sp, int sr, TermCriteria criteria=TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1), Stream& stream=Stream::Null() )

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dst: Destination image containing the color of mapped points. It has the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

It maps each point of the source image into another point. As a result, you have a new color and new position of each point.



gpu::meanShiftProc
----------------------
Performs a mean-shift procedure and stores information about processed points (their colors and positions) in two images.

.. ocv:function:: void gpu::meanShiftProc( const GpuMat& src, GpuMat& dstr, GpuMat& dstsp, int sp, int sr, TermCriteria criteria=TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1), Stream& stream=Stream::Null() )

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dstr: Destination image containing the color of mapped points. The size and type is the same as  ``src`` .

    :param dstsp: Destination image containing the position of mapped points. The size is the same as  ``src`` size. The type is  ``CV_16SC2`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.

.. seealso:: :ocv:func:`gpu::meanShiftFiltering`



gpu::meanShiftSegmentation
------------------------------
Performs a mean-shift segmentation of the source image and eliminates small segments.

.. ocv:function:: void gpu::meanShiftSegmentation(const GpuMat& src, Mat& dst, int sp, int sr, int minsize, TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1))

    :param src: Source image. Only  ``CV_8UC4`` images are supported for now.

    :param dst: Segmented image with the same size and type as  ``src`` .

    :param sp: Spatial window radius.

    :param sr: Color window radius.

    :param minsize: Minimum segment size. Smaller segments are merged.

    :param criteria: Termination criteria. See :ocv:class:`TermCriteria`.



gpu::integral
-----------------
Computes an integral image.

.. ocv:function:: void gpu::integral(const GpuMat& src, GpuMat& sum, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sum: Integral image containing 32-bit unsigned integer values packed into  ``CV_32SC1`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`integral`



gpu::sqrIntegral
--------------------
Computes a squared integral image.

.. ocv:function:: void gpu::sqrIntegral(const GpuMat& src, GpuMat& sqsum, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` images are supported for now.

    :param sqsum: Squared integral image containing 64-bit unsigned integer values packed into  ``CV_64FC1`` .

    :param stream: Stream for the asynchronous version.



gpu::cornerHarris
---------------------
Computes the Harris cornerness criteria at each image pixel.

.. ocv:function:: void gpu::cornerHarris(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, double k, int borderType=BORDER_REFLECT101)

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. It has the same size as ``src`` and ``CV_32FC1`` type.

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param k: Harris detector free parameter.

    :param borderType: Pixel extrapolation method. Only  ``BORDER_REFLECT101`` and  ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerHarris`



gpu::cornerMinEigenVal
--------------------------
Computes the minimum eigen value of a 2x2 derivative covariation matrix at each pixel (the cornerness criteria).

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, int blockSize, int ksize, int borderType=BORDER_REFLECT101)

.. ocv:function:: void gpu::cornerMinEigenVal(const GpuMat& src, GpuMat& dst, GpuMat& Dx, GpuMat& Dy, GpuMat& buf, int blockSize, int ksize, int borderType=BORDER_REFLECT101, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1`` and  ``CV_32FC1`` images are supported for now.

    :param dst: Destination image containing cornerness values. The size is the same. The type is  ``CV_32FC1`` .

    :param blockSize: Neighborhood size.

    :param ksize: Aperture parameter for the Sobel operator.

    :param borderType: Pixel extrapolation method. Only ``BORDER_REFLECT101`` and ``BORDER_REPLICATE`` are supported for now.

.. seealso:: :ocv:func:`cornerMinEigenVal`



gpu::MatchTemplateBuf
---------------------
.. ocv:struct:: gpu::MatchTemplateBuf

Class providing memory buffers for :ocv:func:`gpu::matchTemplate` function, plus it allows to adjust some specific parameters. ::

    struct CV_EXPORTS MatchTemplateBuf
    {
        Size user_block_size;
        GpuMat imagef, templf;
        std::vector<GpuMat> images;
        std::vector<GpuMat> image_sums;
        std::vector<GpuMat> image_sqsums;
    };

You can use field `user_block_size` to set specific block size for :ocv:func:`gpu::matchTemplate` function. If you leave its default value `Size(0,0)` then automatic estimation of block size will be used (which is optimized for speed). By varying `user_block_size` you can reduce memory requirements at the cost of speed.



gpu::matchTemplate
----------------------
Computes a proximity map for a raster template and an image where the template is searched for.

.. ocv:function:: void gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, Stream &stream = Stream::Null())

.. ocv:function:: void gpu::matchTemplate(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, MatchTemplateBuf &buf, Stream& stream = Stream::Null())

    :param image: Source image.  ``CV_32F`` and  ``CV_8U`` depth images (1..4 channels) are supported for now.

    :param templ: Template image with the size and type the same as  ``image`` .

    :param result: Map containing comparison results ( ``CV_32FC1`` ). If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param method: Specifies the way to compare the template with the image.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:struct:`gpu::MatchTemplateBuf`.

    :param stream: Stream for the asynchronous version.

    The following methods are supported for the ``CV_8U`` depth images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_SQDIFF_NORMED``
    * ``CV_TM_CCORR``
    * ``CV_TM_CCORR_NORMED``
    * ``CV_TM_CCOEFF``
    * ``CV_TM_CCOEFF_NORMED``

    The following methods are supported for the ``CV_32F`` images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_CCORR``

.. seealso:: :ocv:func:`matchTemplate`



gpu::cvtColor
-----------------
Converts an image from one color space to another.

.. ocv:function:: void gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0, Stream& stream = Stream::Null())

    :param src: Source image with  ``CV_8U`` , ``CV_16U`` , or  ``CV_32F`` depth and 1, 3, or 4 channels.

    :param dst: Destination image with the same size and depth as  ``src`` .

    :param code: Color space conversion code. For details, see  :ocv:func:`cvtColor` . Conversion to/from Luv and Bayer color spaces is not supported.

    :param dcn: Number of channels in the destination image. If the parameter is 0, the number of the channels is derived automatically from  ``src`` and the  ``code`` .

    :param stream: Stream for the asynchronous version.

3-channel color spaces (like ``HSV``, ``XYZ``, and so on) can be stored in a 4-channel image for better performance.

.. seealso:: :ocv:func:`cvtColor`



gpu::swapChannels
-----------------
Exchanges the color channels of an image in-place.

.. ocv:function:: void gpu::swapChannels(GpuMat& image, const int dstOrder[4], Stream& stream = Stream::Null())

    :param image: Source image. Supports only ``CV_8UC4`` type.

    :param dstOrder: Integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.

    :param stream: Stream for the asynchronous version.

The methods support arbitrary permutations of the original channels, including replication.



gpu::rectStdDev
-------------------
Computes a standard deviation of integral images.

.. ocv:function:: void gpu::rectStdDev(const GpuMat& src, const GpuMat& sqr, GpuMat& dst, const Rect& rect, Stream& stream = Stream::Null())

    :param src: Source image. Only the ``CV_32SC1`` type is supported.

    :param sqr: Squared source image. Only  the ``CV_32FC1`` type is supported.

    :param dst: Destination image with the same type and size as  ``src`` .

    :param rect: Rectangular window.

    :param stream: Stream for the asynchronous version.



gpu::evenLevels
-------------------
Computes levels with even distribution.

.. ocv:function:: void gpu::evenLevels(GpuMat& levels, int nLevels, int lowerLevel, int upperLevel)

    :param levels: Destination array.  ``levels`` has 1 row, ``nLevels`` columns, and the ``CV_32SC1`` type.

    :param nLevels: Number of computed levels.  ``nLevels`` must be at least 2.

    :param lowerLevel: Lower boundary value of the lowest level.

    :param upperLevel: Upper boundary value of the greatest level.



gpu::histEven
-----------------
Calculates a histogram with evenly distributed bins.

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven(const GpuMat& src, GpuMat& hist, GpuMat& buf, int histSize, int lowerLevel, int upperLevel, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histEven( const GpuMat& src, GpuMat hist[4], int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream=Stream::Null() )

.. ocv:function:: void gpu::histEven( const GpuMat& src, GpuMat hist[4], GpuMat& buf, int histSize[4], int lowerLevel[4], int upperLevel[4], Stream& stream=Stream::Null() )

    :param src: Source image. ``CV_8U``, ``CV_16U``, or ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``histSize`` columns, and the ``CV_32S`` type.

    :param histSize: Size of the histogram.

    :param lowerLevel: Lower boundary of lowest-level bin.

    :param upperLevel: Upper boundary of highest-level bin.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::histRange
------------------
Calculates a histogram with bins determined by the ``levels`` array.

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::histRange(const GpuMat& src, GpuMat& hist, const GpuMat& levels, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8U`` , ``CV_16U`` , or  ``CV_16S`` depth and 1 or 4 channels are supported. For a four-channel image, all channels are processed separately.

    :param hist: Destination histogram with one row, ``(levels.cols-1)`` columns, and the  ``CV_32SC1`` type.

    :param levels: Number of levels in the histogram.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.



gpu::calcHist
------------------
Calculates histogram for one channel 8-bit image.

.. ocv:function:: void gpu::calcHist(const GpuMat& src, GpuMat& hist, Stream& stream = Stream::Null())

    :param src: Source image.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param stream: Stream for the asynchronous version.



gpu::equalizeHist
------------------
Equalizes the histogram of a grayscale image.

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::equalizeHist(const GpuMat& src, GpuMat& dst, GpuMat& hist, GpuMat& buf, Stream& stream = Stream::Null())

    :param src: Source image.

    :param dst: Destination image.

    :param hist: Destination histogram with one row, 256 columns, and the  ``CV_32SC1`` type.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`equalizeHist`



gpu::blendLinear
-------------------
Performs linear blending of two images.

.. ocv:function:: void gpu::blendLinear(const GpuMat& img1, const GpuMat& img2, const GpuMat& weights1, const GpuMat& weights2, GpuMat& result, Stream& stream = Stream::Null())

    :param img1: First image. Supports only ``CV_8U`` and ``CV_32F`` depth.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param weights1: Weights for first image. Must have tha same size as ``img1`` . Supports only ``CV_32F`` type.

    :param weights2: Weights for second image. Must have tha same size as ``img2`` . Supports only ``CV_32F`` type.

    :param result: Destination image.

    :param stream: Stream for the asynchronous version.


gpu::bilateralFilter
--------------------
Performs bilateral filtering of passed image

.. ocv:function:: void gpu::bilateralFilter( const GpuMat& src, GpuMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode=BORDER_DEFAULT, Stream& stream=Stream::Null() )

    :param src: Source image. Supports only (channles != 2 && depth() != CV_8S && depth() != CV_32S && depth() != CV_64F).

    :param dst: Destination imagwe.

    :param kernel_size: Kernel window size.

    :param sigma_color: Filter sigma in the color space.

    :param sigma_spatial:  Filter sigma in the coordinate space.

    :param borderMode:  Border type. See :ocv:func:`borderInterpolate` for details. ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , ``BORDER_CONSTANT`` , ``BORDER_REFLECT`` and ``BORDER_WRAP`` are supported for now.

    :param stream: Stream for the asynchronous version.

.. seealso::

    :ocv:func:`bilateralFilter`



gpu::alphaComp
-------------------
Composites two images using alpha opacity values contained in each image.

.. ocv:function:: void gpu::alphaComp(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, int alpha_op, Stream& stream = Stream::Null())

    :param img1: First image. Supports ``CV_8UC4`` , ``CV_16UC4`` , ``CV_32SC4`` and ``CV_32FC4`` types.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param dst: Destination image.

    :param alpha_op: Flag specifying the alpha-blending operation:

            * **ALPHA_OVER**
            * **ALPHA_IN**
            * **ALPHA_OUT**
            * **ALPHA_ATOP**
            * **ALPHA_XOR**
            * **ALPHA_PLUS**
            * **ALPHA_OVER_PREMUL**
            * **ALPHA_IN_PREMUL**
            * **ALPHA_OUT_PREMUL**
            * **ALPHA_ATOP_PREMUL**
            * **ALPHA_XOR_PREMUL**
            * **ALPHA_PLUS_PREMUL**
            * **ALPHA_PREMUL**

    :param stream: Stream for the asynchronous version.



gpu::Canny
-------------------
Finds edges in an image using the [Canny86]_ algorithm.

.. ocv:function:: void gpu::Canny(const GpuMat& image, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)

.. ocv:function:: void gpu::Canny(const GpuMat& image, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)

.. ocv:function:: void gpu::Canny(const GpuMat& dx, const GpuMat& dy, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false)

.. ocv:function:: void gpu::Canny(const GpuMat& dx, const GpuMat& dy, CannyBuf& buf, GpuMat& edges, double low_thresh, double high_thresh, bool L2gradient = false)

    :param image: Single-channel 8-bit input image.

    :param dx: First derivative of image in the vertical direction. Support only ``CV_32S`` type.

    :param dy: First derivative of image in the horizontal direction. Support only ``CV_32S`` type.

    :param edges: Output edge map. It has the same size and type as  ``image`` .

    :param low_thresh: First threshold for the hysteresis procedure.

    :param high_thresh: Second threshold for the hysteresis procedure.

    :param apperture_size: Aperture size for the  :ocv:func:`Sobel`  operator.

    :param L2gradient: Flag indicating whether a more accurate  :math:`L_2`  norm  :math:`=\sqrt{(dI/dx)^2 + (dI/dy)^2}`  should be used to compute the image gradient magnitude ( ``L2gradient=true`` ), or a faster default  :math:`L_1`  norm  :math:`=|dI/dx|+|dI/dy|`  is enough ( ``L2gradient=false`` ).

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`Canny`



gpu::HoughLines
---------------
Finds lines in a binary image using the classical Hough transform.

.. ocv:function:: void gpu::HoughLines(const GpuMat& src, GpuMat& lines, float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096)

.. ocv:function:: void gpu::HoughLines(const GpuMat& src, GpuMat& lines, HoughLinesBuf& buf, float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096)

    :param src: 8-bit, single-channel binary source image.

    :param lines: Output vector of lines. Each line is represented by a two-element vector  :math:`(\rho, \theta)` .  :math:`\rho`  is the distance from the coordinate origin  :math:`(0,0)`  (top-left corner of the image).  :math:`\theta`  is the line rotation angle in radians ( :math:`0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}` ).

    :param rho: Distance resolution of the accumulator in pixels.

    :param theta: Angle resolution of the accumulator in radians.

    :param threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes ( :math:`>\texttt{threshold}` ).

    :param doSort: Performs lines sort by votes.

    :param maxLines: Maximum number of output lines.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`HoughLines`



gpu::HoughLinesDownload
-----------------------
Downloads results from :ocv:func:`gpu::HoughLines` to host memory.

.. ocv:function:: void gpu::HoughLinesDownload(const GpuMat& d_lines, OutputArray h_lines, OutputArray h_votes = noArray())

    :param d_lines: Result of :ocv:func:`gpu::HoughLines` .

    :param h_lines: Output host array.

    :param h_votes: Optional output array for line's votes.

.. seealso:: :ocv:func:`gpu::HoughLines`



gpu::HoughCircles
-----------------
Finds circles in a grayscale image using the Hough transform.

.. ocv:function:: void gpu::HoughCircles(const GpuMat& src, GpuMat& circles, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

.. ocv:function:: void gpu::HoughCircles(const GpuMat& src, GpuMat& circles, HoughCirclesBuf& buf, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

    :param src: 8-bit, single-channel grayscale input image.

    :param circles: Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  :math:`(x, y, radius)` .

    :param method: Detection method to use. Currently, the only implemented method is  ``CV_HOUGH_GRADIENT`` , which is basically  *21HT* , described in  [Yuen90]_.

    :param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if  ``dp=1`` , the accumulator has the same resolution as the input image. If  ``dp=2`` , the accumulator has half as big width and height.

    :param minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.

    :param cannyThreshold: The higher threshold of the two passed to  the :ocv:func:`gpu::Canny`  edge detector (the lower one is twice smaller).

    :param votesThreshold: The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.

    :param minRadius: Minimum circle radius.

    :param maxRadius: Maximum circle radius.

    :param maxCircles: Maximum number of output circles.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`HoughCircles`



gpu::HoughCirclesDownload
-------------------------
Downloads results from :ocv:func:`gpu::HoughCircles` to host memory.

.. ocv:function:: void gpu::HoughCirclesDownload(const GpuMat& d_circles, OutputArray h_circles)

    :param d_circles: Result of :ocv:func:`gpu::HoughCircles` .

    :param h_circles: Output host array.

.. seealso:: :ocv:func:`gpu::HoughCircles`



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
