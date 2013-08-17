Image Processing
=============================

.. highlight:: cpp

ocl::cornerHarris
------------------
Returns void

.. ocv:function:: void ocl::cornerHarris(const oclMat &src, oclMat &dst, int blockSize, int ksize, double k, int bordertype = cv::BORDER_DEFAULT)

    :param src: Source image. Only CV_8UC1 and CV_32FC1 images are supported now.

    :param dst: Destination image containing cornerness values. It has the same size as src and CV_32FC1 type.

    :param blockSize: Neighborhood size

    :param ksize: Aperture parameter for the Sobel operator

    :param k: Harris detector free parameter

    :param bordertype: Pixel extrapolation method. Only BORDER_REFLECT101, BORDER_REFLECT, BORDER_CONSTANT and BORDER_REPLICATE are supported now.

Calculate Harris corner.

ocl::cornerMinEigenVal
------------------------
Returns void

.. ocv:function:: void ocl::cornerMinEigenVal(const oclMat &src, oclMat &dst, int blockSize, int ksize, int bordertype = cv::BORDER_DEFAULT)

    :param src: Source image. Only CV_8UC1 and CV_32FC1 images are supported now.

    :param dst: Destination image containing cornerness values. It has the same size as src and CV_32FC1 type.

    :param blockSize: Neighborhood size

    :param ksize: Aperture parameter for the Sobel operator

    :param bordertype: Pixel extrapolation method. Only BORDER_REFLECT101, BORDER_REFLECT, BORDER_CONSTANT and BORDER_REPLICATE are supported now.

Calculate MinEigenVal.

ocl::calcHist
------------------
Returns void

.. ocv:function:: void ocl::calcHist(const oclMat &mat_src, oclMat &mat_hist)

    :param src: Source arrays. They all should have the same depth, CV 8U, and the same size. Each of them can have an arbitrary number of channels.

    :param dst: The output histogram, a dense or sparse dims-dimensional

Calculates histogram of one or more arrays. Supports only 8UC1 data type.

ocl::remap
------------------
Returns void

.. ocv:function:: void ocl::remap(const oclMat &src, oclMat &dst, oclMat &map1, oclMat &map2, int interpolation, int bordertype, const Scalar &value = Scalar())

    :param src: Source image. Only CV_8UC1 and CV_32FC1 images are supported now.

    :param dst: Destination image containing cornerness values. It has the same size as src and CV_32FC1 type.

    :param map1: The first map of either (x,y) points or just x values having the type CV_16SC2 , CV_32FC1 , or CV_32FC2 . See covertMaps() for details on converting a floating point representation to fixed-point for speed.

    :param map2: The second map of y values having the type CV_32FC1 , or none (empty map if map1 is (x,y) points), respectively.

    :param interpolation: The interpolation method

    :param bordertype: Pixel extrapolation method. Only BORDER_CONSTANT are supported now.

    :param value: The border value if borderType==BORDER CONSTANT

The function remap transforms the source image using the specified map: dst (x ,y) = src (map1(x , y) , map2(x , y)) where values of pixels with non-integer coordinates are computed using one of available interpolation methods. map1 and map2 can be encoded as separate floating-point maps in map1 and map2 respectively, or interleaved floating-point maps of (x,y) in map1. Supports CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1 , CV_32FC3 and CV_32FC4 data types.

ocl::resize
------------------
Returns void

.. ocv:function:: void ocl::resize(const oclMat &src, oclMat &dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR)

    :param src: Source image.

    :param dst: Destination image.

    :param dsize: he destination image size. If it is zero, then it is computed as: dsize = Size(round(fx*src.cols), round(fy*src.rows)). Either dsize or both fx or fy must be non-zero.

    :param fx: The scale factor along the horizontal axis. When 0, it is computed as (double)dsize.width/src.cols

    :param fy: The scale factor along the vertical axis. When 0, it is computed as (double)dsize.height/src.rows

    :param interpolation: The interpolation method: INTER NEAREST or INTER LINEAR

Resizes an image. Supports CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1 , CV_32FC3 and CV_32FC4 data types.

ocl::warpAffine
------------------
Returns void

.. ocv:function:: void ocl::warpAffine(const oclMat &src, oclMat &dst, const Mat &M, Size dsize, int flags = INTER_LINEAR)

    :param src: Source image.

    :param dst: Destination image.

    :param M: 2times 3 transformation matrix

    :param dsize: Size of the destination image

    :param flags: A combination of interpolation methods, see cv::resize, and the optional flag WARP INVERSE MAP that means that M is the inverse transformation (dst to $src)

The function warpAffine transforms the source image using the specified matrix. Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC types.

ocl::warpPerspective
---------------------
Returns void

.. ocv:function:: void ocl::warpPerspective(const oclMat &src, oclMat &dst, const Mat &M, Size dsize, int flags = INTER_LINEAR)

    :param src: Source image.

    :param dst: Destination image.

    :param M: 2times 3 transformation matrix

    :param dsize: Size of the destination image

    :param flags: A combination of interpolation methods, see cv::resize, and the optional flag WARP INVERSE MAP that means that M is the inverse transformation (dst to $src)

Applies a perspective transformation to an image. Supports INTER_NEAREST, INTER_LINEAR, INTER_CUBIC types.

ocl::cvtColor
------------------
Returns void

.. ocv:function:: void ocl::cvtColor(const oclMat &src, oclMat &dst, int code , int dcn = 0)

    :param src: Source image.

    :param dst: Destination image.

    :param code:The color space conversion code

    :param dcn: The number of channels in the destination image; if the parameter is 0, the number of the channels will be derived automatically from src and the code

Converts image from one color space to another.For now, only RGB2GRAY is supportted. Supports.CV_8UC1,CV_8UC4,CV_32SC1,CV_32SC4,CV_32FC1,CV_32FC4

ocl::threshold
------------------
Returns Threshold value

.. ocv:function:: double ocl::threshold(const oclMat &src, oclMat &dst, double thresh, double maxVal, int type = THRESH_TRUNC)

    :param src: The source array

    :param dst: Destination array; will have the same size and the same type as src

    :param thresh: Threshold value

    :param maxVal: Maximum value to use with THRESH BINARY and THRESH BINARY INV thresholding types

    :param type: Thresholding type

The function applies fixed-level thresholding to a single-channel array. The function is typically used to get a bi-level (binary) image out of a grayscale image or for removing a noise, i.e. filtering out pixels with too small or too large values. There are several types of thresholding that the function supports that are determined by thresholdType. Supports only CV_32FC1 and CV_8UC1 data type.

ocl::buildWarpPlaneMaps
-----------------------
Builds plane warping maps.

.. ocv:function:: void ocl::buildWarpPlaneMaps( Size src_size, Rect dst_roi, const Mat& K, const Mat& R, const Mat& T, float scale, oclMat& map_x, oclMat& map_y )



ocl::buildWarpCylindricalMaps
-----------------------------
Builds cylindrical warping maps.

.. ocv:function:: void ocl::buildWarpCylindricalMaps( Size src_size, Rect dst_roi, const Mat& K, const Mat& R, float scale, oclMat& map_x, oclMat& map_y )




ocl::buildWarpSphericalMaps
---------------------------
Builds spherical warping maps.

.. ocv:function:: void ocl::buildWarpSphericalMaps( Size src_size, Rect dst_roi, const Mat& K, const Mat& R, float scale, oclMat& map_x, oclMat& map_y )


ocl::buildWarpPerspectiveMaps
-----------------------------
Builds transformation maps for perspective transformation.

.. ocv:function:: void ocl::buildWarpAffineMaps(const Mat& M, bool inverse, Size dsize, oclMat& xmap, oclMat& ymap)

    :param M: *3x3*  transformation matrix.

    :param inverse: Flag  specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ).

    :param dsize: Size of the destination image.

    :param xmap: X values with  ``CV_32FC1`` type.

    :param ymap: Y values with  ``CV_32FC1`` type.

.. seealso:: :ocv:func:`ocl::warpPerspective` , :ocv:func:`ocl::remap`


ocl::buildWarpAffineMaps
------------------------
Builds transformation maps for affine transformation.

.. ocv:function:: void ocl::buildWarpAffineMaps(const Mat& M, bool inverse, Size dsize, oclMat& xmap, oclMat& ymap)

    :param M: *2x3*  transformation matrix.

    :param inverse: Flag  specifying that  ``M`` is an inverse transformation ( ``dst=>src`` ).

    :param dsize: Size of the destination image.

    :param xmap: X values with  ``CV_32FC1`` type.

    :param ymap: Y values with  ``CV_32FC1`` type.

.. seealso:: :ocv:func:`ocl::warpAffine` , :ocv:func:`ocl::remap`

ocl::PyrLKOpticalFlow
---------------------
.. ocv:class:: ocl::PyrLKOpticalFlow

Class used for calculating an optical flow. ::

    class PyrLKOpticalFlow
    {
    public:
        PyrLKOpticalFlow();

        void sparse(const oclMat& prevImg, const oclMat& nextImg, const oclMat& prevPts, oclMat& nextPts,
            oclMat& status, oclMat* err = 0);

        void dense(const oclMat& prevImg, const oclMat& nextImg, oclMat& u, oclMat& v, oclMat* err = 0);

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

.. note::

   (Ocl) An example the Lucas Kanade optical flow pyramid method can be found at opencv_source_code/samples/ocl/pyrlk_optical_flow.cpp
   (Ocl) An example for square detection can be found at opencv_source_code/samples/ocl/squares.cpp

ocl::PyrLKOpticalFlow::sparse
-----------------------------
Calculate an optical flow for a sparse feature set.

.. ocv:function:: void ocl::PyrLKOpticalFlow::sparse(const oclMat& prevImg, const oclMat& nextImg, const oclMat& prevPts, oclMat& nextPts, oclMat& status, oclMat* err = 0)

    :param prevImg: First 8-bit input image (supports both grayscale and color images).

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param prevPts: Vector of 2D points for which the flow needs to be found. It must be one row matrix with CV_32FC2 type.

    :param nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When ``useInitialFlow`` is true, the vector must have the same size as in the input.

    :param status: Output status vector (CV_8UC1 type). Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`



ocl::PyrLKOpticalFlow::dense
-----------------------------
Calculate dense optical flow.

.. ocv:function:: void ocl::PyrLKOpticalFlow::dense(const oclMat& prevImg, const oclMat& nextImg, oclMat& u, oclMat& v, oclMat* err = 0)

    :param prevImg: First 8-bit grayscale input image.

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param u: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param v: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.



ocl::PyrLKOpticalFlow::releaseMemory
------------------------------------
Releases inner buffers memory.

.. ocv:function:: void ocl::PyrLKOpticalFlow::releaseMemory()


ocl::interpolateFrames
----------------------
Interpolate frames (images) using provided optical flow (displacement field).

.. ocv:function:: void ocl::interpolateFrames(const oclMat& frame0, const oclMat& frame1, const oclMat& fu, const oclMat& fv, const oclMat& bu, const oclMat& bv, float pos, oclMat& newFrame, oclMat& buf)

    :param frame0: First frame (32-bit floating point images, single channel).

    :param frame1: Second frame. Must have the same type and size as ``frame0`` .

    :param fu: Forward horizontal displacement.

    :param fv: Forward vertical displacement.

    :param bu: Backward horizontal displacement.

    :param bv: Backward vertical displacement.

    :param pos: New frame position.

    :param newFrame: Output image.

    :param buf: Temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 oclMat: occlusion masks for first frame, occlusion masks for second, interpolated forward horizontal flow, interpolated forward vertical flow, interpolated backward horizontal flow, interpolated backward vertical flow.


ocl::HoughCircles
-----------------
Finds circles in a grayscale image using the Hough transform.

.. ocv:function:: void ocl::HoughCircles(const oclMat& src, oclMat& circles, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

.. ocv:function:: void ocl::HoughCircles(const oclMat& src, oclMat& circles, HoughCirclesBuf& buf, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

    :param src: 8-bit, single-channel grayscale input image.

    :param circles: Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  :math:`(x, y, radius)` .

    :param method: Detection method to use. Currently, the only implemented method is  ``CV_HOUGH_GRADIENT`` , which is basically  *21HT* , described in  [Yuen90]_.

    :param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if  ``dp=1`` , the accumulator has the same resolution as the input image. If  ``dp=2`` , the accumulator has half as big width and height.

    :param minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.

    :param cannyThreshold: The higher threshold of the two passed to  the :ocv:func:`ocl::Canny`  edge detector (the lower one is twice smaller).

    :param votesThreshold: The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.

    :param minRadius: Minimum circle radius.

    :param maxRadius: Maximum circle radius.

    :param maxCircles: Maximum number of output circles.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. note:: Currently only non-ROI oclMat is supported for src.
.. seealso:: :ocv:func:`HoughCircles`

