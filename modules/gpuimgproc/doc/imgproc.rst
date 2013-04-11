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
