Image Filtering
=============================

.. highlight:: cpp

ocl::BaseRowFilter_GPU
--------------------------
.. ocv:class:: ocl::BaseRowFilter_GPU

Base class for linear or non-linear filters that processes rows of 2D arrays. Such filters are used for the "horizontal" filtering passes in separable filters. ::

    class CV_EXPORTS BaseRowFilter_GPU
    {
    public:
        BaseRowFilter_GPU(int ksize_, int anchor_, int bordertype_) : ksize(ksize_), anchor(anchor_), bordertype(bordertype_) {}
        virtual ~BaseRowFilter_GPU() {}
        virtual void operator()(const oclMat &src, oclMat &dst) = 0;
        int ksize, anchor, bordertype;
    };

.. note:: This class does not allocate memory for a destination image. Usually this class is used inside :ocv:class:`ocl::FilterEngine_GPU`.

ocl::BaseColumnFilter_GPU
-----------------------------
.. ocv:class:: ocl::BaseColumnFilter_GPU

Base class for linear or non-linear filters that processes columns of 2D arrays. Such filters are used for the "vertical" filtering passes in separable filters. ::

    class CV_EXPORTS BaseColumnFilter_GPU
    {
    public:
        BaseColumnFilter_GPU(int ksize_, int anchor_, int bordertype_) : ksize(ksize_), anchor(anchor_), bordertype(bordertype_) {}
        virtual ~BaseColumnFilter_GPU() {}
        virtual void operator()(const oclMat &src, oclMat &dst) = 0;
        int ksize, anchor, bordertype;
    };

.. note:: This class does not allocate memory for a destination image. Usually this class is used inside :ocv:class:`ocl::FilterEngine_GPU`.

ocl::BaseFilter_GPU
-----------------------
.. ocv:class:: ocl::BaseFilter_GPU

Base class for non-separable 2D filters. ::

    class CV_EXPORTS BaseFilter_GPU
    {
    public:
        BaseFilter_GPU(const Size &ksize_, const Point &anchor_, const int &borderType_)
            : ksize(ksize_), anchor(anchor_), borderType(borderType_) {}
        virtual ~BaseFilter_GPU() {}
        virtual void operator()(const oclMat &src, oclMat &dst) = 0;
        Size ksize;
        Point anchor;
        int borderType;
    };

.. note:: This class does not allocate memory for a destination image. Usually this class is used inside :ocv:class:`ocl::FilterEngine_GPU`

ocl::FilterEngine_GPU
------------------------
.. ocv:class:: ocl::FilterEngine_GPU

Base class for the Filter Engine. ::

    class CV_EXPORTS FilterEngine_GPU
    {
    public:
        virtual ~FilterEngine_GPU() {}

        virtual void apply(const oclMat &src, oclMat &dst, Rect roi = Rect(0, 0, -1, -1)) = 0;
    };

The class can be used to apply an arbitrary filtering operation to an image. It contains all the necessary intermediate buffers. Pointers to the initialized ``FilterEngine_GPU`` instances are returned by various ``create*Filter_GPU`` functions (see below), and they are used inside high-level functions such as :ocv:func:`ocl::filter2D`, :ocv:func:`ocl::erode`, :ocv:func:`ocl::Sobel` , and others.

By using ``FilterEngine_GPU`` instead of functions you can avoid unnecessary memory allocation for intermediate buffers and get better performance: ::

    while (...)
    {
        ocl::oclMat src = getImg();
        ocl::oclMat dst;
        // Allocate and release buffers at each iterations
        ocl::GaussianBlur(src, dst, ksize, sigma1);
    }

    // Allocate buffers only once
    cv::Ptr<ocl::FilterEngine_GPU> filter =
        ocl::createGaussianFilter_GPU(CV_8UC4, ksize, sigma1);
    while (...)
    {
        ocl::oclMat src = getImg();
        ocl::oclMat dst;
        filter->apply(src, dst, cv::Rect(0, 0, src.cols, src.rows));
    }
    // Release buffers only once
    filter.release();


``FilterEngine_GPU`` can process a rectangular sub-region of an image. By default, if ``roi == Rect(0,0,-1,-1)`` , ``FilterEngine_GPU`` processes the inner region of an image ( ``Rect(anchor.x, anchor.y, src_size.width - ksize.width, src_size.height - ksize.height)`` ) because some filters do not check whether indices are outside the image for better performance. See below to understand which filters support processing the whole image and which do not and identify image type limitations.

.. note:: The GPU filters do not support the in-place mode.

.. seealso:: :ocv:class:`ocl::BaseRowFilter_GPU`, :ocv:class:`ocl::BaseColumnFilter_GPU`, :ocv:class:`ocl::BaseFilter_GPU`, :ocv:func:`ocl::createFilter2D_GPU`, :ocv:func:`ocl::createSeparableFilter_GPU`, :ocv:func:`ocl::createBoxFilter_GPU`, :ocv:func:`ocl::createMorphologyFilter_GPU`, :ocv:func:`ocl::createLinearFilter_GPU`, :ocv:func:`ocl::createSeparableLinearFilter_GPU`, :ocv:func:`ocl::createDerivFilter_GPU`, :ocv:func:`ocl::createGaussianFilter_GPU`

ocl::createFilter2D_GPU
---------------------------
Creates a non-separable filter engine with the specified filter.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createFilter2D_GPU( const Ptr<BaseFilter_GPU> filter2D)

    :param filter2D: Non-separable 2D filter.

Usually this function is used inside such high-level functions as :ocv:func:`ocl::createLinearFilter_GPU`, :ocv:func:`ocl::createBoxFilter_GPU`.


ocl::createSeparableFilter_GPU
----------------------------------
Creates a separable filter engine with the specified filters.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createSeparableFilter_GPU(const Ptr<BaseRowFilter_GPU> &rowFilter, const Ptr<BaseColumnFilter_GPU> &columnFilter)

    :param rowFilter: "Horizontal" 1D filter.

    :param columnFilter: "Vertical" 1D filter.

Usually this function is used inside such high-level functions as :ocv:func:`ocl::createSeparableLinearFilter_GPU`.

ocl::createBoxFilter_GPU
----------------------------
Creates a normalized 2D box filter.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createBoxFilter_GPU(int srcType, int dstType, const Size &ksize, const Point &anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)

.. ocv:function:: Ptr<BaseFilter_GPU> ocl::getBoxFilter_GPU(int srcType, int dstType, const Size &ksize, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)

    :param srcType: Input image type.

    :param dstType: Output image type.  It supports only the same values as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

    :param borderType: Border type.

.. seealso:: :ocv:func:`boxFilter`

ocl::boxFilter
------------------
Smooths the image using the normalized box filter.

.. ocv:function:: void ocl::boxFilter(const oclMat &src, oclMat &dst, int ddepth, Size ksize, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)

    :param src: Input image.

    :param dst: Output image type. The size and type is the same as ``src`` .

    :param ddepth: Desired depth of the destination image. If it is negative, it is the same as  ``src.depth()`` . It supports only the same depth as the source image depth.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

    :param borderType: Border type.

Smoothes image using box filter.

ocl::blur
-------------
Acts as a synonym for the normalized box filter.

.. ocv:function:: void ocl::blur(const oclMat &src, oclMat &dst, Size ksize, Point anchor = Point(-1, -1), int borderType = BORDER_CONSTANT)

    :param src: Input image.

    :param dst: Output image type with the same size and type as  ``src`` .

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

    :param borderType: Border type.

.. seealso:: :ocv:func:`blur`, :ocv:func:`ocl::boxFilter`

ocl::createMorphologyFilter_GPU
-----------------------------------
Creates a 2D morphological filter.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createMorphologyFilter_GPU(int op, int type, const Mat &kernel, const Point &anchor = Point(-1, -1), int iterations = 1)

.. ocv:function:: Ptr<BaseFilter_GPU> ocl::getMorphologyFilter_GPU(int op, int type, const Mat &kernel, const Size &ksize, Point anchor = Point(-1, -1))

    :param op: Morphology operation id. Only ``MORPH_ERODE`` and ``MORPH_DILATE`` are supported.

    :param type: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4``  are supported.

    :param kernel: 2D 8-bit structuring element for the morphological operation.

    :param ksize: Size of a horizontal or vertical structuring element used for separable morphological operations.

    :param anchor: Anchor position within the structuring element. Negative values mean that the anchor is at the center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`createMorphologyFilter`

ocl::createLinearFilter_GPU
-------------------------------
Creates a non-separable linear filter.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createLinearFilter_GPU(int srcType, int dstType, const Mat &kernel, const Point &anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)

    :param srcType: Input image type..

    :param dstType: Output image type. The same type as ``src`` is supported.

    :param kernel: 2D array of filter coefficients.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

    :param borderType: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

.. seealso:: :ocv:func:`createLinearFilter`


ocl::filter2D
-----------------
Applies the non-separable 2D linear filter to an image.

.. ocv:function:: void ocl::filter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernel, Point anchor = Point(-1, -1), double delta = 0.0, int borderType = BORDER_DEFAULT)

    :param src: Source image.

    :param dst: Destination image. The size and the number of channels is the same as  ``src`` .

    :param ddepth: Desired depth of the destination image. If it is negative, it is the same as  ``src.depth()`` . It supports only the same depth as the source image depth.

    :param kernel: 2D array of filter coefficients.

    :param anchor: Anchor of the kernel that indicates the relative position of a filtered point within the kernel. The anchor resides within the kernel. The special default value (-1,-1) means that the anchor is at the kernel center.

    :param delta: optional value added to the filtered pixels before storing them in ``dst``. Value '0' is supported only.

    :param borderType: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

ocl::getLinearRowFilter_GPU
-------------------------------
Creates a primitive row filter with the specified kernel.

.. ocv:function:: Ptr<BaseRowFilter_GPU> ocl::getLinearRowFilter_GPU(int srcType, int bufType, const Mat &rowKernel, int anchor = -1, int bordertype = BORDER_DEFAULT)

    :param srcType: Source array type. Only  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param bufType: Intermediate buffer type with as many channels as  ``srcType`` .

    :param rowKernel: Filter coefficients. Support kernels with ``size <= 16`` .

    :param anchor: Anchor position within the kernel. Negative values mean that the anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate`.

.. seealso:: :ocv:func:`createSeparableLinearFilter` .


ocl::getLinearColumnFilter_GPU
----------------------------------
Creates a primitive column filter with the specified kernel.

.. ocv:function:: Ptr<BaseColumnFilter_GPU> ocl::getLinearColumnFilter_GPU(int bufType, int dstType, const Mat &columnKernel, int anchor = -1, int bordertype = BORDER_DEFAULT, double delta = 0.0)

    :param bufType: Intermediate buffer type with as many channels as  ``dstType`` .

    :param dstType: Destination array type. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1`` destination types are supported.

    :param columnKernel: Filter coefficients. Support kernels with ``size <= 16`` .

    :param anchor: Anchor position within the kernel. Negative values mean that the anchor is positioned at the aperture center.

    :param bordertype: Pixel extrapolation method. For details, see  :ocv:func:`borderInterpolate` .

    :param delta: default value is 0.0.

.. seealso:: :ocv:func:`ocl::getLinearRowFilter_GPU`, :ocv:func:`createSeparableLinearFilter`

ocl::createSeparableLinearFilter_GPU
----------------------------------------
Creates a separable linear filter engine.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat &rowKernel, const Mat &columnKernel, const Point &anchor = Point(-1, -1), double delta = 0.0, int bordertype = BORDER_DEFAULT)

    :param srcType: Source array type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dstType: Destination array type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  destination types are supported.

    :param rowKernel: Horizontal filter coefficients. Support kernels with ``size <= 16`` .

    :param columnKernel: Vertical filter coefficients. Support kernels with ``size <= 16`` .

    :param anchor: Anchor position within the kernel. Negative values mean that anchor is positioned at the aperture center.

    :param delta: default value is 0.0.

    :param bordertype: Pixel extrapolation method.

.. seealso:: :ocv:func:`ocl::getLinearRowFilter_GPU`, :ocv:func:`ocl::getLinearColumnFilter_GPU`, :ocv:func:`createSeparableLinearFilter`


ocl::sepFilter2D
--------------------
Applies a separable 2D linear filter to an image.

.. ocv:function:: void ocl::sepFilter2D(const oclMat &src, oclMat &dst, int ddepth, const Mat &kernelX, const Mat &kernelY, Point anchor = Point(-1, -1), double delta = 0.0, int bordertype = BORDER_DEFAULT)

    :param src: Source image.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as  ``src`` .

    :param ddepth: Destination image depth.  ``CV_8U`` , ``CV_16S`` , ``CV_32S`` , and  ``CV_32F`` are supported.

    :param kernelX: Horizontal filter coefficients.

    :param kernelY: Vertical filter coefficients.

    :param anchor: Anchor position within the kernel. The default value ``(-1, 1)`` means that the anchor is at the kernel center.

    :param delta: default value is 0.0.

    :param bordertype: Pixel extrapolation method. For details, see  :ocv:func:`borderInterpolate`.

.. seealso:: :ocv:func:`ocl::createSeparableLinearFilter_GPU`, :ocv:func:`sepFilter2D`

ocl::createDerivFilter_GPU
------------------------------
Creates a filter engine for the generalized Sobel operator.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createDerivFilter_GPU( int srcType, int dstType, int dx, int dy, int ksize, int borderType = BORDER_DEFAULT )

    :param srcType: Source image type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dstType: Destination image type with as many channels as  ``srcType`` ,  ``CV_8U`` , ``CV_16S`` , ``CV_32S`` , and  ``CV_32F``  depths are supported.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Aperture size. See  :ocv:func:`getDerivKernels` for details.

    :param borderType: Pixel extrapolation method. For details, see  :ocv:func:`borderInterpolate`.

.. seealso:: :ocv:func:`ocl::createSeparableLinearFilter_GPU`, :ocv:func:`createDerivFilter`


ocl::Sobel
------------------
Returns void

.. ocv:function:: void ocl::Sobel(const oclMat &src, oclMat &dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0.0, int bordertype = BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; It will have the same size as src

    :param ddepth: The destination image depth

    :param dx: Order of the derivative x

    :param dy: Order of the derivative y

    :param ksize: Size of the extended Sobel kernel

    :param scale: The optional scale factor for the computed derivative values(by default, no scaling is applied)

    :param delta: The optional delta value, added to the results prior to storing them in dst

    :param bordertype: Pixel extrapolation method.

The function computes the first x- or y- spatial image derivative using Sobel operator. Surpport 8UC1 8UC4 32SC1 32SC4 32FC1 32FC4 data type.

ocl::Scharr
------------------
Returns void

.. ocv:function:: void ocl::Scharr(const oclMat &src, oclMat &dst, int ddepth, int dx, int dy, double scale = 1, double delta = 0.0, int bordertype = BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; It will have the same size as src

    :param ddepth: The destination image depth

    :param dx: Order of the derivative x

    :param dy: Order of the derivative y

    :param scale: The optional scale factor for the computed derivative values(by default, no scaling is applied)

    :param delta: The optional delta value, added to the results prior to storing them in dst

    :param bordertype: Pixel extrapolation method.

The function computes the first x- or y- spatial image derivative using Scharr operator. Surpport 8UC1 8UC4 32SC1 32SC4 32FC1 32FC4 data type.

ocl::createGaussianFilter_GPU
---------------------------------
Creates a Gaussian filter engine.

.. ocv:function:: Ptr<FilterEngine_GPU> ocl::createGaussianFilter_GPU(int type, Size ksize, double sigma1, double sigma2 = 0, int bordertype = BORDER_DEFAULT)

    :param type: Source and destination image type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1`` are supported.

    :param ksize: Aperture size. See  :ocv:func:`getGaussianKernel` for details.

    :param sigma1: Gaussian sigma in the horizontal direction. See  :ocv:func:`getGaussianKernel` for details.

    :param sigma2: Gaussian sigma in the vertical direction. If 0, then  :math:`\texttt{sigma2}\leftarrow\texttt{sigma1}` .

    :param bordertype: Pixel extrapolation method. For details, see  :ocv:func:`borderInterpolate`.

.. seealso:: :ocv:func:`ocl::createSeparableLinearFilter_GPU`, :ocv:func:`createGaussianFilter`

ocl::GaussianBlur
---------------------
Returns void

.. ocv:function:: void ocl::GaussianBlur(const oclMat &src, oclMat &dst, Size ksize, double sigma1, double sigma2 = 0, int bordertype = BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param ksize: The Gaussian kernel size; ksize.width and ksize.height can differ, but they both must be positive and odd. Or, they can be zero's, then they are computed from sigma

    :param sigma1sigma2: The Gaussian kernel standard deviations in X and Y direction. If sigmaY is zero, it is set to be equal to sigmaX. If they are both zeros, they are computed from ksize.width and ksize.height. To fully control the result regardless of possible future modification of all this semantics, it is recommended to specify all of ksize, sigmaX and sigmaY

    :param bordertype: Pixel extrapolation method.

The function convolves the source image with the specified Gaussian kernel. In-place filtering is supported.  Surpport 8UC1 8UC4 32SC1 32SC4 32FC1 32FC4 data type.

ocl::Laplacian
------------------
Returns void

.. ocv:function:: void ocl::Laplacian(const oclMat &src, oclMat &dst, int ddepth, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param ddepth: The desired depth of the destination image

    :param ksize: The aperture size used to compute the second-derivative filters. It must be positive and odd

    :param scale: The optional scale factor for the computed Laplacian values (by default, no scaling is applied

    :param delta: Optional delta value that is added to the results prior to storing them in  ``dst`` . Supported value is 0 only.

    :param bordertype: Pixel extrapolation method.

The function calculates the Laplacian of the source image by adding up the second x and y derivatives calculated using the Sobel operator.

ocl::ConvolveBuf
----------------
.. ocv:struct:: ocl::ConvolveBuf

Class providing a memory buffer for :ocv:func:`ocl::convolve` function, plus it allows to adjust some specific parameters. ::

    struct CV_EXPORTS ConvolveBuf
    {
        Size result_size;
        Size block_size;
        Size user_block_size;
        Size dft_size;
        int spect_len;

        oclMat image_spect, templ_spect, result_spect;
        oclMat image_block, templ_block, result_data;

        void create(Size image_size, Size templ_size);
        static Size estimateBlockSize(Size result_size, Size templ_size);
    };

You can use field `user_block_size` to set specific block size for :ocv:func:`ocl::convolve` function. If you leave its default value `Size(0,0)` then automatic estimation of block size will be used (which is optimized for speed). By varying `user_block_size` you can reduce memory requirements at the cost of speed.

ocl::ConvolveBuf::create
------------------------
.. ocv:function:: ocl::ConvolveBuf::create(Size image_size, Size templ_size)

Constructs a buffer for :ocv:func:`ocl::convolve` function with respective arguments.

ocl::convolve
------------------
Returns void

.. ocv:function:: void ocl::convolve(const oclMat &image, const oclMat &temp1, oclMat &result, bool ccorr=false)

.. ocv:function:: void ocl::convolve(const oclMat &image, const oclMat &temp1, oclMat &result, bool ccorr, ConvolveBuf& buf)

    :param image: The source image. Only  ``CV_32FC1`` images are supported for now.

    :param temp1: Convolution kernel, a single-channel floating point matrix. The size is not greater than the  ``image`` size. The type is the same as  ``image``.

    :param result: The destination image

    :param ccorr: Flags to evaluate cross-correlation instead of convolution.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:struct:`ocl::ConvolveBuf`.

Convolves an image with the kernel. Supports only CV_32FC1 data types and do not support ROI.

ocl::bilateralFilter
------------------------
Returns void

.. ocv:function:: void ocl::bilateralFilter(const oclMat &src, oclMat &dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; will have the same size and the same type as src

    :param d: The diameter of each pixel neighborhood, that is used during filtering. If it is non-positive, it's computed from sigmaSpace

    :param sigmaColor: Filter sigma in the color space. Larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color

    :param sigmaSpave: Filter sigma in the coordinate space. Larger value of the parameter means that farther pixels will influence each other (as long as their colors are close enough; see sigmaColor). Then d>0, it specifies the neighborhood size regardless of sigmaSpace, otherwise d is proportional to sigmaSpace.

    :param borderType: Pixel extrapolation method.

Applies bilateral filter to the image. Supports 8UC1 8UC4 data types.

ocl::adaptiveBilateralFilter
--------------------------------
Returns void

.. ocv:function:: void ocl::adaptiveBilateralFilter(const oclMat& src, oclMat& dst, Size ksize, double sigmaSpace, double maxSigmaColor = 20.0, Point anchor = Point(-1, -1), int borderType=BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; will have the same size and the same type as src

    :param ksize: The kernel size. This is the neighborhood where the local variance will be calculated, and where pixels will contribute (in a weighted manner).

    :param sigmaSpace: Filter sigma in the coordinate space. Larger value of the parameter means that farther pixels will influence each other (as long as their colors are close enough; see sigmaColor). Then d>0, it specifies the neighborhood size regardless of sigmaSpace, otherwise d is proportional to sigmaSpace.

    :param maxSigmaColor: Maximum allowed sigma color (will clamp the value calculated in the ksize neighborhood. Larger value of the parameter means that more dissimilar pixels will influence each other (as long as their colors are close enough; see sigmaColor). Then d>0, it specifies the neighborhood size regardless of sigmaSpace, otherwise d is proportional to sigmaSpace.

    :param borderType: Pixel extrapolation method.

A main part of our strategy will be to load each raw pixel once, and reuse it to calculate all pixels in the output (filtered) image that need this pixel value. The math of the filter is that of the usual bilateral filter, except that the sigma color is calculated in the neighborhood, and clamped by the optional input value.

Local memory organization


.. image:: images/adaptiveBilateralFilter.jpg
                 :height: 250pt
                 :width:  350pt
                 :alt: Introduction Icon

.. note:: We partition the image to non-overlapping blocks of size (Ux, Uy). Each such block will correspond to the pixel locations where we will calculate the filter result in one workgroup. Considering neighbourhoods of sizes (kx, ky), where kx = 2 dx + 1, and ky = 2 dy + 1 (in image ML, dx = dy = 1, and kx = ky = 3), it is clear that we need to load data of size Wx = Ux + 2 dx, Wy = Uy + 2 dy. Furthermore, if (Sx, Sy) is the top left pixel coordinates for a particular block, and (Sx + Ux - 1, Sy + Uy -1) is to botom right coordinate of the block, we need to load data starting at top left coordinate (PSx, PSy) = (Sx - dx, Sy - dy), and ending at bottom right coordinate (Sx + Ux - 1 + dx, Sy + Uy - 1 + dy). The workgroup layout is (Wx,1). However, to take advantage of the natural hardware properties (preferred wavefront sizes), we restrict Wx to be a multiple of that preferred wavefront size (for current AMD hardware this is typically 64). Each thread in the workgroup will load Wy elements (under the constraint that Wx*Wy*pixel width <= max local memory).

Applies bilateral filter to the image. Supports 8UC1 8UC3 data types.

ocl::copyMakeBorder
-----------------------
Returns void

.. ocv:function:: void ocl::copyMakeBorder(const oclMat &src, oclMat &dst, int top, int bottom, int left, int right, int boardtype, const Scalar &value = Scalar())

    :param src: The source image

    :param dst: The destination image; will have the same type as src and the size size(src.cols+left+right, src.rows+top+bottom)

    :param topbottomleftright: Specify how much pixels in each direction from the source image rectangle one needs to extrapolate, e.g. top=1, bottom=1, left=1, right=1mean that 1 pixel-wide border needs to be built

    :param bordertype: Pixel extrapolation method.

    :param value: The border value if borderType==BORDER CONSTANT

Forms a border around the image. Supports 8UC1 8UC4 32SC1 32SC4 32FC1 32FC4 data types.

ocl::dilate
------------------
Returns void

.. ocv:function:: void ocl::dilate( const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor = Point(-1, -1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar &borderValue = morphologyDefaultBorderValue())

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param kernel: The structuring element used for dilation. If element=Mat(), a 3times 3 rectangular structuring element is used

    :param anchor: Position of the anchor within the element. The default value (-1, -1) means that the anchor is at the element center, only default value is supported

    :param iterations: The number of times dilation is applied

    :param bordertype: Pixel extrapolation method.

    :param value: The border value if borderType==BORDER CONSTANT

The function dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken. Supports 8UC1 8UC4 data types.

ocl::erode
------------------
Returns void

.. ocv:function:: void ocl::erode( const oclMat &src, oclMat &dst, const Mat &kernel, Point anchor = Point(-1, -1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar &borderValue = morphologyDefaultBorderValue())

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param kernel: The structuring element used for dilation. If element=Mat(), a 3times 3 rectangular structuring element is used

    :param anchor: Position of the anchor within the element. The default value (-1, -1) means that the anchor is at the element center, only default value is supported

    :param iterations: The number of times dilation is applied

    :param bordertype: Pixel extrapolation method.

    :param value: The border value if borderType==BORDER CONSTANT

The function erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken. Supports 8UC1 8UC4 data types.

ocl::morphologyEx
---------------------
Returns void

.. ocv:function:: void ocl::morphologyEx( const oclMat &src, oclMat &dst, int op, const Mat &kernel, Point anchor = Point(-1, -1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar &borderValue = morphologyDefaultBorderValue())

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param op: Type of morphological operation, one of the following: ERODE DILTATE OPEN CLOSE GRADIENT TOPHAT BLACKHAT

    :param kernel: The structuring element used for dilation. If element=Mat(), a 3times 3 rectangular structuring element is used

    :param anchor: Position of the anchor within the element. The default value (-1, -1) means that the anchor is at the element center, only default value is supported

    :param iterations: The number of times dilation is applied

    :param bordertype: Pixel extrapolation method.

    :param value: The border value if borderType==BORDER CONSTANT

A wrapper for erode and dilate. Supports 8UC1 8UC4 data types.

ocl::pyrDown
-------------------
Smoothes an image and downsamples it.

.. ocv:function:: void ocl::pyrDown(const oclMat& src, oclMat& dst)

    :param src: Source image.

    :param dst: Destination image. Will have ``Size((src.cols+1)/2, (src.rows+1)/2)`` size and the same type as ``src`` .

.. seealso:: :ocv:func:`pyrDown`


ocl::pyrUp
-------------------
Upsamples an image and then smoothes it.

.. ocv:function:: void ocl::pyrUp(const oclMat& src, oclMat& dst)

    :param src: Source image.

    :param dst: Destination image. Will have ``Size(src.cols*2, src.rows*2)`` size and the same type as ``src`` .

.. seealso:: :ocv:func:`pyrUp`

ocl::columnSum
------------------
Computes a vertical (column) sum.

.. ocv:function:: void ocl::columnSum(const oclMat& src, oclMat& sum)

    :param src: Source image. Only  ``CV_32FC1`` images are supported for now.

    :param sum: Destination image of the  ``CV_32FC1`` type.


ocl::blendLinear
--------------------
Performs linear blending of two images.

.. ocv:function:: void ocl::blendLinear(const oclMat& img1, const oclMat& img2, const oclMat& weights1, const oclMat& weights2, oclMat& result)

    :param img1: First image. Supports only ``CV_8U`` and ``CV_32F`` depth.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param weights1: Weights for first image. Must have tha same size as ``img1`` . Supports only ``CV_32F`` type.

    :param weights2: Weights for second image. Must have tha same size as ``img2`` . Supports only ``CV_32F`` type.

    :param result: Destination image.

ocl::medianFilter
--------------------
Blurs an image using the median filter.

.. ocv:function:: void ocl::medianFilter(const oclMat &src, oclMat &dst, int m)

    :param src: input ```1-``` or ```4```-channel image; the image depth should be ```CV_8U```, ```CV_32F```.

    :param dst: destination array of the same size and type as ```src```.

    :param m: aperture linear size; it must be odd and greater than ```1```. Currently only ```3```, ```5``` are supported.

The function smoothes an image using the median filter with the \texttt{m} \times \texttt{m} aperture. Each channel of a multi-channel image is processed independently. In-place operation is supported.
