Image Filtering
=============================

.. highlight:: cpp

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

ocl::GaussianBlur
------------------
Returns void

.. ocv:function:: void ocl::GaussianBlur(const oclMat &src, oclMat &dst, Size ksize, double sigma1, double sigma2 = 0, int bordertype = BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param ksize: The Gaussian kernel size; ksize.width and ksize.height can differ, but they both must be positive and odd. Or, they can be zero's, then they are computed from sigma

    :param sigma1sigma2: The Gaussian kernel standard deviations in X and Y direction. If sigmaY is zero, it is set to be equal to sigmaX. If they are both zeros, they are computed from ksize.width and ksize.height. To fully control the result regardless of possible future modification of all this semantics, it is recommended to specify all of ksize, sigmaX and sigmaY

    :param bordertype: Pixel extrapolation method.

The function convolves the source image with the specified Gaussian kernel. In-place filtering is supported.  Surpport 8UC1 8UC4 32SC1 32SC4 32FC1 32FC4 data type.

ocl::boxFilter
------------------
Returns void

.. ocv:function:: void ocl::boxFilter(const oclMat &src, oclMat &dst, int ddepth, Size ksize, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param ddepth: The desired depth of the destination image

    :param ksize: The smoothing kernel size. It must be positive and odd

    :param anchor: The anchor point. The default value Point(-1,-1) means that the anchor is at the kernel center.

    :param bordertype: Pixel extrapolation method.

Smoothes image using box filter.Supports data type: CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4.

ocl::Laplacian
------------------
Returns void

.. ocv:function:: void ocl::Laplacian(const oclMat &src, oclMat &dst, int ddepth, int ksize = 1, double scale = 1)

    :param src: The source image

    :param dst: The destination image; It will have the same size and the same type as src

    :param ddepth: The desired depth of the destination image

    :param ksize: The aperture size used to compute the second-derivative filters. It must be positive and odd

    :param scale: The optional scale factor for the computed Laplacian values (by default, no scaling is applied

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
--------------------
Returns void

.. ocv:function:: void ocl::bilateralFilter(const oclMat &src, oclMat &dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT)

    :param src: The source image

    :param dst: The destination image; will have the same size and the same type as src

    :param d: The diameter of each pixel neighborhood, that is used during filtering. If it is non-positive, it's computed from sigmaSpace

    :param sigmaColor: Filter sigma in the color space. Larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color

    :param sigmaSpave: Filter sigma in the coordinate space. Larger value of the parameter means that farther pixels will influence each other (as long as their colors are close enough; see sigmaColor). Then d>0, it specifies the neighborhood size regardless of sigmaSpace, otherwise d is proportional to sigmaSpace.

    :param borderType: Pixel extrapolation method.

Applies bilateral filter to the image. Supports 8UC1 8UC4 data types.

ocl::copyMakeBorder
--------------------
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
------------------
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
-------------------
Performs linear blending of two images.

.. ocv:function:: void ocl::blendLinear(const oclMat& img1, const oclMat& img2, const oclMat& weights1, const oclMat& weights2, oclMat& result)

    :param img1: First image. Supports only ``CV_8U`` and ``CV_32F`` depth.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param weights1: Weights for first image. Must have tha same size as ``img1`` . Supports only ``CV_32F`` type.

    :param weights2: Weights for second image. Must have tha same size as ``img2`` . Supports only ``CV_32F`` type.

    :param result: Destination image.
