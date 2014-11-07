Image Filtering
===============

.. highlight:: cpp

Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images (represented as
:ocv:func:`Mat`'s). It means that for each pixel location
:math:`(x,y)` in the source image (normally, rectangular), its neighborhood is considered and used to compute the response. In case of a linear filter, it is a weighted sum of pixel values. In case of morphological operations, it is the minimum or maximum values, and so on. The computed response is stored in the destination image at the same location
:math:`(x,y)` . It means that the output image will be of the same size as the input image. Normally, the functions support multi-channel arrays, in which case every channel is processed independently. Therefore, the output image will also have the same number of channels as the input one.

Another common feature of the functions and classes described in this section is that, unlike simple arithmetic functions, they need to extrapolate values of some non-existing pixels. For example, if you want to smooth an image using a Gaussian
:math:`3 \times 3` filter, then, when processing the left-most pixels in each row, you need pixels to the left of them, that is, outside of the image. You can let these pixels be the same as the left-most image pixels ("replicated border" extrapolation method), or assume that all the non-existing pixels are zeros ("constant border" extrapolation method), and so on.
OpenCV enables you to specify the extrapolation method. For details, see the function  ``borderInterpolate``  and discussion of the  ``borderType``  parameter in the section and various functions below. ::

   /*
    Various border types, image boundaries are denoted with '|'

    * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
    * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
    * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
    * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
    * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
    */

.. note::

   * (Python) A complete example illustrating different morphological operations like erode/dilate, open/close, blackhat/tophat ... can be found at opencv_source_code/samples/python2/morphology.py

bilateralFilter
-------------------
Applies the bilateral filter to an image.

.. ocv:function:: void bilateralFilter( InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst

    :param src: Source 8-bit or floating-point, 1-channel or 3-channel image.

    :param dst: Destination image of the same size and type as  ``src`` .

    :param d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from  ``sigmaSpace`` .

    :param sigmaColor: Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see  ``sigmaSpace`` ) will be mixed together, resulting in larger areas of semi-equal color.

    :param sigmaSpace: Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see  ``sigmaColor`` ). When  ``d>0`` , it specifies the neighborhood size regardless of  ``sigmaSpace`` . Otherwise,  ``d``  is proportional to  ``sigmaSpace`` .

The function applies bilateral filtering to the input image, as described in
http://www.dai.ed.ac.uk/CVonline/LOCAL\_COPIES/MANDUCHI1/Bilateral\_Filtering.html
``bilateralFilter`` can reduce unwanted noise very well while keeping edges fairly sharp. However, it is very slow compared to most filters.

*Sigma values*: For simplicity, you can set the 2 sigma values to be the same. If they are small (< 10), the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect, making the image look "cartoonish".

*Filter size*: Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering.

This filter does not work inplace.


blur
----
Blurs an image using the normalized box filter.

.. ocv:function:: void blur( InputArray src, OutputArray dst, Size ksize, Point anchor=Point(-1,-1),           int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst

    :param src: input image; it can have any number of channels, which are processed independently, but the depth should be ``CV_8U``, ``CV_16U``, ``CV_16S``, ``CV_32F`` or ``CV_64F``.

    :param dst: output image of the same size and type as ``src``.

    :param ksize: blurring kernel size.

    :param anchor: anchor point; default value ``Point(-1,-1)`` means that the anchor is at the kernel center.

    :param borderType: border mode used to extrapolate pixels outside of the image.

The function smoothes an image using the kernel:

.. math::

    \texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}

The call ``blur(src, dst, ksize, anchor, borderType)`` is equivalent to ``boxFilter(src, dst, src.type(), anchor, true, borderType)`` .

.. seealso::

   :ocv:func:`boxFilter`,
   :ocv:func:`bilateralFilter`,
   :ocv:func:`GaussianBlur`,
   :ocv:func:`medianBlur`


boxFilter
---------
Blurs an image using the box filter.

.. ocv:function:: void boxFilter( InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor=Point(-1,-1), bool normalize=true, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) -> dst

    :param src: input image.

    :param dst: output image of the same size and type as ``src``.

    :param ddepth: the output image depth (-1 to use ``src.depth()``).

    :param ksize: blurring kernel size.

    :param anchor: anchor point; default value ``Point(-1,-1)`` means that the anchor is at the kernel center.

    :param normalize: flag, specifying whether the kernel is normalized by its area or not.

    :param borderType: border mode used to extrapolate pixels outside of the image.

The function smoothes an image using the kernel:

.. math::

    \texttt{K} =  \alpha \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1 \end{bmatrix}

where

.. math::

    \alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}

Unnormalized box filter is useful for computing various integral characteristics over each pixel neighborhood, such as covariance matrices of image derivatives (used in dense optical flow algorithms, and so on). If you need to compute pixel sums over variable-size windows, use :ocv:func:`integral` .

.. seealso::

    :ocv:func:`blur`,
    :ocv:func:`bilateralFilter`,
    :ocv:func:`GaussianBlur`,
    :ocv:func:`medianBlur`,
    :ocv:func:`integral`



buildPyramid
------------
Constructs the Gaussian pyramid for an image.

.. ocv:function:: void buildPyramid( InputArray src, OutputArrayOfArrays dst, int maxlevel, int borderType=BORDER_DEFAULT )

    :param src: Source image. Check  :ocv:func:`pyrDown`  for the list of supported types.

    :param dst: Destination vector of  ``maxlevel+1``  images of the same type as  ``src`` . ``dst[0]``  will be the same as  ``src`` .  ``dst[1]``  is the next pyramid layer, a smoothed and down-sized  ``src``  , and so on.

    :param maxlevel: 0-based index of the last (the smallest) pyramid layer. It must be non-negative.

    :param borderType: Pixel extrapolation method (BORDER_CONSTANT don't supported). See  ``borderInterpolate`` for details.

The function constructs a vector of images and builds the Gaussian pyramid by recursively applying
:ocv:func:`pyrDown` to the previously built pyramid layers, starting from ``dst[0]==src`` .


dilate
------
Dilates an image by using a specific structuring element.

.. ocv:function:: void dilate( InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )

.. ocv:pyfunction:: cv2.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

.. ocv:cfunction:: void cvDilate( const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1 )

    :param src: input image; the number of channels can be arbitrary, but the depth should be one of ``CV_8U``, ``CV_16U``, ``CV_16S``,  ``CV_32F` or ``CV_64F``.

    :param dst: output image of the same size and type as ``src``.

    :param kernel: structuring element used for dilation; if  ``elemenat=Mat()`` , a  ``3 x 3`` rectangular structuring element is used. Kernel can be created using :ocv:func:`getStructuringElement`

    :param anchor: position of the anchor within the element; default value ``(-1, -1)`` means that the anchor is at the element center.

    :param iterations: number of times dilation is applied.

    :param borderType: pixel extrapolation method (see  ``borderInterpolate`` for details).

    :param borderValue: border value in case of a constant border

The function dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken:

.. math::

    \texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')

The function supports the in-place mode. Dilation can be applied several ( ``iterations`` ) times. In case of multi-channel images, each channel is processed independently.

.. seealso::

    :ocv:func:`erode`,
    :ocv:func:`morphologyEx`,
    :ocv:func:`getStructuringElement`


.. note::

   * An example using the morphological dilate operation can be found at opencv_source_code/samples/cpp/morphology2.cpp


erode
-----
Erodes an image by using a specific structuring element.

.. ocv:function:: void erode( InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )

.. ocv:pyfunction:: cv2.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

.. ocv:cfunction:: void cvErode( const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1)

    :param src: input image; the number of channels can be arbitrary, but the depth should be one of ``CV_8U``, ``CV_16U``, ``CV_16S``,  ``CV_32F` or ``CV_64F``.

    :param dst: output image of the same size and type as ``src``.

    :param kernel: structuring element used for erosion; if  ``element=Mat()`` , a  ``3 x 3``  rectangular structuring element is used. Kernel can be created using :ocv:func:`getStructuringElement`.

    :param anchor: position of the anchor within the element; default value  ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: number of times erosion is applied.

    :param borderType: pixel extrapolation method (see  ``borderInterpolate`` for details).

    :param borderValue: border value in case of a constant border

The function erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken:

.. math::

    \texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')

The function supports the in-place mode. Erosion can be applied several ( ``iterations`` ) times. In case of multi-channel images, each channel is processed independently.

.. seealso::

    :ocv:func:`dilate`,
    :ocv:func:`morphologyEx`,
    :ocv:func:`getStructuringElement`

.. note::

   * An example using the morphological erode operation can be found at opencv_source_code/samples/cpp/morphology2.cpp

filter2D
--------
Convolves an image with the kernel.

.. ocv:function:: void filter2D( InputArray src, OutputArray dst, int ddepth, InputArray kernel, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst

.. ocv:cfunction:: void cvFilter2D( const CvArr* src, CvArr* dst, const CvMat* kernel, CvPoint anchor=cvPoint(-1,-1) )

    :param src: input image.

    :param dst: output image of the same size and the same number of channels as ``src``.


    :param ddepth: desired depth of the destination image; if it is negative, it will be the same as ``src.depth()``; the following combinations of ``src.depth()`` and ``ddepth`` are supported:
         * ``src.depth()`` = ``CV_8U``, ``ddepth`` = -1/``CV_16S``/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_16U``/``CV_16S``, ``ddepth`` = -1/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_32F``, ``ddepth`` = -1/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_64F``, ``ddepth`` = -1/``CV_64F``

        when ``ddepth=-1``, the output image will have the same depth as the source.

    :param kernel: convolution kernel (or rather a correlation kernel), a single-channel floating point matrix; if you want to apply different kernels to different channels, split the image into separate color planes using  :ocv:func:`split`  and process them individually.

    :param anchor: anchor of the kernel that indicates the relative position of a filtered point within the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor is at the kernel center.

    :param delta: optional value added to the filtered pixels before storing them in ``dst``.

    :param borderType: pixel extrapolation method (see  ``borderInterpolate`` for details).

The function applies an arbitrary linear filter to an image. In-place operation is supported. When the aperture is partially outside the image, the function interpolates outlier pixel values according to the specified border mode.

The function does actually compute correlation, not the convolution:

.. math::

    \texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' < \texttt{kernel.cols},}{0\leq y' < \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )

That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip the kernel using
:ocv:func:`flip` and set the new anchor to ``(kernel.cols - anchor.x - 1, kernel.rows - anchor.y - 1)`` .

The function uses the DFT-based algorithm in case of sufficiently large kernels (~``11 x 11`` or larger) and the direct algorithm for small kernels.

.. seealso::

    :ocv:func:`sepFilter2D`,
    :ocv:func:`dft`,
    :ocv:func:`matchTemplate`



GaussianBlur
------------
Blurs an image using a Gaussian filter.

.. ocv:function:: void GaussianBlur( InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst

    :param src: input image; the image can have any number of channels, which are processed independently, but the depth should be ``CV_8U``, ``CV_16U``, ``CV_16S``, ``CV_32F`` or ``CV_64F``.

    :param dst: output image of the same size and type as ``src``.

    :param ksize: Gaussian kernel size.  ``ksize.width``  and  ``ksize.height``  can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from  ``sigma*`` .

    :param sigmaX: Gaussian kernel standard deviation in X direction.

    :param sigmaY: Gaussian kernel standard deviation in Y direction; if  ``sigmaY``  is zero, it is set to be equal to  ``sigmaX``, if both sigmas are zeros, they are computed from  ``ksize.width``  and  ``ksize.height`` , respectively (see  :ocv:func:`getGaussianKernel` for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ``ksize``, ``sigmaX``, and ``sigmaY``.

    :param borderType: pixel extrapolation method (see  ``borderInterpolate`` for details).

The function convolves the source image with the specified Gaussian kernel. In-place filtering is supported.

.. seealso::

   :ocv:func:`sepFilter2D`,
   :ocv:func:`filter2D`,
   :ocv:func:`blur`,
   :ocv:func:`boxFilter`,
   :ocv:func:`bilateralFilter`,
   :ocv:func:`medianBlur`


getDerivKernels
---------------
Returns filter coefficients for computing spatial image derivatives.

.. ocv:function:: void getDerivKernels( OutputArray kx, OutputArray ky, int dx, int dy, int ksize,                      bool normalize=false, int ktype=CV_32F )

.. ocv:pyfunction:: cv2.getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) -> kx, ky

    :param kx: Output matrix of row filter coefficients. It has the type  ``ktype`` .

    :param ky: Output matrix of column filter coefficients. It has the type  ``ktype`` .

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Aperture size. It can be  ``CV_SCHARR`` , 1, 3, 5, or 7.

    :param normalize: Flag indicating whether to normalize (scale down) the filter coefficients or not. Theoretically, the coefficients should have the denominator  :math:`=2^{ksize*2-dx-dy-2}` . If you are going to filter floating-point images, you are likely to use the normalized kernels. But if you compute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve all the fractional bits, you may want to set  ``normalize=false`` .

    :param ktype: Type of filter coefficients. It can be  ``CV_32f``  or  ``CV_64F`` .

The function computes and returns the filter coefficients for spatial image derivatives. When ``ksize=CV_SCHARR`` , the Scharr
:math:`3 \times 3` kernels are generated (see
:ocv:func:`Scharr` ). Otherwise, Sobel kernels are generated (see
:ocv:func:`Sobel` ). The filters are normally passed to
:ocv:func:`sepFilter2D` or to


getGaussianKernel
-----------------
Returns Gaussian filter coefficients.

.. ocv:function:: Mat getGaussianKernel( int ksize, double sigma, int ktype=CV_64F )

.. ocv:pyfunction:: cv2.getGaussianKernel(ksize, sigma[, ktype]) -> retval

    :param ksize: Aperture size. It should be odd ( :math:`\texttt{ksize} \mod 2 = 1` ) and positive.

    :param sigma: Gaussian standard deviation. If it is non-positive, it is computed from  ``ksize``  as  \ ``sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`` .
    :param ktype: Type of filter coefficients. It can be  ``CV_32F``  or  ``CV_64F`` .

The function computes and returns the
:math:`\texttt{ksize} \times 1` matrix of Gaussian filter coefficients:

.. math::

    G_i= \alpha *e^{-(i-( \texttt{ksize} -1)/2)^2/(2* \texttt{sigma} )^2},

where
:math:`i=0..\texttt{ksize}-1` and
:math:`\alpha` is the scale factor chosen so that
:math:`\sum_i G_i=1`.

Two of such generated kernels can be passed to
:ocv:func:`sepFilter2D`. Those functions automatically recognize smoothing kernels (a symmetrical kernel with sum of weights equal to 1) and handle them accordingly. You may also use the higher-level
:ocv:func:`GaussianBlur`.

.. seealso::

   :ocv:func:`sepFilter2D`,
   :ocv:func:`getDerivKernels`,
   :ocv:func:`getStructuringElement`,
   :ocv:func:`GaussianBlur`



getGaborKernel
-----------------
Returns Gabor filter coefficients.

.. ocv:function:: Mat getGaborKernel( Size ksize, double sigma, double theta, double lambd, double gamma, double psi = CV_PI*0.5, int ktype = CV_64F )

.. ocv:pyfunction:: cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]) -> retval

    :param ksize: Size of the filter returned.

    :param sigma: Standard deviation of the gaussian envelope.

    :param theta: Orientation of the normal to the parallel stripes of a Gabor function.

    :param lambd: Wavelength of the sinusoidal factor.

    :param gamma: Spatial aspect ratio.

    :param psi: Phase offset.

    :param ktype: Type of filter coefficients. It can be  ``CV_32F``  or  ``CV_64F`` .

For more details about gabor filter equations and parameters, see: `Gabor Filter <http://en.wikipedia.org/wiki/Gabor_filter>`_.


getStructuringElement
---------------------
Returns a structuring element of the specified size and shape for morphological operations.

.. ocv:function:: Mat getStructuringElement(int shape, Size ksize, Point anchor=Point(-1,-1))

.. ocv:pyfunction:: cv2.getStructuringElement(shape, ksize[, anchor]) -> retval

.. ocv:cfunction:: IplConvKernel* cvCreateStructuringElementEx( int cols, int rows, int anchor_x, int anchor_y, int shape, int* values=NULL )

    :param shape: Element shape that could be one of the following:

      * **MORPH_RECT**         - a rectangular structuring element:

        .. math::

            E_{ij}=1

      * **MORPH_ELLIPSE**         - an elliptic structuring element, that is, a filled ellipse inscribed into the rectangle ``Rect(0, 0, esize.width, 0.esize.height)``

      * **MORPH_CROSS**         - a cross-shaped structuring element:

        .. math::

            E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}

      * **CV_SHAPE_CUSTOM**     - custom structuring element (OpenCV 1.x API)

    :param ksize: Size of the structuring element.

    :param cols: Width of the structuring element

    :param rows: Height of the structuring element

    :param anchor: Anchor position within the element. The default value  :math:`(-1, -1)`  means that the anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor position. In other cases the anchor just regulates how much the result of the morphological operation is shifted.

    :param anchor_x: x-coordinate of the anchor

    :param anchor_y: y-coordinate of the anchor

    :param values: integer array of ``cols``*``rows`` elements that specifies the custom shape of the structuring element, when ``shape=CV_SHAPE_CUSTOM``.

The function constructs and returns the structuring element that can be further passed to
:ocv:func:`erode`,
:ocv:func:`dilate` or
:ocv:func:`morphologyEx` . But you can also construct an arbitrary binary mask yourself and use it as the structuring element.

.. note:: When using OpenCV 1.x C API, the created structuring element ``IplConvKernel* element`` must be released in the end using ``cvReleaseStructuringElement(&element)``.


medianBlur
----------
Blurs an image using the median filter.

.. ocv:function:: void medianBlur( InputArray src, OutputArray dst, int ksize )

.. ocv:pyfunction:: cv2.medianBlur(src, ksize[, dst]) -> dst

    :param src: input 1-, 3-, or 4-channel image; when  ``ksize``  is 3 or 5, the image depth should be ``CV_8U``, ``CV_16U``, or ``CV_32F``, for larger aperture sizes, it can only be ``CV_8U``.

    :param dst: destination array of the same size and type as ``src``.

    :param ksize: aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...

The function smoothes an image using the median filter with the
:math:`\texttt{ksize} \times \texttt{ksize}` aperture. Each channel of a multi-channel image is processed independently. In-place operation is supported.

.. seealso::

    :ocv:func:`bilateralFilter`,
    :ocv:func:`blur`,
    :ocv:func:`boxFilter`,
    :ocv:func:`GaussianBlur`



morphologyEx
------------
Performs advanced morphological transformations.

.. ocv:function:: void morphologyEx( InputArray src, OutputArray dst, int op, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )

.. ocv:pyfunction:: cv2.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst

.. ocv:cfunction:: void cvMorphologyEx( const CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* element, int operation, int iterations=1 )

    :param src: Source image. The number of channels can be arbitrary. The depth should be one of ``CV_8U``, ``CV_16U``, ``CV_16S``,  ``CV_32F` or ``CV_64F``.

    :param dst: Destination image of the same size and type as  ``src`` .

    :param kernel: Structuring element. It can be created using :ocv:func:`getStructuringElement`.

    :param anchor: Anchor position with the kernel. Negative values mean that the anchor is at the kernel center.

    :param op: Type of a morphological operation that can be one of the following:

            * **MORPH_OPEN** - an opening operation

            * **MORPH_CLOSE** - a closing operation

            * **MORPH_GRADIENT** - a morphological gradient

            * **MORPH_TOPHAT** - "top hat"

            * **MORPH_BLACKHAT** - "black hat"

    :param iterations: Number of times erosion and dilation are applied.

    :param borderType: Pixel extrapolation method. See  ``borderInterpolate`` for details.

    :param borderValue: Border value in case of a constant border. The default value has a special meaning.

The function can perform advanced morphological transformations using an erosion and dilation as basic operations.

Opening operation:

.. math::

    \texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))

Closing operation:

.. math::

    \texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))

Morphological gradient:

.. math::

    \texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )

"Top hat":

.. math::

    \texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )

"Black hat":

.. math::

    \texttt{dst} = \mathrm{blackhat} ( \texttt{src} , \texttt{element} )= \mathrm{close} ( \texttt{src} , \texttt{element} )- \texttt{src}

Any of the operations can be done in-place. In case of multi-channel images, each channel is processed independently.

.. seealso::

    :ocv:func:`dilate`,
    :ocv:func:`erode`,
    :ocv:func:`getStructuringElement`

.. note::

   * An example using the morphologyEx function for the morphological opening and closing operations can be found at opencv_source_code/samples/cpp/morphology2.cpp

Laplacian
---------
Calculates the Laplacian of an image.

.. ocv:function:: void Laplacian( InputArray src, OutputArray dst, int ddepth, int ksize=1, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

.. ocv:cfunction:: void cvLaplace( const CvArr* src, CvArr* dst, int aperture_size=3 )

    :param src: Source image.

    :param dst: Destination image of the same size and the same number of channels as  ``src`` .

    :param ddepth: Desired depth of the destination image.

    :param ksize: Aperture size used to compute the second-derivative filters. See  :ocv:func:`getDerivKernels` for details. The size must be positive and odd.

    :param scale: Optional scale factor for the computed Laplacian values. By default, no scaling is applied. See  :ocv:func:`getDerivKernels` for details.

    :param delta: Optional delta value that is added to the results prior to storing them in  ``dst`` .

    :param borderType: Pixel extrapolation method. See  ``borderInterpolate`` for details.

The function calculates the Laplacian of the source image by adding up the second x and y derivatives calculated using the Sobel operator:

.. math::

    \texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}

This is done when ``ksize > 1`` . When ``ksize == 1`` , the Laplacian is computed by filtering the image with the following
:math:`3 \times 3` aperture:

.. math::

    \vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}

.. seealso::

    :ocv:func:`Sobel`,
    :ocv:func:`Scharr`

.. note::

   * An example using the Laplace transformation for edge detection can be found at opencv_source_code/samples/cpp/laplace.cpp

pyrDown
-------
Blurs an image and downsamples it.

.. ocv:function:: void pyrDown( InputArray src, OutputArray dst, const Size& dstsize=Size(), int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst

.. ocv:cfunction:: void cvPyrDown( const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5 )

    :param src: input image.

    :param dst: output image; it has the specified size and the same type as ``src``.

    :param dstsize: size of the output image.

    :param borderType: Pixel extrapolation method (BORDER_CONSTANT don't supported). See  ``borderInterpolate`` for details.

By default, size of the output image is computed as ``Size((src.cols+1)/2, (src.rows+1)/2)``, but in any case, the following conditions should be satisfied:

.. math::

    \begin{array}{l}
    | \texttt{dstsize.width} *2-src.cols| \leq  2  \\ | \texttt{dstsize.height} *2-src.rows| \leq  2 \end{array}

The function performs the downsampling step of the Gaussian pyramid construction. First, it convolves the source image with the kernel:

.. math::

    \frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}

Then, it downsamples the image by rejecting even rows and columns.

pyrUp
-----
Upsamples an image and then blurs it.

.. ocv:function:: void pyrUp( InputArray src, OutputArray dst, const Size& dstsize=Size(), int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst

.. ocv:cfunction:: cvPyrUp( const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5 )

    :param src: input image.

    :param dst: output image. It has the specified size and the same type as  ``src`` .

    :param dstsize: size of the output image.

    :param borderType: Pixel extrapolation method (only BORDER_DEFAULT supported). See  ``borderInterpolate`` for details.

By default, size of the output image is computed as ``Size(src.cols*2, (src.rows*2)``, but in any case, the following conditions should be satisfied:

.. math::

    \begin{array}{l}
    | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}

The function performs the upsampling step of the Gaussian pyramid construction, though it can actually be used to construct the Laplacian pyramid. First, it upsamples the source image by injecting even zero rows and columns and then convolves the result with the same kernel as in
:ocv:func:`pyrDown`  multiplied by 4.

.. note::

   * (Python) An example of Laplacian Pyramid construction and merging can be found at opencv_source_code/samples/python2/lappyr.py


pyrMeanShiftFiltering
---------------------
Performs initial step of meanshift segmentation of an image.

.. ocv:function:: void pyrMeanShiftFiltering( InputArray src, OutputArray dst, double sp, double sr, int maxLevel=1, TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1) )

.. ocv:pyfunction:: cv2.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst

.. ocv:cfunction:: void cvPyrMeanShiftFiltering( const CvArr* src, CvArr* dst, double sp,  double sr,  int max_level=1, CvTermCriteria termcrit= cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1))

    :param src: The source 8-bit, 3-channel image.

    :param dst: The destination image of the same format and the same size as the source.

    :param sp: The spatial window radius.

    :param sr: The color window radius.

    :param maxLevel: Maximum level of the pyramid for the segmentation.

    :param termcrit: Termination criteria: when to stop meanshift iterations.


The function implements the filtering stage of meanshift segmentation, that is, the output of the function is the filtered "posterized" image with color gradients and fine-grain texture flattened. At every pixel
``(X,Y)`` of the input image (or down-sized input image, see below) the function executes meanshift
iterations, that is, the pixel ``(X,Y)`` neighborhood in the joint space-color hyperspace is considered:

    .. math::

        (x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr}


where  ``(R,G,B)`` and  ``(r,g,b)`` are the vectors of color components at ``(X,Y)`` and  ``(x,y)``, respectively (though, the algorithm does not depend on the color space used, so any 3-component color space can be used instead). Over the neighborhood the average spatial value  ``(X',Y')`` and average color vector  ``(R',G',B')`` are found and they act as the neighborhood center on the next iteration:

    .. math::

        (X,Y)~(X',Y'), (R,G,B)~(R',G',B').

After the iterations over, the color components of the initial pixel (that is, the pixel from where the iterations started) are set to the final value (average color at the last iteration):

    .. math::

        I(X,Y) <- (R*,G*,B*)

When ``maxLevel > 0``, the gaussian pyramid of ``maxLevel+1`` levels is built, and the above procedure is run on the smallest layer first. After that, the results are propagated to the larger layer and the iterations are run again only on those pixels where the layer colors differ by more than ``sr`` from the lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the results will be actually different from the ones obtained by running the meanshift procedure on the whole original image (i.e. when ``maxLevel==0``).

.. note::

   * An example using mean-shift image segmentation can be found at opencv_source_code/samples/cpp/meanshift_segmentation.cpp

sepFilter2D
-----------
Applies a separable linear filter to an image.

.. ocv:function:: void sepFilter2D( InputArray src, OutputArray dst, int ddepth, InputArray kernelX, InputArray kernelY, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) -> dst

    :param src: Source image.

    :param dst: Destination image of the same size and the same number of channels as  ``src`` .

    :param ddepth: Destination image depth. The following combination of ``src.depth()`` and ``ddepth`` are supported:
         * ``src.depth()`` = ``CV_8U``, ``ddepth`` = -1/``CV_16S``/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_16U``/``CV_16S``, ``ddepth`` = -1/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_32F``, ``ddepth`` = -1/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_64F``, ``ddepth`` = -1/``CV_64F``

        when ``ddepth=-1``, the destination image will have the same depth as the source.

    :param kernelX: Coefficients for filtering each row.

    :param kernelY: Coefficients for filtering each column.

    :param anchor: Anchor position within the kernel. The default value  :math:`(-1,-1)`  means that the anchor is at the kernel center.

    :param delta: Value added to the filtered results before storing them.

    :param borderType: Pixel extrapolation method. See  ``borderInterpolate`` for details.

The function applies a separable linear filter to the image. That is, first, every row of ``src`` is filtered with the 1D kernel ``kernelX`` . Then, every column of the result is filtered with the 1D kernel ``kernelY`` . The final result shifted by ``delta`` is stored in ``dst`` .

.. seealso::

   :ocv:func:`filter2D`,
   :ocv:func:`Sobel`,
   :ocv:func:`GaussianBlur`,
   :ocv:func:`boxFilter`,
   :ocv:func:`blur`


Smooth
------
Smooths the image in one of several ways.

.. ocv:cfunction:: void cvSmooth( const CvArr* src, CvArr* dst, int smoothtype=CV_GAUSSIAN, int size1=3, int size2=0, double sigma1=0, double sigma2=0 )

    :param src: The source image

    :param dst: The destination image

    :param smoothtype: Type of the smoothing:

            * **CV_BLUR_NO_SCALE** linear convolution with  :math:`\texttt{size1}\times\texttt{size2}`  box kernel (all 1's). If you want to smooth different pixels with different-size box kernels, you can use the integral image that is computed using  :ocv:func:`integral`


            * **CV_BLUR** linear convolution with  :math:`\texttt{size1}\times\texttt{size2}`  box kernel (all 1's) with subsequent scaling by  :math:`1/(\texttt{size1}\cdot\texttt{size2})`


            * **CV_GAUSSIAN** linear convolution with a  :math:`\texttt{size1}\times\texttt{size2}`  Gaussian kernel


            * **CV_MEDIAN** median filter with a  :math:`\texttt{size1}\times\texttt{size1}`  square aperture


            * **CV_BILATERAL** bilateral filter with a  :math:`\texttt{size1}\times\texttt{size1}`  square aperture, color sigma= ``sigma1``  and spatial sigma= ``sigma2`` . If  ``size1=0`` , the aperture square side is set to  ``cvRound(sigma2*1.5)*2+1`` . Information about bilateral filtering can be found at  http://www.dai.ed.ac.uk/CVonline/LOCAL\_COPIES/MANDUCHI1/Bilateral\_Filtering.html


    :param size1: The first parameter of the smoothing operation, the aperture width. Must be a positive odd number (1, 3, 5, ...)

    :param size2: The second parameter of the smoothing operation, the aperture height. Ignored by  ``CV_MEDIAN``  and  ``CV_BILATERAL``  methods. In the case of simple scaled/non-scaled and Gaussian blur if  ``size2``  is zero, it is set to  ``size1`` . Otherwise it must be a positive odd number.

    :param sigma1: In the case of a Gaussian parameter this parameter may specify Gaussian  :math:`\sigma`  (standard deviation). If it is zero, it is calculated from the kernel size:

        .. math::

            \sigma  = 0.3 (n/2 - 1) + 0.8  \quad   \text{where}   \quad  n= \begin{array}{l l} \mbox{\texttt{size1} for horizontal kernel} \\ \mbox{\texttt{size2} for vertical kernel} \end{array}

        Using standard sigma for small kernels ( :math:`3\times 3`  to  :math:`7\times 7` ) gives better speed. If  ``sigma1``  is not zero, while  ``size1``  and  ``size2``  are zeros, the kernel size is calculated from the sigma (to provide accurate enough operation).

The function smooths an image using one of several methods. Every of the methods has some features and restrictions listed below:

 * Blur with no scaling works with single-channel images only and supports accumulation of 8-bit to 16-bit format (similar to :ocv:func:`Sobel` and :ocv:func:`Laplacian`) and 32-bit floating point to 32-bit floating-point format.

 * Simple blur and Gaussian blur support 1- or 3-channel, 8-bit and 32-bit floating point images. These two methods can process images in-place.

 * Median and bilateral filters work with 1- or 3-channel 8-bit images and can not process images in-place.

.. note:: The function is now obsolete. Use :ocv:func:`GaussianBlur`, :ocv:func:`blur`, :ocv:func:`medianBlur` or :ocv:func:`bilateralFilter`.


Sobel
-----
Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

.. ocv:function:: void Sobel( InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst

.. ocv:cfunction:: void cvSobel( const CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size=3 )

    :param src: input image.

    :param dst: output image of the same size and the same number of channels as  ``src`` .

    :param ddepth: output image depth; the following combinations of ``src.depth()`` and ``ddepth`` are supported:
         * ``src.depth()`` = ``CV_8U``, ``ddepth`` = -1/``CV_16S``/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_16U``/``CV_16S``, ``ddepth`` = -1/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_32F``, ``ddepth`` = -1/``CV_32F``/``CV_64F``
         * ``src.depth()`` = ``CV_64F``, ``ddepth`` = -1/``CV_64F``

        when ``ddepth=-1``, the destination image will have the same depth as the source; in the case of 8-bit input images it will result in truncated derivatives.

    :param xorder: order of the derivative x.

    :param yorder: order of the derivative y.

    :param ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.

    :param scale: optional scale factor for the computed derivative values; by default, no scaling is applied (see  :ocv:func:`getDerivKernels` for details).

    :param delta: optional delta value that is added to the results prior to storing them in ``dst``.

    :param borderType: pixel extrapolation method (see  ``borderInterpolate`` for details).

In all cases except one, the
:math:`\texttt{ksize} \times
\texttt{ksize}` separable kernel is used to calculate the
derivative. When
:math:`\texttt{ksize = 1}` , the
:math:`3 \times 1` or
:math:`1 \times 3` kernel is used (that is, no Gaussian smoothing is done). ``ksize = 1`` can only be used for the first or the second x- or y- derivatives.

There is also the special value ``ksize = CV_SCHARR`` (-1) that corresponds to the
:math:`3\times3` Scharr
filter that may give more accurate results than the
:math:`3\times3` Sobel. The Scharr aperture is

.. math::

    \vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}

for the x-derivative, or transposed for the y-derivative.

The function calculates an image derivative by convolving the image with the appropriate kernel:

.. math::

    \texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}

The Sobel operators combine Gaussian smoothing and differentiation,
so the result is more or less resistant to the noise. Most often,
the function is called with ( ``xorder`` = 1, ``yorder`` = 0, ``ksize`` = 3) or ( ``xorder`` = 0, ``yorder`` = 1, ``ksize`` = 3) to calculate the first x- or y- image
derivative. The first case corresponds to a kernel of:

.. math::

    \vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}

The second case corresponds to a kernel of:

.. math::

    \vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}

.. seealso::

    :ocv:func:`Scharr`,
    :ocv:func:`Laplacian`,
    :ocv:func:`sepFilter2D`,
    :ocv:func:`filter2D`,
    :ocv:func:`GaussianBlur`,
    :ocv:func:`cartToPolar`



Scharr
------
Calculates the first x- or y- image derivative using Scharr operator.

.. ocv:function:: void Scharr( InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )

.. ocv:pyfunction:: cv2.Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst

    :param src: input image.

    :param dst: output image of the same size and the same number of channels as ``src``.

    :param ddepth: output image depth (see :ocv:func:`Sobel` for the list of supported combination of ``src.depth()`` and ``ddepth``).

    :param dx: order of the derivative x.

    :param dy: order of the derivative y.

    :param scale: optional scale factor for the computed derivative values; by default, no scaling is applied (see  :ocv:func:`getDerivKernels` for details).

    :param delta: optional delta value that is added to the results prior to storing them in ``dst``.

    :param borderType: pixel extrapolation method (see  ``borderInterpolate`` for details).

The function computes the first x- or y- spatial image derivative using the Scharr operator. The call

.. math::

    \texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}

is equivalent to

.. math::

    \texttt{Sobel(src, dst, ddepth, dx, dy, CV\_SCHARR, scale, delta, borderType)} .

.. seealso::

    :ocv:func:`cartToPolar`
