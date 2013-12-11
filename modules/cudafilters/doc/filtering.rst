Image Filtering
===============

.. highlight:: cpp

Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images.

.. note::

   * An example containing all basic morphology operators like erode and dilate can be found at opencv_source_code/samples/gpu/morphology.cpp



cuda::Filter
------------
.. ocv:class:: cuda::Filter

Common interface for all CUDA filters ::

    class CV_EXPORTS Filter : public Algorithm
    {
    public:
        virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
    };



cuda::Filter::apply
-------------------
Applies the specified filter to the image.

.. ocv:function:: void cuda::Filter::apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0

    :param src: Input image.

    :param dst: Output image.

    :param stream: Stream for the asynchronous version.



cuda::createBoxFilter
---------------------
Creates a normalized 2D box filter.

.. ocv:function:: Ptr<Filter> cuda::createBoxFilter(int srcType, int dstType, Size ksize, Point anchor = Point(-1,-1), int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input image type. Only ``CV_8UC1`` and ``CV_8UC4`` are supported for now.

    :param dstType: Output image type. Only the same type as ``src`` is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.

.. seealso:: :ocv:func:`boxFilter`



cuda::createLinearFilter
------------------------
Creates a non-separable linear 2D filter.

.. ocv:function:: Ptr<Filter> cuda::createLinearFilter(int srcType, int dstType, InputArray kernel, Point anchor = Point(-1,-1), int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input image type. Supports  ``CV_8U``  ,  ``CV_16U``  and  ``CV_32F``  one and four channel image.

    :param dstType: Output image type. Only the same type as ``src`` is supported for now.

    :param kernel: 2D array of filter coefficients.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.

.. seealso:: :ocv:func:`filter2D`



cuda::createLaplacianFilter
---------------------------
Creates a Laplacian operator.

.. ocv:function:: Ptr<Filter> cuda::createLaplacianFilter(int srcType, int dstType, int ksize = 1, double scale = 1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input image type. Supports  ``CV_8U``  ,  ``CV_16U``  and  ``CV_32F``  one and four channel image.

    :param dstType: Output image type. Only the same type as ``src`` is supported for now.

    :param ksize: Aperture size used to compute the second-derivative filters (see :ocv:func:`getDerivKernels`). It must be positive and odd. Only  ``ksize``  = 1 and  ``ksize``  = 3 are supported.

    :param scale: Optional scale factor for the computed Laplacian values. By default, no scaling is applied (see  :ocv:func:`getDerivKernels` ).

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.

.. seealso:: :ocv:func:`Laplacian`



cuda::createSeparableLinearFilter
---------------------------------
Creates a separable linear filter.

.. ocv:function:: Ptr<Filter> cuda::createSeparableLinearFilter(int srcType, int dstType, InputArray rowKernel, InputArray columnKernel, Point anchor = Point(-1,-1), int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)

    :param srcType: Source array type.

    :param dstType: Destination array type.

    :param rowKernel: Horizontal filter coefficients. Support kernels with ``size <= 32`` .

    :param columnKernel: Vertical filter coefficients. Support kernels with ``size <= 32`` .

    :param anchor: Anchor position within the kernel. Negative values mean that anchor is positioned at the aperture center.

    :param rowBorderMode: Pixel extrapolation method in the vertical direction For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderMode: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`sepFilter2D`



cuda::createDerivFilter
-----------------------
Creates a generalized Deriv operator.

.. ocv:function:: Ptr<Filter> cuda::createDerivFilter(int srcType, int dstType, int dx, int dy, int ksize, bool normalize = false, double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)

    :param srcType: Source image type.

    :param dstType: Destination array type.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Aperture size. See  :ocv:func:`getDerivKernels` for details.

    :param normalize: Flag indicating whether to normalize (scale down) the filter coefficients or not. See  :ocv:func:`getDerivKernels` for details.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. For details, see  :ocv:func:`getDerivKernels` .

    :param rowBorderMode: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderMode: Pixel extrapolation method in the horizontal direction.



cuda::createSobelFilter
-----------------------
Creates a Sobel operator.

.. ocv:function:: Ptr<Filter> cuda::createSobelFilter(int srcType, int dstType, int dx, int dy, int ksize = 3, double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)

    :param srcType: Source image type.

    :param dstType: Destination array type.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Size of the extended Sobel kernel. Possible values are 1, 3, 5 or 7.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. For details, see  :ocv:func:`getDerivKernels` .

    :param rowBorderMode: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderMode: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`Sobel`



cuda::createScharrFilter
------------------------
Creates a vertical or horizontal Scharr operator.

.. ocv:function:: Ptr<Filter> cuda::createScharrFilter(int srcType, int dstType, int dx, int dy, double scale = 1, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)

    :param srcType: Source image type.

    :param dstType: Destination array type.

    :param dx: Order of the derivative x.

    :param dy: Order of the derivative y.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. See  :ocv:func:`getDerivKernels`  for details.

    :param rowBorderMode: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderMode: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`Scharr`



cuda::createGaussianFilter
--------------------------
Creates a Gaussian filter.

.. ocv:function:: Ptr<Filter> cuda::createGaussianFilter(int srcType, int dstType, Size ksize, double sigma1, double sigma2 = 0, int rowBorderMode = BORDER_DEFAULT, int columnBorderMode = -1)

    :param srcType: Source image type.

    :param dstType: Destination array type.

    :param ksize: Aperture size. See  :ocv:func:`getGaussianKernel` for details.

    :param sigma1: Gaussian sigma in the horizontal direction. See  :ocv:func:`getGaussianKernel` for details.

    :param sigma2: Gaussian sigma in the vertical direction. If 0, then  :math:`\texttt{sigma2}\leftarrow\texttt{sigma1}` .

    :param rowBorderMode: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderMode: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`GaussianBlur`



cuda::createMorphologyFilter
----------------------------
Creates a 2D morphological filter.

.. ocv:function:: Ptr<Filter> cuda::createMorphologyFilter(int op, int srcType, InputArray kernel, Point anchor = Point(-1, -1), int iterations = 1)

    :param op: Type of morphological operation. The following types are possible:

        * **MORPH_ERODE** erode

        * **MORPH_DILATE** dilate

        * **MORPH_OPEN** opening

        * **MORPH_CLOSE** closing

        * **MORPH_GRADIENT** morphological gradient

        * **MORPH_TOPHAT** "top hat"

        * **MORPH_BLACKHAT** "black hat"

    :param srcType: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4``  are supported.

    :param kernel: 2D 8-bit structuring element for the morphological operation.

    :param anchor: Anchor position within the structuring element. Negative values mean that the anchor is at the center.

    :param iterations: Number of times erosion and dilation to be applied.

.. seealso:: :ocv:func:`morphologyEx`



cuda::createBoxMaxFilter
------------------------
Creates the maximum filter.

.. ocv:function:: Ptr<Filter> cuda::createBoxMaxFilter(int srcType, Size ksize, Point anchor = Point(-1, -1), int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4`` are supported.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.



cuda::createBoxMinFilter
------------------------
Creates the minimum filter.

.. ocv:function:: Ptr<Filter> cuda::createBoxMinFilter(int srcType, Size ksize, Point anchor = Point(-1, -1), int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4`` are supported.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.



cuda::createRowSumFilter
------------------------
Creates a horizontal 1D box filter.

.. ocv:function:: Ptr<Filter> cuda::createRowSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param sumType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.



cuda::createColumnSumFilter
---------------------------
Creates a vertical 1D box filter.

.. ocv:function:: Ptr<Filter> cuda::createColumnSumFilter(int srcType, int dstType, int ksize, int anchor = -1, int borderMode = BORDER_DEFAULT, Scalar borderVal = Scalar::all(0))

    :param srcType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param sumType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

    :param borderMode: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate` .

    :param borderVal: Default border value.
