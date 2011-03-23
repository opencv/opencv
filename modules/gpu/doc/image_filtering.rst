Image Filtering
===============

.. highlight:: cpp



Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images.

See also: :ref:`ImageFiltering`.



.. index:: gpu::BaseRowFilter_GPU

gpu::BaseRowFilter_GPU
----------------------
.. cpp:class:: gpu::BaseRowFilter_GPU

The base class for linear or non-linear filters that processes rows of 2D arrays. Such filters are used for the "horizontal" filtering passes in separable filters. ::

    class BaseRowFilter_GPU
    {
    public:
        BaseRowFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseRowFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        int ksize, anchor;
    };

**Please note:** This class doesn't allocate memory for destination image. Usually this class is used inside :cpp:class:`gpu::FilterEngine_GPU`.



.. index:: gpu::BaseColumnFilter_GPU

gpu::BaseColumnFilter_GPU
-------------------------
.. cpp:class:: gpu::BaseColumnFilter_GPU

The base class for linear or non-linear filters that processes columns of 2D arrays. Such filters are used for the "vertical" filtering passes in separable filters. ::

    class BaseColumnFilter_GPU
    {
    public:
        BaseColumnFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseColumnFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        int ksize, anchor;
    };

**Please note:** This class doesn't allocate memory for destination image. Usually this class is used inside :cpp:class:`gpu::FilterEngine_GPU`.



.. index:: gpu::BaseFilter_GPU

gpu::BaseFilter_GPU
-------------------
.. cpp:class:: gpu::BaseFilter_GPU

The base class for non-separable 2D filters. ::

    class CV_EXPORTS BaseFilter_GPU
    {
    public:
        BaseFilter_GPU(const Size& ksize_, const Point& anchor_);
        virtual ~BaseFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        Size ksize;
        Point anchor;
    };


**Please note:** This class doesn't allocate memory for destination image. Usually this class is used inside :cpp:class:`gpu::FilterEngine_GPU`.



.. index:: gpu::FilterEngine_GPU

gpu::FilterEngine_GPU
---------------------
.. cpp:class:: gpu::FilterEngine_GPU

The base class for Filter Engine. ::

    class CV_EXPORTS FilterEngine_GPU
    {
    public:
        virtual ~FilterEngine_GPU() {}

        virtual void apply(const GpuMat& src, GpuMat& dst,
                           Rect roi = Rect(0,0,-1,-1)) = 0;
    };

The class can be used to apply an arbitrary filtering operation to an image. It contains all the necessary intermediate buffers. Pointers to the initialized ``FilterEngine_GPU`` instances are returned by various ``create*Filter_GPU`` functions, see below, and they are used inside high-level functions such as :cpp:func:`gpu::filter2D`, :cpp:func:`gpu::erode`, :cpp:func:`gpu::Sobel` etc.

By using ``FilterEngine_GPU`` instead of functions you can avoid unnecessary memory allocation for intermediate buffers and get much better performance: ::

    while (...)
    {
        gpu::GpuMat src = getImg();
        gpu::GpuMat dst;
        // Allocate and release buffers at each iterations
        gpu::GaussianBlur(src, dst, ksize, sigma1);
    }

    // Allocate buffers only once
    cv::Ptr<gpu::FilterEngine_GPU> filter =
        gpu::createGaussianFilter_GPU(CV_8UC4, ksize, sigma1);
    while (...)
    {
        gpu::GpuMat src = getImg();
        gpu::GpuMat dst;
        filter->apply(src, dst, cv::Rect(0, 0, src.cols, src.rows));
    }
    // Release buffers only once
    filter.release();

``FilterEngine_GPU`` can process a rectangular sub-region of an image. By default, if ``roi == Rect(0,0,-1,-1)``, ``FilterEngine_GPU`` processes inner region of image (``Rect(anchor.x, anchor.y, src_size.width - ksize.width, src_size.height - ksize.height)``), because some filters doesn't check if indices are outside the image for better perfomace. See below which filters supports processing the whole image and which not and image type limitations.

**Please note:** The GPU filters doesn't support the in-place mode.

See also: :cpp:class:`gpu::BaseRowFilter_GPU`, :cpp:class:`gpu::BaseColumnFilter_GPU`, :cpp:class:`gpu::BaseFilter_GPU`, :cpp:func:`gpu::createFilter2D_GPU`, :cpp:func:`gpu::createSeparableFilter_GPU`, :cpp:func:`gpu::createBoxFilter_GPU`, :cpp:func:`gpu::createMorphologyFilter_GPU`, :cpp:func:`gpu::createLinearFilter_GPU`, :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :cpp:func:`gpu::createDerivFilter_GPU`, :cpp:func:`gpu::createGaussianFilter_GPU`.



.. index:: gpu::createFilter2D_GPU

gpu::createFilter2D_GPU
---------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createFilter2D_GPU(const Ptr<BaseFilter_GPU>& filter2D, int srcType, int dstType)

    Creates non-separable filter engine with the specified filter.
    
    :param filter2D: Non-separable 2D filter.

    :param srcType: Input image type. It must be supported by ``filter2D``.

    :param dstType: Output image type. It must be supported by ``filter2D``.

Usually this function is used inside high-level functions, like :cpp:func:`gpu::createLinearFilter_GPU`, :cpp:func:`gpu::createBoxFilter_GPU`.



.. index:: gpu::createSeparableFilter_GPU

gpu::createSeparableFilter_GPU
----------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createSeparableFilter_GPU( const Ptr<BaseRowFilter_GPU>& rowFilter, const Ptr<BaseColumnFilter_GPU>& columnFilter, int srcType, int bufType, int dstType)

    Creates separable filter engine with the specified filters.
    
    :param rowFilter: "Horizontal" 1D filter.

    :param columnFilter: "Vertical" 1D filter.

    :param srcType: Input image type. It must be supported by ``rowFilter``.

    :param bufType: Buffer image type. It must be supported by ``rowFilter`` and ``columnFilter``.

    :param dstType: Output image type. It must be supported by ``columnFilter``.

Usually this function is used inside high-level functions, like :cpp:func:`gpu::createSeparableLinearFilter_GPU`.



.. index:: gpu::getRowSumFilter_GPU

gpu::getRowSumFilter_GPU
----------------------------
.. cpp:function:: Ptr<BaseRowFilter_GPU> gpu::getRowSumFilter_GPU(int srcType, int sumType, int ksize, int anchor = -1)

    Creates horizontal 1D box filter.

    :param srcType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param sumType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.



.. index:: gpu::getColumnSumFilter_GPU

gpu::getColumnSumFilter_GPU
-------------------------------
.. cpp:function:: Ptr<BaseColumnFilter_GPU> gpu::getColumnSumFilter_GPU(int sumType,  int dstType, int ksize, int anchor = -1)

    Creates vertical 1D box filter.

    :param sumType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param dstType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.



.. index:: gpu::createBoxFilter_GPU

gpu::createBoxFilter_GPU
----------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createBoxFilter_GPU(int srcType, int dstType, const Size& ksize, const Point& anchor = Point(-1,-1))

.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getBoxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1, -1))

    Creates normalized 2D box filter.

    :param srcType: Input image type. Supports ``CV_8UC1`` and ``CV_8UC4``.

    :param dstType: Output image type. Supports only the same as source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`boxFilter`.



.. index:: gpu::boxFilter

gpu::boxFilter
------------------
.. cpp:function:: void gpu::boxFilter(const GpuMat& src, GpuMat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1))

    Smooths the image using the normalized box filter.

    :param src: Input image. Supports ``CV_8UC1`` and ``CV_8UC4`` source types.

    :param dst: Output image type. Will have the same size and the same type as ``src``.

    :param ddepth: Output image depth. Support only the same as source depth (``CV_8U``) or -1 what means use source depth.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`boxFilter`, :cpp:func:`gpu::createBoxFilter_GPU`.



.. index:: gpu::blur

gpu::blur
-------------
.. cpp:function:: void gpu::blur(const GpuMat& src, GpuMat& dst, Size ksize,  Point anchor = Point(-1,-1))

    A synonym for normalized box filter.

    :param src: Input image. Supports ``CV_8UC1`` and ``CV_8UC4`` source type.

    :param dst: Output image type. Will have the same size and the same type as ``src``.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`blur`, :cpp:func:`gpu::boxFilter`.



.. index:: gpu::createMorphologyFilter_GPU

gpu::createMorphologyFilter_GPU
-----------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1)

.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Size& ksize, Point anchor=Point(-1,-1))

    Creates 2D morphological filter.
    
    :param op: Morphology operation id. Only ``MORPH_ERODE`` and ``MORPH_DILATE`` are supported.

    :param type: Input/output image type. Only ``CV_8UC1`` and ``CV_8UC4`` are supported.

    :param kernel: 2D 8-bit structuring element for the morphological operation.

    :param size: Horizontal or vertical structuring element size for separable morphological operations.

    :param anchor: Anchor position within the structuring element; negative values mean that the anchor is at the center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`createMorphologyFilter`.



.. index:: gpu::erode

gpu::erode
--------------
.. cpp:function:: void gpu::erode(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1)

    Erodes an image by using a specific structuring element.

    :param src: Source image. Only ``CV_8UC1`` and ``CV_8UC4`` types are supported.

    :param dst: Destination image. It will have the same size and the same type as ``src``.

    :param kernel: Structuring element used for dilation. If ``kernel=Mat()``, a :math:`3 \times 3` rectangular structuring element is used.

    :param anchor: Position of the anchor within the element. The default value ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: Number of times erosion to be applied.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`erode`, :cpp:func:`gpu::createMorphologyFilter_GPU`.



.. index:: gpu::dilate

gpu::dilate
---------------
.. cpp:function:: void gpu::dilate(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1)

    Dilates an image by using a specific structuring element.

    :param src: Source image. Supports ``CV_8UC1`` and ``CV_8UC4`` source types.

    :param dst: Destination image. It will have the same size and the same type as ``src``.

    :param kernel: Structuring element used for dilation. If ``kernel=Mat()``, a :math:`3 \times 3` rectangular structuring element is used.

    :param anchor: Position of the anchor within the element. The default value ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: Number of times dilation to be applied.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`dilate`, :cpp:func:`gpu::createMorphologyFilter_GPU`.



.. index:: gpu::morphologyEx

gpu::morphologyEx
---------------------
.. cpp:function:: void gpu::morphologyEx(const GpuMat& src, GpuMat& dst, int op,  const Mat& kernel,  Point anchor = Point(-1, -1),  int iterations = 1)

    Applies an advanced morphological operation to image.

    :param src: Source image. Supports ``CV_8UC1`` and ``CV_8UC4`` source type.

    :param dst: Destination image. It will have the same size and the same type as ``src``.

    :param op: Type of morphological operation, one of the following:
        
            * **MORPH_OPEN** opening
            
            * **MORPH_CLOSE** closing
            
            * **MORPH_GRADIENT** morphological gradient
            
            * **MORPH_TOPHAT** "top hat"
            
            * **MORPH_BLACKHAT** "black hat"

    :param kernel: Structuring element.

    :param anchor: Position of the anchor within the element. The default value ``(-1, -1)`` means that the anchor is at the element center.

    :param iterations: Number of times erosion and dilation to be applied.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`morphologyEx`.



.. index:: gpu::createLinearFilter_GPU

gpu::createLinearFilter_GPU
-------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Point& anchor = Point(-1,-1))

.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Size& ksize, Point anchor = Point(-1, -1))

    Creates the non-separable linear filter.

    :param srcType: Input image type. Supports ``CV_8UC1`` and ``CV_8UC4``.

    :param dstType: Output image type. Supports only the same as source type.

    :param kernel: 2D array of filter coefficients. This filter works with integers kernels, if ``kernel`` has ``float`` or ``double`` type it will be used fixed point arithmetic.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``(-1, -1)`` means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`createLinearFilter`.



.. index:: gpu::filter2D

gpu::filter2D
-----------------
.. cpp:function:: void gpu::filter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernel, Point anchor=Point(-1,-1))

    Applies non-separable 2D linear filter to image.

    :param src: Source image. Supports ``CV_8UC1`` and ``CV_8UC4`` source types.

    :param dst: Destination image. It will have the same size and the same number of channels as ``src``.

    :param ddepth: The desired depth of the destination image. If it is negative, it will be the same as ``src.depth()``. Supports only the same depth as source image.

    :param kernel: 2D array of filter coefficients. This filter works with integers kernels, if ``kernel`` has ``float`` or ``double`` type it will use fixed point arithmetic.

    :param anchor: Anchor of the kernel that indicates the relative position of a filtered point within the kernel. The anchor should lie within the kernel. The special default value ``(-1,-1)`` means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`filter2D`, :cpp:func:`gpu::createLinearFilter_GPU`.



.. index:: gpu::Laplacian

gpu::Laplacian
------------------
.. cpp:function:: void gpu::Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize = 1, double scale = 1)

    Applies Laplacian operator to image.

    :param src: Source image. Supports ``CV_8UC1`` and ``CV_8UC4`` source types.

    :param dst: Destination image; will have the same size and the same number of channels as ``src``.

    :param ddepth: Desired depth of the destination image. Supports only tha same depth as source image depth.

    :param ksize: Aperture size used to compute the second-derivative filters, see :c:func:`getDerivKernels`. It must be positive and odd. Supports only ``ksize`` = 1 and ``ksize`` = 3.

    :param scale: Optional scale factor for the computed Laplacian values (by default, no scaling is applied, see  :c:func:`getDerivKernels`).

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also: :c:func:`Laplacian`, :cpp:func:`gpu::filter2D`.



.. index:: gpu::getLinearRowFilter_GPU

gpu::getLinearRowFilter_GPU
-------------------------------
.. cpp:function:: Ptr<BaseRowFilter_GPU> gpu::getLinearRowFilter_GPU(int srcType, int bufType, const Mat& rowKernel, int anchor = -1, int borderType = BORDER_CONSTANT)

    Creates primitive row filter with the specified kernel.

    :param srcType: Source array type. Supports only ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param bufType: Inermediate buffer type; must have as many channels as ``srcType``.

    :param rowKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel; negative values mean that anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method; see :c:func:`borderInterpolate`. About limitation see below.

There are two version of algorithm: NPP and OpenCV. NPP calls when ``srcType == CV_8UC1`` or ``srcType == CV_8UC4`` and ``bufType == srcType``, otherwise calls OpenCV version. NPP supports only ``BORDER_CONSTANT`` border type and doesn't check indices outside image. OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101``,``BORDER_REPLICATE`` and ``BORDER_CONSTANT`` border types and checks indices outside image.

See also: :cpp:func:`gpu::getLinearColumnFilter_GPU`, :c:func:`createSeparableLinearFilter`.



.. index:: gpu::getLinearColumnFilter_GPU

gpu::getLinearColumnFilter_GPU
----------------------------------
.. cpp:function:: Ptr<BaseColumnFilter_GPU> gpu::getLinearColumnFilter_GPU(int bufType, int dstType, const Mat& columnKernel, int anchor = -1, int borderType = BORDER_CONSTANT)

    Creates the primitive column filter with the specified kernel.

    :param bufType: Inermediate buffer type; must have as many channels as ``dstType``.

    :param dstType: Destination array type. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` destination types.

    :param columnKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel; negative values mean that anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method; see :c:func:`borderInterpolate`. About limitation see below.

There are two version of algorithm: NPP and OpenCV. NPP calls when ``dstType == CV_8UC1`` or ``dstType == CV_8UC4`` and ``bufType == dstType``, otherwise calls OpenCV version. NPP supports only ``BORDER_CONSTANT`` border type and doesn't check indices outside image. OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101``,``BORDER_REPLICATE`` and ``BORDER_CONSTANT`` border types and checks indices outside image.

See also: :cpp:func:`gpu::getLinearRowFilter_GPU`, :c:func:`createSeparableLinearFilter`.



.. index:: gpu::createSeparableLinearFilter_GPU

gpu::createSeparableLinearFilter_GPU
----------------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createSeparableLinearFilter_GPU(int srcType,  int dstType, const Mat& rowKernel, const Mat& columnKernel, const Point& anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Creates the separable linear filter engine.

    :param srcType: Source array type. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param dstType: Destination array type. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` destination types.

    :param rowKernel, columnKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel; negative values mean that anchor is positioned at the aperture center.

    :param rowBorderType, columnBorderType: Pixel extrapolation method in the horizontal and the vertical directions; see :c:func:`borderInterpolate`. About limitation see :cpp:func:`gpu::getLinearRowFilter_GPU`, cpp:func:`gpu::getLinearColumnFilter_GPU`.

See also: :cpp:func:`gpu::getLinearRowFilter_GPU`, :cpp:func:`gpu::getLinearColumnFilter_GPU`, :c:func:`createSeparableLinearFilter`.



.. index:: gpu::sepFilter2D

gpu::sepFilter2D
--------------------
.. cpp:function:: void gpu::sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY, Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Applies separable 2D linear filter to the image.

    :param src: Source image. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param dst: Destination image; will have the same size and the same number of channels as ``src``.

    :param ddepth: Destination image depth. Supports ``CV_8U``, ``CV_16S``, ``CV_32S`` and ``CV_32F``.

    :param kernelX, kernelY: Filter coefficients.

    :param anchor: Anchor position within the kernel; The default value ``(-1, 1)`` means that the anchor is at the kernel center.

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see :c:func:`borderInterpolate`.

See also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:func:`sepFilter2D`.



.. index:: gpu::createDerivFilter_GPU

gpu::createDerivFilter_GPU
------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Creates filter engine for the generalized Sobel operator.

    :param srcType: Source image type. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param dstType: Destination image type; must have as many channels as ``srcType``. Supports ``CV_8U``, ``CV_16S``, ``CV_32S`` and ``CV_32F`` depths.

    :param dx: Derivative order in respect with x.

    :param dy: Derivative order in respect with y.

    :param ksize: Aperture size; see :c:func:`getDerivKernels`.

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see :c:func:`borderInterpolate`.

See also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:func:`createDerivFilter`.



.. index:: gpu::Sobel

gpu::Sobel
--------------
.. cpp:function:: void gpu::Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Applies generalized Sobel operator to the image.

    :param src: Source image. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param dst: Destination image. Will have the same size and number of channels as source image.

    :param ddepth: Destination image depth. Supports ``CV_8U``, ``CV_16S``, ``CV_32S`` and ``CV_32F``.

    :param dx: Derivative order in respect with x.

    :param dy: Derivative order in respect with y.

    :param ksize: Size of the extended Sobel kernel, must be 1, 3, 5 or 7.

    :param scale: Optional scale factor for the computed derivative values (by default, no scaling is applied, see :c:func:`getDerivKernels`).

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see :c:func:`borderInterpolate`.

See also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:func:`Sobel`.



.. index:: gpu::Scharr

gpu::Scharr
---------------
.. cpp:function:: void gpu::Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, double scale = 1, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Calculates the first x- or y- image derivative using Scharr operator.

    :param src: Source image. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param dst: Destination image; will have the same size and the same number of channels as ``src``.

    :param ddepth: Destination image depth. Supports ``CV_8U``, ``CV_16S``, ``CV_32S`` and ``CV_32F``.

    :param xorder: Order of the derivative x.

    :param yorder: Order of the derivative y.

    :param scale: Optional scale factor for the computed derivative values (by default, no scaling is applied, see :c:func:`getDerivKernels`).

    :param rowBorderType, columnBorderType: Pixel extrapolation method, see :c:func:`borderInterpolate`.

See also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:func:`Scharr`.



.. index:: gpu::createGaussianFilter_GPU

gpu::createGaussianFilter_GPU
---------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createGaussianFilter_GPU(int type, Size ksize, double sigmaX, double sigmaY = 0, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Creates Gaussian filter engine.

    :param type: Source and the destination image type. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``.

    :param ksize: Aperture size; see :c:func:`getGaussianKernel`.

    :param sigmaX: Gaussian sigma in the horizontal direction; see :c:func:`getGaussianKernel`.

    :param sigmaY: Gaussian sigma in the vertical direction; if 0, then :math:`\texttt{sigmaY}\leftarrow\texttt{sigmaX}`.

    :param rowBorderType, columnBorderType: Which border type to use; see :c:func:`borderInterpolate`.

See also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:func:`createGaussianFilter`.



.. index:: gpu::GaussianBlur

gpu::GaussianBlur
---------------------
.. cpp:function:: void gpu::GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigmaX, double sigmaY = 0, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Smooths the image using Gaussian filter.

    :param src: Source image. Supports ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` source types.

    :param dst: Destination image; will have the same size and the same type as ``src``.

    :param ksize: Gaussian kernel size; ``ksize.width`` and ``ksize.height`` can differ, but they both must be positive and odd. Or, they can be zero's, then they are computed from ``sigmaX`` amd ``sigmaY``.

    :param sigmaX, sigmaY: Gaussian kernel standard deviations in X and Y direction. If ``sigmaY`` is zero, it is set to be equal to ``sigmaX``. If they are both zeros, they are computed from ``ksize.width`` and ``ksize.height``, respectively, see :c:func:`getGaussianKernel`. To fully control the result regardless of possible future modification of all this semantics, it is recommended to specify all of ``ksize``, ``sigmaX`` and ``sigmaY``.

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see :c:func:`borderInterpolate`.

See also: :cpp:func:`gpu::createGaussianFilter_GPU`, :c:func:`GaussianBlur`.



.. index:: gpu::getMaxFilter_GPU

gpu::getMaxFilter_GPU
-------------------------
.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getMaxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1))

    Creates maximum filter.

    :param srcType: Input image type. Supports only ``CV_8UC1`` and ``CV_8UC4``.

    :param dstType: Output image type. Supports only the same type as source.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.



.. index:: gpu::getMinFilter_GPU

gpu::getMinFilter_GPU
-------------------------
.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getMinFilter_GPU(int srcType, int dstType,  const Size& ksize, Point anchor = Point(-1,-1))

    Creates minimum filter.

    :param srcType: Input image type. Supports only ``CV_8UC1`` and ``CV_8UC4``.

    :param dstType: Output image type. Supports only the same type as source.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:** This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.
