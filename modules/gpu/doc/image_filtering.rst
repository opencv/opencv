Image Filtering
===============

.. highlight:: cpp

Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images.



gpu::BaseRowFilter_GPU
----------------------
.. ocv:class:: gpu::BaseRowFilter_GPU

Base class for linear or non-linear filters that processes rows of 2D arrays. Such filters are used for the "horizontal" filtering passes in separable filters. ::

    class BaseRowFilter_GPU
    {
    public:
        BaseRowFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseRowFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
        int ksize, anchor;
    };


.. note:: This class does not allocate memory for a destination image. Usually this class is used inside :ocv:class:`gpu::FilterEngine_GPU`.



gpu::BaseColumnFilter_GPU
-------------------------
.. ocv:class:: gpu::BaseColumnFilter_GPU

Base class for linear or non-linear filters that processes columns of 2D arrays. Such filters are used for the "vertical" filtering passes in separable filters. ::

    class BaseColumnFilter_GPU
    {
    public:
        BaseColumnFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseColumnFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
        int ksize, anchor;
    };


.. note:: This class does not allocate memory for a destination image. Usually this class is used inside :ocv:class:`gpu::FilterEngine_GPU`.



gpu::BaseFilter_GPU
-------------------
.. ocv:class:: gpu::BaseFilter_GPU

Base class for non-separable 2D filters. ::

    class CV_EXPORTS BaseFilter_GPU
    {
    public:
        BaseFilter_GPU(const Size& ksize_, const Point& anchor_);
        virtual ~BaseFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst, Stream& stream = Stream::Null()) = 0;
        Size ksize;
        Point anchor;
    };


.. note:: This class does not allocate memory for a destination image. Usually this class is used inside :ocv:class:`gpu::FilterEngine_GPU`.



gpu::FilterEngine_GPU
---------------------
.. ocv:class:: gpu::FilterEngine_GPU

Base class for the Filter Engine. ::

    class CV_EXPORTS FilterEngine_GPU
    {
    public:
        virtual ~FilterEngine_GPU() {}

        virtual void apply(const GpuMat& src, GpuMat& dst,
                           Rect roi = Rect(0,0,-1,-1), Stream& stream = Stream::Null()) = 0;
    };


The class can be used to apply an arbitrary filtering operation to an image. It contains all the necessary intermediate buffers. Pointers to the initialized ``FilterEngine_GPU`` instances are returned by various ``create*Filter_GPU`` functions (see below), and they are used inside high-level functions such as :ocv:func:`gpu::filter2D`, :ocv:func:`gpu::erode`, :ocv:func:`gpu::Sobel` , and others.

By using ``FilterEngine_GPU`` instead of functions you can avoid unnecessary memory allocation for intermediate buffers and get better performance: ::

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


``FilterEngine_GPU`` can process a rectangular sub-region of an image. By default, if ``roi == Rect(0,0,-1,-1)`` , ``FilterEngine_GPU`` processes the inner region of an image ( ``Rect(anchor.x, anchor.y, src_size.width - ksize.width, src_size.height - ksize.height)`` ) because some filters do not check whether indices are outside the image for better perfomance. See below to understand which filters support processing the whole image and which do not and identify image type limitations.

.. note:: The GPU filters do not support the in-place mode.

.. seealso:: :ocv:class:`gpu::BaseRowFilter_GPU`, :ocv:class:`gpu::BaseColumnFilter_GPU`, :ocv:class:`gpu::BaseFilter_GPU`, :ocv:func:`gpu::createFilter2D_GPU`, :ocv:func:`gpu::createSeparableFilter_GPU`, :ocv:func:`gpu::createBoxFilter_GPU`, :ocv:func:`gpu::createMorphologyFilter_GPU`, :ocv:func:`gpu::createLinearFilter_GPU`, :ocv:func:`gpu::createSeparableLinearFilter_GPU`, :ocv:func:`gpu::createDerivFilter_GPU`, :ocv:func:`gpu::createGaussianFilter_GPU`



gpu::createFilter2D_GPU
---------------------------
Creates a non-separable filter engine with the specified filter.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createFilter2D_GPU( const Ptr<BaseFilter_GPU>& filter2D, int srcType, int dstType)

    :param filter2D: Non-separable 2D filter.

    :param srcType: Input image type. It must be supported by  ``filter2D`` .

    :param dstType: Output image type. It must be supported by  ``filter2D`` .

Usually this function is used inside such high-level functions as :ocv:func:`gpu::createLinearFilter_GPU`, :ocv:func:`gpu::createBoxFilter_GPU`.



gpu::createSeparableFilter_GPU
----------------------------------
Creates a separable filter engine with the specified filters.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createSeparableFilter_GPU( const Ptr<BaseRowFilter_GPU>& rowFilter, const Ptr<BaseColumnFilter_GPU>& columnFilter, int srcType, int bufType, int dstType)

    :param rowFilter: "Horizontal" 1D filter.

    :param columnFilter: "Vertical" 1D filter.

    :param srcType: Input image type. It must be supported by  ``rowFilter`` .

    :param bufType: Buffer image type. It must be supported by  ``rowFilter``  and  ``columnFilter`` .

    :param dstType: Output image type. It must be supported by  ``columnFilter`` .

Usually this function is used inside such high-level functions as :ocv:func:`gpu::createSeparableLinearFilter_GPU`.



gpu::getRowSumFilter_GPU
----------------------------
Creates a horizontal 1D box filter.

.. ocv:function:: Ptr<BaseRowFilter_GPU> gpu::getRowSumFilter_GPU(int srcType, int sumType, int ksize, int anchor = -1)

    :param srcType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param sumType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.



gpu::getColumnSumFilter_GPU
-------------------------------
Creates a vertical 1D box filter.

.. ocv:function:: Ptr<BaseColumnFilter_GPU> gpu::getColumnSumFilter_GPU(int sumType, int dstType, int ksize, int anchor = -1)

    :param sumType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param dstType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.



gpu::createBoxFilter_GPU
----------------------------
Creates a normalized 2D box filter.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createBoxFilter_GPU(int srcType, int dstType, const Size& ksize, const Point& anchor = Point(-1,-1))

.. ocv:function:: Ptr<BaseFilter_GPU> getBoxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1, -1))

    :param srcType: Input image type supporting ``CV_8UC1`` and ``CV_8UC4`` .

    :param dstType: Output image type.  It supports only the same values as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`boxFilter`



gpu::boxFilter
------------------
Smooths the image using the normalized box filter.

.. ocv:function:: void gpu::boxFilter(const GpuMat& src, GpuMat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), Stream& stream = Stream::Null())

    :param src: Input image. ``CV_8UC1`` and ``CV_8UC4`` source types are supported.

    :param dst: Output image type. The size and type is the same as ``src`` .

    :param ddepth: Output image depth. If -1, the output image has the same depth as the input one. The only values allowed here are ``CV_8U`` and -1.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

    :param stream: Stream for the asynchronous version.

.. note::    This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`boxFilter`



gpu::blur
-------------
Acts as a synonym for the normalized box filter.

.. ocv:function:: void gpu::blur(const GpuMat& src, GpuMat& dst, Size ksize, Point anchor = Point(-1,-1), Stream& stream = Stream::Null())

    :param src: Input image.  ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Output image type with the same size and type as  ``src`` .

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

    :param stream: Stream for the asynchronous version.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`blur`, :ocv:func:`gpu::boxFilter`



gpu::createMorphologyFilter_GPU
-----------------------------------
Creates a 2D morphological filter.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1)

.. ocv:function:: Ptr<BaseFilter_GPU> getMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Size& ksize, Point anchor=Point(-1,-1))

    :param op: Morphology operation id. Only ``MORPH_ERODE`` and ``MORPH_DILATE`` are supported.

    :param type: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4``  are supported.

    :param kernel: 2D 8-bit structuring element for the morphological operation.

    :param size: Size of a horizontal or vertical structuring element used for separable morphological operations.

    :param anchor: Anchor position within the structuring element. Negative values mean that the anchor is at the center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`createMorphologyFilter`



gpu::erode
--------------
Erodes an image by using a specific structuring element.

.. ocv:function:: void gpu::erode(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null())

    :param src: Source image. Only  ``CV_8UC1``  and  ``CV_8UC4``  types are supported.

    :param dst: Destination image with the same size and type as  ``src`` .

    :param kernel: Structuring element used for erosion. If  ``kernel=Mat()``, a  3x3 rectangular structuring element is used.

    :param anchor: Position of an anchor within the element. The default value  ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: Number of times erosion to be applied.

    :param stream: Stream for the asynchronous version.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`erode`



gpu::dilate
---------------
Dilates an image by using a specific structuring element.

.. ocv:function:: void gpu::dilate(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8UC1`` and ``CV_8UC4`` source types are supported.

    :param dst: Destination image with the same size and type as ``src``.

    :param kernel: Structuring element used for dilation. If  ``kernel=Mat()``, a  3x3 rectangular structuring element is used.

    :param anchor: Position of an anchor within the element. The default value  ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: Number of times dilation to be applied.

    :param stream: Stream for the asynchronous version.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`dilate`



gpu::morphologyEx
---------------------
Applies an advanced morphological operation to an image.

.. ocv:function:: void gpu::morphologyEx(const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1, Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Destination image with the same size and type as  ``src`` .

    :param op: Type of morphological operation. The following types are possible:

        * **MORPH_OPEN** opening

        * **MORPH_CLOSE** closing

        * **MORPH_GRADIENT** morphological gradient

        * **MORPH_TOPHAT** "top hat"

        * **MORPH_BLACKHAT** "black hat"

    :param kernel: Structuring element.

    :param anchor: Position of an anchor within the element. The default value ``Point(-1, -1)`` means that the anchor is at the element center.

    :param iterations: Number of times erosion and dilation to be applied.

    :param stream: Stream for the asynchronous version.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`morphologyEx`



gpu::createLinearFilter_GPU
-------------------------------
Creates a non-separable linear filter.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Point& anchor = Point(-1,-1))

.. ocv:function:: Ptr<BaseFilter_GPU> gpu::getLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Size& ksize, Point anchor = Point(-1, -1))

    :param srcType: Input image type. ``CV_8UC1``  and  ``CV_8UC4`` types are supported.

    :param dstType: Output image type. The same type as ``src`` is supported.

    :param kernel: 2D array of filter coefficients. Floating-point coefficients will be converted to fixed-point representation before the actual processing.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`createLinearFilter`



gpu::filter2D
-----------------
Applies the non-separable 2D linear filter to an image.

.. ocv:function:: void gpu::filter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernel, Point anchor=Point(-1,-1), Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Destination image. The size and the number of channels is the same as  ``src`` .

    :param ddepth: Desired depth of the destination image. If it is negative, it is the same as  ``src.depth()`` . It supports only the same depth as the source image depth.

    :param kernel: 2D array of filter coefficients. This filter works with integers kernels. If  ``kernel``  has a ``float``  or  ``double``  type, it uses fixed-point arithmetic.

    :param anchor: Anchor of the kernel that indicates the relative position of a filtered point within the kernel. The anchor resides within the kernel. The special default value (-1,-1) means that the anchor is at the kernel center.

    :param stream: Stream for the asynchronous version.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`filter2D`



gpu::Laplacian
------------------
Applies the Laplacian operator to an image.

.. ocv:function:: void gpu::Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize = 1, double scale = 1, Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Destination image. The size and number of channels is the same as  ``src`` .

    :param ddepth: Desired depth of the destination image. It supports only the same depth as the source image depth.

    :param ksize: Aperture size used to compute the second-derivative filters (see :ocv:func:`getDerivKernels`). It must be positive and odd. Only  ``ksize``  = 1 and  ``ksize``  = 3 are supported.

    :param scale: Optional scale factor for the computed Laplacian values. By default, no scaling is applied (see  :ocv:func:`getDerivKernels` ).

    :param stream: Stream for the asynchronous version.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. seealso:: :ocv:func:`Laplacian`, :ocv:func:`gpu::filter2D`



gpu::getLinearRowFilter_GPU
-------------------------------
Creates a primitive row filter with the specified kernel.

.. ocv:function:: Ptr<BaseRowFilter_GPU> gpu::getLinearRowFilter_GPU(int srcType, int bufType, const Mat& rowKernel, int anchor = -1, int borderType = BORDER_CONSTANT)

    :param srcType: Source array type. Only  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param bufType: Intermediate buffer type with as many channels as  ``srcType`` .

    :param rowKernel: Filter coefficients. Support kernels with ``size <= 16`` .

    :param anchor: Anchor position within the kernel. Negative values mean that the anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method. For details, see :ocv:func:`borderInterpolate`. For details on limitations, see below.

There are two versions of the algorithm: NPP and OpenCV.

    * NPP version is called when ``srcType == CV_8UC1`` or ``srcType == CV_8UC4`` and ``bufType == srcType`` . Otherwise, the OpenCV version is called. NPP supports only ``BORDER_CONSTANT`` border type and does not check indices outside the image.

    * OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , and ``BORDER_CONSTANT`` border types. It checks indices outside the image.

.. seealso:: :ocv:func:`createSeparableLinearFilter` .



gpu::getLinearColumnFilter_GPU
----------------------------------
Creates a primitive column filter with the specified kernel.

.. ocv:function:: Ptr<BaseColumnFilter_GPU> gpu::getLinearColumnFilter_GPU(int bufType, int dstType, const Mat& columnKernel, int anchor = -1, int borderType = BORDER_CONSTANT)

    :param bufType: Inermediate buffer type with as many channels as  ``dstType`` .

    :param dstType: Destination array type. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1`` destination types are supported.

    :param columnKernel: Filter coefficients. Support kernels with ``size <= 16`` .

    :param anchor: Anchor position within the kernel. Negative values mean that the anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method. For details, see  :ocv:func:`borderInterpolate` . For details on limitations, see below.

There are two versions of the algorithm: NPP and OpenCV.

    * NPP version is called when ``dstType == CV_8UC1`` or ``dstType == CV_8UC4`` and ``bufType == dstType`` . Otherwise, the OpenCV version is called. NPP supports only ``BORDER_CONSTANT`` border type and does not check indices outside the image.

    * OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , and ``BORDER_CONSTANT`` border types. It checks indices outside image.

.. seealso:: :ocv:func:`gpu::getLinearRowFilter_GPU`, :ocv:func:`createSeparableLinearFilter`



gpu::createSeparableLinearFilter_GPU
----------------------------------------
Creates a separable linear filter engine.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat& rowKernel, const Mat& columnKernel, const Point& anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    :param srcType: Source array type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dstType: Destination array type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  destination types are supported.

    :param rowKernel: Horizontal filter coefficients. Support kernels with ``size <= 16`` .

    :param columnKernel: Vertical filter coefficients. Support kernels with ``size <= 16`` .

    :param anchor: Anchor position within the kernel. Negative values mean that anchor is positioned at the aperture center.

    :param rowBorderType: Pixel extrapolation method in the vertical direction For details, see  :ocv:func:`borderInterpolate`. For details on limitations, see :ocv:func:`gpu::getLinearRowFilter_GPU`, cpp:ocv:func:`gpu::getLinearColumnFilter_GPU`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`gpu::getLinearRowFilter_GPU`, :ocv:func:`gpu::getLinearColumnFilter_GPU`, :ocv:func:`createSeparableLinearFilter`



gpu::sepFilter2D
--------------------
Applies a separable 2D linear filter to an image.

.. ocv:function:: void gpu::sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY, Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as  ``src`` .

    :param ddepth: Destination image depth.  ``CV_8U`` , ``CV_16S`` , ``CV_32S`` , and  ``CV_32F`` are supported.

    :param kernelX: Horizontal filter coefficients.

    :param kernelY: Vertical filter coefficients.

    :param anchor: Anchor position within the kernel. The default value ``(-1, 1)`` means that the anchor is at the kernel center.

    :param rowBorderType: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::createSeparableLinearFilter_GPU`, :ocv:func:`sepFilter2D`



gpu::createDerivFilter_GPU
------------------------------
Creates a filter engine for the generalized Sobel operator.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    :param srcType: Source image type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dstType: Destination image type with as many channels as  ``srcType`` ,  ``CV_8U`` , ``CV_16S`` , ``CV_32S`` , and  ``CV_32F``  depths are supported.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Aperture size. See  :ocv:func:`getDerivKernels` for details.

    :param rowBorderType: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`gpu::createSeparableLinearFilter_GPU`, :ocv:func:`createDerivFilter`



gpu::Sobel
--------------
Applies the generalized Sobel operator to an image.

.. ocv:function:: void gpu::Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as source image.

    :param ddepth: Destination image depth.  ``CV_8U`` , ``CV_16S`` , ``CV_32S`` , and  ``CV_32F`` are supported.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Size of the extended Sobel kernel. Possible valies are 1, 3, 5 or 7.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. For details, see  :ocv:func:`getDerivKernels` .

    :param rowBorderType: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::createSeparableLinearFilter_GPU`, :ocv:func:`Sobel`



gpu::Scharr
---------------
Calculates the first x- or y- image derivative using the Scharr operator.

.. ocv:function:: void gpu::Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, double scale = 1, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as  ``src`` has.

    :param ddepth: Destination image depth.  ``CV_8U`` , ``CV_16S`` , ``CV_32S`` , and  ``CV_32F`` are supported.

    :param xorder: Order of the derivative x.

    :param yorder: Order of the derivative y.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. See  :ocv:func:`getDerivKernels`  for details.

    :param rowBorderType: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::createSeparableLinearFilter_GPU`, :ocv:func:`Scharr`



gpu::createGaussianFilter_GPU
---------------------------------
Creates a Gaussian filter engine.

.. ocv:function:: Ptr<FilterEngine_GPU> gpu::createGaussianFilter_GPU(int type, Size ksize, double sigmaX, double sigmaY = 0, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    :param type: Source and destination image type.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1`` are supported.

    :param ksize: Aperture size. See  :ocv:func:`getGaussianKernel` for details.

    :param sigmaX: Gaussian sigma in the horizontal direction. See  :ocv:func:`getGaussianKernel` for details.

    :param sigmaY: Gaussian sigma in the vertical direction. If 0, then  :math:`\texttt{sigmaY}\leftarrow\texttt{sigmaX}` .

    :param rowBorderType: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

.. seealso:: :ocv:func:`gpu::createSeparableLinearFilter_GPU`, :ocv:func:`createGaussianFilter`



gpu::GaussianBlur
---------------------
Smooths an image using the Gaussian filter.

.. ocv:function:: void gpu::GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigmaX, double sigmaY = 0, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1, Stream& stream = Stream::Null())

    :param src: Source image.  ``CV_8UC1`` , ``CV_8UC4`` , ``CV_16SC1`` , ``CV_16SC2`` , ``CV_16SC3`` , ``CV_32SC1`` , ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and type as  ``src`` .

    :param ksize: Gaussian kernel size.  ``ksize.width``  and  ``ksize.height``  can differ but they both must be positive and odd. If they are zeros, they are computed from  ``sigmaX``  and  ``sigmaY`` .

    :param sigmaX: Gaussian kernel standard deviation in X direction.

    :param sigmaY: Gaussian kernel standard deviation in Y direction. If  ``sigmaY``  is zero, it is set to be equal to  ``sigmaX`` . If they are both zeros, they are computed from  ``ksize.width``  and  ``ksize.height``, respectively. See  :ocv:func:`getGaussianKernel` for details. To fully control the result regardless of possible future modification of all this semantics, you are recommended to specify all of  ``ksize`` , ``sigmaX`` , and  ``sigmaY`` .

    :param rowBorderType: Pixel extrapolation method in the vertical direction. For details, see  :ocv:func:`borderInterpolate`.

    :param columnBorderType: Pixel extrapolation method in the horizontal direction.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`gpu::createGaussianFilter_GPU`, :ocv:func:`GaussianBlur`



gpu::getMaxFilter_GPU
-------------------------
Creates the maximum filter.

.. ocv:function:: Ptr<BaseFilter_GPU> gpu::getMaxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1))

    :param srcType: Input image type. Only  ``CV_8UC1``  and  ``CV_8UC4`` are supported.

    :param dstType: Output image type. It supports only the same type as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.



gpu::getMinFilter_GPU
-------------------------
Creates the minimum filter.

.. ocv:function:: Ptr<BaseFilter_GPU> gpu::getMinFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1))

    :param srcType: Input image type. Only  ``CV_8UC1``  and  ``CV_8UC4`` are supported.

    :param dstType: Output image type. It supports only the same type as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

.. note:: This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.
