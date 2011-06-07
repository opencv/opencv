Image Filtering
===============

.. highlight:: cpp

Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images.

See also: :ref:`ImageFiltering`.

.. index:: gpu::BaseRowFilter_GPU

gpu::BaseRowFilter_GPU
----------------------
.. cpp:class:: gpu::BaseRowFilter_GPU

This is a base class for linear or non-linear filters that processes rows of 2D arrays. Such filters are used for the "horizontal" filtering passes in separable filters. ::

    class BaseRowFilter_GPU
    {
    public:
        BaseRowFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseRowFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        int ksize, anchor;
    };


**Note:** 

This class does not allocate memory for a destination image. Usually this class is used inside :cpp:class:`gpu::FilterEngine_GPU`.

.. index:: gpu::BaseColumnFilter_GPU

gpu::BaseColumnFilter_GPU
-------------------------
.. cpp:class:: gpu::BaseColumnFilter_GPU

This is a base class for linear or non-linear filters that processes columns of 2D arrays. Such filters are used for the "vertical" filtering passes in separable filters. ::

    class BaseColumnFilter_GPU
    {
    public:
        BaseColumnFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseColumnFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        int ksize, anchor;
    };


**Note:**

This class does not allocate memory for a destination image. Usually this class is used inside :cpp:class:`gpu::FilterEngine_GPU`.

.. index:: gpu::BaseFilter_GPU

gpu::BaseFilter_GPU
-------------------
.. cpp:class:: gpu::BaseFilter_GPU

This is a base class for non-separable 2D filters. ::

    class CV_EXPORTS BaseFilter_GPU
    {
    public:
        BaseFilter_GPU(const Size& ksize_, const Point& anchor_);
        virtual ~BaseFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        Size ksize;
        Point anchor;
    };


**Note:**

This class does not allocate memory for a destination image. Usually this class is used inside :cpp:class:`gpu::FilterEngine_GPU`.

.. index:: gpu::FilterEngine_GPU

gpu::FilterEngine_GPU
---------------------
.. cpp:class:: gpu::FilterEngine_GPU

This is a base class for Filter Engine. ::

    class CV_EXPORTS FilterEngine_GPU
    {
    public:
        virtual ~FilterEngine_GPU() {}

        virtual void apply(const GpuMat& src, GpuMat& dst,
                           Rect roi = Rect(0,0,-1,-1)) = 0;
    };


The class can be used to apply an arbitrary filtering operation to an image. It contains all the necessary intermediate buffers. Pointers to the initialized ``FilterEngine_GPU`` instances are returned by various ``create*Filter_GPU`` functions (see below), and they are used inside high-level functions such as
:cpp:func:`gpu::filter2D`, :cpp:func:`gpu::erode`, :cpp:func:`gpu::Sobel` , and others.

By using ``FilterEngine_GPU`` instead of functions you can avoid unnecessary memory allocation for intermediate buffers and get much better performance: 
::

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

 ``FilterEngine_GPU`` can process a rectangular sub-region of an image. By default, if ``roi == Rect(0,0,-1,-1)``, ``FilterEngine_GPU`` processes the inner region of an image ( ``Rect(anchor.x, anchor.y, src_size.width - ksize.width, src_size.height - ksize.height)`` ), because some filters do not check whether indices are outside the image for better perfomance. See below to understand which filters support processing the whole image and which do not and identify image type limitations.

**Note:** 

The GPU filters do not support the in-place mode.

See also: :cpp:class:`gpu::BaseRowFilter_GPU`, :cpp:class:`gpu::BaseColumnFilter_GPU`, :cpp:class:`gpu::BaseFilter_GPU`, :cpp:func:`gpu::createFilter2D_GPU`, :cpp:func:`gpu::createSeparableFilter_GPU`, :cpp:func:`gpu::createBoxFilter_GPU`, :cpp:func:`gpu::createMorphologyFilter_GPU`, :cpp:func:`gpu::createLinearFilter_GPU`, :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :cpp:func:`gpu::createDerivFilter_GPU`, :cpp:func:`gpu::createGaussianFilter_GPU`.

.. index:: gpu::createFilter2D_GPU

gpu::createFilter2D_GPU
---------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createFilter2D_GPU( const Ptr<BaseFilter_GPU>& filter2D, int srcType, int dstType)

    Creates a non-separable filter engine with the specified filter.

    :param filter2D: Non-separable 2D filter.

    :param srcType: Input image type. It must be supported by  ``filter2D`` .

    :param dstType: Output image type. It must be supported by  ``filter2D`` .

	Usually this function is used inside such high-level functions as :cpp:func:`gpu::createLinearFilter_GPU`, :cpp:func:`gpu::createBoxFilter_GPU`.

.. index:: gpu::createSeparableFilter_GPU

gpu::createSeparableFilter_GPU
----------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createSeparableFilter_GPU( const Ptr<BaseRowFilter_GPU>& rowFilter, const Ptr<BaseColumnFilter_GPU>& columnFilter, int srcType, int bufType, int dstType)

    Creates a separable filter engine with the specified filters.

    :param rowFilter: "Horizontal" 1D filter.
    
    :param columnFilter: "Vertical" 1D filter.

    :param srcType: Input image type. It must be supported by  ``rowFilter``.

    :param bufType: Buffer image type. It must be supported by  ``rowFilter``  and  ``columnFilter``.

    :param dstType: Output image type. It must be supported by  ``columnFilter``.

	Usually this function is used inside such high-level functions as :cpp:func:`gpu::createSeparableLinearFilter_GPU`.

.. index:: gpu::getRowSumFilter_GPU

gpu::getRowSumFilter_GPU
----------------------------
.. cpp:function:: Ptr<BaseRowFilter_GPU> gpu::getRowSumFilter_GPU(int srcType, int sumType, int ksize, int anchor = -1)

    Creates a horizontal 1D box filter.

    :param srcType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param sumType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. index:: gpu::getColumnSumFilter_GPU

gpu::getColumnSumFilter_GPU
-------------------------------
.. cpp:function:: Ptr<BaseColumnFilter_GPU> gpu::getColumnSumFilter_GPU(int sumType, int dstType, int ksize, int anchor = -1)

    Creates a vertical 1D box filter.

    :param sumType: Input image type. Only ``CV_8UC1`` type is supported for now.

    :param dstType: Output image type. Only ``CV_32FC1`` type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. index:: gpu::createBoxFilter_GPU

gpu::createBoxFilter_GPU
----------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createBoxFilter_GPU(int srcType, int dstType, const Size& ksize, const Point& anchor = Point(-1,-1))

    Creates a normalized 2D box filter.

.. cpp:function:: Ptr<BaseFilter_GPU> getBoxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1, -1))

    :param srcType: Input image type. Supports ``CV_8UC1`` and ``CV_8UC4``.

    :param dstType: Output image type.  It supports only the same as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`boxFilter`

.. index:: gpu::boxFilter

gpu::boxFilter
------------------
.. cpp:function:: void gpu::boxFilter(const GpuMat& src, GpuMat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1))

    Smooths the image using the normalized box filter.

    :param src: Input image. ``CV_8UC1`` and ``CV_8UC4`` source types are supported.

    :param dst: Output image type. The size and type is the same as ``src``.

    :param ddepth: Output image depth. If -1, the output image has the same depth as the input one. The only values allowed here are ``CV_8U`` and -1.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value ``Point(-1, -1)`` means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`boxFilter`

.. index:: gpu::blur

gpu::blur
-------------
.. cpp:function:: void gpu::blur(const GpuMat& src, GpuMat& dst, Size ksize, Point anchor = Point(-1,-1))

    Acts as a synonym for the normalized box filter.

    :param src: Input image.  ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Output image type with the same size and type as  ``src`` .

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`blur`, :cpp:func:`gpu::boxFilter`

.. index:: gpu::createMorphologyFilter_GPU

gpu::createMorphologyFilter_GPU
-----------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Point& anchor = Point(-1,-1), int iterations = 1)

    Creates a 2D morphological filter.

.. cpp:function:: Ptr<BaseFilter_GPU> getMorphologyFilter_GPU(int op, int type, const Mat& kernel, const Size& ksize, Point anchor=Point(-1,-1))

    {Morphology operation id. Only ``MORPH_ERODE``     and ``MORPH_DILATE``     are supported.}

    :param type: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4``  are supported.

    :param kernel: 2D 8-bit structuring element for the morphological operation.

    :param size: Size of a horizontal or vertical structuring element used for separable morphological operations.

    :param anchor: Anchor position within the structuring element. Negative values mean that the anchor is at the center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`createMorphologyFilter`

.. index:: gpu::erode

gpu::erode
--------------
.. cpp:function:: void gpu::erode(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1)

    Erodes an image by using a specific structuring element.

    :param src: Source image. Only  ``CV_8UC1``  and  ``CV_8UC4``  types are supported.

    :param dst: Destination image with the same size and type as  ``src`` .

    :param kernel: Structuring element used for erosion. If  ``kernel=Mat()``, a  3x3 rectangular structuring element is used.

    :param anchor: Position of an anchor within the element. The default value  ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: Number of times erosion to be applied.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`erode`

.. index:: gpu::dilate

gpu::dilate
---------------
.. cpp:function:: void gpu::dilate(const GpuMat& src, GpuMat& dst, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1)

    Dilates an image by using a specific structuring element.

    :param src: Source image. ``CV_8UC1`` and ``CV_8UC4`` source types are supported.

    :param dst: Destination image with the same size and type as ``src``.

    :param kernel: Structuring element used for dilation. If  ``kernel=Mat()``, a  3x3 rectangular structuring element is used.

    :param anchor: Position of an anchor within the element. The default value  ``(-1, -1)``  means that the anchor is at the element center.

    :param iterations: Number of times dilation to be applied.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`dilate`

.. index:: gpu::morphologyEx

gpu::morphologyEx
---------------------
.. cpp:function:: void gpu::morphologyEx(const GpuMat& src, GpuMat& dst, int op, const Mat& kernel, Point anchor = Point(-1, -1), int iterations = 1)

    Applies an advanced morphological operation to an image.

    :param src: Source image.  ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Destination image with the same size and type as  ``src``
    
    :param op: Type of morphological operation. The following types are possible:
        
        * **MORPH_OPEN** opening
            
        * **MORPH_CLOSE** closing
            
        * **MORPH_GRADIENT** morphological gradient
            
        * **MORPH_TOPHAT** "top hat"
            
        * **MORPH_BLACKHAT** "black hat"
            

    :param kernel: Structuring element.

    :param anchor: Position of an anchor within the element. The default value ``Point(-1, -1)`` means that the anchor is at the element center.

    :param iterations: Number of times erosion and dilation to be applied.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`morphologyEx` 

.. index:: gpu::createLinearFilter_GPU

gpu::createLinearFilter_GPU
-------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Point& anchor = Point(-1,-1))

    Creates a non-separable linear filter.

.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getLinearFilter_GPU(int srcType, int dstType, const Mat& kernel, const Size& ksize, Point anchor = Point(-1, -1))

    :param srcType: Input image type. ``CV_8UC1``  and  ``CV_8UC4`` types are supported.

    :param dstType: Output image type. The same type as ``src`` is supported.

    :param kernel: 2D array of filter coefficients. Floating-point coefficients will be converted to fixed-point representation before the actual processing.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`createLinearFilter`

.. index:: gpu::filter2D

gpu::filter2D
-----------------
.. cpp:function:: void gpu::filter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernel, Point anchor=Point(-1,-1))

    Applies the non-separable 2D linear filter to an image.

    :param src: Source image.  ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Destination image. The size and the number of channels is the same as  ``src`` .

    :param ddepth: Desired depth of the destination image. If it is negative, it is the same as  ``src.depth()`` . It supports only the same depth as the source image depth.

    :param kernel: 2D array of filter coefficients. This filter works with integers kernels. If  ``kernel``  has a ``float``  or  ``double``  type, it uses fixed-point arithmetic.

    :param anchor: Anchor of the kernel that indicates the relative position of a filtered point within the kernel. The anchor resides within the kernel. The special default value (-1,-1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`filter2D`

.. index:: gpu::Laplacian

gpu::Laplacian
------------------
.. cpp:function:: void gpu::Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize = 1, double scale = 1)

    Applies the Laplacian operator to an image.

    :param src: Source image. ``CV_8UC1``  and  ``CV_8UC4``  source types are supported.

    :param dst: Destination image. The size and number of channels is the same as  ``src`` .

    :param ddepth: Desired depth of the destination image. It supports only the same depth as the source image depth.

    :param ksize: Aperture size used to compute the second-derivative filters (see :c:cpp:func:`getDerivKernels`). It must be positive and odd. Only  ``ksize``  = 1 and  ``ksize``  = 3 are supported.

    :param scale: Optional scale factor for the computed Laplacian values. By default, no scaling is applied (see  :c:cpp:func:`getDerivKernels` ).

	**Note:**
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

See Also: :c:cpp:func:`Laplacian`,:cpp:func:`gpu::filter2D` .

.. index:: gpu::getLinearRowFilter_GPU

gpu::getLinearRowFilter_GPU
-------------------------------
.. cpp:function:: Ptr<BaseRowFilter_GPU> gpu::getLinearRowFilter_GPU(int srcType, int bufType, const Mat& rowKernel, int anchor = -1, int borderType = BORDER_CONSTANT)

    Creates a primitive row filter with the specified kernel.

    :param srcType: Source array type. Only  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param bufType: Intermediate buffer type with as many channels as  ``srcType`` .

    :param rowKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel. Negative values mean that the anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method. For details, see :c:cpp:func:`borderInterpolate`. For details on limitations, see below.

	There are two versions of the algorithm: NPP and OpenCV.
	* NPP version is called when ``srcType == CV_8UC1`` or ``srcType == CV_8UC4`` and ``bufType == srcType`` . Otherwise, the OpenCV version is called. NPP supports only ``BORDER_CONSTANT`` border type and does not check indices outside the image. 
	* OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101``,``BORDER_REPLICATE``, and ``BORDER_CONSTANT`` border types. It checks indices outside the image.

See Also:,:cpp:func:`createSeparableLinearFilter` .

.. index:: gpu::getLinearColumnFilter_GPU

gpu::getLinearColumnFilter_GPU
----------------------------------
.. cpp:function:: Ptr<BaseColumnFilter_GPU> gpu::getLinearColumnFilter_GPU(int bufType, int dstType, const Mat& columnKernel, int anchor = -1, int borderType = BORDER_CONSTANT)

    Creates a primitive column filter with the specified kernel.

    :param bufType: Inermediate buffer type with as many channels as  ``dstType`` .

    :param dstType: Destination array type. ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` destination types are supported.

    :param columnKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel. Negative values mean that the anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method. For details, see  :c:cpp:func:`borderInterpolate` . For details on limitations, see below.

	There are two versions of the algorithm: NPP and OpenCV.
	* NPP version is called when ``dstType == CV_8UC1`` or ``dstType == CV_8UC4`` and ``bufType == dstType`` . Otherwise, the OpenCV version is called. NPP supports only ``BORDER_CONSTANT`` border type and does not check indices outside the image. 
	* OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101``, ``BORDER_REPLICATE``, and ``BORDER_CONSTANT`` border types. It checks indices outside image.
	
See Also: :cpp:func:`gpu::getLinearRowFilter_GPU`, :c:cpp:func:`createSeparableLinearFilter`

.. index:: gpu::createSeparableLinearFilter_GPU

gpu::createSeparableLinearFilter_GPU
----------------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createSeparableLinearFilter_GPU(int srcType, int dstType, const Mat& rowKernel, const Mat& columnKernel, const Point& anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Creates a separable linear filter engine.

    :param srcType: Source array type.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param dstType: Destination array type.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  destination types are supported.

    :param rowKernel, columnKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel. Negative values mean that anchor is positioned at the aperture center.

    :param rowBorderType, columnBorderType: Pixel extrapolation method in the horizontal and vertical directions For details, see  :c:cpp:func:`borderInterpolate`. For details on limitations, see :cpp:func:`gpu::getLinearRowFilter_GPU`, cpp:cpp:func:`gpu::getLinearColumnFilter_GPU`.


See Also: :cpp:func:`gpu::getLinearRowFilter_GPU`, :cpp:func:`gpu::getLinearColumnFilter_GPU`, :c:cpp:func:`createSeparableLinearFilter`

.. index:: gpu::sepFilter2D

gpu::sepFilter2D
--------------------
.. cpp:function:: void gpu::sepFilter2D(const GpuMat& src, GpuMat& dst, int ddepth, const Mat& kernelX, const Mat& kernelY, Point anchor = Point(-1,-1), int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Applies a separable 2D linear filter to an image.

    :param src: Source image.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as  ``src`` .

    :param ddepth: Destination image depth.  ``CV_8U``, ``CV_16S``, ``CV_32S``, and  ``CV_32F`` are supported.

    :param kernelX, kernelY: Filter coefficients.

    :param anchor: Anchor position within the kernel. The default value ``(-1, 1)`` means that the anchor is at the kernel center.

    :param rowBorderType, columnBorderType: Pixel extrapolation method. For details, see  :c:cpp:func:`borderInterpolate`.

See Also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:cpp:func:`sepFilter2D`

.. index:: gpu::createDerivFilter_GPU

gpu::createDerivFilter_GPU
------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> createDerivFilter_GPU(int srcType, int dstType, int dx, int dy, int ksize, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Creates a filter engine for the generalized Sobel operator.

    :param srcType: Source image type.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param dstType: Destination image type with as many channels as  ``srcType`` .  ``CV_8U``, ``CV_16S``, ``CV_32S``, and  ``CV_32F``  depths are supported.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Aperture size. See  :c:cpp:func:`getDerivKernels` for details.

    :param rowBorderType, columnBorderType: Pixel extrapolation method. See  :c:cpp:func:`borderInterpolate` for details.

See Also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:cpp:func:`createDerivFilter`

.. index:: gpu::Sobel

gpu::Sobel
--------------
.. cpp:function:: void gpu::Sobel(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Applies the generalized Sobel operator to an image.

    :param src: Source image.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as source image.

    :param ddepth: Destination image depth.  ``CV_8U``, ``CV_16S``, ``CV_32S``, and  ``CV_32F`` are supported.

    :param dx: Derivative order in respect of x.

    :param dy: Derivative order in respect of y.

    :param ksize: Size of the extended Sobel kernel. Possible valies are 1, 3, 5 or 7.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. For details, see  :c:cpp:func:`getDerivKernels` .

    :param rowBorderType, columnBorderType: Pixel extrapolation method. See  :c:cpp:func:`borderInterpolate` for details.

See Also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:cpp:func:`Sobel`

.. index:: gpu::Scharr

gpu::Scharr
---------------
.. cpp:function:: void gpu::Scharr(const GpuMat& src, GpuMat& dst, int ddepth, int dx, int dy, double scale = 1, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Calculates the first x- or y- image derivative using the Scharr operator.

    :param src: Source image.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and number of channels as  ``src`` has.

    :param ddepth: Destination image depth.  ``CV_8U``, ``CV_16S``, ``CV_32S``, and  ``CV_32F`` are supported.

    :param xorder: Order of the derivative x.

    :param yorder: Order of the derivative y.

    :param scale: Optional scale factor for the computed derivative values. By default, no scaling is applied. See  :c:cpp:func:`getDerivKernels`  for details.

    :param rowBorderType, columnBorderType: Pixel extrapolation method. For details, see  :c:cpp:func:`borderInterpolate`  and :c:cpp:func:`Scharr` .

See Also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:cpp:func:`Scharr`

.. index:: gpu::createGaussianFilter_GPU

gpu::createGaussianFilter_GPU
---------------------------------
.. cpp:function:: Ptr<FilterEngine_GPU> gpu::createGaussianFilter_GPU(int type, Size ksize, double sigmaX, double sigmaY = 0, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Creates a Gaussian filter engine.

    :param type: Source and destination image type.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1`` are supported.

    :param ksize: Aperture size. See  :c:cpp:func:`getGaussianKernel` for details.

    :param sigmaX: Gaussian sigma in the horizontal direction. See  :c:cpp:func:`getGaussianKernel` for details.

    :param sigmaY: Gaussian sigma in the vertical direction. If 0, then  :math:`\texttt{sigmaY}\leftarrow\texttt{sigmaX}` .

    :param rowBorderType, columnBorderType: Border type to use. See  :c:cpp:func:`borderInterpolate` for details.

See Also: :cpp:func:`gpu::createSeparableLinearFilter_GPU`, :c:cpp:func:`createGaussianFilter`

.. index:: gpu::GaussianBlur

gpu::GaussianBlur
---------------------
.. cpp:function:: void gpu::GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigmaX, double sigmaY = 0, int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1)

    Smooths an image using the Gaussian filter.

    :param src: Source image.  ``CV_8UC1``, ``CV_8UC4``, ``CV_16SC1``, ``CV_16SC2``, ``CV_32SC1``, ``CV_32FC1``  source types are supported.

    :param dst: Destination image with the same size and type as  ``src``.

    :param ksize: Gaussian kernel size.  ``ksize.width``  and  ``ksize.height``  can differ but they both must be positive and odd. If they are zeros, they are computed from  ``sigmaX``  and  ``sigmaY`` .

    :param sigmaX, sigmaY: Gaussian kernel standard deviations in X and Y direction. If  ``sigmaY``  is zero, it is set to be equal to  ``sigmaX`` . If they are both zeros, they are computed from  ``ksize.width``  and  ``ksize.height``, respectively. See  :c:cpp:func:`getGaussianKernel` for details. To fully control the result regardless of possible future modification of all this semantics, you are recommended to specify all of  ``ksize``, ``sigmaX``, and  ``sigmaY`` .

    :param rowBorderType, columnBorderType: Pixel extrapolation method. See  :c:cpp:func:`borderInterpolate` for details.

See Also: :cpp:func:`gpu::createGaussianFilter_GPU`, :c:cpp:func:`GaussianBlur`

.. index:: gpu::getMaxFilter_GPU

gpu::getMaxFilter_GPU
-------------------------
.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getMaxFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1))

    Creates the maximum filter.

    :param srcType: Input image type. Only  ``CV_8UC1``  and  ``CV_8UC4`` are supported.

    :param dstType: Output image type. It supports only the same type as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.

.. index:: gpu::getMinFilter_GPU

gpu::getMinFilter_GPU
-------------------------
.. cpp:function:: Ptr<BaseFilter_GPU> gpu::getMinFilter_GPU(int srcType, int dstType, const Size& ksize, Point anchor = Point(-1,-1))

    Creates the minimum filter.

    :param srcType: Input image type. Only  ``CV_8UC1``  and  ``CV_8UC4`` are supported.

    :param dstType: Output image type. It supports only the same type as the source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

	**Note:** 
	
	This filter does not check out-of-border accesses, so only a proper sub-matrix of a bigger matrix has to be passed to it.
