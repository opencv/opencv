Image Filtering
===============

.. highlight:: cpp

Functions and classes described in this section are used to perform various linear or non-linear filtering operations on 2D images.

See also:

.. index:: gpu::BaseRowFilter_GPU

.. _gpu::BaseRowFilter_GPU:

gpu::BaseRowFilter_GPU
----------------------
.. ctype:: gpu::BaseRowFilter_GPU

The base class for linear or non-linear filters that processes rows of 2D arrays. Such filters are used for the "horizontal" filtering passes in separable filters. ::

    class BaseRowFilter_GPU
    {
    public:
        BaseRowFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseRowFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        int ksize, anchor;
    };
..

**Please note:**
This class doesn't allocate memory for destination image. Usually this class is used inside
.

.. index:: gpu::BaseColumnFilter_GPU

.. _gpu::BaseColumnFilter_GPU:

gpu::BaseColumnFilter_GPU
-------------------------
.. ctype:: gpu::BaseColumnFilter_GPU

The base class for linear or non-linear filters that processes columns of 2D arrays. Such filters are used for the "vertical" filtering passes in separable filters. ::

    class BaseColumnFilter_GPU
    {
    public:
        BaseColumnFilter_GPU(int ksize_, int anchor_);
        virtual ~BaseColumnFilter_GPU() {}
        virtual void operator()(const GpuMat& src, GpuMat& dst) = 0;
        int ksize, anchor;
    };
..

**Please note:**
This class doesn't allocate memory for destination image. Usually this class is used inside
.

.. index:: gpu::BaseFilter_GPU

.. _gpu::BaseFilter_GPU:

gpu::BaseFilter_GPU
-------------------
.. ctype:: gpu::BaseFilter_GPU

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
..

**Please note:**
This class doesn't allocate memory for destination image. Usually this class is used inside
.

.. index:: gpu::FilterEngine_GPU

.. _gpu::FilterEngine_GPU:

gpu::FilterEngine_GPU
---------------------
.. ctype:: gpu::FilterEngine_GPU

The base class for Filter Engine. ::

    class CV_EXPORTS FilterEngine_GPU
    {
    public:
        virtual ~FilterEngine_GPU() {}

        virtual void apply(const GpuMat& src, GpuMat& dst,
                           Rect roi = Rect(0,0,-1,-1)) = 0;
    };
..

The class can be used to apply an arbitrary filtering operation to an image. It contains all the necessary intermediate buffers. Pointers to the initialized ``FilterEngine_GPU`` instances are returned by various ``create*Filter_GPU`` functions, see below, and they are used inside high-level functions such as
:func:`gpu::filter2D`,:func:`gpu::erode`,:func:`gpu::Sobel` etc.

By using ``FilterEngine_GPU`` instead of functions you can avoid unnecessary memory allocation for intermediate buffers and get much better performance: ::

    while (...)
    {
        cv::gpu::GpuMat src = getImg();
        cv::gpu::GpuMat dst;
        // Allocate and release buffers at each iterations
        cv::gpu::GaussianBlur(src, dst, ksize, sigma1);
    }

    // Allocate buffers only once
    cv::Ptr<cv::gpu::FilterEngine_GPU> filter =
        cv::gpu::createGaussianFilter_GPU(CV_8UC4, ksize, sigma1);
    while (...)
    {
        cv::gpu::GpuMat src = getImg();
        cv::gpu::GpuMat dst;
        filter->apply(src, dst, cv::Rect(0, 0, src.cols, src.rows));
    }
    // Release buffers only once
    filter.release();
..
 ``FilterEngine_GPU`` can process a rectangular sub-region of an image. By default, if ``roi == Rect(0,0,-1,-1)``,``FilterEngine_GPU`` processes inner region of image ( ``Rect(anchor.x, anchor.y, src_size.width - ksize.width, src_size.height - ksize.height)`` ), because some filters doesn't check if indices are outside the image for better perfomace. See below which filters supports processing the whole image and which not and image type limitations.

**Please note:**
The GPU filters doesn't support the in-place mode.

See also:,,,,,,,,,,

.. index:: cv::gpu::createFilter2D_GPU

.. _cv::gpu::createFilter2D_GPU:

cv::gpu::createFilter2D_GPU
---------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createFilter2D_GPU( const Ptr<BaseFilter_GPU>\& filter2D,  int srcType, int dstType)

    Creates non-separable filter engine with the specified filter.

    {Non-separable 2D filter.}

    :param srcType: Input image type. It must be supported by  ``filter2D`` .

    :param dstType: Output image type. It must be supported by  ``filter2D`` .

Usually this function is used inside high-level functions, like,.

.. index:: cv::gpu::createSeparableFilter_GPU

.. _cv::gpu::createSeparableFilter_GPU:

cv::gpu::createSeparableFilter_GPU
----------------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createSeparableFilter_GPU( const Ptr<BaseRowFilter_GPU>\& rowFilter,  const Ptr<BaseColumnFilter_GPU>\& columnFilter,  int srcType, int bufType, int dstType)

    Creates separable filter engine with the specified filters.

    {"Horizontal" 1D filter.}
    {"Vertical" 1D filter.}

    :param srcType: Input image type. It must be supported by  ``rowFilter`` .

    :param bufType: Buffer image type. It must be supported by  ``rowFilter``  and  ``columnFilter`` .

    :param dstType: Output image type. It must be supported by  ``columnFilter`` .

Usually this function is used inside high-level functions, like
.

.. index:: cv::gpu::getRowSumFilter_GPU

.. _cv::gpu::getRowSumFilter_GPU:

cv::gpu::getRowSumFilter_GPU
----------------------------
.. cfunction:: Ptr<BaseRowFilter_GPU> getRowSumFilter_GPU(int srcType, int sumType,  int ksize, int anchor = -1)

    Creates horizontal 1D box filter.

    :param srcType: Input image type. Only  ``CV_8UC1``  type is supported for now.

    :param sumType: Output image type. Only  ``CV_32FC1``  type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

.. index:: cv::gpu::getColumnSumFilter_GPU

.. _cv::gpu::getColumnSumFilter_GPU:

cv::gpu::getColumnSumFilter_GPU
-------------------------------
.. cfunction:: Ptr<BaseColumnFilter_GPU> getColumnSumFilter_GPU(int sumType,  int dstType, int ksize, int anchor = -1)

    Creates vertical 1D box filter.

    :param sumType: Input image type. Only  ``CV_8UC1``  type is supported for now.

    :param dstType: Output image type. Only  ``CV_32FC1``  type is supported for now.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

.. index:: cv::gpu::createBoxFilter_GPU

.. _cv::gpu::createBoxFilter_GPU:

cv::gpu::createBoxFilter_GPU
----------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createBoxFilter_GPU(int srcType, int dstType,  const Size\& ksize,  const Point\& anchor = Point(-1,-1))

    Creates normalized 2D box filter.

.. cfunction:: Ptr<BaseFilter_GPU> getBoxFilter_GPU(int srcType, int dstType,  const Size\& ksize,  Point anchor = Point(-1, -1))

    :param srcType: Input image type. Supports  ``CV_8UC1``  and  ``CV_8UC4`` .

    :param dstType: Output image type. Supports only the same as source type.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`boxFilter` .

.. index:: gpu::boxFilter

cv::gpu::boxFilter
------------------
.. cfunction:: void boxFilter(const GpuMat\& src, GpuMat\& dst, int ddepth, Size ksize,  Point anchor = Point(-1,-1))

    Smooths the image using the normalized box filter.

    :param src: Input image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  source types.

    :param dst: Output image type. Will have the same size and the same type as  ``src`` .

    :param ddepth: Output image depth. Support only the same as source depth ( ``CV_8U`` ) or -1 what means use source depth.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`boxFilter`,.

.. index:: gpu::blur

cv::gpu::blur
-------------
.. cfunction:: void blur(const GpuMat\& src, GpuMat\& dst, Size ksize,  Point anchor = Point(-1,-1))

    A synonym for normalized box filter.

    :param src: Input image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  source type.

    :param dst: Output image type. Will have the same size and the same type as  ``src`` .

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`blur`,:func:`gpu::boxFilter` .

.. index:: cv::gpu::createMorphologyFilter_GPU

.. _cv::gpu::createMorphologyFilter_GPU:

cv::gpu::createMorphologyFilter_GPU
-----------------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createMorphologyFilter_GPU(int op, int type,  const Mat\& kernel,  const Point\& anchor = Point(-1,-1),  int iterations = 1)

    Creates 2D morphological filter.

.. cfunction:: Ptr<BaseFilter_GPU> getMorphologyFilter_GPU(int op, int type,  const Mat\& kernel, const Size\& ksize,  Point anchor=Point(-1,-1))

    {Morphology operation id. Only ``MORPH_ERODE``     and ``MORPH_DILATE``     are supported.}

    :param type: Input/output image type. Only  ``CV_8UC1``  and  ``CV_8UC4``  are supported.

    :param kernel: 2D 8-bit structuring element for the morphological operation.

    :param size: Horizontal or vertical structuring element size for separable morphological operations.

    :param anchor: Anchor position within the structuring element; negative values mean that the anchor is at the center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`createMorphologyFilter` .

.. index:: gpu::erode

cv::gpu::erode
--------------
.. cfunction:: void erode(const GpuMat\& src, GpuMat\& dst, const Mat\& kernel,  Point anchor = Point(-1, -1),  int iterations = 1)

    Erodes an image by using a specific structuring element.

    :param src: Source image. Only  ``CV_8UC1``  and  ``CV_8UC4``  types are supported.

    :param dst: Destination image. It will have the same size and the same type as  ``src`` .

    :param kernel: Structuring element used for dilation. If  ``kernel=Mat()`` , a  :math:`3 \times 3`  rectangular structuring element is used.

    :param anchor: Position of the anchor within the element. The default value  :math:`(-1, -1)`  means that the anchor is at the element center.

    :param iterations: Number of times erosion to be applied.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`erode`,.

.. index:: gpu::dilate

cv::gpu::dilate
---------------
.. cfunction:: void dilate(const GpuMat\& src, GpuMat\& dst, const Mat\& kernel,  Point anchor = Point(-1, -1),  int iterations = 1)

    Dilates an image by using a specific structuring element.

    :param src: Source image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  source types.

    :param dst: Destination image. It will have the same size and the same type as  ``src`` .

    :param kernel: Structuring element used for dilation. If  ``kernel=Mat()`` , a  :math:`3 \times 3`  rectangular structuring element is used.

    :param anchor: Position of the anchor within the element. The default value  :math:`(-1, -1)`  means that the anchor is at the element center.

    :param iterations: Number of times dilation to be applied.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`dilate`,.

.. index:: gpu::morphologyEx

cv::gpu::morphologyEx
---------------------
.. cfunction:: void morphologyEx(const GpuMat\& src, GpuMat\& dst, int op,  const Mat\& kernel,  Point anchor = Point(-1, -1),  int iterations = 1)

    Applies an advanced morphological operation to image.

    :param src: Source image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  source type.

    :param dst: Destination image. It will have the same size and the same type as  ``src``
    :param op: Type of morphological operation, one of the following:
        
            * **MORPH_OPEN** opening
            
            * **MORPH_CLOSE** closing
            
            * **MORPH_GRADIENT** morphological gradient
            
            * **MORPH_TOPHAT** "top hat"
            
            * **MORPH_BLACKHAT** "black hat"
            

    :param kernel: Structuring element.

    :param anchor: Position of the anchor within the element. The default value Point(-1, -1) means that the anchor is at the element center.

    :param iterations: Number of times erosion and dilation to be applied.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`morphologyEx` .

.. index:: cv::gpu::createLinearFilter_GPU

.. _cv::gpu::createLinearFilter_GPU:

cv::gpu::createLinearFilter_GPU
-------------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createLinearFilter_GPU(int srcType, int dstType,  const Mat\& kernel,  const Point\& anchor = Point(-1,-1))

    Creates the non-separable linear filter.

.. cfunction:: Ptr<BaseFilter_GPU> getLinearFilter_GPU(int srcType, int dstType,  const Mat\& kernel, const Size\& ksize,  Point anchor = Point(-1, -1))

    :param srcType: Input image type. Supports  ``CV_8UC1``  and  ``CV_8UC4`` .

    :param dstType: Output image type. Supports only the same as source type.

    :param kernel: 2D array of filter coefficients. This filter works with integers kernels, if  ``kernel``  has  ``float``  or  ``double``  type it will be used fixed point arithmetic.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`createLinearFilter` .

.. index:: gpu::filter2D

cv::gpu::filter2D
-----------------
.. cfunction:: void filter2D(const GpuMat\& src, GpuMat\& dst, int ddepth,  const Mat\& kernel,  Point anchor=Point(-1,-1))

    Applies non-separable 2D linear filter to image.

    :param src: Source image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  source types.

    :param dst: Destination image. It will have the same size and the same number of channels as  ``src`` .

    :param ddepth: The desired depth of the destination image. If it is negative, it will be the same as  ``src.depth()`` . Supports only the same depth as source image.

    :param kernel: 2D array of filter coefficients. This filter works with integers kernels, if  ``kernel``  has  ``float``  or  ``double``  type it will use fixed point arithmetic.

    :param anchor: Anchor of the kernel that indicates the relative position of a filtered point within the kernel. The anchor should lie within the kernel. The special default value (-1,-1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`filter2D`,.

.. index:: gpu::Laplacian

cv::gpu::Laplacian
------------------
.. cfunction:: void Laplacian(const GpuMat\& src, GpuMat\& dst, int ddepth,  int ksize = 1, double scale = 1)

    Applies Laplacian operator to image.

    :param src: Source image. Supports  ``CV_8UC1``  and  ``CV_8UC4``  source types.

    :param dst: Destination image; will have the same size and the same number of channels as  ``src`` .

    :param ddepth: Desired depth of the destination image. Supports only tha same depth as source image depth.

    :param ksize: Aperture size used to compute the second-derivative filters, see  :func:`getDerivKernels` . It must be positive and odd. Supports only  ``ksize``  = 1 and  ``ksize``  = 3.

    :param scale: Optional scale factor for the computed Laplacian values (by default, no scaling is applied, see  :func:`getDerivKernels` ).

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

See also:
:func:`Laplacian`,:func:`gpu::filter2D` .

.. index:: cv::gpu::getLinearRowFilter_GPU

.. _cv::gpu::getLinearRowFilter_GPU:

cv::gpu::getLinearRowFilter_GPU
-------------------------------
.. cfunction:: Ptr<BaseRowFilter_GPU> getLinearRowFilter_GPU(int srcType,  int bufType, const Mat\& rowKernel, int anchor = -1,  int borderType = BORDER_CONSTANT)

    Creates primitive row filter with the specified kernel.

    :param srcType: Source array type. Supports only  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param bufType: Inermediate buffer type; must have as many channels as  ``srcType`` .

    :param rowKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel; negative values mean that anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method; see  :func:`borderInterpolate` . About limitation see below.

There are two version of algorithm: NPP and OpenCV. NPP calls when ``srcType == CV_8UC1`` or ``srcType == CV_8UC4`` and ``bufType == srcType`` , otherwise calls OpenCV version. NPP supports only ``BORDER_CONSTANT`` border type and doesn't check indices outside image. OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101``,``BORDER_REPLICATE`` and ``BORDER_CONSTANT`` border types and checks indices outside image.

See also:,:func:`createSeparableLinearFilter` .

.. index:: cv::gpu::getLinearColumnFilter_GPU

.. _cv::gpu::getLinearColumnFilter_GPU:

cv::gpu::getLinearColumnFilter_GPU
----------------------------------
.. cfunction:: Ptr<BaseColumnFilter_GPU> getLinearColumnFilter_GPU(int bufType,  int dstType, const Mat\& columnKernel, int anchor = -1,  int borderType = BORDER_CONSTANT)

    Creates the primitive column filter with the specified kernel.

    :param bufType: Inermediate buffer type; must have as many channels as  ``dstType`` .

    :param dstType: Destination array type. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  destination types.

    :param columnKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel; negative values mean that anchor is positioned at the aperture center.

    :param borderType: Pixel extrapolation method; see  :func:`borderInterpolate` . About limitation see below.

There are two version of algorithm: NPP and OpenCV. NPP calls when ``dstType == CV_8UC1`` or ``dstType == CV_8UC4`` and ``bufType == dstType`` , otherwise calls OpenCV version. NPP supports only ``BORDER_CONSTANT`` border type and doesn't check indices outside image. OpenCV version supports only ``CV_32F`` buffer depth and ``BORDER_REFLECT101``,``BORDER_REPLICATE`` and ``BORDER_CONSTANT`` border types and checks indices outside image.
See also:,:func:`createSeparableLinearFilter` .

.. index:: cv::gpu::createSeparableLinearFilter_GPU

.. _cv::gpu::createSeparableLinearFilter_GPU:

cv::gpu::createSeparableLinearFilter_GPU
----------------------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createSeparableLinearFilter_GPU(int srcType,  int dstType, const Mat\& rowKernel, const Mat\& columnKernel,  const Point\& anchor = Point(-1,-1),  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Creates the separable linear filter engine.

    :param srcType: Source array type. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param dstType: Destination array type. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  destination types.

    :param rowKernel, columnKernel: Filter coefficients.

    :param anchor: Anchor position within the kernel; negative values mean that anchor is positioned at the aperture center.

    :param rowBorderType, columnBorderType: Pixel extrapolation method in the horizontal and the vertical directions; see  :func:`borderInterpolate` . About limitation see  ,  .

See also:,,
:func:`createSeparableLinearFilter` .

.. index:: gpu::sepFilter2D

cv::gpu::sepFilter2D
--------------------
.. cfunction:: void sepFilter2D(const GpuMat\& src, GpuMat\& dst, int ddepth,  const Mat\& kernelX, const Mat\& kernelY,  Point anchor = Point(-1,-1),  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Applies separable 2D linear filter to the image.

    :param src: Source image. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param dst: Destination image; will have the same size and the same number of channels as  ``src`` .

    :param ddepth: Destination image depth. Supports  ``CV_8U`` ,  ``CV_16S`` ,  ``CV_32S``  and  ``CV_32F`` .

    :param kernelX, kernelY: Filter coefficients.

    :param anchor: Anchor position within the kernel; The default value  :math:`(-1, 1)`  means that the anchor is at the kernel center.

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see  :func:`borderInterpolate` .

See also:,:func:`sepFilter2D` .

.. index:: cv::gpu::createDerivFilter_GPU

.. _cv::gpu::createDerivFilter_GPU:

cv::gpu::createDerivFilter_GPU
------------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createDerivFilter_GPU(int srcType, int dstType,  int dx, int dy, int ksize,  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Creates filter engine for the generalized Sobel operator.

    :param srcType: Source image type. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param dstType: Destination image type; must have as many channels as  ``srcType`` . Supports  ``CV_8U`` ,  ``CV_16S`` ,  ``CV_32S``  and  ``CV_32F``  depths.

    :param dx: Derivative order in respect with x.

    :param dy: Derivative order in respect with y.

    :param ksize: Aperture size; see  :func:`getDerivKernels` .

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see  :func:`borderInterpolate` .

See also:,:func:`createDerivFilter` .

.. index:: gpu::Sobel

cv::gpu::Sobel
--------------
.. cfunction:: void Sobel(const GpuMat\& src, GpuMat\& dst, int ddepth, int dx, int dy,  int ksize = 3, double scale = 1,  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Applies generalized Sobel operator to the image.

    :param src: Source image. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param dst: Destination image. Will have the same size and number of channels as source image.

    :param ddepth: Destination image depth. Supports  ``CV_8U`` ,  ``CV_16S`` ,  ``CV_32S``  and  ``CV_32F`` .

    :param dx: Derivative order in respect with x.

    :param dy: Derivative order in respect with y.

    :param ksize: Size of the extended Sobel kernel, must be 1, 3, 5 or 7.

    :param scale: Optional scale factor for the computed derivative values (by default, no scaling is applied, see  :func:`getDerivKernels` ).

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see  :func:`borderInterpolate` .

See also:,:func:`Sobel` .

.. index:: gpu::Scharr

cv::gpu::Scharr
---------------
.. cfunction:: void Scharr(const GpuMat\& src, GpuMat\& dst, int ddepth,  int dx, int dy, double scale = 1,  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Calculates the first x- or y- image derivative using Scharr operator.

    :param src: Source image. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param dst: Destination image; will have the same size and the same number of channels as  ``src`` .

    :param ddepth: Destination image depth. Supports  ``CV_8U`` ,  ``CV_16S`` ,  ``CV_32S``  and  ``CV_32F`` .

    :param xorder: Order of the derivative x.

    :param yorder: Order of the derivative y.

    :param scale: Optional scale factor for the computed derivative values (by default, no scaling is applied, see  :func:`getDerivKernels` ).

    :param rowBorderType, columnBorderType: Pixel extrapolation method, see  :func:`borderInterpolate`
See also:,:func:`Scharr` .

.. index:: cv::gpu::createGaussianFilter_GPU

.. _cv::gpu::createGaussianFilter_GPU:

cv::gpu::createGaussianFilter_GPU
---------------------------------
.. cfunction:: Ptr<FilterEngine_GPU> createGaussianFilter_GPU(int type, Size ksize,  double sigmaX, double sigmaY = 0,  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Creates Gaussian filter engine.

    :param type: Source and the destination image type. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1`` .

    :param ksize: Aperture size; see  :func:`getGaussianKernel` .

    :param sigmaX: Gaussian sigma in the horizontal direction; see  :func:`getGaussianKernel` .

    :param sigmaY: Gaussian sigma in the vertical direction; if 0, then  :math:`\texttt{sigmaY}\leftarrow\texttt{sigmaX}` .

    :param rowBorderType, columnBorderType: Which border type to use; see  :func:`borderInterpolate` .

See also:,:func:`createGaussianFilter` .

.. index:: gpu::GaussianBlur

cv::gpu::GaussianBlur
---------------------
.. cfunction:: void GaussianBlur(const GpuMat\& src, GpuMat\& dst, Size ksize,  double sigmaX, double sigmaY = 0,  int rowBorderType = BORDER_DEFAULT,  int columnBorderType = -1)

    Smooths the image using Gaussian filter.

    :param src: Source image. Supports  ``CV_8UC1`` ,  ``CV_8UC4`` ,  ``CV_16SC1`` ,  ``CV_16SC2`` ,  ``CV_32SC1`` ,  ``CV_32FC1``  source types.

    :param dst: Destination image; will have the same size and the same type as  ``src`` .

    :param ksize: Gaussian kernel size;  ``ksize.width``  and  ``ksize.height``  can differ, but they both must be positive and odd. Or, they can be zero's, then they are computed from  ``sigmaX``  amd  ``sigmaY`` .

    :param sigmaX, sigmaY: Gaussian kernel standard deviations in X and Y direction. If  ``sigmaY``  is zero, it is set to be equal to  ``sigmaX`` . If they are both zeros, they are computed from  ``ksize.width``  and  ``ksize.height`` , respectively, see  :func:`getGaussianKernel` . To fully control the result regardless of possible future modification of all this semantics, it is recommended to specify all of  ``ksize`` ,  ``sigmaX``  and  ``sigmaY`` .

    :param rowBorderType, columnBorderType: Pixel extrapolation method; see  :func:`borderInterpolate` .

See also:,:func:`GaussianBlur` .

.. index:: cv::gpu::getMaxFilter_GPU

.. _cv::gpu::getMaxFilter_GPU:

cv::gpu::getMaxFilter_GPU
-------------------------
.. cfunction:: Ptr<BaseFilter_GPU> getMaxFilter_GPU(int srcType, int dstType,  const Size\& ksize, Point anchor = Point(-1,-1))

    Creates maximum filter.

    :param srcType: Input image type. Supports only  ``CV_8UC1``  and  ``CV_8UC4`` .

    :param dstType: Output image type. Supports only the same type as source.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.

.. index:: cv::gpu::getMinFilter_GPU

.. _cv::gpu::getMinFilter_GPU:

cv::gpu::getMinFilter_GPU
-------------------------
.. cfunction:: Ptr<BaseFilter_GPU> getMinFilter_GPU(int srcType, int dstType,  const Size\& ksize, Point anchor = Point(-1,-1))

    Creates minimum filter.

    :param srcType: Input image type. Supports only  ``CV_8UC1``  and  ``CV_8UC4`` .

    :param dstType: Output image type. Supports only the same type as source.

    :param ksize: Kernel size.

    :param anchor: Anchor point. The default value (-1) means that the anchor is at the kernel center.

**Please note:**
This filter doesn't check out of border accesses, so only proper submatrix of bigger matrix have to be passed to it.
