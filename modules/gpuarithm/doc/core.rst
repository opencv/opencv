Core Operations on Matrices
===========================

.. highlight:: cpp



gpu::merge
--------------
Makes a multi-channel matrix out of several single-channel matrices.

.. ocv:function:: void gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::merge(const vector<GpuMat>& src, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Array/vector of source matrices.

    :param n: Number of source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`merge`



gpu::split
--------------
Copies each plane of a multi-channel matrix into an array.

.. ocv:function:: void gpu::split(const GpuMat& src, GpuMat* dst, Stream& stream = Stream::Null())

.. ocv:function:: void gpu::split(const GpuMat& src, vector<GpuMat>& dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination array/vector of single-channel matrices.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`split`



gpu::copyMakeBorder
-----------------------
Forms a border around an image.

.. ocv:function:: void gpu::copyMakeBorder(const GpuMat& src, GpuMat& dst, int top, int bottom, int left, int right, int borderType, const Scalar& value = Scalar(), Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and  ``CV_32FC1`` types are supported.

    :param dst: Destination image with the same type as  ``src``. The size is  ``Size(src.cols+left+right, src.rows+top+bottom)`` .

    :param top:

    :param bottom:

    :param left:

    :param right: Number of pixels in each direction from the source image rectangle to extrapolate. For example:  ``top=1, bottom=1, left=1, right=1`` mean that 1 pixel-wide border needs to be built.

    :param borderType: Border type. See  :ocv:func:`borderInterpolate` for details. ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , ``BORDER_CONSTANT`` , ``BORDER_REFLECT`` and ``BORDER_WRAP`` are supported for now.

    :param value: Border value.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`copyMakeBorder`



gpu::transpose
------------------
Transposes a matrix.

.. ocv:function:: void gpu::transpose( const GpuMat& src1, GpuMat& dst, Stream& stream=Stream::Null() )

    :param src1: Source matrix. 1-, 4-, 8-byte element sizes are supported for now (CV_8UC1, CV_8UC4, CV_16UC2, CV_32FC1, etc).

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`transpose`



gpu::flip
-------------
Flips a 2D matrix around vertical, horizontal, or both axes.

.. ocv:function:: void gpu::flip( const GpuMat& a, GpuMat& b, int flipCode, Stream& stream=Stream::Null() )

    :param a: Source matrix. Supports 1, 3 and 4 channels images with ``CV_8U``, ``CV_16U``, ``CV_32S`` or ``CV_32F`` depth.

    :param b: Destination matrix.

    :param flipCode: Flip mode for the source:

        * ``0`` Flips around x-axis.

        * ``>0`` Flips around y-axis.

        * ``<0`` Flips around both axes.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`flip`



gpu::LUT
------------
Transforms the source matrix into the destination matrix using the given look-up table: ``dst(I) = lut(src(I))``

.. ocv:function:: void gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst, Stream& stream = Stream::Null())

    :param src: Source matrix.  ``CV_8UC1``  and  ``CV_8UC3``  matrices are supported for now.

    :param lut: Look-up table of 256 elements. It is a continuous ``CV_8U`` matrix.

    :param dst: Destination matrix with the same depth as  ``lut``  and the same number of channels as  ``src`` .

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`LUT`
