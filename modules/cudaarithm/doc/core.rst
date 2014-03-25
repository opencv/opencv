Core Operations on Matrices
===========================

.. highlight:: cpp



cuda::merge
-----------
Makes a multi-channel matrix out of several single-channel matrices.

.. ocv:function:: void cuda::merge(const GpuMat* src, size_t n, OutputArray dst, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::merge(const std::vector<GpuMat>& src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Array/vector of source matrices.

    :param n: Number of source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`merge`



cuda::split
-----------
Copies each plane of a multi-channel matrix into an array.

.. ocv:function:: void cuda::split(InputArray src, GpuMat* dst, Stream& stream = Stream::Null())

.. ocv:function:: void cuda::split(InputArray src, vector<GpuMat>& dst, Stream& stream = Stream::Null())

    :param src: Source matrix.

    :param dst: Destination array/vector of single-channel matrices.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`split`



cuda::transpose
---------------
Transposes a matrix.

.. ocv:function:: void cuda::transpose(InputArray src1, OutputArray dst, Stream& stream = Stream::Null())

    :param src1: Source matrix. 1-, 4-, 8-byte element sizes are supported for now.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`transpose`



cuda::flip
----------
Flips a 2D matrix around vertical, horizontal, or both axes.

.. ocv:function:: void cuda::flip(InputArray src, OutputArray dst, int flipCode, Stream& stream = Stream::Null())

    :param src: Source matrix. Supports 1, 3 and 4 channels images with ``CV_8U``, ``CV_16U``, ``CV_32S`` or ``CV_32F`` depth.

    :param dst: Destination matrix.

    :param flipCode: Flip mode for the source:

        * ``0`` Flips around x-axis.

        * ``> 0`` Flips around y-axis.

        * ``< 0`` Flips around both axes.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`flip`



cuda::LookUpTable
-----------------
.. ocv:class:: cuda::LookUpTable : public Algorithm

Base class for transform using lookup table. ::

    class CV_EXPORTS LookUpTable : public Algorithm
    {
    public:
        virtual void transform(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
    };

.. seealso:: :ocv:func:`LUT`



cuda::LookUpTable::transform
----------------------------
Transforms the source matrix into the destination matrix using the given look-up table: ``dst(I) = lut(src(I))`` .

.. ocv:function:: void cuda::LookUpTable::transform(InputArray src, OutputArray dst, Stream& stream = Stream::Null())

    :param src: Source matrix.  ``CV_8UC1``  and  ``CV_8UC3``  matrices are supported for now.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.



cuda::createLookUpTable
-----------------------
Creates implementation for :ocv:class:`cuda::LookUpTable` .

.. ocv:function:: Ptr<LookUpTable> createLookUpTable(InputArray lut)

    :param lut: Look-up table of 256 elements. It is a continuous ``CV_8U`` matrix.



cuda::copyMakeBorder
--------------------
Forms a border around an image.

.. ocv:function:: void cuda::copyMakeBorder(InputArray src, OutputArray dst, int top, int bottom, int left, int right, int borderType, Scalar value = Scalar(), Stream& stream = Stream::Null())

    :param src: Source image. ``CV_8UC1`` , ``CV_8UC4`` , ``CV_32SC1`` , and ``CV_32FC1`` types are supported.

    :param dst: Destination image with the same type as  ``src``. The size is ``Size(src.cols+left+right, src.rows+top+bottom)`` .

    :param top:

    :param bottom:

    :param left:

    :param right: Number of pixels in each direction from the source image rectangle to extrapolate. For example:  ``top=1, bottom=1, left=1, right=1`` mean that 1 pixel-wide border needs to be built.

    :param borderType: Border type. See  :ocv:func:`borderInterpolate` for details. ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , ``BORDER_CONSTANT`` , ``BORDER_REFLECT`` and ``BORDER_WRAP`` are supported for now.

    :param value: Border value.

    :param stream: Stream for the asynchronous version.

.. seealso:: :ocv:func:`copyMakeBorder`
