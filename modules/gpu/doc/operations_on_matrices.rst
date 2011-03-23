Operations on Matrices
======================

.. highlight:: cpp



.. index:: gpu::transpose

gpu::transpose
------------------
.. cpp:function:: void gpu::transpose(const GpuMat& src, GpuMat& dst)

    Transposes a matrix.

    :param src: Source matrix. 1, 4, 8 bytes element sizes are supported for now.

    :param dst: Destination matrix.

See also: :c:func:`transpose`.



.. index:: gpu::flip

gpu::flip
-------------
.. cpp:function:: void gpu::flip(const GpuMat& a, GpuMat& b, int flipCode)

    Flips a 2D matrix around vertical, horizontal or both axes.

    :param a: Source matrix. Only ``CV_8UC1`` and ``CV_8UC4`` matrices are supported for now.

    :param b: Destination matrix.

    :param flipCode: Specifies how to flip the source:
        
            * **0** Flip around x-axis.
            
            * **:math:`>`0** Flip around y-axis.
            
            * **:math:`<`0** Flip around both axes.

See also: :c:func:`flip`.



.. index:: gpu::LUT

gpu::LUT
------------

.. cpp:function:: void gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst)

    Transforms the source matrix into the destination matrix using given look-up table:

    .. math::

        dst(I) = lut(src(I))

    :param src: Source matrix. ``CV_8UC1`` and ``CV_8UC3`` matrixes are supported for now.

    :param lut: Look-up table. Must be continuous, ``CV_8U`` depth matrix. Its area must satisfy to ``lut.rows`` :math:`\times` ``lut.cols`` = 256 condition.

    :param dst: Destination matrix. Will have the same depth as ``lut`` and the same number of channels as ``src``.

See also: :c:func:`LUT`.



.. index:: gpu::merge

gpu::merge
--------------
.. cpp:function:: void gpu::merge(const GpuMat* src, size_t n, GpuMat& dst)

.. cpp:function:: void gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, const Stream& stream)

    Makes a multi-channel matrix out of several single-channel matrices.

    :param src: Pointer to array of the source matrices.

    :param n: Number of source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. cpp:function:: void gpu::merge(const vector<GpuMat>& src, GpuMat& dst)

.. cpp:function:: void gpu::merge(const vector<GpuMat>& src, GpuMat& dst, const Stream& stream)

    :param src: Vector of the source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`merge`.



.. index:: gpu::split

gpu::split
--------------
.. cpp:function:: void gpu::split(const GpuMat& src, GpuMat* dst)

.. cpp:function:: void gpu::split(const GpuMat& src, GpuMat* dst, const Stream& stream)

    Copies each plane of a multi-channel matrix into an array.

    :param src: Source matrix.

    :param dst: Pointer to array of single-channel matrices.

    :param stream: Stream for the asynchronous version.

.. cpp:function:: void gpu::split(const GpuMat& src, vector<GpuMat>& dst)

.. cpp:function:: void gpu::split(const GpuMat& src, vector<GpuMat>& dst, const Stream& stream)

    :param src: Source matrix.

    :param dst: Destination vector of single-channel matrices.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`split`.



.. index:: gpu::magnitude

gpu::magnitude
------------------
.. cpp:function:: void gpu::magnitude(const GpuMat& x, GpuMat& magnitude)

    Computes magnitudes of complex matrix elements.

    :param x: Source complex matrix in the interleaved format (``CV_32FC2``).

    :param magnitude: Destination matrix of float magnitudes (``CV_32FC1``).

.. cpp:function:: void gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& magnitude)

.. cpp:function:: void gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, const Stream& stream)

    :param x: Source matrix, containing real components (``CV_32FC1``).

    :param y: Source matrix, containing imaginary components (``CV_32FC1``).

    :param magnitude: Destination matrix of float magnitudes (``CV_32FC1``).

    :param stream: Stream for the asynchronous version.

See also: :c:func:`magnitude`.



.. index:: gpu::magnitudeSqr

gpu::magnitudeSqr
---------------------
.. cpp:function:: void gpu::magnitudeSqr(const GpuMat& x, GpuMat& magnitude)

    Computes squared magnitudes of complex matrix elements.

    :param x: Source complex matrix in the interleaved format (``CV_32FC2``).

    :param magnitude: Destination matrix of float magnitude squares (``CV_32FC1``).

.. cpp:function:: void gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& magnitude)

.. cpp:function:: void gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, const Stream& stream)

    :param x: Source matrix, containing real components (``CV_32FC1``).

    :param y: Source matrix, containing imaginary components (``CV_32FC1``).

    :param magnitude: Destination matrix of float magnitude squares (``CV_32FC1``).

    :param stream: Stream for the asynchronous version.



.. index:: gpu::phase

gpu::phase
--------------
.. cpp:function:: void gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees=false)

.. cpp:function:: void gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees, const Stream& stream)

    Computes polar angles of complex matrix elements.

    :param x: Source matrix, containing real components (``CV_32FC1``).

    :param y: Source matrix, containing imaginary components (``CV_32FC1``).

    :param angle: Destionation matrix of angles (``CV_32FC1``).

    :param angleInDegress: Flag which indicates angles must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`phase`.



.. index:: gpu::cartToPolar

gpu::cartToPolar
--------------------
.. cpp:function:: void gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees=false)

.. cpp:function:: void gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees, const Stream& stream)

    Converts Cartesian coordinates into polar.

    :param x: Source matrix, containing real components (``CV_32FC1``).

    :param y: Source matrix, containing imaginary components (``CV_32FC1``).

    :param magnitude: Destination matrix of float magnituds (``CV_32FC1``).

    :param angle: Destionation matrix of angles (``CV_32FC1``).

    :param angleInDegress: Flag which indicates angles must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`cartToPolar`.



.. index:: gpu::polarToCart

gpu::polarToCart
--------------------
.. cpp:function:: void gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees=false)

.. cpp:function:: void gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, const Stream& stream)

    Converts polar coordinates into Cartesian.

    :param magnitude: Source matrix, containing magnitudes (``CV_32FC1``).

    :param angle: Source matrix, containing angles (``CV_32FC1``).

    :param x: Destination matrix of real components (``CV_32FC1``).

    :param y: Destination matrix of imaginary components (``CV_32FC1``).

    :param angleInDegress: Flag which indicates angles are in degress.

    :param stream: Stream for the asynchronous version.

See also: :c:func:`polarToCart`.
