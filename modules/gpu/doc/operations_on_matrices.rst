Operations on Matrices
======================

.. highlight:: cpp

.. index:: gpu::transpose

cv::gpu::transpose
------------------
.. c:function:: void transpose(const GpuMat\& src, GpuMat\& dst)

    Transposes a matrix.

    :param src: Source matrix. 1, 4, 8 bytes element sizes are supported for now.

    :param dst: Destination matrix.

See also:
:func:`transpose` .

.. index:: gpu::flip

cv::gpu::flip
-------------
.. c:function:: void flip(const GpuMat\& a, GpuMat\& b, int flipCode)

    Flips a 2D matrix around vertical, horizontal or both axes.

    :param a: Source matrix. Only  ``CV_8UC1``  and  ``CV_8UC4``  matrices are supported for now.

    :param b: Destination matrix.

    :param flipCode: Specifies how to flip the source:
        
            * **0** Flip around x-axis.
            
            * **:math:`>`0** Flip around y-axis.
            
            * **:math:`<`0** Flip around both axes.
            

See also:
:func:`flip` .

.. index:: gpu::LUT

cv::gpu::LUT
------------
.. math::

    dst(I) = lut(src(I))

.. c:function:: void LUT(const GpuMat\& src, const Mat\& lut, GpuMat\& dst)

    Transforms the source matrix into the destination matrix using given look-up table:

    :param src: Source matrix.  ``CV_8UC1``  and  ``CV_8UC3``  matrixes are supported for now.

    :param lut: Look-up table. Must be continuous,  ``CV_8U``  depth matrix. Its area must satisfy to  ``lut.rows``   :math:`\times`   ``lut.cols``  = 256 condition.

    :param dst: Destination matrix. Will have the same depth as  ``lut``  and the same number of channels as  ``src`` .

See also:
:func:`LUT` .

.. index:: gpu::merge

cv::gpu::merge
--------------
.. c:function:: void merge(const GpuMat* src, size_t n, GpuMat\& dst)

.. c:function:: void merge(const GpuMat* src, size_t n, GpuMat\& dst,
   const Stream\& stream)

    Makes a multi-channel matrix out of several single-channel matrices.

    :param src: Pointer to array of the source matrices.

    :param n: Number of source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. c:function:: void merge(const vector$<$GpuMat$>$\& src, GpuMat\& dst)

.. c:function:: void merge(const vector$<$GpuMat$>$\& src, GpuMat\& dst,
   const Stream\& stream)

    * **src** Vector of the source matrices.

    * **dst** Destination matrix.

    * **stream** Stream for the asynchronous version.

See also:
:func:`merge` .

.. index:: gpu::split

cv::gpu::split
--------------
.. c:function:: void split(const GpuMat\& src, GpuMat* dst)

.. c:function:: void split(const GpuMat\& src, GpuMat* dst, const Stream\& stream)

    Copies each plane of a multi-channel matrix into an array.

    :param src: Source matrix.

    :param dst: Pointer to array of single-channel matrices.

    :param stream: Stream for the asynchronous version.

.. c:function:: void split(const GpuMat\& src, vector$<$GpuMat$>$\& dst)

.. c:function:: void split(const GpuMat\& src, vector$<$GpuMat$>$\& dst,
   const Stream\& stream)

    * **src** Source matrix.

    * **dst** Destination vector of single-channel matrices.

    * **stream** Stream for the asynchronous version.

See also:
:func:`split` .

.. index:: gpu::magnitude

cv::gpu::magnitude
------------------
.. c:function:: void magnitude(const GpuMat\& x, GpuMat\& magnitude)

    Computes magnitudes of complex matrix elements.

    :param x: Source complex matrix in the interleaved format ( ``CV_32FC2`` ).

    :param magnitude: Destination matrix of float magnitudes ( ``CV_32FC1`` ).

.. c:function:: void magnitude(const GpuMat\& x, const GpuMat\& y, GpuMat\& magnitude)

.. c:function:: void magnitude(const GpuMat\& x, const GpuMat\& y, GpuMat\& magnitude,
   const Stream\& stream)

    * **x** Source matrix, containing real components ( ``CV_32FC1`` ).

    * **y** Source matrix, containing imaginary components ( ``CV_32FC1`` ).

    * **magnitude** Destination matrix of float magnitudes ( ``CV_32FC1`` ).

    * **stream** Stream for the asynchronous version.

See also:
:func:`magnitude` .

.. index:: gpu::magnitudeSqr

cv::gpu::magnitudeSqr
---------------------
.. c:function:: void magnitudeSqr(const GpuMat\& x, GpuMat\& magnitude)

    Computes squared magnitudes of complex matrix elements.

    :param x: Source complex matrix in the interleaved format ( ``CV_32FC2`` ).

    :param magnitude: Destination matrix of float magnitude squares ( ``CV_32FC1`` ).

.. c:function:: void magnitudeSqr(const GpuMat\& x, const GpuMat\& y, GpuMat\& magnitude)

.. c:function:: void magnitudeSqr(const GpuMat\& x, const GpuMat\& y, GpuMat\& magnitude,
   const Stream\& stream)

    * **x** Source matrix, containing real components ( ``CV_32FC1`` ).

    * **y** Source matrix, containing imaginary components ( ``CV_32FC1`` ).

    * **magnitude** Destination matrix of float magnitude squares ( ``CV_32FC1`` ).

    * **stream** Stream for the asynchronous version.

.. index:: gpu::phase

cv::gpu::phase
--------------
.. c:function:: void phase(const GpuMat\& x, const GpuMat\& y, GpuMat\& angle,
   bool angleInDegrees=false)

.. c:function:: void phase(const GpuMat\& x, const GpuMat\& y, GpuMat\& angle,
   bool angleInDegrees, const Stream\& stream)

    Computes polar angles of complex matrix elements.

    :param x: Source matrix, containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix, containing imaginary components ( ``CV_32FC1`` ).

    :param angle: Destionation matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegress: Flag which indicates angles must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

See also:
:func:`phase` .

.. index:: gpu::cartToPolar

cv::gpu::cartToPolar
--------------------
.. c:function:: void cartToPolar(const GpuMat\& x, const GpuMat\& y, GpuMat\& magnitude,
   GpuMat\& angle, bool angleInDegrees=false)

.. c:function:: void cartToPolar(const GpuMat\& x, const GpuMat\& y, GpuMat\& magnitude,
   GpuMat\& angle, bool angleInDegrees, const Stream\& stream)

    Converts Cartesian coordinates into polar.

    :param x: Source matrix, containing real components ( ``CV_32FC1`` ).

    :param y: Source matrix, containing imaginary components ( ``CV_32FC1`` ).

    :param magnitude: Destination matrix of float magnituds ( ``CV_32FC1`` ).

    :param angle: Destionation matrix of angles ( ``CV_32FC1`` ).

    :param angleInDegress: Flag which indicates angles must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

See also:
:func:`cartToPolar` .

.. index:: gpu::polarToCart

cv::gpu::polarToCart
--------------------
.. c:function:: void polarToCart(const GpuMat\& magnitude, const GpuMat\& angle,
   GpuMat\& x, GpuMat\& y, bool angleInDegrees=false)

.. c:function:: void polarToCart(const GpuMat\& magnitude, const GpuMat\& angle,
   GpuMat\& x, GpuMat\& y, bool angleInDegrees,
   const Stream\& stream)

    Converts polar coordinates into Cartesian.

    :param magnitude: Source matrix, containing magnitudes ( ``CV_32FC1`` ).

    :param angle: Source matrix, containing angles ( ``CV_32FC1`` ).

    :param x: Destination matrix of real components ( ``CV_32FC1`` ).

    :param y: Destination matrix of imaginary components ( ``CV_32FC1`` ).

    :param angleInDegress: Flag which indicates angles are in degress.

    :param stream: Stream for the asynchronous version.

See also:
:func:`polarToCart` .
