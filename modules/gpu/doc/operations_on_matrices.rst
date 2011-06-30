Operations on Matrices
======================

.. highlight:: cpp

.. index:: gpu::transpose

gpu::transpose
------------------
.. ocv:function:: void gpu::transpose(const GpuMat& src, GpuMat& dst)

    Transposes a matrix.

    :param src: Source matrix. 1-, 4-, 8-byte element sizes are supported for now.

    :param dst: Destination matrix.

.. seealso::
   :ocv:func:`transpose` 

.. index:: gpu::flip

gpu::flip
-------------
.. ocv:function:: void gpu::flip(const GpuMat& src, GpuMat& dst, int flipCode)

    Flips a 2D matrix around vertical, horizontal, or both axes.

    :param src: Source matrix. Only  ``CV_8UC1``  and  ``CV_8UC4``  matrices are supported for now.

    :param dst: Destination matrix.

    :param flipCode: Flip mode for the source:
        
            * ``0`` Flips around x-axis.
            
            * ``>0`` Flips around y-axis.
            
            * ``<0`` Flips around both axes.
            

.. seealso::
   :ocv:func:`flip` 

.. index:: gpu::LUT

gpu::LUT
------------
.. ocv:function:: void gpu::LUT(const GpuMat& src, const Mat& lut, GpuMat& dst)

    Transforms the source matrix into the destination matrix using the given look-up table: ``dst(I) = lut(src(I))``

    :param src: Source matrix.  ``CV_8UC1``  and  ``CV_8UC3``  matrices are supported for now.

    :param lut: Look-up table of 256 elements. It is a continuous ``CV_8U`` matrix.

    :param dst: Destination matrix with the same depth as  ``lut``  and the same number of channels as  ``src``.
            

.. seealso:: 
   :ocv:func:`LUT` 

.. index:: gpu::merge

gpu::merge
--------------
.. ocv:function:: void gpu::merge(const GpuMat* src, size_t n, GpuMat& dst)

.. ocv:function:: void gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, const Stream& stream)

.. ocv:function:: void gpu::merge(const vector<GpuMat>& src, GpuMat& dst)

.. ocv:function:: void gpu::merge(const vector<GpuMat>& src, GpuMat& dst, const Stream& stream)

    Makes a multi-channel matrix out of several single-channel matrices.

    :param src: Array/vector of source matrices.

    :param n: Number of source matrices.

    :param dst: Destination matrix.

    :param stream: Stream for the asynchronous version.

.. seealso:: 
   :ocv:func:`merge` 

.. index:: gpu::split

gpu::split
--------------
.. ocv:function:: void gpu::split(const GpuMat& src, GpuMat* dst)

.. ocv:function:: void gpu::split(const GpuMat& src, GpuMat* dst, const Stream& stream)

.. ocv:function:: void gpu::split(const GpuMat& src, vector<GpuMat>& dst)

.. ocv:function:: void gpu::split(const GpuMat& src, vector<GpuMat>& dst, const Stream& stream)

    Copies each plane of a multi-channel matrix into an array.

    :param src: Source matrix.

    :param dst: Destination array/vector of single-channel matrices.

    :param stream: Stream for the asynchronous version.

.. seealso:: 
   :ocv:func:`split`

.. index:: gpu::magnitude

gpu::magnitude
------------------
.. ocv:function:: void gpu::magnitude(const GpuMat& xy, GpuMat& magnitude)

.. ocv:function:: void gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& magnitude)

.. ocv:function:: void gpu::magnitude(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, const Stream& stream)

    Computes magnitudes of complex matrix elements.

    :param xy: Source complex matrix in the interleaved format (``CV_32FC2``).
    
    :param x: Source matrix containing real components (``CV_32FC1``).

    :param y: Source matrix containing imaginary components (``CV_32FC1``).

    :param magnitude: Destination matrix of float magnitudes (``CV_32FC1``).

    :param stream: Stream for the asynchronous version.

.. seealso::
   :ocv:func:`magnitude` 

.. index:: gpu::magnitudeSqr

gpu::magnitudeSqr
---------------------
.. ocv:function:: void gpu::magnitudeSqr(const GpuMat& xy, GpuMat& magnitude)

.. ocv:function:: void gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& magnitude)

.. ocv:function:: void gpu::magnitudeSqr(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, const Stream& stream)

    Computes squared magnitudes of complex matrix elements.

    :param xy: Source complex matrix in the interleaved format (``CV_32FC2``).

    :param x: Source matrix containing real components (``CV_32FC1``).

    :param y: Source matrix containing imaginary components (``CV_32FC1``).

    :param magnitude: Destination matrix of float magnitude squares (``CV_32FC1``).

    :param stream: Stream for the asynchronous version.

.. index:: gpu::phase

gpu::phase
--------------
.. ocv:function:: void gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees=false)

.. ocv:function:: void gpu::phase(const GpuMat& x, const GpuMat& y, GpuMat& angle, bool angleInDegrees, const Stream& stream)

    Computes polar angles of complex matrix elements.

    :param x: Source matrix containing real components (``CV_32FC1``).

    :param y: Source matrix containing imaginary components (``CV_32FC1``).

    :param angle: Destionation matrix of angles (``CV_32FC1``).

    :param angleInDegress: Flag for angles that must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

.. seealso::
   :ocv:func:`phase` 

.. index:: gpu::cartToPolar

gpu::cartToPolar
--------------------
.. ocv:function:: void gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees=false)

.. ocv:function:: void gpu::cartToPolar(const GpuMat& x, const GpuMat& y, GpuMat& magnitude, GpuMat& angle, bool angleInDegrees, const Stream& stream)

    Converts Cartesian coordinates into polar.

    :param x: Source matrix containing real components (``CV_32FC1``).

    :param y: Source matrix containing imaginary components (``CV_32FC1``).

    :param magnitude: Destination matrix of float magnitudes (``CV_32FC1``).

    :param angle: Destionation matrix of angles (``CV_32FC1``).

    :param angleInDegress: Flag for angles that must be evaluated in degress.

    :param stream: Stream for the asynchronous version.

.. seealso::
   :ocv:func:`cartToPolar` 

.. index:: gpu::polarToCart

gpu::polarToCart
--------------------
.. ocv:function:: void gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees=false)

.. ocv:function:: void gpu::polarToCart(const GpuMat& magnitude, const GpuMat& angle, GpuMat& x, GpuMat& y, bool angleInDegrees, const Stream& stream)

    Converts polar coordinates into Cartesian.

    :param magnitude: Source matrix containing magnitudes (``CV_32FC1``).

    :param angle: Source matrix containing angles (``CV_32FC1``).

    :param x: Destination matrix of real components (``CV_32FC1``).

    :param y: Destination matrix of imaginary components (``CV_32FC1``).

    :param angleInDegress: Flag that indicates angles in degress.

    :param stream: Stream for the asynchronous version.

.. seealso::
   :ocv:func:`polarToCart` 
