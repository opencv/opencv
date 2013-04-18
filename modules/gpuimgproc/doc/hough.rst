Hough Transform
===============

.. highlight:: cpp



gpu::HoughLines
---------------
Finds lines in a binary image using the classical Hough transform.

.. ocv:function:: void gpu::HoughLines(const GpuMat& src, GpuMat& lines, float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096)

.. ocv:function:: void gpu::HoughLines(const GpuMat& src, GpuMat& lines, HoughLinesBuf& buf, float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096)

    :param src: 8-bit, single-channel binary source image.

    :param lines: Output vector of lines. Each line is represented by a two-element vector  :math:`(\rho, \theta)` .  :math:`\rho`  is the distance from the coordinate origin  :math:`(0,0)`  (top-left corner of the image).  :math:`\theta`  is the line rotation angle in radians ( :math:`0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}` ).

    :param rho: Distance resolution of the accumulator in pixels.

    :param theta: Angle resolution of the accumulator in radians.

    :param threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes ( :math:`>\texttt{threshold}` ).

    :param doSort: Performs lines sort by votes.

    :param maxLines: Maximum number of output lines.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`HoughLines`



gpu::HoughLinesDownload
-----------------------
Downloads results from :ocv:func:`gpu::HoughLines` to host memory.

.. ocv:function:: void gpu::HoughLinesDownload(const GpuMat& d_lines, OutputArray h_lines, OutputArray h_votes = noArray())

    :param d_lines: Result of :ocv:func:`gpu::HoughLines` .

    :param h_lines: Output host array.

    :param h_votes: Optional output array for line's votes.

.. seealso:: :ocv:func:`gpu::HoughLines`



gpu::HoughCircles
-----------------
Finds circles in a grayscale image using the Hough transform.

.. ocv:function:: void gpu::HoughCircles(const GpuMat& src, GpuMat& circles, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

.. ocv:function:: void gpu::HoughCircles(const GpuMat& src, GpuMat& circles, HoughCirclesBuf& buf, int method, float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

    :param src: 8-bit, single-channel grayscale input image.

    :param circles: Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  :math:`(x, y, radius)` .

    :param method: Detection method to use. Currently, the only implemented method is  ``CV_HOUGH_GRADIENT`` , which is basically  *21HT* , described in  [Yuen90]_.

    :param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if  ``dp=1`` , the accumulator has the same resolution as the input image. If  ``dp=2`` , the accumulator has half as big width and height.

    :param minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.

    :param cannyThreshold: The higher threshold of the two passed to  the :ocv:func:`gpu::Canny`  edge detector (the lower one is twice smaller).

    :param votesThreshold: The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.

    :param minRadius: Minimum circle radius.

    :param maxRadius: Maximum circle radius.

    :param maxCircles: Maximum number of output circles.

    :param buf: Optional buffer to avoid extra memory allocations (for many calls with the same sizes).

.. seealso:: :ocv:func:`HoughCircles`



gpu::HoughCirclesDownload
-------------------------
Downloads results from :ocv:func:`gpu::HoughCircles` to host memory.

.. ocv:function:: void gpu::HoughCirclesDownload(const GpuMat& d_circles, OutputArray h_circles)

    :param d_circles: Result of :ocv:func:`gpu::HoughCircles` .

    :param h_circles: Output host array.

.. seealso:: :ocv:func:`gpu::HoughCircles`
