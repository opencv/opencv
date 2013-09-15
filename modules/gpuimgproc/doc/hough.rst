Hough Transform
===============

.. highlight:: cpp



gpu::HoughLinesDetector
-----------------------
.. ocv:class:: gpu::HoughLinesDetector : public Algorithm

Base class for lines detector algorithm. ::

    class CV_EXPORTS HoughLinesDetector : public Algorithm
    {
    public:
        virtual void detect(InputArray src, OutputArray lines) = 0;
        virtual void downloadResults(InputArray d_lines, OutputArray h_lines, OutputArray h_votes = noArray()) = 0;

        virtual void setRho(float rho) = 0;
        virtual float getRho() const = 0;

        virtual void setTheta(float theta) = 0;
        virtual float getTheta() const = 0;

        virtual void setThreshold(int threshold) = 0;
        virtual int getThreshold() const = 0;

        virtual void setDoSort(bool doSort) = 0;
        virtual bool getDoSort() const = 0;

        virtual void setMaxLines(int maxLines) = 0;
        virtual int getMaxLines() const = 0;
    };



gpu::HoughLinesDetector::detect
-------------------------------
Finds lines in a binary image using the classical Hough transform.

.. ocv:function:: void gpu::HoughLinesDetector::detect(InputArray src, OutputArray lines)

    :param src: 8-bit, single-channel binary source image.

    :param lines: Output vector of lines. Each line is represented by a two-element vector  :math:`(\rho, \theta)` .  :math:`\rho`  is the distance from the coordinate origin  :math:`(0,0)`  (top-left corner of the image).  :math:`\theta`  is the line rotation angle in radians ( :math:`0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}` ).

.. seealso:: :ocv:func:`HoughLines`



gpu::HoughLinesDetector::downloadResults
----------------------------------------
Downloads results from :ocv:func:`gpu::HoughLinesDetector::detect` to host memory.

.. ocv:function:: void gpu::HoughLinesDetector::downloadResults(InputArray d_lines, OutputArray h_lines, OutputArray h_votes = noArray())

    :param d_lines: Result of :ocv:func:`gpu::HoughLinesDetector::detect` .

    :param h_lines: Output host array.

    :param h_votes: Optional output array for line's votes.



gpu::createHoughLinesDetector
-----------------------------
Creates implementation for :ocv:class:`gpu::HoughLinesDetector` .

.. ocv:function:: Ptr<HoughLinesDetector> gpu::createHoughLinesDetector(float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096)

    :param rho: Distance resolution of the accumulator in pixels.

    :param theta: Angle resolution of the accumulator in radians.

    :param threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes ( :math:`>\texttt{threshold}` ).

    :param doSort: Performs lines sort by votes.

    :param maxLines: Maximum number of output lines.



gpu::HoughSegmentDetector
-------------------------
.. ocv:class:: gpu::HoughSegmentDetector : public Algorithm

Base class for line segments detector algorithm. ::

    class CV_EXPORTS HoughSegmentDetector : public Algorithm
    {
    public:
        virtual void detect(InputArray src, OutputArray lines) = 0;

        virtual void setRho(float rho) = 0;
        virtual float getRho() const = 0;

        virtual void setTheta(float theta) = 0;
        virtual float getTheta() const = 0;

        virtual void setMinLineLength(int minLineLength) = 0;
        virtual int getMinLineLength() const = 0;

        virtual void setMaxLineGap(int maxLineGap) = 0;
        virtual int getMaxLineGap() const = 0;

        virtual void setMaxLines(int maxLines) = 0;
        virtual int getMaxLines() const = 0;
    };

.. note::

   * An example using the Hough segment detector can be found at opencv_source_code/samples/gpu/houghlines.cpp


gpu::HoughSegmentDetector::detect
---------------------------------
Finds line segments in a binary image using the probabilistic Hough transform.

.. ocv:function:: void gpu::HoughSegmentDetector::detect(InputArray src, OutputArray lines)

    :param src: 8-bit, single-channel binary source image.

    :param lines: Output vector of lines. Each line is represented by a 4-element vector  :math:`(x_1, y_1, x_2, y_2)` , where  :math:`(x_1,y_1)`  and  :math:`(x_2, y_2)`  are the ending points of each detected line segment.

.. seealso:: :ocv:func:`HoughLinesP`



gpu::createHoughSegmentDetector
-------------------------------
Creates implementation for :ocv:class:`gpu::HoughSegmentDetector` .

.. ocv:function:: Ptr<HoughSegmentDetector> gpu::createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines = 4096)

    :param rho: Distance resolution of the accumulator in pixels.

    :param theta: Angle resolution of the accumulator in radians.

    :param minLineLength: Minimum line length. Line segments shorter than that are rejected.

    :param maxLineGap: Maximum allowed gap between points on the same line to link them.

    :param maxLines: Maximum number of output lines.



gpu::HoughCirclesDetector
-------------------------
.. ocv:class:: gpu::HoughCirclesDetector : public Algorithm

Base class for circles detector algorithm. ::

    class CV_EXPORTS HoughCirclesDetector : public Algorithm
    {
    public:
        virtual void detect(InputArray src, OutputArray circles) = 0;

        virtual void setDp(float dp) = 0;
        virtual float getDp() const = 0;

        virtual void setMinDist(float minDist) = 0;
        virtual float getMinDist() const = 0;

        virtual void setCannyThreshold(int cannyThreshold) = 0;
        virtual int getCannyThreshold() const = 0;

        virtual void setVotesThreshold(int votesThreshold) = 0;
        virtual int getVotesThreshold() const = 0;

        virtual void setMinRadius(int minRadius) = 0;
        virtual int getMinRadius() const = 0;

        virtual void setMaxRadius(int maxRadius) = 0;
        virtual int getMaxRadius() const = 0;

        virtual void setMaxCircles(int maxCircles) = 0;
        virtual int getMaxCircles() const = 0;
    };



gpu::HoughCirclesDetector::detect
---------------------------------
Finds circles in a grayscale image using the Hough transform.

.. ocv:function:: void gpu::HoughCirclesDetector::detect(InputArray src, OutputArray circles)

    :param src: 8-bit, single-channel grayscale input image.

    :param circles: Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  :math:`(x, y, radius)` .

.. seealso:: :ocv:func:`HoughCircles`



gpu::createHoughCirclesDetector
-------------------------------
Creates implementation for :ocv:class:`gpu::HoughCirclesDetector` .

.. ocv:function:: Ptr<HoughCirclesDetector> gpu::createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096)

    :param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if  ``dp=1`` , the accumulator has the same resolution as the input image. If  ``dp=2`` , the accumulator has half as big width and height.

    :param minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.

    :param cannyThreshold: The higher threshold of the two passed to Canny edge detector (the lower one is twice smaller).

    :param votesThreshold: The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.

    :param minRadius: Minimum circle radius.

    :param maxRadius: Maximum circle radius.

    :param maxCircles: Maximum number of output circles.



gpu::createGeneralizedHoughBallard
----------------------------------
Creates implementation for generalized hough transform from [Ballard1981]_ .

.. ocv:function:: Ptr<GeneralizedHoughBallard> gpu::createGeneralizedHoughBallard()



gpu::createGeneralizedHoughGuil
-------------------------------
Creates implementation for generalized hough transform from [Guil1999]_ .

.. ocv:function:: Ptr<GeneralizedHoughGuil> gpu::createGeneralizedHoughGuil()



.. [Ballard1981] Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
.. [Guil1999] Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
