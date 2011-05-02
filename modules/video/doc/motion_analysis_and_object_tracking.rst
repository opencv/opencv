Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

.. index:: calcOpticalFlowPyrLK

calcOpticalFlowPyrLK
------------------------
.. c:function:: void calcOpticalFlowPyrLK( const Mat\& prevImg, const Mat\& nextImg,        const vector<Point2f>\& prevPts, vector<Point2f>\& nextPts,        vector<uchar>\& status, vector<float>\& err,         Size winSize=Size(15,15), int maxLevel=3,        TermCriteria criteria=TermCriteria(            TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),        double derivLambda=0.5, int flags=0 )

    Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.

    :param prevImg: The first 8-bit single-channel or 3-channel input image.

    :param nextImg: The second input image of the same size and the same type as  ``prevImg`` .
	
    :param prevPts: Vector of points for which the flow needs to be found.

    :param nextPts: Output vector of points containing the calculated new positions of input features in the second image.

    :param status: Output status vector. Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector that contains the difference between patches around the original and moved points.

    :param winSize: Size of the search window at each pyramid level.

    :param maxLevel: 0-based maximal pyramid level number. If set to 0, pyramids are not used (single level). If set to 1, two levels are used, and so on.

    :param criteria: Parameter specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations  ``criteria.maxCount``  or when the search window moves by less than  ``criteria.epsilon`` .
	
    :param derivLambda: Relative weight of the spatial image derivatives impact to the optical flow estimation. If  ``derivLambda=0`` , only the image intensity is used. If  ``derivLambda=1`` , only derivatives are used. Any other values between 0 and 1 mean that both derivatives and the image intensity are used (in the corresponding proportions).

    :param flags: Operation flags:

            * **OPTFLOW_USE_INITIAL_FLOW** Use initial estimations stored in  ``nextPts`` . If the flag is not set, then initially??  :math:`\texttt{nextPts}\leftarrow\texttt{prevPts}` .
            
The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
Bouguet00
.

.. index:: calcOpticalFlowFarneback

calcOpticalFlowFarneback
----------------------------
.. c:function:: void calcOpticalFlowFarneback( const Mat\& prevImg, const Mat\& nextImg,                               Mat\& flow, double pyrScale, int levels, int winsize,                               int iterations, int polyN, double polySigma, int flags )

    Computes a dense optical flow using the Gunnar Farneback's algorithm.

    :param prevImg: The first 8-bit single-channel input image.

    :param nextImg: The second input image of the same size and the same type as  ``prevImg`` .
	
    :param flow: Computed flow image that has the same size as  ``prevImg``  and type  ``CV_32FC2`` .
	
    :param pyrScale: Parameter specifying the image scale (<1) to build pyramids for each image.  ``pyrScale=0.5``  means a classical pyramid, where each next layer is twice smaller than the previous one.

    :param levels: Number of pyramid layers including the initial image.  ``levels=1``  means that no extra layers are created and only the original images are used.

    :param winsize: Averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.

    :param iterations: Number of iterations the algorithm does at each pyramid level.

    :param polyN: Size of the pixel neighborhood used to find polynomial expansion in each pixel. Larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred  motion field. Typically,  ``polyN`` =5 or 7.

    :param polySigma: Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion. For  ``polyN=5`` ,  you can set  ``polySigma=1.1`` . For  ``polyN=7`` , a good value would be  ``polySigma=1.5`` .
	
    :param flags: Operation flags that can be a combination of the following:

            * **OPTFLOW_USE_INITIAL_FLOW** Use the input  ``flow``  as an initial flow approximation.

            * **OPTFLOW_FARNEBACK_GAUSSIAN** Use the Gaussian  :math:`\texttt{winsize}\times\texttt{winsize}`  filter instead of a box filter of the same size for optical flow estimation. Usually, this option gives more accurate flow than with a box filter, at the cost of lower speed. Normally,  ``winsize``  for a Gaussian window should be set to a larger value to achieve the same level of robustness.

The function finds an optical flow for each ``prevImg`` pixel using the alorithm so that

.. math::

    \texttt{prevImg} (x,y)  \sim \texttt{nextImg} ( \texttt{flow} (x,y)[0],  \texttt{flow} (x,y)[1])

.. index:: updateMotionHistory

updateMotionHistory
-----------------------
.. c:function:: void updateMotionHistory( const Mat\& silhouette, Mat\& mhi,                          double timestamp, double duration )

    Updates the motion history image by a moving silhouette.

    :param silhouette: Silhouette mask that has non-zero pixels where the motion occurs.

    :param mhi: Motion history image that is updated by the function (single-channel, 32-bit floating-point).

    :param timestamp: Current time in milliseconds or other units.

    :param duration: Maximal duration of the motion track in the same units as  ``timestamp`` .

The function updates the motion history image as follows:

.. math::

    \texttt{mhi} (x,y)= \forkthree{\texttt{timestamp}}{if $\texttt{silhouette}(x,y) \ne 0$}{0}{if $\texttt{silhouette}(x,y) = 0$ and $\texttt{mhi} < (\texttt{timestamp} - \texttt{duration})$}{\texttt{mhi}(x,y)}{otherwise}

That is, MHI pixels where the motion occurs are set to the current ``timestamp`` , while the pixels where the motion happened last time a long time ago are cleared.

The function, together with
:func:`calcMotionGradient` and
:func:`calcGlobalOrientation` , implements a motion templates technique described in
Davis97
and
Bradski00
.
See also the OpenCV sample ``motempl.c`` that demonstrates the use of all the motion template functions.

.. index:: calcMotionGradient

calcMotionGradient
----------------------
.. c:function:: void calcMotionGradient( const Mat\& mhi, Mat\& mask,                         Mat\& orientation,                         double delta1, double delta2,                         int apertureSize=3 )

    Calculates a gradient orientation of a motion history image.

    :param mhi: Motion history single-channel floating-point image.

    :param mask: Output mask image that has the type  ``CV_8UC1``  and the same size as  ``mhi`` . Its non-zero elements mark pixels where the motion gradient data is correct.

    :param orientation: Output motion gradient orientation image that has the same type and the same size as  ``mhi`` . Each pixel of the image is a motion orientation in degrees, from 0 to 360.??

    :param delta1, delta2: Minimum and maximum allowed difference between  ``mhi``  values within a pixel neighorhood. That is, the function finds the minimum ( :math:`m(x,y)` ) and maximum ( :math:`M(x,y)` )  ``mhi``  values over  :math:`3 \times 3`  neighborhood of each pixel and marks the motion orientation at  :math:`(x, y)`  as valid only if

        .. math::

            \min ( \texttt{delta1}  ,  \texttt{delta2}  )  \le  M(x,y)-m(x,y)  \le   \max ( \texttt{delta1}  , \texttt{delta2} ).

    :param apertureSize: Aperture size of  the :func:`Sobel`  operator.

The function calculates a gradient orientation at each pixel
:math:`(x, y)` as:

.. math::

    \texttt{orientation} (x,y)= \arctan{\frac{d\texttt{mhi}/dy}{d\texttt{mhi}/dx}}

In fact,
:func:`fastArctan` and
:func:`phase` are used so that the computed angle is measured in degrees and covers the full range 0..360. Also, the ``mask`` is filled to indicate pixels where the computed angle is valid.

.. index:: calcGlobalOrientation

calcGlobalOrientation
-------------------------
.. c:function:: double calcGlobalOrientation( const Mat\& orientation, const Mat\& mask,                              const Mat\& mhi, double timestamp,                              double duration )

    Calculates a global motion orientation in a selected region.

    :param orientation: Motion gradient orientation image calculated by the function  :func:`calcMotionGradient` .
    
    :param mask: Mask image. It may be a conjunction of a valid gradient mask, also calculated by  :func:`calcMotionGradient` , and the mask of a region whose direction needs to be calculated.

    :param mhi: Motion history image calculated by  :func:`updateMotionHistory` .
    
    :param timestamp: Timestamp passed to  :func:`updateMotionHistory` .
    
    :param duration: Maximum duration of a motion track in milliseconds, passed to  :func:`updateMotionHistory` .

The function calculates an average
motion direction in the selected region and returns the angle between
0 degrees  and 360 degrees. The average direction is computed from
the weighted orientation histogram, where a recent motion has a larger
weight and the motion occurred in the past has a smaller weight, as recorded in ``mhi`` .

.. index:: CamShift

CamShift
------------
.. c:function:: RotatedRect CamShift( const Mat\& probImage, Rect\& window,                      TermCriteria criteria )

    Finds an object center, size, and orientation.

    :param probImage: Back projection of the object histogram. See  :func:`calcBackProject` .
    
    :param window: Initial search window.

    :param criteria: Stop criteria for the underlying  :func:`meanShift` .

The function implements the CAMSHIFT object tracking algrorithm
Bradski98
.
First, it finds an object center using
:func:`meanShift` and then adjusts the window size and finds the optimal rotation. The function returns the rotated rectangle structure that includes the object position, size, and orientation. The next position of the search window can be obtained with ``RotatedRect::boundingRect()`` .

See the OpenCV sample ``camshiftdemo.c`` that tracks colored objects.

.. index:: meanShift

meanShift
-------------
.. c:function:: int meanShift( const Mat\& probImage, Rect\& window,               TermCriteria criteria )

    Finds an object on a back projection image.

    :param probImage: Back projection of the object histogram. See  :func:`calcBackProject` for details.
	
    :param window: Initial search window.

    :param criteria: Stop criteria for the iterative search algorithm.

The function implements the iterative object search algorithm. It takes the input back projection of an object and the initial position. The mass center in ``window`` of the back projection image is computed and the search window center shifts to the mass center. The procedure is repeated until the specified number of iterations ``criteria.maxCount`` is done or until the window center shifts by less than ``criteria.epsilon`` . The algorithm is used inside
:func:`CamShift` and, unlike
:func:`CamShift` , the search window size or orientation do not change during the search. You can simply pass the output of
:func:`calcBackProject` to this function. But better results can be obtained if you pre-filter the back projection and remove the noise (for example, by retrieving connected components with
:func:`findContours` , throwing away contours with small area (
:func:`contourArea` ), and rendering the  remaining contours with
:func:`drawContours` ).

.. index:: KalmanFilter

.. _KalmanFilter:

KalmanFilter
------------
.. c:type:: KalmanFilter

Kalman filter class ::

    class KalmanFilter
    {
    public:
        KalmanFilter();
        KalmanFilter(int dynamParams, int measureParams, int controlParams=0);
        void init(int dynamParams, int measureParams, int controlParams=0);
        // predicts statePre from statePost
        const Mat& predict(const Mat& control=Mat());
        // corrects statePre based on the input measurement vector
        // and stores the result in statePost.
        const Mat& correct(const Mat& measurement);

        Mat statePre;           // predicted state (x'(k)):
                                //    x(k)=A*x(k-1)+B*u(k)
        Mat statePost;          // corrected state (x(k)):
                                //    x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
        Mat transitionMatrix;   // state transition matrix (A)
        Mat controlMatrix;      // control matrix (B)
                                //   (it is not used if there is no control)
        Mat measurementMatrix;  // measurement matrix (H)
        Mat processNoiseCov;    // process noise covariance matrix (Q)
        Mat measurementNoiseCov;// measurement noise covariance matrix (R)
        Mat errorCovPre;        // priori error estimate covariance matrix (P'(k)):
                                //    P'(k)=A*P(k-1)*At + Q)*/
        Mat gain;               // Kalman gain matrix (K(k)):
                                //    K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
        Mat errorCovPost;       // posteriori error estimate covariance matrix (P(k)):
                                //    P(k)=(I-K(k)*H)*P'(k)
        ...
    };


The class implements a standard Kalman filter
http://en.wikipedia.org/wiki/Kalman_filter
. However, you can modify ``transitionMatrix``,``controlMatrix`` , and ``measurementMatrix`` to get an extended Kalman filter functionality. See the OpenCV sample ``kalman.c`` .