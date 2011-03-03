Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

.. index:: calcOpticalFlowPyrLK

calcOpticalFlowPyrLK
------------------------
.. c:function:: void calcOpticalFlowPyrLK( const Mat\& prevImg, const Mat\& nextImg,        const vector<Point2f>\& prevPts, vector<Point2f>\& nextPts,        vector<uchar>\& status, vector<float>\& err,         Size winSize=Size(15,15), int maxLevel=3,        TermCriteria criteria=TermCriteria(            TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),        double derivLambda=0.5, int flags=0 )

    Calculates the optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids

    :param prevImg: The first 8-bit single-channel or 3-channel input image

    :param nextImg: The second input image of the same size and the same type as  ``prevImg``
    :param prevPts: Vector of points for which the flow needs to be found

    :param nextPts: The output vector of points containing the calculated new positions of the input features in the second image

    :param status: The output status vector. Each element of the vector is set to 1 if the flow for the corresponding features has been found, 0 otherwise

    :param err: The output vector that will contain the difference between patches around the original and moved points

    :param winSize: Size of the search window at each pyramid level

    :param maxLevel: 0-based maximal pyramid level number. If 0, pyramids are not used (single level), if 1, two levels are used etc.

    :param criteria: Specifies the termination criteria of the iterative search algorithm (after the specified maximum number of iterations  ``criteria.maxCount``  or when the search window moves by less than  ``criteria.epsilon``
    :param derivLambda: The relative weight of the spatial image derivatives impact to the optical flow estimation. If  ``derivLambda=0`` , only the image intensity is used, if  ``derivLambda=1`` , only derivatives are used. Any other values between 0 and 1 means that both derivatives and the image intensity are used (in the corresponding proportions).

    :param flags: The operation flags:

            * **OPTFLOW_USE_INITIAL_FLOW** use initial estimations stored in  ``nextPts`` . If the flag is not set, then initially  :math:`\texttt{nextPts}\leftarrow\texttt{prevPts}`
            
The function implements the sparse iterative version of the Lucas-Kanade optical flow in pyramids, see
Bouguet00
.

.. index:: calcOpticalFlowFarneback

calcOpticalFlowFarneback
----------------------------
.. c:function:: void calcOpticalFlowFarneback( const Mat\& prevImg, const Mat\& nextImg,                               Mat\& flow, double pyrScale, int levels, int winsize,                               int iterations, int polyN, double polySigma, int flags )

    Computes dense optical flow using Gunnar Farneback's algorithm

    :param prevImg: The first 8-bit single-channel input image

    :param nextImg: The second input image of the same size and the same type as  ``prevImg``
    :param flow: The computed flow image; will have the same size as  ``prevImg``  and type  ``CV_32FC2``
    :param pyrScale: Specifies the image scale (<1) to build the pyramids for each image.  ``pyrScale=0.5``  means the classical pyramid, where each next layer is twice smaller than the previous

    :param levels: The number of pyramid layers, including the initial image.  ``levels=1``  means that no extra layers are created and only the original images are used

    :param winsize: The averaging window size; The larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field

    :param iterations: The number of iterations the algorithm does at each pyramid level

    :param polyN: Size of the pixel neighborhood used to find polynomial expansion in each pixel. The larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred  motion field. Typically,  ``polyN`` =5 or 7

    :param polySigma: Standard deviation of the Gaussian that is used to smooth derivatives that are used as a basis for the polynomial expansion. For  ``polyN=5``  you can set  ``polySigma=1.1`` , for  ``polyN=7``  a good value would be  ``polySigma=1.5``
    :param flags: The operation flags; can be a combination of the following:

            * **OPTFLOW_USE_INITIAL_FLOW** Use the input  ``flow``  as the initial flow approximation

            * **OPTFLOW_FARNEBACK_GAUSSIAN** Use a Gaussian  :math:`\texttt{winsize}\times\texttt{winsize}`  filter instead of box filter of the same size for optical flow estimation. Usually, this option gives more accurate flow than with a box filter, at the cost of lower speed (and normally  ``winsize``  for a Gaussian window should be set to a larger value to achieve the same level of robustness)

The function finds optical flow for each ``prevImg`` pixel using the alorithm so that

.. math::

    \texttt{prevImg} (x,y)  \sim \texttt{nextImg} ( \texttt{flow} (x,y)[0],  \texttt{flow} (x,y)[1])

.. index:: updateMotionHistory

updateMotionHistory
-----------------------
.. c:function:: void updateMotionHistory( const Mat\& silhouette, Mat\& mhi,                          double timestamp, double duration )

    Updates the motion history image by a moving silhouette.

    :param silhouette: Silhouette mask that has non-zero pixels where the motion occurs

    :param mhi: Motion history image, that is updated by the function (single-channel, 32-bit floating-point)

    :param timestamp: Current time in milliseconds or other units

    :param duration: Maximal duration of the motion track in the same units as  ``timestamp``

The function updates the motion history image as following:

.. math::

    \texttt{mhi} (x,y)= \forkthree{\texttt{timestamp}}{if $\texttt{silhouette}(x,y) \ne 0$}{0}{if $\texttt{silhouette}(x,y) = 0$ and $\texttt{mhi} < (\texttt{timestamp} - \texttt{duration})$}{\texttt{mhi}(x,y)}{otherwise}

That is, MHI pixels where motion occurs are set to the current ``timestamp`` , while the pixels where motion happened last time a long time ago are cleared.

The function, together with
:func:`calcMotionGradient` and
:func:`calcGlobalOrientation` , implements the motion templates technique, described in
Davis97
and
Bradski00
.
See also the OpenCV sample ``motempl.c`` that demonstrates the use of all the motion template functions.

.. index:: calcMotionGradient

calcMotionGradient
----------------------
.. c:function:: void calcMotionGradient( const Mat\& mhi, Mat\& mask,                         Mat\& orientation,                         double delta1, double delta2,                         int apertureSize=3 )

    Calculates the gradient orientation of a motion history image.

    :param mhi: Motion history single-channel floating-point image

    :param mask: The output mask image; will have the type  ``CV_8UC1``  and the same size as  ``mhi`` . Its non-zero elements will mark pixels where the motion gradient data is correct

    :param orientation: The output motion gradient orientation image; will have the same type and the same size as  ``mhi`` . Each pixel of it will the motion orientation in degrees, from 0 to 360.

    :param delta1, delta2: The minimal and maximal allowed difference between  ``mhi``  values within a pixel neighorhood. That is, the function finds the minimum ( :math:`m(x,y)` ) and maximum ( :math:`M(x,y)` )  ``mhi``  values over  :math:`3 \times 3`  neighborhood of each pixel and marks the motion orientation at  :math:`(x, y)`  as valid only if

        .. math::

            \min ( \texttt{delta1}  ,  \texttt{delta2}  )  \le  M(x,y)-m(x,y)  \le   \max ( \texttt{delta1}  , \texttt{delta2} ).

    :param apertureSize: The aperture size of  :func:`Sobel`  operator

The function calculates the gradient orientation at each pixel
:math:`(x, y)` as:

.. math::

    \texttt{orientation} (x,y)= \arctan{\frac{d\texttt{mhi}/dy}{d\texttt{mhi}/dx}}

(in fact,
:func:`fastArctan` and
:func:`phase` are used, so that the computed angle is measured in degrees and covers the full range 0..360). Also, the ``mask`` is filled to indicate pixels where the computed angle is valid.

.. index:: calcGlobalOrientation

calcGlobalOrientation
-------------------------
.. c:function:: double calcGlobalOrientation( const Mat\& orientation, const Mat\& mask,                              const Mat\& mhi, double timestamp,                              double duration )

    Calculates the global motion orientation in some selected region.

    :param orientation: Motion gradient orientation image, calculated by the function  :func:`calcMotionGradient`
    
    :param mask: Mask image. It may be a conjunction of a valid gradient mask, also calculated by  :func:`calcMotionGradient` , and the mask of the region, whose direction needs to be calculated

    :param mhi: The motion history image, calculated by  :func:`updateMotionHistory`
    
    :param timestamp: The timestamp passed to  :func:`updateMotionHistory`
    
    :param duration: Maximal duration of motion track in milliseconds, passed to  :func:`updateMotionHistory`

The function calculates the average
motion direction in the selected region and returns the angle between
0 degrees  and 360 degrees. The average direction is computed from
the weighted orientation histogram, where a recent motion has larger
weight and the motion occurred in the past has smaller weight, as recorded in ``mhi`` .

.. index:: CamShift

CamShift
------------
.. c:function:: RotatedRect CamShift( const Mat\& probImage, Rect\& window,                      TermCriteria criteria )

    Finds the object center, size, and orientation

    :param probImage: Back projection of the object histogram; see  :func:`calcBackProject`
    
    :param window: Initial search window

    :param criteria: Stop criteria for the underlying  :func:`meanShift`

The function implements the CAMSHIFT object tracking algrorithm
Bradski98
.
First, it finds an object center using
:func:`meanShift` and then adjust the window size and finds the optimal rotation. The function returns the rotated rectangle structure that includes the object position, size and the orientation. The next position of the search window can be obtained with ``RotatedRect::boundingRect()`` .

See the OpenCV sample ``camshiftdemo.c`` that tracks colored objects.

.. index:: meanShift

meanShift
-------------
.. c:function:: int meanShift( const Mat\& probImage, Rect\& window,               TermCriteria criteria )

    Finds the object on a back projection image.

    :param probImage: Back projection of the object histogram; see  :func:`calcBackProject`
    :param window: Initial search window

    :param criteria: The stop criteria for the iterative search algorithm

The function implements iterative object search algorithm. It takes the object back projection on input and the initial position. The mass center in ``window`` of the back projection image is computed and the search window center shifts to the mass center. The procedure is repeated until the specified number of iterations ``criteria.maxCount`` is done or until the window center shifts by less than ``criteria.epsilon`` . The algorithm is used inside
:func:`CamShift` and, unlike
:func:`CamShift` , the search window size or orientation do not change during the search. You can simply pass the output of
:func:`calcBackProject` to this function, but better results can be obtained if you pre-filter the back projection and remove the noise (e.g. by retrieving connected components with
:func:`findContours` , throwing away contours with small area (
:func:`contourArea` ) and rendering the  remaining contours with
:func:`drawContours` )

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
        // and stores the result to statePost.
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


The class implements standard Kalman filter
http://en.wikipedia.org/wiki/Kalman_filter
. However, you can modify ``transitionMatrix``,``controlMatrix`` and ``measurementMatrix`` to get the extended Kalman filter functionality. See the OpenCV sample ``kalman.c`` 