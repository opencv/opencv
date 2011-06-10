Motion Analysis and Object Tracking
===================================

.. highlight:: cpp

.. index:: calcOpticalFlowPyrLK

calcOpticalFlowPyrLK
------------------------
.. cpp:function:: void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, OutputArray status, OutputArray err, Size winSize=Size(15,15), int maxLevel=3,        TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), double derivLambda=0.5, int flags=0 )

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

            * **OPTFLOW_USE_INITIAL_FLOW** Use initial estimations stored in  ``nextPts`` . If the flag is not set, then ``prevPts`` is copied to ``nextPts`` and is considered as the initial estimate.
            
The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See
Bouguet00
.

.. index:: calcOpticalFlowFarneback

calcOpticalFlowFarneback
----------------------------
.. cpp:function:: void calcOpticalFlowFarneback( InputArray prevImg, InputArray nextImg,                               InputOutputArray flow, double pyrScale, int levels, int winsize, int iterations, int polyN, double polySigma, int flags )

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


.. index:: estimateRigidTransform

estimateRigidTransform
--------------------------
.. cpp:function:: Mat estimateRigidTransform( InputArray src, InputArray dst, bool fullAffine )

    Computes an optimal affine transformation between two 2D point sets.

    :param src: The first input 2D point set, stored in ``std::vector`` or ``Mat``, or an image, stored in ``Mat``

    :param dst: The second input 2D point set of the same size and the same type as ``A``, or another image.

    :param fullAffine: If true, the function finds an optimal affine transformation with no additional resrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (5 degrees of freedom).

The function finds an optimal affine transform *[A|b]* (a ``2 x 3`` floating-point matrix) that approximates best the affine transformation between:

  #.
      two point sets
  #.
      or between 2 raster images. In this case, the function first finds some features in the ``src`` image and finds the corresponding features in ``dst`` image, after which the problem is reduced to the first case.
      
In the case of point sets, the problem is formulated in the following way. We need to find such 2x2 matrix *A* and 2x1 vector *b*, such that:

    .. math::

        [A^*|b^*] = arg  \min _{[A|b]}  \sum _i  \| \texttt{dst}[i] - A { \texttt{src}[i]}^T - b  \| ^2

    where ``src[i]`` and ``dst[i]`` are the i-th points in ``src`` and ``dst``, respectively
    
    :math:`[A|b]` can be either arbitrary (when ``fullAffine=true`` ) or have form

    .. math::

        \begin{bmatrix} a_{11} & a_{12} & b_1  \\ -a_{12} & a_{11} & b_2  \end{bmatrix}

    when ``fullAffine=false`` .

    See Also:
    :cpp:func:`getAffineTransform`,
    :cpp:func:`getPerspectiveTransform`,
    :cpp:func:`findHomography`


.. index:: updateMotionHistory

updateMotionHistory
-----------------------
.. cpp:function:: void updateMotionHistory( InputArray silhouette, InputOutputArray mhi, double timestamp, double duration )

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
:cpp:func:`calcMotionGradient` and
:cpp:func:`calcGlobalOrientation` , implements a motion templates technique described in
Davis97
and
Bradski00
.
See also the OpenCV sample ``motempl.c`` that demonstrates the use of all the motion template functions.

.. index:: calcMotionGradient

calcMotionGradient
----------------------
.. cpp:function:: void calcMotionGradient( InputArray mhi, OutputArray mask, OutputArray orientation,                         double delta1, double delta2, int apertureSize=3 )

    Calculates a gradient orientation of a motion history image.

    :param mhi: Motion history single-channel floating-point image.

    :param mask: Output mask image that has the type  ``CV_8UC1``  and the same size as  ``mhi`` . Its non-zero elements mark pixels where the motion gradient data is correct.

    :param orientation: Output motion gradient orientation image that has the same type and the same size as  ``mhi`` . Each pixel of the image is a motion orientation, from 0 to 360 degrees.

    :param delta1, delta2: Minimum and maximum allowed difference between  ``mhi``  values within a pixel neighorhood. That is, the function finds the minimum ( :math:`m(x,y)` ) and maximum ( :math:`M(x,y)` )  ``mhi``  values over  :math:`3 \times 3`  neighborhood of each pixel and marks the motion orientation at  :math:`(x, y)`  as valid only if

        .. math::

            \min ( \texttt{delta1}  ,  \texttt{delta2}  )  \le  M(x,y)-m(x,y)  \le   \max ( \texttt{delta1}  , \texttt{delta2} ).

    :param apertureSize: Aperture size of  the :cpp:func:`Sobel`  operator.

The function calculates a gradient orientation at each pixel
:math:`(x, y)` as:

.. math::

    \texttt{orientation} (x,y)= \arctan{\frac{d\texttt{mhi}/dy}{d\texttt{mhi}/dx}}

In fact,
:cpp:func:`fastArctan` and
:cpp:func:`phase` are used so that the computed angle is measured in degrees and covers the full range 0..360. Also, the ``mask`` is filled to indicate pixels where the computed angle is valid.

.. index:: calcGlobalOrientation

calcGlobalOrientation
-------------------------
.. cpp:function:: double calcGlobalOrientation( InputArray orientation, InputArray mask, InputArray mhi, double timestamp, double duration )

    Calculates a global motion orientation in a selected region.

    :param orientation: Motion gradient orientation image calculated by the function  :cpp:func:`calcMotionGradient` .
    
    :param mask: Mask image. It may be a conjunction of a valid gradient mask, also calculated by  :cpp:func:`calcMotionGradient` , and the mask of a region whose direction needs to be calculated.

    :param mhi: Motion history image calculated by  :cpp:func:`updateMotionHistory` .
    
    :param timestamp: Timestamp passed to  :cpp:func:`updateMotionHistory` .
    
    :param duration: Maximum duration of a motion track in milliseconds, passed to  :cpp:func:`updateMotionHistory` .

The function calculates an average
motion direction in the selected region and returns the angle between
0 degrees  and 360 degrees. The average direction is computed from
the weighted orientation histogram, where a recent motion has a larger
weight and the motion occurred in the past has a smaller weight, as recorded in ``mhi`` .

.. index:: CamShift

CamShift
------------
.. cpp:function:: RotatedRect CamShift( InputArray probImage, Rect& window, TermCriteria criteria )

    Finds an object center, size, and orientation.

    :param probImage: Back projection of the object histogram. See  :cpp:func:`calcBackProject` .
    
    :param window: Initial search window.

    :param criteria: Stop criteria for the underlying  :cpp:func:`meanShift` .

The function implements the CAMSHIFT object tracking algrorithm
Bradski98
.
First, it finds an object center using
:cpp:func:`meanShift` and then adjusts the window size and finds the optimal rotation. The function returns the rotated rectangle structure that includes the object position, size, and orientation. The next position of the search window can be obtained with ``RotatedRect::boundingRect()`` .

See the OpenCV sample ``camshiftdemo.c`` that tracks colored objects.

.. index:: meanShift

meanShift
-------------
.. cpp:function:: int meanShift( InputArray probImage, Rect& window, TermCriteria criteria )

    Finds an object on a back projection image.

    :param probImage: Back projection of the object histogram. See  :cpp:func:`calcBackProject` for details.
	
    :param window: Initial search window.

    :param criteria: Stop criteria for the iterative search algorithm.

The function implements the iterative object search algorithm. It takes the input back projection of an object and the initial position. The mass center in ``window`` of the back projection image is computed and the search window center shifts to the mass center. The procedure is repeated until the specified number of iterations ``criteria.maxCount`` is done or until the window center shifts by less than ``criteria.epsilon`` . The algorithm is used inside
:cpp:func:`CamShift` and, unlike
:cpp:func:`CamShift` , the search window size or orientation do not change during the search. You can simply pass the output of
:cpp:func:`calcBackProject` to this function. But better results can be obtained if you pre-filter the back projection and remove the noise (for example, by retrieving connected components with
:cpp:func:`findContours` , throwing away contours with small area (
:cpp:func:`contourArea` ), and rendering the  remaining contours with
:cpp:func:`drawContours` ).

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



BackgroundSubtractor
--------------------

.. cpp:class: BackgroundSubtractor

The base class for background/foreground segmentation. ::

    class BackgroundSubtractor
    {
    public:
        virtual ~BackgroundSubtractor();
        virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=0);
        virtual void getBackgroundImage(OutputArray backgroundImage) const;
    };


The class is only used to define the common interface for the whole family of background/foreground segmentation algorithms.

BackgroundSubtractor::operator()
-------------------------------

.. cpp:function:: virtual void BackgroundSubtractor::operator()(InputArray image, OutputArray fgmask, double learningRate=0)

    Computes foreground mask.

    :param image: The next video frame.

    :param fgmask: The foreground mask as 8-bit binary image


BackgroundSubtractor::getBackgroundImage
----------------------------------------

.. cpp:function:: virtual void BackgroundSubtractor::getBackgroundImage(OutputArray backgroundImage) const

This method computes a background image.


BackgroundSubtractorMOG
-----------------------

.. cpp:class: BackgroundSubtractorMOG : public BackgroundSubtractor

    Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm.

The class implements the following algorithm: P. KadewTraKuPong and R. Bowden, An improved adaptive background mixture model for real-time tracking with shadow detection, Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001: http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf


BackgroundSubtractorMOG::BackgroundSubtractorMOG
------------------------------------------------

.. cpp:function:: BackgroundSubtractorMOG::BackgroundSubtractorMOG()

.. cpp:function:: BackgroundSubtractorMOG::BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma=0)

    :param history: The length of the history.

    :param nmixtures: The number of gaussian mixtures.

    :param backgroundRatio: Background ratio.

    :param noiseSigma: The noise strength.

Default constructor sets all parameters to some default values.


BackgroundSubtractorMOG::operator()
-----------------------------------

.. cpp:function:: virtual void BackgroundSubtractorMOG::operator()(InputArray image, OutputArray fgmask, double learningRate=0)

    The update operator.


BackgroundSubtractorMOG::initialize
------------------------------------------------

.. cpp:function: virtual void BackgroundSubtractorMOG::initialize(Size frameSize, int frameType)

    Re-initiaization method.


BackgroundSubtractorMOG2
------------------------

.. cpp:class: BackgroundSubtractorMOG2 : public BackgroundSubtractor

    Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm.

The class implements the Gaussian mixture model background subtraction from: 

  * Z.Zivkovic, Improved adaptive Gausian mixture model for background subtraction, International Conference Pattern Recognition, UK, August, 2004, http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf. The code is very fast and performs also shadow detection. Number of Gausssian components is adapted per pixel.

  * Z.Zivkovic, F. van der Heijden, Efficient Adaptive Density Estimapion per Image Pixel for the Task of Background Subtraction, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006. The algorithm similar to the standard Stauffer&Grimson algorithm with additional selection of the number of the Gaussian components based on: Z.Zivkovic, F.van der Heijden, Recursive unsupervised learning of finite mixture models, IEEE Trans. on Pattern Analysis and Machine Intelligence, vol.26, no.5, pages 651-656, 2004.


BackgroundSubtractorMOG2::BackgroundSubtractorMOG2
--------------------------------------------------

.. cpp:function: BackgroundSubtractorMOG2::BackgroundSubtractorMOG2()

.. cpp:function: BackgroundSubtractorMOG2::BackgroundSubtractorMOG2(int history, float varThreshold, bool bShadowDetection=1)

    :param history: The length of the history.

    :param varThreshold: Threshold on the squared Mahalanobis distance to decide if it is well described by the background model or not. Related to Cthr from the paper. This does not influence the update of the background. A typical value could be 4 sigma and that is varThreshold=4*4=16; Corresponds to Tb in the paper.

    :param bShadowDetection: Do shadow detection (true) or not (false).


The class has an important public parameter:

    :param nmixtures: The maximum allowed number of mixture comonents. Actual number is determined dynamically per pixel.

Also the class has several less important parameters - things you might change but be carefull:

    :param backgroundRatio: Corresponds to fTB=1-cf from the paper. TB - threshold when the component becomes significant enough to be included into the background model. It is the TB=1-cf from the paper. Default is cf=0.1 => TB=0.9. For alpha=0.001 it means that the mode should exist for approximately 105 frames before it is considered foreground.

    :param varThresholdGen: Correspondts to Tg - threshold on the squared Mahalanobis distance to decide when a sample is close to the existing components. If it is not close to any a new component will be generated. Default is 3 sigma => Tg=3*3=9. Smaller Tg leads to more generated components and higher Tg might make lead to small number of components but they can grow too large.

    :param fVarInit: Initial variance for the newly generated components. It will will influence the speed of adaptation. A good guess should be made. A simple way is to estimate the typical standard deviation from the images. OpenCV uses here 15 as a reasonable value.

    :param fVarMin: Used to further control the variance.

    :param fVarMax: Used to further control the variance.

    :param fCT: Complexity reduction prior. This is related to the number of samples needed to accept that a component actually exists. Default is CT=0.05 of all the samples. By setting CT=0 you get the standard Stauffer&Grimson algorithm (maybe not exact but very similar).

    :param nShadowDetection: This value is inserted as the shadow detection result. Default value is 127.

    :param fTau: Shadow threshold. The shadow is detected if the pixel is darker version of the background. Tau is a threshold on how much darker the shadow can be. Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow. See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.
                 

BackgroundSubtractorMOG2::operator()
-----------------------------------

.. cpp:function:: virtual void BackgroundSubtractorMOG2::operator()(InputArray image, OutputArray fgmask, double learningRate=-1)

    The update operator.


BackgroundSubtractorMOG2::initialize
------------------------------------

.. cpp:function: virtual void BackgroundSubtractorMOG2::initialize(Size frameSize, int frameType)

     Re-initiaization method.


BackgroundSubtractorMOG2::getBackgroundImage
--------------------------------------------

.. cpp:function: virtual void BackgroundSubtractorMOG2::getBackgroundImage(OutputArray backgroundImage) const

    Computes a background image which are the mean of all background gaussians.


