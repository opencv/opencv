Motion Analysis and Object Tracking
===================================

.. highlight:: cpp


calcOpticalFlowPyrLK
------------------------
Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.

.. ocv:function:: void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, OutputArray status, OutputArray err, Size winSize=Size(15,15), int maxLevel=3, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags=0, double minEigThreshold=1e-4)

.. ocv:pyfunction:: cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]]) -> nextPts, status, err

.. ocv:cfunction:: void cvCalcOpticalFlowPyrLK( const CvArr* prev, const CvArr* curr, CvArr* prevPyr, CvArr* currPyr, const CvPoint2D32f* prevFeatures, CvPoint2D32f* currFeatures, int count, CvSize winSize, int level, char* status, float* trackError, CvTermCriteria criteria, int flags )
.. ocv:pyoldfunction:: cv.CalcOpticalFlowPyrLK( prev, curr, prevPyr, currPyr, prevFeatures, winSize, level, criteria, flags, guesses=None) -> (currFeatures, status, trackError)

    :param prevImg: First 8-bit input image or pyramid constructed by :ocv:func:`buildOpticalFlowPyramid`.

    :param nextImg: Second input image or pyramid of the same size and the same type as  ``prevImg``.

    :param prevPts: Vector of 2D points for which the flow needs to be found. The point coordinates must be single-precision floating-point numbers.

    :param nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When ``OPTFLOW_USE_INITIAL_FLOW`` flag is passed, the vector must have the same size as in the input.

    :param status: Output status vector. Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector of errors. Each element of the vector is set to a error for the corresponding feature. A type of the error measure can be set in ``flags`` parameter. If the flow wasn't found then the error is not defined (use the ``status`` parameter to find such cases).

    :param winSize: Size of the search window at each pyramid level.

    :param maxLevel: 0-based maximal pyramid level number. If set to 0, pyramids are not used (single level). If set to 1, two levels are used, and so on. If pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than ``maxLevel``.

    :param criteria: Parameter specifying the termination criteria of the iterative search algorithm (after the specified maximum number of iterations  ``criteria.maxCount``  or when the search window moves by less than  ``criteria.epsilon`` .
    
    :param flags: Operation flags:
        
        * **OPTFLOW_USE_INITIAL_FLOW** Use initial estimations stored in  ``nextPts`` . If the flag is not set, then ``prevPts`` is copied to ``nextPts`` and is considered as the initial estimate.
        * **OPTFLOW_LK_GET_MIN_EIGENVALS** Use minimum eigen values as a error measure (see ``minEigThreshold`` description). If the flag is not set, then L1 distance between patches around the original and a moved point divided by number of pixels in a window is used as a error measure.

    :param minEigThreshold: The algorithm computes a minimum eigen value of a 2x2 normal matrix of optical flow equations (this matrix is called a spatial gradient matrix in [Bouguet00]_) divided by number of pixels in a window. If this value is less then ``minEigThreshold`` then a corresponding feature is filtered out and its flow is not computed. So it allows to remove bad points earlier and speed up the computation.
            
The function implements a sparse iterative version of the Lucas-Kanade optical flow in pyramids. See [Bouguet00]_. The function is parallelized with the TBB library.

buildOpticalFlowPyramid
-----------------------
Constructs the image pyramid which can be passed to :ocv:func:`calcOpticalFlowPyrLK`.

.. ocv:function:: int buildOpticalFlowPyramid(InputArray img, OutputArrayOfArrays pyramid, Size winSize, int maxLevel, bool withDerivatives = true, int pyrBorder = BORDER_REFLECT_101, int derivBorder = BORDER_CONSTANT, bool tryReuseInputImage = true)

.. ocv:pyfunction:: cv2.buildOpticalFlowPyramid(img, winSize, maxLevel[, pyramid[, withDerivatives[, pyrBorder[, derivBorder[, tryReuseInputImage]]]]]) -> retval, pyramid

    :param img: 8-bit input image.

    :param pyramid: output pyramid.

    :param winSize: window size of optical flow algorithm. Must be not less than ``winSize`` argument of :ocv:func:`calcOpticalFlowPyrLK`. It is needed to calculate required padding for pyramid levels.

    :param maxLevel: 0-based maximal pyramid level number.

    :param withDerivatives: set to precompute gradients for the every pyramid level. If pyramid is constructed without the gradients then :ocv:func:`calcOpticalFlowPyrLK` will calculate them internally.

    :param pyrBorder: the border mode for pyramid layers.

    :param derivBorder: the border mode for gradients.

    :param tryReuseInputImage: put ROI of input image into the pyramid if possible. You can pass ``false`` to force data copying.

    :return: number of levels in constructed pyramid. Can be less than ``maxLevel``.


calcOpticalFlowFarneback
----------------------------
Computes a dense optical flow using the Gunnar Farneback's algorithm.

.. ocv:function:: void calcOpticalFlowFarneback( InputArray prevImg, InputArray nextImg, InputOutputArray flow, double pyrScale, int levels, int winsize, int iterations, int polyN, double polySigma, int flags )

.. ocv:cfunction:: void cvCalcOpticalFlowFarneback( const CvArr* prevImg, const CvArr* nextImg, CvArr* flow, double pyrScale, int levels, int winsize, int iterations, int polyN, double polySigma, int flags )

.. ocv:pyfunction:: cv2.calcOpticalFlowFarneback(prevImg, nextImg, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]) -> flow

    :param prevImg: First 8-bit single-channel input image.

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param flow: Computed flow image that has the same size as  ``prevImg``  and type  ``CV_32FC2`` .

    :param pyrScale: Parameter specifying the image scale (<1) to build pyramids for each image.  ``pyrScale=0.5``  means a classical pyramid, where each next layer is twice smaller than the previous one.

    :param levels: Number of pyramid layers including the initial image.  ``levels=1``  means that no extra layers are created and only the original images are used.

    :param winsize: Averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.

    :param iterations: Number of iterations the algorithm does at each pyramid level.

    :param polyN: Size of the pixel neighborhood used to find polynomial expansion in each pixel. Larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred  motion field. Typically,  ``polyN`` =5 or 7.

    :param polySigma: Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion. For  ``polyN=5`` ,  you can set  ``polySigma=1.1`` . For  ``polyN=7`` , a good value would be  ``polySigma=1.5`` .
    
    :param flags: Operation flags that can be a combination of the following:

            * **OPTFLOW_USE_INITIAL_FLOW** Use the input  ``flow``  as an initial flow approximation.

            * **OPTFLOW_FARNEBACK_GAUSSIAN** Use the Gaussian  :math:`\texttt{winsize}\times\texttt{winsize}`  filter instead of a box filter of the same size for optical flow estimation. Usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed. Normally,  ``winsize``  for a Gaussian window should be set to a larger value to achieve the same level of robustness.

The function finds an optical flow for each ``prevImg`` pixel using the [Farneback2003]_ algorithm so that

.. math::

    \texttt{prevImg} (y,x)  \sim \texttt{nextImg} ( y + \texttt{flow} (y,x)[1],  x + \texttt{flow} (y,x)[0])


estimateRigidTransform
--------------------------
Computes an optimal affine transformation between two 2D point sets.

.. ocv:function:: Mat estimateRigidTransform( InputArray src, InputArray dst, bool fullAffine )

.. ocv:pyfunction:: cv2.estimateRigidTransform(src, dst, fullAffine) -> retval

    :param src: First input 2D point set stored in ``std::vector`` or ``Mat``, or an image stored in ``Mat``.

    :param dst: Second input 2D point set of the same size and the same type as ``A``, or another image.

    :param fullAffine: If true, the function finds an optimal affine transformation with no additional restrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (5 degrees of freedom).

The function finds an optimal affine transform *[A|b]* (a ``2 x 3`` floating-point matrix) that approximates best the affine transformation between:

  *
      Two point sets
  *
      Two raster images. In this case, the function first finds some features in the ``src`` image and finds the corresponding features in ``dst`` image. After that, the problem is reduced to the first case.
      
In case of point sets, the problem is formulated as follows: you need to find a 2x2 matrix *A* and 2x1 vector *b* so that:

    .. math::

        [A^*|b^*] = arg  \min _{[A|b]}  \sum _i  \| \texttt{dst}[i] - A { \texttt{src}[i]}^T - b  \| ^2

    where ``src[i]`` and ``dst[i]`` are the i-th points in ``src`` and ``dst``, respectively
    
    :math:`[A|b]` can be either arbitrary (when ``fullAffine=true`` ) or have a form of

    .. math::

        \begin{bmatrix} a_{11} & a_{12} & b_1  \\ -a_{12} & a_{11} & b_2  \end{bmatrix}

    when ``fullAffine=false`` .

.. seealso::

    :ocv:func:`getAffineTransform`,
    :ocv:func:`getPerspectiveTransform`,
    :ocv:func:`findHomography`



updateMotionHistory
-----------------------
Updates the motion history image by a moving silhouette.

.. ocv:function:: void updateMotionHistory( InputArray silhouette, InputOutputArray mhi, double timestamp, double duration )

.. ocv:pyfunction:: cv2.updateMotionHistory(silhouette, mhi, timestamp, duration) -> None

.. ocv:cfunction:: void cvUpdateMotionHistory( const CvArr* silhouette, CvArr* mhi, double timestamp, double duration )
.. ocv:pyoldfunction:: cv.UpdateMotionHistory(silhouette, mhi, timestamp, duration)-> None

    :param silhouette: Silhouette mask that has non-zero pixels where the motion occurs.

    :param mhi: Motion history image that is updated by the function (single-channel, 32-bit floating-point).

    :param timestamp: Current time in milliseconds or other units.

    :param duration: Maximal duration of the motion track in the same units as  ``timestamp`` .

The function updates the motion history image as follows:

.. math::

    \texttt{mhi} (x,y)= \forkthree{\texttt{timestamp}}{if $\texttt{silhouette}(x,y) \ne 0$}{0}{if $\texttt{silhouette}(x,y) = 0$ and $\texttt{mhi} < (\texttt{timestamp} - \texttt{duration})$}{\texttt{mhi}(x,y)}{otherwise}

That is, MHI pixels where the motion occurs are set to the current ``timestamp`` , while the pixels where the motion happened last time a long time ago are cleared.

The function, together with
:ocv:func:`calcMotionGradient` and
:ocv:func:`calcGlobalOrientation` , implements a motion templates technique described in
[Davis97]_ and [Bradski00]_.
See also the OpenCV sample ``motempl.c`` that demonstrates the use of all the motion template functions.


calcMotionGradient
----------------------
Calculates a gradient orientation of a motion history image.

.. ocv:function:: void calcMotionGradient( InputArray mhi, OutputArray mask, OutputArray orientation,                         double delta1, double delta2, int apertureSize=3 )

.. ocv:pyfunction:: cv2.calcMotionGradient(mhi, delta1, delta2[, mask[, orientation[, apertureSize]]]) -> mask, orientation

.. ocv:cfunction:: void cvCalcMotionGradient( const CvArr* mhi, CvArr* mask, CvArr* orientation, double delta1, double delta2, int apertureSize=3 )
.. ocv:pyoldfunction:: cv.CalcMotionGradient(mhi, mask, orientation, delta1, delta2, apertureSize=3)-> None

    :param mhi: Motion history single-channel floating-point image.

    :param mask: Output mask image that has the type  ``CV_8UC1``  and the same size as  ``mhi`` . Its non-zero elements mark pixels where the motion gradient data is correct.

    :param orientation: Output motion gradient orientation image that has the same type and the same size as  ``mhi`` . Each pixel of the image is a motion orientation, from 0 to 360 degrees.

    :param delta1: Minimal (or maximal) allowed difference between  ``mhi``  values within a pixel neighborhood.
    
    :param delta2: Maximal (or minimal) allowed difference between  ``mhi``  values within a pixel neighborhood. That is, the function finds the minimum ( :math:`m(x,y)` ) and maximum ( :math:`M(x,y)` )  ``mhi``  values over  :math:`3 \times 3`  neighborhood of each pixel and marks the motion orientation at  :math:`(x, y)`  as valid only if

        .. math::

            \min ( \texttt{delta1}  ,  \texttt{delta2}  )  \le  M(x,y)-m(x,y)  \le   \max ( \texttt{delta1}  , \texttt{delta2} ).

    :param apertureSize: Aperture size of  the :ocv:func:`Sobel`  operator.

The function calculates a gradient orientation at each pixel
:math:`(x, y)` as:

.. math::

    \texttt{orientation} (x,y)= \arctan{\frac{d\texttt{mhi}/dy}{d\texttt{mhi}/dx}}

In fact,
:ocv:func:`fastAtan2` and
:ocv:func:`phase` are used so that the computed angle is measured in degrees and covers the full range 0..360. Also, the ``mask`` is filled to indicate pixels where the computed angle is valid.



calcGlobalOrientation
-------------------------
Calculates a global motion orientation in a selected region.

.. ocv:function:: double calcGlobalOrientation( InputArray orientation, InputArray mask, InputArray mhi, double timestamp, double duration )

.. ocv:pyfunction:: cv2.calcGlobalOrientation(orientation, mask, mhi, timestamp, duration) -> retval

.. ocv:cfunction:: double cvCalcGlobalOrientation( const CvArr* orientation, const CvArr* mask, const CvArr* mhi, double timestamp, double duration )
.. ocv:pyoldfunction:: cv.CalcGlobalOrientation(orientation, mask, mhi, timestamp, duration)-> float

    :param orientation: Motion gradient orientation image calculated by the function  :ocv:func:`calcMotionGradient` .
    
    :param mask: Mask image. It may be a conjunction of a valid gradient mask, also calculated by  :ocv:func:`calcMotionGradient` , and the mask of a region whose direction needs to be calculated.

    :param mhi: Motion history image calculated by  :ocv:func:`updateMotionHistory` .
    
    :param timestamp: Timestamp passed to  :ocv:func:`updateMotionHistory` .
    
    :param duration: Maximum duration of a motion track in milliseconds, passed to  :ocv:func:`updateMotionHistory` .

The function calculates an average
motion direction in the selected region and returns the angle between
0 degrees  and 360 degrees. The average direction is computed from
the weighted orientation histogram, where a recent motion has a larger
weight and the motion occurred in the past has a smaller weight, as recorded in ``mhi`` .




segmentMotion
-------------
Splits a motion history image into a few parts corresponding to separate independent motions (for example, left hand, right hand).

.. ocv:function:: void segmentMotion(InputArray mhi, OutputArray segmask, vector<Rect>& boundingRects, double timestamp, double segThresh)

.. ocv:pyfunction:: cv2.segmentMotion(mhi, timestamp, segThresh[, segmask]) -> segmask, boundingRects

.. ocv:cfunction:: CvSeq* cvSegmentMotion( const CvArr* mhi, CvArr* segMask, CvMemStorage* storage, double timestamp, double segThresh )
.. ocv:pyoldfunction:: cv.SegmentMotion(mhi, segMask, storage, timestamp, segThresh)-> None

    :param mhi: Motion history image.

    :param segmask: Image where the found mask should be stored, single-channel, 32-bit floating-point.

    :param boundingRects: Vector containing ROIs of motion connected components.

    :param timestamp: Current time in milliseconds or other units.

    :param segThresh: Segmentation threshold that is recommended to be equal to the interval between motion history "steps" or greater.
 

The function finds all of the motion segments and marks them in ``segmask`` with individual values (1,2,...). It also computes a vector with ROIs of motion connected components. After that the motion direction for every component can be calculated with :ocv:func:`calcGlobalOrientation` using the extracted mask of the particular component.




CamShift
------------
Finds an object center, size, and orientation.

.. ocv:function:: RotatedRect CamShift( InputArray probImage, Rect& window, TermCriteria criteria )

.. ocv:pyfunction:: cv2.CamShift(probImage, window, criteria) -> retval, window

.. ocv:cfunction:: int cvCamShift( const CvArr* probImage, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp, CvBox2D* box=NULL )

.. ocv:pyoldfunction:: cv.CamShift(probImage, window, criteria)-> (int, comp, box)

    :param probImage: Back projection of the object histogram. See  :ocv:func:`calcBackProject` .
    
    :param window: Initial search window.

    :param criteria: Stop criteria for the underlying  :ocv:func:`meanShift` .

    :returns: (in old interfaces) Number of iterations CAMSHIFT took to converge 

The function implements the CAMSHIFT object tracking algorithm
[Bradski98]_.
First, it finds an object center using
:ocv:func:`meanShift` and then adjusts the window size and finds the optimal rotation. The function returns the rotated rectangle structure that includes the object position, size, and orientation. The next position of the search window can be obtained with ``RotatedRect::boundingRect()`` .

See the OpenCV sample ``camshiftdemo.c`` that tracks colored objects.



meanShift
-------------
Finds an object on a back projection image.

.. ocv:function:: int meanShift( InputArray probImage, Rect& window, TermCriteria criteria )

.. ocv:pyfunction:: cv2.meanShift(probImage, window, criteria) -> retval, window

.. ocv:cfunction:: int cvMeanShift( const CvArr* probImage, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp )
.. ocv:pyoldfunction:: cv.MeanShift(probImage, window, criteria)-> comp

    :param probImage: Back projection of the object histogram. See  :ocv:func:`calcBackProject` for details.
    
    :param window: Initial search window.

    :param criteria: Stop criteria for the iterative search algorithm.

    :returns: Number of iterations CAMSHIFT took to converge.

The function implements the iterative object search algorithm. It takes the input back projection of an object and the initial position. The mass center in ``window`` of the back projection image is computed and the search window center shifts to the mass center. The procedure is repeated until the specified number of iterations ``criteria.maxCount`` is done or until the window center shifts by less than ``criteria.epsilon`` . The algorithm is used inside
:ocv:func:`CamShift` and, unlike
:ocv:func:`CamShift` , the search window size or orientation do not change during the search. You can simply pass the output of
:ocv:func:`calcBackProject` to this function. But better results can be obtained if you pre-filter the back projection and remove the noise. For example, you can do this by retrieving connected components with
:ocv:func:`findContours` , throwing away contours with small area (
:ocv:func:`contourArea` ), and rendering the  remaining contours with
:ocv:func:`drawContours` .



KalmanFilter
------------
.. ocv:class:: KalmanFilter

    Kalman filter class.

The class implements a standard Kalman filter
http://en.wikipedia.org/wiki/Kalman_filter, [Welch95]_. However, you can modify ``transitionMatrix``, ``controlMatrix``, and ``measurementMatrix`` to get an extended Kalman filter functionality. See the OpenCV sample ``kalman.cpp`` .




KalmanFilter::KalmanFilter
--------------------------
The constructors.

.. ocv:function:: KalmanFilter::KalmanFilter()

.. ocv:function:: KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F)

.. ocv:pyfunction:: cv2.KalmanFilter(dynamParams, measureParams[, controlParams[, type]]) -> <KalmanFilter object>

.. ocv:cfunction:: CvKalman* cvCreateKalman( int dynamParams, int measureParams, int controlParams=0 )
.. ocv:pyoldfunction:: cv.CreateKalman(dynamParams, measureParams, controlParams=0) -> CvKalman

    The full constructor.
    
    :param dynamParams: Dimensionality of the state.
    
    :param measureParams: Dimensionality of the measurement.
    
    :param controlParams: Dimensionality of the control vector.

    :param type: Type of the created matrices that should be ``CV_32F`` or ``CV_64F``.

.. note:: In C API when ``CvKalman* kalmanFilter`` structure is not needed anymore, it should be released with ``cvReleaseKalman(&kalmanFilter)``

KalmanFilter::init
------------------
Re-initializes Kalman filter. The previous content is destroyed.

.. ocv:function:: void KalmanFilter::init(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F)

    :param dynamParams: Dimensionalityensionality of the state.
    
    :param measureParams: Dimensionality of the measurement.
    
    :param controlParams: Dimensionality of the control vector.

    :param type: Type of the created matrices that should be ``CV_32F`` or ``CV_64F``.


KalmanFilter::predict
---------------------
Computes a predicted state.

.. ocv:function:: const Mat& KalmanFilter::predict(const Mat& control=Mat())

.. ocv:pyfunction:: cv2.KalmanFilter.predict([, control]) -> retval

.. ocv:cfunction:: const CvMat* cvKalmanPredict( CvKalman* kalman, const CvMat* control=NULL)
.. ocv:pyoldfunction:: cv.KalmanPredict(kalman, control=None) -> cvmat

    :param control: The optional input control


KalmanFilter::correct
---------------------
Updates the predicted state from the measurement.

.. ocv:function:: const Mat& KalmanFilter::correct(const Mat& measurement)

.. ocv:pyfunction:: cv2.KalmanFilter.correct(measurement) -> retval

.. ocv:cfunction:: const CvMat* cvKalmanCorrect( CvKalman* kalman, const CvMat* measurement )

.. ocv:pyoldfunction:: cv.KalmanCorrect(kalman, measurement) -> cvmat

    :param measurement: The measured system parameters


BackgroundSubtractor
--------------------

.. ocv:class:: BackgroundSubtractor : public Algorithm

Base class for background/foreground segmentation. ::

    class BackgroundSubtractor : public Algorithm
    {
    public:
        virtual ~BackgroundSubtractor();
        virtual void operator()(InputArray image, OutputArray fgmask, double learningRate=0);
        virtual void getBackgroundImage(OutputArray backgroundImage) const;
    };


The class is only used to define the common interface for the whole family of background/foreground segmentation algorithms.


BackgroundSubtractor::operator()
--------------------------------
Computes a foreground mask.

.. ocv:function:: void BackgroundSubtractor::operator()(InputArray image, OutputArray fgmask, double learningRate=0)

.. ocv:pyfunction:: cv2.BackgroundSubtractor.apply(image[, fgmask[, learningRate]]) -> fgmask

    :param image: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.


BackgroundSubtractor::getBackgroundImage
----------------------------------------
Computes a background image.

.. ocv:function:: void BackgroundSubtractor::getBackgroundImage(OutputArray backgroundImage) const

    :param backgroundImage: The output background image.
    
.. note:: Sometimes the background image can be very blurry, as it contain the average background statistics.

BackgroundSubtractorMOG
-----------------------

.. ocv:class:: BackgroundSubtractorMOG : public BackgroundSubtractor

Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

The class implements the algorithm described in P. KadewTraKuPong and R. Bowden, *An improved adaptive background mixture model for real-time tracking with shadow detection*, Proc. 2nd European Workshop on Advanced Video-Based Surveillance Systems, 2001: http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf




BackgroundSubtractorMOG::BackgroundSubtractorMOG
------------------------------------------------
The constructors.

.. ocv:function:: BackgroundSubtractorMOG::BackgroundSubtractorMOG()

.. ocv:function:: BackgroundSubtractorMOG::BackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma=0)

.. ocv:pyfunction:: cv2.BackgroundSubtractorMOG(history, nmixtures, backgroundRatio[, noiseSigma]) -> <BackgroundSubtractorMOG object>

    :param history: Length of the history.

    :param nmixtures: Number of Gaussian mixtures.

    :param backgroundRatio: Background ratio.

    :param noiseSigma: Noise strength.

Default constructor sets all parameters to default values.




BackgroundSubtractorMOG::operator()
-----------------------------------
Updates the background model and returns the foreground mask

.. ocv:function:: void BackgroundSubtractorMOG::operator()(InputArray image, OutputArray fgmask, double learningRate=0)

Parameters are the same as in :ocv:funcx:`BackgroundSubtractor::operator()`


BackgroundSubtractorMOG2
------------------------
Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

.. ocv:class:: BackgroundSubtractorMOG2 : public BackgroundSubtractor

    Here are important members of the class that control the algorithm, which you can set after constructing the class instance:

    .. ocv:member:: int nmixtures
    
        Maximum allowed number of mixture components. Actual number is determined dynamically per pixel.

    .. ocv:member:: float backgroundRatio
    
        Threshold defining whether the component is significant enough to be included into the background model ( corresponds to ``TB=1-cf`` from the paper??which paper??). ``cf=0.1 => TB=0.9`` is default. For ``alpha=0.001``, it means that the mode should exist for approximately 105 frames before it is considered foreground.

    .. ocv:member:: float varThresholdGen
    
        Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the existing components (corresponds to ``Tg``). If it is not close to any component, a new component is generated. ``3 sigma => Tg=3*3=9`` is default. A smaller ``Tg`` value generates more components. A higher ``Tg`` value may result in a small number of components but they can grow too large.

    .. ocv:member:: float fVarInit
    
        Initial variance for the newly generated components. It affects the speed of adaptation. The parameter value is based on your estimate of the typical standard deviation from the images. OpenCV uses 15 as a reasonable value.

    .. ocv:member:: float fVarMin 
    
        Parameter used to further control the variance.

    .. ocv:member:: float fVarMax
    
        Parameter used to further control the variance.

    .. ocv:member:: float fCT
        
        Complexity reduction parameter. This parameter defines the number of samples needed to accept to prove the component exists. ``CT=0.05`` is a default value for all the samples. By setting ``CT=0`` you get an algorithm very similar to the standard Stauffer&Grimson algorithm.

    .. ocv:member:: uchar nShadowDetection
    
        The value for marking shadow pixels in the output foreground mask. Default value is 127.

    .. ocv:member:: float fTau
        
        Shadow threshold. The shadow is detected if the pixel is a darker version of the background. ``Tau`` is a threshold defining how much darker the shadow can be. ``Tau= 0.5`` means that if a pixel is more than twice darker then it is not shadow. See Prati,Mikic,Trivedi,Cucchiarra, *Detecting Moving Shadows...*, IEEE PAMI,2003.


The class implements the Gaussian mixture model background subtraction described in:

  * Z.Zivkovic, *Improved adaptive Gausian mixture model for background subtraction*, International Conference Pattern Recognition, UK, August, 2004, http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf. The code is very fast and performs also shadow detection. Number of Gausssian components is adapted per pixel.

  * Z.Zivkovic, F. van der Heijden, *Efficient Adaptive Density Estimapion per Image Pixel for the Task of Background Subtraction*, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006. The algorithm similar to the standard Stauffer&Grimson algorithm with additional selection of the number of the Gaussian components based on: Z.Zivkovic, F.van der Heijden, Recursive unsupervised learning of finite mixture models, IEEE Trans. on Pattern Analysis and Machine Intelligence, vol.26, no.5, pages 651-656, 2004.


BackgroundSubtractorMOG2::BackgroundSubtractorMOG2
--------------------------------------------------
The constructors.

.. ocv:function:: BackgroundSubtractorMOG2::BackgroundSubtractorMOG2()

.. ocv:function:: BackgroundSubtractorMOG2::BackgroundSubtractorMOG2(int history, float varThreshold, bool bShadowDetection=1)

    :param history: Length of the history.

    :param varThreshold: Threshold on the squared Mahalanobis distance to decide whether it is well described by the background model (see Cthr??). This parameter does not affect the background update. A typical value could be 4 sigma, that is, ``varThreshold=4*4=16;`` (see Tb??).

    :param bShadowDetection: Parameter defining whether shadow detection should be enabled (``true`` or ``false``).



BackgroundSubtractorMOG2::operator()
------------------------------------
Updates the background model and computes the foreground mask

.. ocv:function:: void BackgroundSubtractorMOG2::operator()(InputArray image, OutputArray fgmask, double learningRate=-1)

    See :ocv:funcx:`BackgroundSubtractor::operator()`.


BackgroundSubtractorMOG2::getBackgroundImage
--------------------------------------------
Returns background image

.. ocv:function:: void BackgroundSubtractorMOG2::getBackgroundImage(OutputArray backgroundImage)

See :ocv:func:`BackgroundSubtractor::getBackgroundImage`.


.. [Bouguet00] Jean-Yves Bouguet. Pyramidal Implementation of the Lucas Kanade Feature Tracker.

.. [Bradski98] Bradski, G.R. "Computer Vision Face Tracking for Use in a Perceptual User Interface", Intel, 1998

.. [Bradski00] Davis, J.W. and Bradski, G.R. “Motion Segmentation and Pose Recognition with Motion History Gradients”, WACV00, 2000

.. [Davis97] Davis, J.W. and Bobick, A.F. “The Representation and Recognition of Action Using Temporal Templates”, CVPR97, 1997

.. [Farneback2003] Gunnar Farneback, Two-frame motion estimation based on polynomial expansion, Lecture Notes in Computer Science, 2003, (2749), , 363-370. 

.. [Horn81] Berthold K.P. Horn and Brian G. Schunck. Determining Optical Flow. Artificial Intelligence, 17, pp. 185-203, 1981.

.. [Lucas81] Lucas, B., and Kanade, T. An Iterative Image Registration Technique with an Application to Stereo Vision, Proc. of 7th International Joint Conference on Artificial Intelligence (IJCAI), pp. 674-679.

.. [Welch95] Greg Welch and Gary Bishop “An Introduction to the Kalman Filter”, 1995
