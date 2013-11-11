Video Analysis
=============================

.. highlight:: cpp

ocl::GoodFeaturesToTrackDetector_OCL
----------------------------------------
.. ocv:class:: ocl::GoodFeaturesToTrackDetector_OCL

Class used for strong corners detection on an image. ::

    class GoodFeaturesToTrackDetector_OCL
    {
    public:
        explicit GoodFeaturesToTrackDetector_OCL(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0,
            int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04);

        //! return 1 rows matrix with CV_32FC2 type
        void operator ()(const oclMat& image, oclMat& corners, const oclMat& mask = oclMat());
        //! download points of type Point2f to a vector. the vector's content will be erased
        void downloadPoints(const oclMat &points, std::vector<Point2f> &points_v);

        int maxCorners;
        double qualityLevel;
        double minDistance;

        int blockSize;
        bool useHarrisDetector;
        double harrisK;
        void releaseMemory()
        {
            Dx_.release();
            Dy_.release();
            eig_.release();
            minMaxbuf_.release();
            tmpCorners_.release();
        }
    };

The class finds the most prominent corners in the image.

.. seealso:: :ocv:func:`goodFeaturesToTrack()`

ocl::GoodFeaturesToTrackDetector_OCL::GoodFeaturesToTrackDetector_OCL
-------------------------------------------------------------------------
Constructor.

.. ocv:function:: ocl::GoodFeaturesToTrackDetector_OCL::GoodFeaturesToTrackDetector_OCL(int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0, int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04)

    :param maxCorners: Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned.

    :param qualityLevel: Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see  :ocv:func:`ocl::cornerMinEigenVal` ) or the Harris function response (see  :ocv:func:`ocl::cornerHarris` ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the  ``qualityLevel=0.01`` , then all the corners with the quality measure less than 15 are rejected.

    :param minDistance: Minimum possible Euclidean distance between the returned corners.

    :param blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See  :ocv:func:`cornerEigenValsAndVecs` .

    :param useHarrisDetector: Parameter indicating whether to use a Harris detector (see :ocv:func:`ocl::cornerHarris`) or :ocv:func:`ocl::cornerMinEigenVal`.

    :param harrisK: Free parameter of the Harris detector.

ocl::GoodFeaturesToTrackDetector_OCL::operator ()
-------------------------------------------------------
Finds the most prominent corners in the image.

.. ocv:function:: void ocl::GoodFeaturesToTrackDetector_OCL::operator ()(const oclMat& image, oclMat& corners, const oclMat& mask = oclMat())

    :param image: Input 8-bit, single-channel image.

    :param corners: Output vector of detected corners (it will be one row matrix with CV_32FC2 type).

    :param mask: Optional region of interest. If the image is not empty (it needs to have the type  ``CV_8UC1``  and the same size as  ``image`` ), it  specifies the region in which the corners are detected.

.. seealso:: :ocv:func:`goodFeaturesToTrack`

ocl::GoodFeaturesToTrackDetector_OCL::releaseMemory
--------------------------------------------------------
Releases inner buffers memory.

.. ocv:function:: void ocl::GoodFeaturesToTrackDetector_OCL::releaseMemory()

ocl::FarnebackOpticalFlow
-------------------------------
.. ocv:class:: ocl::FarnebackOpticalFlow

Class computing a dense optical flow using the Gunnar Farneback's algorithm. ::

    class CV_EXPORTS FarnebackOpticalFlow
    {
    public:
        FarnebackOpticalFlow();

        int numLevels;
        double pyrScale;
        bool fastPyramids;
        int winSize;
        int numIters;
        int polyN;
        double polySigma;
        int flags;

        void operator ()(const oclMat &frame0, const oclMat &frame1, oclMat &flowx, oclMat &flowy);

        void releaseMemory();

    private:
        /* hidden */
    };

ocl::FarnebackOpticalFlow::operator ()
------------------------------------------
Computes a dense optical flow using the Gunnar Farneback's algorithm.

.. ocv:function:: void ocl::FarnebackOpticalFlow::operator ()(const oclMat &frame0, const oclMat &frame1, oclMat &flowx, oclMat &flowy)

    :param frame0: First 8-bit gray-scale input image
    :param frame1: Second 8-bit gray-scale input image
    :param flowx: Flow horizontal component
    :param flowy: Flow vertical component

.. seealso:: :ocv:func:`calcOpticalFlowFarneback`

ocl::FarnebackOpticalFlow::releaseMemory
--------------------------------------------
Releases unused auxiliary memory buffers.

.. ocv:function:: void ocl::FarnebackOpticalFlow::releaseMemory()


ocl::PyrLKOpticalFlow
-------------------------
.. ocv:class:: ocl::PyrLKOpticalFlow

Class used for calculating an optical flow. ::

    class PyrLKOpticalFlow
    {
    public:
        PyrLKOpticalFlow();

        void sparse(const oclMat& prevImg, const oclMat& nextImg, const oclMat& prevPts, oclMat& nextPts,
            oclMat& status, oclMat* err = 0);

        void dense(const oclMat& prevImg, const oclMat& nextImg, oclMat& u, oclMat& v, oclMat* err = 0);

        Size winSize;
        int maxLevel;
        int iters;
        double derivLambda;
        bool useInitialFlow;
        float minEigThreshold;
        bool getMinEigenVals;

        void releaseMemory();

    private:
        /* hidden */
    };

The class can calculate an optical flow for a sparse feature set or dense optical flow using the iterative Lucas-Kanade method with pyramids.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`

ocl::PyrLKOpticalFlow::sparse
---------------------------------
Calculate an optical flow for a sparse feature set.

.. ocv:function:: void ocl::PyrLKOpticalFlow::sparse(const oclMat& prevImg, const oclMat& nextImg, const oclMat& prevPts, oclMat& nextPts, oclMat& status, oclMat* err = 0)

    :param prevImg: First 8-bit input image (supports both grayscale and color images).

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param prevPts: Vector of 2D points for which the flow needs to be found. It must be one row matrix with CV_32FC2 type.

    :param nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new positions of input features in the second image. When ``useInitialFlow`` is true, the vector must have the same size as in the input.

    :param status: Output status vector (CV_8UC1 type). Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.

.. seealso:: :ocv:func:`calcOpticalFlowPyrLK`


ocl::PyrLKOpticalFlow::dense
---------------------------------
Calculate dense optical flow.

.. ocv:function:: void ocl::PyrLKOpticalFlow::dense(const oclMat& prevImg, const oclMat& nextImg, oclMat& u, oclMat& v, oclMat* err = 0)

    :param prevImg: First 8-bit grayscale input image.

    :param nextImg: Second input image of the same size and the same type as  ``prevImg`` .

    :param u: Horizontal component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param v: Vertical component of the optical flow of the same size as input images, 32-bit floating-point, single-channel

    :param err: Output vector (CV_32FC1 type) that contains the difference between patches around the original and moved points or min eigen value if ``getMinEigenVals`` is checked. It can be NULL, if not needed.


ocl::PyrLKOpticalFlow::releaseMemory
----------------------------------------
Releases inner buffers memory.

.. ocv:function:: void ocl::PyrLKOpticalFlow::releaseMemory()

ocl::interpolateFrames
--------------------------
Interpolates frames (images) using provided optical flow (displacement field).

.. ocv:function:: void ocl::interpolateFrames(const oclMat& frame0, const oclMat& frame1, const oclMat& fu, const oclMat& fv, const oclMat& bu, const oclMat& bv, float pos, oclMat& newFrame, oclMat& buf)

    :param frame0: First frame (32-bit floating point images, single channel).

    :param frame1: Second frame. Must have the same type and size as ``frame0`` .

    :param fu: Forward horizontal displacement.

    :param fv: Forward vertical displacement.

    :param bu: Backward horizontal displacement.

    :param bv: Backward vertical displacement.

    :param pos: New frame position.

    :param newFrame: Output image.

    :param buf: Temporary buffer, will have width x 6*height size, CV_32FC1 type and contain 6 oclMat: occlusion masks for first frame, occlusion masks for second, interpolated forward horizontal flow, interpolated forward vertical flow, interpolated backward horizontal flow, interpolated backward vertical flow.

ocl::KalmanFilter
--------------------
.. ocv:class:: ocl::KalmanFilter

Kalman filter class. ::

    class CV_EXPORTS KalmanFilter
    {
    public:
        KalmanFilter();
        //! the full constructor taking the dimensionality of the state, of the measurement and of the control vector
        KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);
        //! re-initializes Kalman filter. The previous content is destroyed.
        void init(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);

        const oclMat& predict(const oclMat& control=oclMat());
        const oclMat& correct(const oclMat& measurement);

        oclMat statePre; //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
        oclMat statePost; //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
        oclMat transitionMatrix; //!< state transition matrix (A)
        oclMat controlMatrix; //!< control matrix (B) (not used if there is no control)
        oclMat measurementMatrix; //!< measurement matrix (H)
        oclMat processNoiseCov; //!< process noise covariance matrix (Q)
        oclMat measurementNoiseCov;//!< measurement noise covariance matrix (R)
        oclMat errorCovPre; //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
        oclMat gain; //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
        oclMat errorCovPost; //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
    private:
        /* hidden */
    };

ocl::KalmanFilter::KalmanFilter
----------------------------------
The constructors.

.. ocv:function:: ocl::KalmanFilter::KalmanFilter()

.. ocv:function:: ocl::KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F)

    The full constructor.

    :param dynamParams: Dimensionality of the state.

    :param measureParams: Dimensionality of the measurement.

    :param controlParams: Dimensionality of the control vector.

    :param type: Type of the created matrices that should be ``CV_32F`` or ``CV_64F``.


ocl::KalmanFilter::init
---------------------------
Re-initializes Kalman filter. The previous content is destroyed.

.. ocv:function:: void ocl::KalmanFilter::init(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F)

    :param dynamParams: Dimensionalityensionality of the state.

    :param measureParams: Dimensionality of the measurement.

    :param controlParams: Dimensionality of the control vector.

    :param type: Type of the created matrices that should be ``CV_32F`` or ``CV_64F``.


ocl::KalmanFilter::predict
------------------------------
Computes a predicted state.

.. ocv:function:: const oclMat& ocl::KalmanFilter::predict(const oclMat& control=oclMat())

    :param control: The optional input control


ocl::KalmanFilter::correct
-----------------------------
Updates the predicted state from the measurement.

.. ocv:function:: const oclMat& ocl::KalmanFilter::correct(const oclMat& measurement)

    :param measurement: The measured system parameters


ocl::BackgroundSubtractor
----------------------------
.. ocv:class:: ocl::BackgroundSubtractor

Base class for background/foreground segmentation. ::

    class CV_EXPORTS BackgroundSubtractor
    {
    public:
        //! the virtual destructor
        virtual ~BackgroundSubtractor();
        //! the update operator that takes the next video frame and returns the current foreground mask as 8-bit binary image.
        virtual void operator()(const oclMat& image, oclMat& fgmask, float learningRate);

        //! computes a background image
        virtual void getBackgroundImage(oclMat& backgroundImage) const = 0;
    };


The class is only used to define the common interface for the whole family of background/foreground segmentation algorithms.


ocl::BackgroundSubtractor::operator()
-----------------------------------------
Computes a foreground mask.

.. ocv:function:: void ocl::BackgroundSubtractor::operator()(const oclMat& image, oclMat& fgmask, float learningRate)

    :param image: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.


ocl::BackgroundSubtractor::getBackgroundImage
-------------------------------------------------
Computes a background image.

.. ocv:function:: void ocl::BackgroundSubtractor::getBackgroundImage(oclMat& backgroundImage) const

    :param backgroundImage: The output background image.

.. note:: Sometimes the background image can be very blurry, as it contain the average background statistics.

ocl::MOG
------------
.. ocv:class:: ocl::MOG : public ocl::BackgroundSubtractor

Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm. ::

    class CV_EXPORTS MOG: public cv::ocl::BackgroundSubtractor
    {
    public:
        //! the default constructor
        MOG(int nmixtures = -1);

        //! re-initiaization method
        void initialize(Size frameSize, int frameType);

        //! the update operator
        void operator()(const oclMat& frame, oclMat& fgmask, float learningRate = 0.f);

        //! computes a background image which are the mean of all background gaussians
        void getBackgroundImage(oclMat& backgroundImage) const;

        //! releases all inner buffers
        void release();

        int history;
        float varThreshold;
        float backgroundRatio;
        float noiseSigma;

    private:
        /* hidden */
    };

The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [MOG2001]_.

.. seealso:: :ocv:class:`BackgroundSubtractorMOG`


ocl::MOG::MOG
---------------------
The constructor.

.. ocv:function:: ocl::MOG::MOG(int nmixtures = -1)

    :param nmixtures: Number of Gaussian mixtures.

Default constructor sets all parameters to default values.


ocl::MOG::operator()
------------------------
Updates the background model and returns the foreground mask.

.. ocv:function:: void ocl::MOG::operator()(const oclMat& frame, oclMat& fgmask, float learningRate = 0.f)

    :param frame: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.


ocl::MOG::getBackgroundImage
--------------------------------
Computes a background image.

.. ocv:function:: void ocl::MOG::getBackgroundImage(oclMat& backgroundImage) const

    :param backgroundImage: The output background image.


ocl::MOG::release
---------------------
Releases all inner buffer's memory.

.. ocv:function:: void ocl::MOG::release()


ocl::MOG2
-------------
.. ocv:class:: ocl::MOG2 : public ocl::BackgroundSubtractor

  Gaussian Mixture-based Background/Foreground Segmentation Algorithm.

  The class discriminates between foreground and background pixels by building and maintaining a model of the background. Any pixel which does not fit this model is then deemed to be foreground. The class implements algorithm described in [MOG2004]_. ::

    class CV_EXPORTS MOG2: public cv::ocl::BackgroundSubtractor
    {
    public:
        //! the default constructor
        MOG2(int nmixtures = -1);

        //! re-initiaization method
        void initialize(Size frameSize, int frameType);

        //! the update operator
        void operator()(const oclMat& frame, oclMat& fgmask, float learningRate = -1.0f);

        //! computes a background image which are the mean of all background gaussians
        void getBackgroundImage(oclMat& backgroundImage) const;

        //! releases all inner buffers
        void release();

        int history;

        float varThreshold;

        float backgroundRatio;

        float varThresholdGen;

        float fVarInit;
        float fVarMin;
        float fVarMax;

        float fCT;

        bool bShadowDetection;
        unsigned char nShadowDetection;
        float fTau;

    private:
        /* hidden */
    };

  .. ocv:member:: float backgroundRatio

      Threshold defining whether the component is significant enough to be included into the background model. ``cf=0.1 => TB=0.9`` is default. For ``alpha=0.001``, it means that the mode should exist for approximately 105 frames before it is considered foreground.

  .. ocv:member:: float varThreshold

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

      Shadow threshold. The shadow is detected if the pixel is a darker version of the background. ``Tau`` is a threshold defining how much darker the shadow can be. ``Tau= 0.5`` means that if a pixel is more than twice darker then it is not shadow. See [ShadowDetect2003]_.

  .. ocv:member:: bool bShadowDetection

      Parameter defining whether shadow detection should be enabled.


.. seealso:: :ocv:class:`BackgroundSubtractorMOG2`


ocl::MOG2::MOG2
-----------------------
The constructor.

.. ocv:function:: ocl::MOG2::MOG2(int nmixtures = -1)

    :param nmixtures: Number of Gaussian mixtures.

Default constructor sets all parameters to default values.


ocl::MOG2::operator()
-------------------------
Updates the background model and returns the foreground mask.

.. ocv:function:: void ocl::MOG2::operator()( const oclMat& frame, oclMat& fgmask, float learningRate=-1.0f)

    :param frame: Next video frame.

    :param fgmask: The output foreground mask as an 8-bit binary image.


ocl::MOG2::getBackgroundImage
---------------------------------
Computes a background image.

.. ocv:function:: void ocl::MOG2::getBackgroundImage(oclMat& backgroundImage) const

    :param backgroundImage: The output background image.


ocl::MOG2::release
----------------------
Releases all inner buffer's memory.

.. ocv:function:: void ocl::MOG2::release()


.. [ShadowDetect2003] Prati, Mikic, Trivedi and Cucchiarra. *Detecting Moving Shadows...*. IEEE PAMI, 2003
