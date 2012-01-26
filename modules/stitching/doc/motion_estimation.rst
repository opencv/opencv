Rotation Estimation
===================

.. highlight:: cpp

detail::Estimator
-----------------
.. ocv:class:: detail::Estimator

Rotation estimator base class. It takes features of all images, pairwise matches between all images and estimates rotations of all cameras. 

.. note:: The coordinate system origin is implementation-dependent, but you can always normalize the rotations in respect to the first camera, for instance. 

::

    class CV_EXPORTS Estimator
    {
    public:
        virtual ~Estimator() {}

        void operator ()(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, 
                         std::vector<CameraParams> &cameras)
            { estimate(features, pairwise_matches, cameras); }

    protected:
        virtual void estimate(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, 
                              std::vector<CameraParams> &cameras) = 0;
    };

detail::Estimator::operator()
-----------------------------

Estimates camera parameters.

.. ocv:function:: detail::Estimator::operator ()(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, std::vector<CameraParams> &cameras)

    :param features: Features of images

    :param pairwise_matches: Pairwise matches of images

    :param cameras: Estimated camera parameters

detail::Estimator::estimate
---------------------------

This method must implement camera parameters estimation logic in order to make the wrapper `detail::Estimator::operator()`_ work.

.. ocv:function:: void detail::Estimator::estimate(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches, std::vector<CameraParams> &cameras)

    :param features: Features of images

    :param pairwise_matches: Pairwise matches of images

    :param cameras: Estimated camera parameters

detail::HomographyBasedEstimator
--------------------------------
.. ocv:class:: detail::HomographyBasedEstimator

Homography based rotation estimator. ::

    class CV_EXPORTS HomographyBasedEstimator : public Estimator
    {
    public:
        HomographyBasedEstimator(bool is_focals_estimated = false)
            : is_focals_estimated_(is_focals_estimated) {}

    private:
        /* hidden */
    };

detail::BundleAdjusterBase
--------------------------
.. ocv:class:: detail::BundleAdjusterBase

Base class for all camera parameters refinement methods. ::

    class CV_EXPORTS BundleAdjusterBase : public Estimator
    {
    public:
        const Mat refinementMask() const { return refinement_mask_.clone(); }
        void setRefinementMask(const Mat &mask) 
        { 
            CV_Assert(mask.type() == CV_8U && mask.size() == Size(3, 3));
            refinement_mask_ = mask.clone(); 
        }

        double confThresh() const { return conf_thresh_; }
        void setConfThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

        CvTermCriteria termCriteria() { return term_criteria_; }
        void setTermCriteria(const CvTermCriteria& term_criteria) { term_criteria_ = term_criteria; }

    protected:
        BundleAdjusterBase(int num_params_per_cam, int num_errs_per_measurement) 
            : num_params_per_cam_(num_params_per_cam), 
              num_errs_per_measurement_(num_errs_per_measurement) 
        {    
            setRefinementMask(Mat::ones(3, 3, CV_8U));
            setConfThresh(1.); 
            setTermCriteria(cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, DBL_EPSILON));
        }

        // Runs bundle adjustment
        virtual void estimate(const std::vector<ImageFeatures> &features, 
                              const std::vector<MatchesInfo> &pairwise_matches,
                              std::vector<CameraParams> &cameras);

        virtual void setUpInitialCameraParams(const std::vector<CameraParams> &cameras) = 0;
        virtual void obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const = 0;
        virtual void calcError(Mat &err) = 0;
        virtual void calcJacobian(Mat &jac) = 0;

        // 3x3 8U mask, where 0 means don't refine respective parameter, != 0 means refine
        Mat refinement_mask_;

        int num_images_;
        int total_num_matches_;

        int num_params_per_cam_;
        int num_errs_per_measurement_;

        const ImageFeatures *features_;
        const MatchesInfo *pairwise_matches_;

        // Threshold to filter out poorly matched image pairs
        double conf_thresh_;

        //Levenberg–Marquardt algorithm termination criteria
        CvTermCriteria term_criteria_;

        // Camera parameters matrix (CV_64F)
        Mat cam_params_;

        // Connected images pairs
        std::vector<std::pair<int,int> > edges_;
    };

.. seealso:: :ocv:class:`detail::Estimator`

detail::BundleAdjusterBase::BundleAdjusterBase
----------------------------------------------

Construct a bundle adjuster base instance.

.. ocv:function:: detail::BundleAdjusterBase::BundleAdjusterBase(int num_params_per_cam, int num_errs_per_measurement)

    :param num_params_per_cam: Number of parameters per camera
    
    :param num_errs_per_measurement: Number of error terms (components) per match

detail::BundleAdjusterBase::setUpInitialCameraParams
----------------------------------------------------

Sets initial camera parameter to refine.

.. ocv:function:: void detail::BundleAdjusterBase::setUpInitialCameraParams(const std::vector<CameraParams> &cameras)

    :param cameras: Camera parameters

detail::BundleAdjusterBase::calcError
-------------------------------------

Calculates error vector.

.. ocv:function:: void detail::BundleAdjusterBase::calcError(Mat &err)

    :param err: Error column-vector of length ``total_num_matches * num_errs_per_measurement``

detail::BundleAdjusterBase::calcJacobian
----------------------------------------

Calculates the cost function jacobian.

.. ocv:function:: void detail::BundleAdjusterBase::calcJacobian(Mat &jac)

    :param jac: Jacobian matrix of dimensions ``(total_num_matches * num_errs_per_measurement) x (num_images * num_params_per_cam)``

detail::BundleAdjusterBase::obtainRefinedCameraParams
-----------------------------------------------------

Gets the refined camera parameters.

.. ocv:function:: void detail::BundleAdjusterBase::obtainRefinedCameraParams(std::vector<CameraParams> &cameras) const

    :param cameras: Refined camera parameters

detail::BundleAdjusterReproj
----------------------------
.. ocv:class:: detail::BundleAdjusterReproj

Implementation of the camera parameters refinement algorithm which minimizes sum of the reprojection error squares. ::

    class CV_EXPORTS BundleAdjusterReproj : public BundleAdjusterBase
    {
    public:
        BundleAdjusterReproj() : BundleAdjusterBase(7, 2) {}

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::BundleAdjusterBase`, :ocv:class:`detail::Estimator`

detail::BundleAdjusterRay
-------------------------

Implementation of the camera parameters refinement algorithm which minimizes sum of the distances between the rays passing through the camera center and a feature. ::

    class CV_EXPORTS BundleAdjusterRay : public BundleAdjusterBase
    {
    public:
        BundleAdjusterRay() : BundleAdjusterBase(4, 3) {}

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::BundleAdjusterBase`

detail::WaveCorrectKind
-----------------------
.. ocv:class:: detail::WaveCorrectKind

Wave correction kind. ::

    enum CV_EXPORTS WaveCorrectKind
    {
        WAVE_CORRECT_HORIZ,
        WAVE_CORRECT_VERT
    };

detail::waveCorrect
-------------------
Tries to make panorama more horizontal (or verical).

.. ocv:function:: void waveCorrect(std::vector<Mat> &rmats, WaveCorrectKind kind)

    :param rmats: Camera rotation matrices.

    :param kind: Correction kind, see :ocv:class:`detail::WaveCorrectKind`.
