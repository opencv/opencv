Global Motion Estimation
========================

.. highlight:: cpp

The video stabilization module contains a set of functions and classes for global motion estimation between point clouds or between images. In the last case  features are extracted and matched internally. For the sake of convenience the motion estimation functions are wrapped into classes. Both the functions and the classes are available.

videostab::MotionModel
----------------------

.. ocv:class:: videostab::MotionModel

Describes motion model between two point clouds. 

::

    enum MotionModel
    {
        MM_TRANSLATION = 0,
        MM_TRANSLATION_AND_SCALE = 1,
        MM_ROTATION = 2,
        MM_RIGID = 3,
        MM_SIMILARITY = 4,
        MM_AFFINE = 5,
        MM_HOMOGRAPHY = 6,
        MM_UNKNOWN = 7
    };


videostab::RansacParams
-----------------------

.. ocv:class:: videostab::RansacParams

Describes RANSAC method parameters.

::

    struct CV_EXPORTS RansacParams
    {
        int size; // subset size
        float thresh; // max error to classify as inlier
        float eps; // max outliers ratio
        float prob; // probability of success

        RansacParams() : size(0), thresh(0), eps(0), prob(0) {}
        RansacParams(int size, float thresh, float eps, float prob);

        int niters() const;

        static RansacParams default2dMotion(MotionModel model);
    };


videostab::RansacParams::RansacParams
-------------------------------------

.. ocv:function:: RansacParams::RansacParams()

    :return: RANSAC method empty parameters object.


videostab::RansacParams::RansacParams
-------------------------------------

.. ocv:function:: RansacParams::RansacParams(int size, float thresh, float eps, float prob)

    :param size: Subset size.

    :param thresh: Maximum re-projection error value to classify as inlier.

    :param eps: Maximum ratio of incorrect correspondences.

    :param prob: Required success probability.

    :return: RANSAC method parameters object.


videostab::RansacParams::niters
-------------------------------

.. ocv:function:: int RansacParams::niters() const

    :return: Number of iterations that'll be performed by RANSAC method.


videostab::RansacParams::default2dMotion
----------------------------------------

.. ocv:function:: static RansacParams RansacParams::default2dMotion(MotionModel model)

    :param model: Motion model. See :ocv:class:`videostab::MotionModel`.

    :return: Default RANSAC method parameters for the given motion model.


videostab::estimateGlobalMotionLeastSquares
-------------------------------------------

Estimates best global motion between two 2D point clouds in the least-squares sense.

.. note:: Works in-place and changes input point arrays.

.. ocv:function:: Mat estimateGlobalMotionLeastSquares(InputOutputArray points0, InputOutputArray points1, int model = MM_AFFINE, float *rmse = 0)

    :param points0: Source set of 2D points (``32F``).

    :param points1: Destination set of 2D points (``32F``).

    :param model: Motion model (up to ``MM_AFFINE``).

    :param rmse: Final root-mean-square error.

    :return: 3x3 2D transformation matrix (``32F``).


videostab::estimateGlobalMotionRansac
-------------------------------------

Estimates best global motion between two 2D point clouds robustly (using RANSAC method).

.. ocv:function:: Mat estimateGlobalMotionRansac(InputArray points0, InputArray points1, int model = MM_AFFINE, const RansacParams &params = RansacParams::default2dMotion(MM_AFFINE), float *rmse = 0, int *ninliers = 0)

    :param points0: Source set of 2D points (``32F``).

    :param points1: Destination set of 2D points (``32F``).

    :param model: Motion model. See :ocv:class:`videostab::MotionModel`.

    :param params: RANSAC method parameters. See :ocv:class:`videostab::RansacParams`.

    :param rmse: Final root-mean-square error.

    :param ninliers: Final number of inliers.


videostab::getMotion
--------------------

Computes motion between two frames assuming that all the intermediate motions are known.

.. ocv:function:: Mat getMotion(int from, int to, const std::vector<Mat> &motions)

    :param from: Source frame index.

    :param to: Destination frame index.

    :param motions: Pair-wise motions. ``motions[i]`` denotes motion from the frame ``i`` to the frame ``i+1``

    :return: Motion from the frame ``from`` to the frame ``to``.


videostab::MotionEstimatorBase
------------------------------

.. ocv:class:: videostab::MotionEstimatorBase

Base class for all global motion estimation methods.

::

    class CV_EXPORTS MotionEstimatorBase
    {
    public:
        virtual ~MotionEstimatorBase();

        virtual void setMotionModel(MotionModel val);
        virtual MotionModel motionModel() const;

        virtual Mat estimate(InputArray points0, InputArray points1, bool *ok = 0) = 0;
    };


videostab::MotionEstimatorBase::setMotionModel
----------------------------------------------

Sets motion model.

.. ocv:function:: void MotionEstimatorBase::setMotionModel(MotionModel val)

    :param val: Motion model. See :ocv:class:`videostab::MotionModel`.



videostab::MotionEstimatorBase::motionModel
----------------------------------------------

.. ocv:function:: MotionModel MotionEstimatorBase::motionModel() const

    :return: Motion model. See :ocv:class:`videostab::MotionModel`.    


videostab::MotionEstimatorBase::estimate
----------------------------------------

Estimates global motion between two 2D point clouds.

.. ocv:function:: Mat MotionEstimatorBase::estimate(InputArray points0, InputArray points1, bool *ok = 0)

    :param points0: Source set of 2D points (``32F``).

    :param points1: Destination set of 2D points (``32F``).

    :param ok: Indicates whether motion was estimated successfully.

    :return: 3x3 2D transformation matrix (``32F``).


videostab::MotionEstimatorRansacL2
----------------------------------

.. ocv:class:: videostab::MotionEstimatorRansacL2

Describes a robust RANSAC-based global 2D motion estimation method which minimizes L2 error.

::

    class CV_EXPORTS MotionEstimatorRansacL2 : public MotionEstimatorBase
    {
    public:
        MotionEstimatorRansacL2(MotionModel model = MM_AFFINE);

        void setRansacParams(const RansacParams &val);
        RansacParams ransacParams() const;

        void setMinInlierRatio(float val);
        float minInlierRatio() const;

        virtual Mat estimate(InputArray points0, InputArray points1, bool *ok = 0);
    };


videostab::MotionEstimatorL1
----------------------------

.. ocv:class:: videostab::MotionEstimatorL1

Describes a global 2D motion estimation method which minimizes L1 error.

.. note:: To be able to use this method you must build OpenCV with CLP library support.

::

    class CV_EXPORTS MotionEstimatorL1 : public MotionEstimatorBase
    {
    public:
        MotionEstimatorL1(MotionModel model = MM_AFFINE);

        virtual Mat estimate(InputArray points0, InputArray points1, bool *ok = 0);
    };


videostab::ImageMotionEstimatorBase
-----------------------------------

.. ocv:class:: videostab::ImageMotionEstimatorBase

Base class for global 2D motion estimation methods which take frames as input.

::

    class CV_EXPORTS ImageMotionEstimatorBase
    {
    public:
        virtual ~ImageMotionEstimatorBase();

        virtual void setMotionModel(MotionModel val);
        virtual MotionModel motionModel() const;

        virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0) = 0;
    };


videostab::KeypointBasedMotionEstimator
---------------------------------------

.. ocv:class:: videostab::KeypointBasedMotionEstimator

Describes a global 2D motion estimation method which uses keypoints detection and optical flow for matching.

::

    class CV_EXPORTS KeypointBasedMotionEstimator : public ImageMotionEstimatorBase
    {
    public:
        KeypointBasedMotionEstimator(Ptr<MotionEstimatorBase> estimator);

        virtual void setMotionModel(MotionModel val);
        virtual MotionModel motionModel() const;

        void setDetector(Ptr<FeatureDetector> val);
        Ptr<FeatureDetector> detector() const;

        void setOpticalFlowEstimator(Ptr<ISparseOptFlowEstimator> val);
        Ptr<ISparseOptFlowEstimator> opticalFlowEstimator() const;

        void setOutlierRejector(Ptr<IOutlierRejector> val);
        Ptr<IOutlierRejector> outlierRejector() const;

        virtual Mat estimate(const Mat &frame0, const Mat &frame1, bool *ok = 0);
    };
