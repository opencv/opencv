// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEO_DETAIL_TRACKING_HPP
#define OPENCV_VIDEO_DETAIL_TRACKING_HPP

/*
 * Partially based on:
 * ====================================================================================================================
 *  - [AAM] S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - [AMVOT] X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/tracking/doc/uml
 *
 */

#include "opencv2/core.hpp"

#include "tracking_feature.private.hpp"  // CvHaarEvaluator
#include "tracking_online_boosting.private.hpp"  // StrongClassifierDirectSelection
#include "tracking_online_mil.private.hpp"  // ClfMilBoost

namespace cv {
namespace detail {
inline namespace tracking {

/** @addtogroup tracking_detail
@{
*/

/************************************ TrackerFeature Base Classes ************************************/

/** @brief Abstract base class for TrackerFeature that represents the feature.
 */
class CV_EXPORTS TrackerFeature
{
public:
    virtual ~TrackerFeature();

    /** @brief Compute the features in the images collection
    @param images The images
    @param response The output response
    */
    void compute(const std::vector<Mat>& images, Mat& response);

    /** @brief Create TrackerFeature by tracker feature type
    @param trackerFeatureType The TrackerFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D
    */
    static Ptr<TrackerFeature> create(const String& trackerFeatureType);

    /** @brief Identify most effective features
    @param response Collection of response for the specific TrackerFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
    */
    virtual void selection(Mat& response, int npoints) = 0;

    /** @brief Get the name of the specific TrackerFeature
    */
    String getClassName() const;

protected:
    virtual bool computeImpl(const std::vector<Mat>& images, Mat& response) = 0;

    String className;
};

/** @brief Class that manages the extraction and selection of features

@cite AAM Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection).
See table I and section III C @cite AMVOT Appearance modelling -\> Visual representation (Table II,
section 3.1 - 3.2)

TrackerFeatureSet is an aggregation of TrackerFeature

@sa
   TrackerFeature

 */
class CV_EXPORTS TrackerFeatureSet
{
public:
    TrackerFeatureSet();

    ~TrackerFeatureSet();

    /** @brief Extract features from the images collection
    @param images The input images
    */
    void extraction(const std::vector<Mat>& images);

    /** @brief Identify most effective features for all feature types (optional)
    */
    void selection();

    /** @brief Remove outliers for all feature types (optional)
    */
    void removeOutliers();

    /** @brief Add TrackerFeature in the collection. Return true if TrackerFeature is added, false otherwise
    @param trackerFeatureType The TrackerFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D

    Example TrackerFeatureSet::addTrackerFeature : :
    @code
        //sample usage:

        Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
        featureSet->addTrackerFeature( trackerFeature );

        //or add CSC sampler with default parameters
        //featureSet->addTrackerFeature( "HAAR" );
    @endcode
    @note If you use the second method, you must initialize the TrackerFeature
    */
    bool addTrackerFeature(String trackerFeatureType);

    /** @overload
    @param feature The TrackerFeature class
    */
    bool addTrackerFeature(Ptr<TrackerFeature>& feature);

    /** @brief Get the TrackerFeature collection (TrackerFeature name, TrackerFeature pointer)
    */
    const std::vector<std::pair<String, Ptr<TrackerFeature>>>& getTrackerFeature() const;

    /** @brief Get the responses

    @note Be sure to call extraction before getResponses Example TrackerFeatureSet::getResponses : :
    */
    const std::vector<Mat>& getResponses() const;

private:
    void clearResponses();
    bool blockAddTrackerFeature;

    std::vector<std::pair<String, Ptr<TrackerFeature>>> features;  //list of features
    std::vector<Mat> responses;  //list of response after compute
};

/************************************ TrackerSampler Base Classes ************************************/

/** @brief Abstract base class for TrackerSamplerAlgorithm that represents the algorithm for the specific
sampler.
 */
class CV_EXPORTS TrackerSamplerAlgorithm
{
public:
    virtual ~TrackerSamplerAlgorithm();

    /** @brief Create TrackerSamplerAlgorithm by tracker sampler type.
    @param trackerSamplerType The trackerSamplerType name

    The modes available now:

    -   "CSC" -- Current State Center
    -   "CS" -- Current State
    */
    static Ptr<TrackerSamplerAlgorithm> create(const String& trackerSamplerType);

    /** @brief Computes the regions starting from a position in an image.

    Return true if samples are computed, false otherwise

    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated

    @param sample The computed samples @cite AAM Fig. 1 variable Sk
    */
    bool sampling(const Mat& image, Rect boundingBox, std::vector<Mat>& sample);

    /** @brief Get the name of the specific TrackerSamplerAlgorithm
    */
    String getClassName() const;

protected:
    String className;

    virtual bool samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample) = 0;
};

/**
 * @brief Class that manages the sampler in order to select regions for the update the model of the tracker
 * [AAM] Sampling e Labeling. See table I and section III B
 */

/** @brief Class that manages the sampler in order to select regions for the update the model of the tracker

@cite AAM Sampling e Labeling. See table I and section III B

TrackerSampler is an aggregation of TrackerSamplerAlgorithm
@sa
   TrackerSamplerAlgorithm
 */
class CV_EXPORTS TrackerSampler
{
public:
    TrackerSampler();

    ~TrackerSampler();

    /** @brief Computes the regions starting from a position in an image
    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated
    */
    void sampling(const Mat& image, Rect boundingBox);

    /** @brief Return the collection of the TrackerSamplerAlgorithm
    */
    const std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm>>>& getSamplers() const;

    /** @brief Return the samples from all TrackerSamplerAlgorithm, @cite AAM Fig. 1 variable Sk
    */
    const std::vector<Mat>& getSamples() const;

    /** @brief Add TrackerSamplerAlgorithm in the collection. Return true if sampler is added, false otherwise
    @param trackerSamplerAlgorithmType The TrackerSamplerAlgorithm name

    The modes available now:
    -   "CSC" -- Current State Center
    -   "CS" -- Current State
    -   "PF" -- Particle Filtering

    Example TrackerSamplerAlgorithm::addTrackerSamplerAlgorithm : :
    @code
         TrackerSamplerCSC::Params CSCparameters;
         Ptr<TrackerSamplerAlgorithm> CSCSampler = new TrackerSamplerCSC( CSCparameters );

         if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
           return false;

         //or add CSC sampler with default parameters
         //sampler->addTrackerSamplerAlgorithm( "CSC" );
    @endcode
    @note If you use the second method, you must initialize the TrackerSamplerAlgorithm
    */
    bool addTrackerSamplerAlgorithm(String trackerSamplerAlgorithmType);

    /** @overload
    @param sampler The TrackerSamplerAlgorithm
    */
    bool addTrackerSamplerAlgorithm(Ptr<TrackerSamplerAlgorithm>& sampler);

private:
    std::vector<std::pair<String, Ptr<TrackerSamplerAlgorithm>>> samplers;
    std::vector<Mat> samples;
    bool blockAddTrackerSampler;

    void clearSamples();
};

/************************************ TrackerModel Base Classes ************************************/

/** @brief Abstract base class for TrackerTargetState that represents a possible state of the target.

See @cite AAM \f$\hat{x}^{i}_{k}\f$ all the states candidates.

Inherits this class with your Target state, In own implementation you can add scale variation,
width, height, orientation, etc.
 */
class CV_EXPORTS TrackerTargetState
{
public:
    virtual ~TrackerTargetState() {}
    /** @brief Get the position
    * @return The position
    */
    Point2f getTargetPosition() const;

    /** @brief Set the position
    * @param position The position
    */
    void setTargetPosition(const Point2f& position);
    /** @brief Get the width of the target
    * @return The width of the target
    */
    int getTargetWidth() const;

    /** @brief Set the width of the target
    * @param width The width of the target
    */
    void setTargetWidth(int width);
    /** @brief Get the height of the target
    * @return The height of the target
    */
    int getTargetHeight() const;

    /** @brief Set the height of the target
    * @param height The height of the target
    */
    void setTargetHeight(int height);

protected:
    Point2f targetPosition;
    int targetWidth;
    int targetHeight;
};

/** @brief Represents the model of the target at frame \f$k\f$ (all states and scores)

See @cite AAM The set of the pair \f$\langle \hat{x}^{i}_{k}, C^{i}_{k} \rangle\f$
@sa TrackerTargetState
 */
typedef std::vector<std::pair<Ptr<TrackerTargetState>, float>> ConfidenceMap;

/** @brief Represents the estimate states for all frames

@cite AAM \f$x_{k}\f$ is the trajectory of the target up to time \f$k\f$

@sa TrackerTargetState
 */
typedef std::vector<Ptr<TrackerTargetState>> Trajectory;

/** @brief Abstract base class for TrackerStateEstimator that estimates the most likely target state.

See @cite AAM State estimator

See @cite AMVOT Statistical modeling (Fig. 3), Table III (generative) - IV (discriminative) - V (hybrid)
 */
class CV_EXPORTS TrackerStateEstimator
{
public:
    virtual ~TrackerStateEstimator();

    /** @brief Estimate the most likely target state, return the estimated state
    @param confidenceMaps The overall appearance model as a list of :cConfidenceMap
    */
    Ptr<TrackerTargetState> estimate(const std::vector<ConfidenceMap>& confidenceMaps);

    /** @brief Update the ConfidenceMap with the scores
    @param confidenceMaps The overall appearance model as a list of :cConfidenceMap
    */
    void update(std::vector<ConfidenceMap>& confidenceMaps);

    /** @brief Create TrackerStateEstimator by tracker state estimator type
    @param trackeStateEstimatorType The TrackerStateEstimator name

    The modes available now:

    -   "BOOSTING" -- Boosting-based discriminative appearance models. See @cite AMVOT section 4.4

    The modes available soon:

    -   "SVM" -- SVM-based discriminative appearance models. See @cite AMVOT section 4.5
    */
    static Ptr<TrackerStateEstimator> create(const String& trackeStateEstimatorType);

    /** @brief Get the name of the specific TrackerStateEstimator
    */
    String getClassName() const;

protected:
    virtual Ptr<TrackerTargetState> estimateImpl(const std::vector<ConfidenceMap>& confidenceMaps) = 0;
    virtual void updateImpl(std::vector<ConfidenceMap>& confidenceMaps) = 0;
    String className;
};

/** @brief Abstract class that represents the model of the target. It must be instantiated by specialized
tracker

See @cite AAM Ak

Inherits this with your TrackerModel
 */
class CV_EXPORTS TrackerModel
{
public:
    TrackerModel();

    virtual ~TrackerModel();

    /** @brief Set TrackerEstimator, return true if the tracker state estimator is added, false otherwise
    @param trackerStateEstimator The TrackerStateEstimator
    @note You can add only one TrackerStateEstimator
    */
    bool setTrackerStateEstimator(Ptr<TrackerStateEstimator> trackerStateEstimator);

    /** @brief Estimate the most likely target location

    @cite AAM ME, Model Estimation table I
    @param responses Features extracted from TrackerFeatureSet
    */
    void modelEstimation(const std::vector<Mat>& responses);

    /** @brief Update the model

    @cite AAM MU, Model Update table I
    */
    void modelUpdate();

    /** @brief Run the TrackerStateEstimator, return true if is possible to estimate a new state, false otherwise
    */
    bool runStateEstimator();

    /** @brief Set the current TrackerTargetState in the Trajectory
    @param lastTargetState The current TrackerTargetState
    */
    void setLastTargetState(const Ptr<TrackerTargetState>& lastTargetState);

    /** @brief Get the last TrackerTargetState from Trajectory
    */
    Ptr<TrackerTargetState> getLastTargetState() const;

    /** @brief Get the list of the ConfidenceMap
    */
    const std::vector<ConfidenceMap>& getConfidenceMaps() const;

    /** @brief Get the last ConfidenceMap for the current frame
    */
    const ConfidenceMap& getLastConfidenceMap() const;

    /** @brief Get the TrackerStateEstimator
    */
    Ptr<TrackerStateEstimator> getTrackerStateEstimator() const;

private:
    void clearCurrentConfidenceMap();

protected:
    std::vector<ConfidenceMap> confidenceMaps;
    Ptr<TrackerStateEstimator> stateEstimator;
    ConfidenceMap currentConfidenceMap;
    Trajectory trajectory;
    int maxCMLength;

    virtual void modelEstimationImpl(const std::vector<Mat>& responses) = 0;
    virtual void modelUpdateImpl() = 0;
};

/************************************ Specific TrackerStateEstimator Classes ************************************/

/** @brief TrackerStateEstimator based on Boosting
    */
class CV_EXPORTS TrackerStateEstimatorMILBoosting : public TrackerStateEstimator
{
public:
    /**
    * Implementation of the target state for TrackerStateEstimatorMILBoosting
    */
    class TrackerMILTargetState : public TrackerTargetState
    {

    public:
        /** @brief Constructor
        * @param position Top left corner of the bounding box
        * @param width Width of the bounding box
        * @param height Height of the bounding box
        * @param foreground label for target or background
        * @param features features extracted
        */
        TrackerMILTargetState(const Point2f& position, int width, int height, bool foreground, const Mat& features);

        ~TrackerMILTargetState() {}

        /** @brief Set label: true for target foreground, false for background
        @param foreground Label for background/foreground
        */
        void setTargetFg(bool foreground);
        /** @brief Set the features extracted from TrackerFeatureSet
        @param features The features extracted
        */
        void setFeatures(const Mat& features);
        /** @brief Get the label. Return true for target foreground, false for background
        */
        bool isTargetFg() const;
        /** @brief Get the features extracted
        */
        Mat getFeatures() const;

    private:
        bool isTarget;
        Mat targetFeatures;
    };

    /** @brief Constructor
    @param nFeatures Number of features for each sample
    */
    TrackerStateEstimatorMILBoosting(int nFeatures = 250);
    ~TrackerStateEstimatorMILBoosting();

    /** @brief Set the current confidenceMap
    @param confidenceMap The current :cConfidenceMap
    */
    void setCurrentConfidenceMap(ConfidenceMap& confidenceMap);

protected:
    Ptr<TrackerTargetState> estimateImpl(const std::vector<ConfidenceMap>& confidenceMaps) CV_OVERRIDE;
    void updateImpl(std::vector<ConfidenceMap>& confidenceMaps) CV_OVERRIDE;

private:
    uint max_idx(const std::vector<float>& v);
    void prepareData(const ConfidenceMap& confidenceMap, Mat& positive, Mat& negative);

    ClfMilBoost boostMILModel;
    bool trained;
    int numFeatures;

    ConfidenceMap currentConfidenceMap;
};

/** @brief TrackerStateEstimatorAdaBoosting based on ADA-Boosting
 */
class CV_EXPORTS TrackerStateEstimatorAdaBoosting : public TrackerStateEstimator
{
public:
    /** @brief Implementation of the target state for TrackerAdaBoostingTargetState
    */
    class TrackerAdaBoostingTargetState : public TrackerTargetState
    {

    public:
        /**
        * @brief Constructor
        * @param position Top left corner of the bounding box
        * @param width Width of the bounding box
        * @param height Height of the bounding box
        * @param foreground label for target or background
        * @param responses list of features
        */
        TrackerAdaBoostingTargetState(const Point2f& position, int width, int height, bool foreground, const Mat& responses);

        ~TrackerAdaBoostingTargetState() {}

        /** @brief Set the features extracted from TrackerFeatureSet
        @param responses The features extracted
        */
        void setTargetResponses(const Mat& responses);
        /** @brief Set label: true for target foreground, false for background
        @param foreground Label for background/foreground
        */
        void setTargetFg(bool foreground);
        /** @brief Get the features extracted
        */
        Mat getTargetResponses() const;
        /** @brief Get the label. Return true for target foreground, false for background
        */
        bool isTargetFg() const;

    private:
        bool isTarget;
        Mat targetResponses;
    };

    /** @brief Constructor
    @param numClassifer Number of base classifiers
    @param initIterations Number of iterations in the initialization
    @param nFeatures Number of features/weak classifiers
    @param patchSize tracking rect
    @param ROI initial ROI
    */
    TrackerStateEstimatorAdaBoosting(int numClassifer, int initIterations, int nFeatures, Size patchSize, const Rect& ROI);

    ~TrackerStateEstimatorAdaBoosting();

    /** @brief Get the sampling ROI
    */
    Rect getSampleROI() const;

    /** @brief Set the sampling ROI
    @param ROI the sampling ROI
    */
    void setSampleROI(const Rect& ROI);

    /** @brief Set the current confidenceMap
    @param confidenceMap The current :cConfidenceMap
    */
    void setCurrentConfidenceMap(ConfidenceMap& confidenceMap);

    /** @brief Get the list of the selected weak classifiers for the classification step
    */
    std::vector<int> computeSelectedWeakClassifier();

    /** @brief Get the list of the weak classifiers that should be replaced
    */
    std::vector<int> computeReplacedClassifier();

    /** @brief Get the list of the weak classifiers that replace those to be replaced
    */
    std::vector<int> computeSwappedClassifier();

protected:
    Ptr<TrackerTargetState> estimateImpl(const std::vector<ConfidenceMap>& confidenceMaps) CV_OVERRIDE;
    void updateImpl(std::vector<ConfidenceMap>& confidenceMaps) CV_OVERRIDE;

    Ptr<StrongClassifierDirectSelection> boostClassifier;

private:
    int numBaseClassifier;
    int iterationInit;
    int numFeatures;
    bool trained;
    Size initPatchSize;
    Rect sampleROI;
    std::vector<int> replacedClassifier;
    std::vector<int> swappedClassifier;

    ConfidenceMap currentConfidenceMap;
};

/**
 * @brief TrackerStateEstimator based on SVM
 */
class CV_EXPORTS TrackerStateEstimatorSVM : public TrackerStateEstimator
{
public:
    TrackerStateEstimatorSVM();
    ~TrackerStateEstimatorSVM();

protected:
    Ptr<TrackerTargetState> estimateImpl(const std::vector<ConfidenceMap>& confidenceMaps) CV_OVERRIDE;
    void updateImpl(std::vector<ConfidenceMap>& confidenceMaps) CV_OVERRIDE;
};

/************************************ Specific TrackerSamplerAlgorithm Classes ************************************/

/** @brief TrackerSampler based on CSC (current state centered), used by MIL algorithm TrackerMIL
 */
class CV_EXPORTS TrackerSamplerCSC : public TrackerSamplerAlgorithm
{
public:
    enum
    {
        MODE_INIT_POS = 1,  //!< mode for init positive samples
        MODE_INIT_NEG = 2,  //!< mode for init negative samples
        MODE_TRACK_POS = 3,  //!< mode for update positive samples
        MODE_TRACK_NEG = 4,  //!< mode for update negative samples
        MODE_DETECT = 5  //!< mode for detect samples
    };

    struct CV_EXPORTS Params
    {
        Params();
        float initInRad;  //!< radius for gathering positive instances during init
        float trackInPosRad;  //!< radius for gathering positive instances during tracking
        float searchWinSize;  //!< size of search window
        int initMaxNegNum;  //!< # negative samples to use during init
        int trackMaxPosNum;  //!< # positive samples to use during training
        int trackMaxNegNum;  //!< # negative samples to use during training
    };

    /** @brief Constructor
    @param parameters TrackerSamplerCSC parameters TrackerSamplerCSC::Params
    */
    TrackerSamplerCSC(const TrackerSamplerCSC::Params& parameters = TrackerSamplerCSC::Params());

    /** @brief Set the sampling mode of TrackerSamplerCSC
    @param samplingMode The sampling mode

    The modes are:

    -   "MODE_INIT_POS = 1" -- for the positive sampling in initialization step
    -   "MODE_INIT_NEG = 2" -- for the negative sampling in initialization step
    -   "MODE_TRACK_POS = 3" -- for the positive sampling in update step
    -   "MODE_TRACK_NEG = 4" -- for the negative sampling in update step
    -   "MODE_DETECT = 5" -- for the sampling in detection step
    */
    void setMode(int samplingMode);

    ~TrackerSamplerCSC();

protected:
    bool samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample) CV_OVERRIDE;

private:
    Params params;
    int mode;
    RNG rng;

    std::vector<Mat> sampleImage(const Mat& img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000);
};

/** @brief TrackerSampler based on CS (current state), used by algorithm TrackerBoosting
 */
class CV_EXPORTS TrackerSamplerCS : public TrackerSamplerAlgorithm
{
public:
    enum
    {
        MODE_POSITIVE = 1,  //!< mode for positive samples
        MODE_NEGATIVE = 2,  //!< mode for negative samples
        MODE_CLASSIFY = 3  //!< mode for classify samples
    };

    struct CV_EXPORTS Params
    {
        Params();
        float overlap;  //!<overlapping for the search windows
        float searchFactor;  //!<search region parameter
    };

    /** @brief Constructor
    @param parameters TrackerSamplerCS parameters TrackerSamplerCS::Params
    */
    TrackerSamplerCS(const TrackerSamplerCS::Params& parameters = TrackerSamplerCS::Params());

    /** @brief Set the sampling mode of TrackerSamplerCS
    @param samplingMode The sampling mode

    The modes are:

    -   "MODE_POSITIVE = 1" -- for the positive sampling
    -   "MODE_NEGATIVE = 2" -- for the negative sampling
    -   "MODE_CLASSIFY = 3" -- for the sampling in classification step
    */
    void setMode(int samplingMode);

    ~TrackerSamplerCS();

    bool samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample) CV_OVERRIDE;
    Rect getROI() const;

private:
    Rect getTrackingROI(float searchFactor);
    Rect RectMultiply(const Rect& rect, float f);
    std::vector<Mat> patchesRegularScan(const Mat& image, Rect trackingROI, Size patchSize);
    void setCheckedROI(Rect imageROI);

    Params params;
    int mode;
    Rect trackedPatch;
    Rect validROI;
    Rect ROI;
};

/** @brief This sampler is based on particle filtering.
*/
class CV_EXPORTS TrackerSamplerPF : public TrackerSamplerAlgorithm
{
public:
    /** @brief This structure contains all the parameters that can be varied during the course of sampling
    algorithm. Below is the structure exposed, together with its members briefly explained with
    reference to the above discussion on algorithm's working.
    */
    struct CV_EXPORTS Params
    {
        Params();
        int iterationNum;  //!< number of selection rounds
        int particlesNum;  //!< number of "perturbed" boxes on each round
        double alpha;  //!< with each new round we exponentially decrease the amount of "perturbing" we allow (like in simulated annealing)
                //!< and this very alpha controls how fast annealing happens, ie. how fast perturbing decreases
        Mat_<double> std;  //!< initial values for perturbing (1-by-4 array, as each rectangle is given by 4 values -- coordinates of opposite vertices,
                //!< hence we have 4 values to perturb)
    };
    /** @brief Constructor
    @param chosenRect Initial rectangle, that is supposed to contain target we'd like to track.
    @param parameters
    */
    TrackerSamplerPF(const Mat& chosenRect, const TrackerSamplerPF::Params& parameters = TrackerSamplerPF::Params());

protected:
    bool samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample) CV_OVERRIDE;

private:
    Params params;
    Ptr<MinProblemSolver> _solver;
    Ptr<MinProblemSolver::Function> _function;
};

/************************************ Specific TrackerFeature Classes ************************************/

/**
 * @brief TrackerFeature based on Feature2D
 */
class CV_EXPORTS TrackerFeatureFeature2d : public TrackerFeature
{
public:
    /**
    * @brief Constructor
    * @param detectorType string of FeatureDetector
    * @param descriptorType string of DescriptorExtractor
    */
    TrackerFeatureFeature2d(String detectorType, String descriptorType);

    ~TrackerFeatureFeature2d() CV_OVERRIDE;

    void selection(Mat& response, int npoints) CV_OVERRIDE;

protected:
    bool computeImpl(const std::vector<Mat>& images, Mat& response) CV_OVERRIDE;

private:
    std::vector<KeyPoint> keypoints;
};

/**
 * @brief TrackerFeature based on HOG
 */
class CV_EXPORTS TrackerFeatureHOG : public TrackerFeature
{
public:
    TrackerFeatureHOG();

    ~TrackerFeatureHOG() CV_OVERRIDE;

    void selection(Mat& response, int npoints) CV_OVERRIDE;

protected:
    bool computeImpl(const std::vector<Mat>& images, Mat& response) CV_OVERRIDE;
};

/** @brief TrackerFeature based on HAAR features, used by TrackerMIL and many others algorithms
@note HAAR features implementation is copied from apps/traincascade and modified according to MIL
 */
class CV_EXPORTS TrackerFeatureHAAR : public TrackerFeature
{
public:
    struct CV_EXPORTS Params
    {
        Params();
        int numFeatures;  //!< # of rects
        Size rectSize;  //!< rect size
        bool isIntegral;  //!< true if input images are integral, false otherwise
    };

    /** @brief Constructor
    @param parameters TrackerFeatureHAAR parameters TrackerFeatureHAAR::Params
    */
    TrackerFeatureHAAR(const TrackerFeatureHAAR::Params& parameters = TrackerFeatureHAAR::Params());

    ~TrackerFeatureHAAR() CV_OVERRIDE;

    /** @brief Compute the features only for the selected indices in the images collection
    @param selFeatures indices of selected features
    @param images The images
    @param response Collection of response for the specific TrackerFeature
    */
    bool extractSelected(const std::vector<int> selFeatures, const std::vector<Mat>& images, Mat& response);

    /** @brief Identify most effective features
    @param response Collection of response for the specific TrackerFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
    */
    void selection(Mat& response, int npoints) CV_OVERRIDE;

    /** @brief Swap the feature in position source with the feature in position target
    @param source The source position
    @param target The target position
    */
    bool swapFeature(int source, int target);

    /** @brief   Swap the feature in position id with the feature input
    @param id The position
    @param feature The feature
    */
    bool swapFeature(int id, CvHaarEvaluator::FeatureHaar& feature);

    /** @brief Get the feature in position id
    @param id The position
    */
    CvHaarEvaluator::FeatureHaar& getFeatureAt(int id);

protected:
    bool computeImpl(const std::vector<Mat>& images, Mat& response) CV_OVERRIDE;

private:
    Params params;
    Ptr<CvHaarEvaluator> featureEvaluator;
};

/**
 * @brief TrackerFeature based on LBP
 */
class CV_EXPORTS TrackerFeatureLBP : public TrackerFeature
{
public:
    TrackerFeatureLBP();

    ~TrackerFeatureLBP();

    void selection(Mat& response, int npoints) CV_OVERRIDE;

protected:
    bool computeImpl(const std::vector<Mat>& images, Mat& response) CV_OVERRIDE;
};

//! @}

}}}  // namespace cv::detail::tracking

#endif  // OPENCV_VIDEO_DETAIL_TRACKING_HPP
