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

protected:
    virtual bool computeImpl(const std::vector<Mat>& images, Mat& response) = 0;
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

    /** @brief Add TrackerFeature in the collection. Return true if TrackerFeature is added, false otherwise
    @param feature The TrackerFeature class
    */
    bool addTrackerFeature(const Ptr<TrackerFeature>& feature);

    /** @brief Get the TrackerFeature collection (TrackerFeature name, TrackerFeature pointer)
    */
    const std::vector<Ptr<TrackerFeature>>& getTrackerFeatures() const;

    /** @brief Get the responses
    @note Be sure to call extraction before getResponses Example TrackerFeatureSet::getResponses
    */
    const std::vector<Mat>& getResponses() const;

private:
    void clearResponses();
    bool blockAddTrackerFeature;

    std::vector<Ptr<TrackerFeature>> features;  // list of features
    std::vector<Mat> responses;  // list of response after compute
};

/************************************ TrackerSampler Base Classes ************************************/

/** @brief Abstract base class for TrackerSamplerAlgorithm that represents the algorithm for the specific
sampler.
*/
class CV_EXPORTS TrackerSamplerAlgorithm
{
public:
    virtual ~TrackerSamplerAlgorithm();

    /** @brief Computes the regions starting from a position in an image.

    Return true if samples are computed, false otherwise

    @param image The current frame
    @param boundingBox The bounding box from which regions can be calculated

    @param sample The computed samples @cite AAM Fig. 1 variable Sk
    */
    virtual bool sampling(const Mat& image, const Rect& boundingBox, std::vector<Mat>& sample) = 0;
};

/**
 * \brief Class that manages the sampler in order to select regions for the update the model of the tracker
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
    const std::vector<Ptr<TrackerSamplerAlgorithm>>& getSamplers() const;

    /** @brief Return the samples from all TrackerSamplerAlgorithm, @cite AAM Fig. 1 variable Sk
    */
    const std::vector<Mat>& getSamples() const;

    /** @brief Add TrackerSamplerAlgorithm in the collection. Return true if sampler is added, false otherwise
    @param sampler The TrackerSamplerAlgorithm
    */
    bool addTrackerSamplerAlgorithm(const Ptr<TrackerSamplerAlgorithm>& sampler);

private:
    std::vector<Ptr<TrackerSamplerAlgorithm>> samplers;
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
    virtual ~TrackerTargetState() {};
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

/** @brief Abstract class that represents the model of the target.

It must be instantiated by specialized tracker

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

// None

/************************************ Specific TrackerSamplerAlgorithm Classes ************************************/

/** @brief TrackerSampler based on CSC (current state centered), used by MIL algorithm TrackerMIL
 */
class CV_EXPORTS TrackerSamplerCSC : public TrackerSamplerAlgorithm
{
public:
    ~TrackerSamplerCSC();

    enum MODE
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

    bool sampling(const Mat& image, const Rect& boundingBox, std::vector<Mat>& sample) CV_OVERRIDE;

private:
    Params params;
    int mode;
    RNG rng;

    std::vector<Mat> sampleImage(const Mat& img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000);
};

//! @}

}}}  // namespace cv::detail::tracking

#endif  // OPENCV_VIDEO_DETAIL_TRACKING_HPP
