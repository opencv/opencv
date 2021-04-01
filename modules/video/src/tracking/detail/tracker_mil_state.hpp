// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEO_DETAIL_TRACKING_MIL_STATE_HPP
#define OPENCV_VIDEO_DETAIL_TRACKING_MIL_STATE_HPP

#include "opencv2/video/detail/tracking.detail.hpp"
#include "tracking_online_mil.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

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
        /**
        * \brief Constructor
        * \param position Top left corner of the bounding box
        * \param width Width of the bounding box
        * \param height Height of the bounding box
        * \param foreground label for target or background
        * \param features features extracted
        */
        TrackerMILTargetState(const Point2f& position, int width, int height, bool foreground, const Mat& features);

        ~TrackerMILTargetState() {};

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

}}}  // namespace cv::detail::tracking

#endif
