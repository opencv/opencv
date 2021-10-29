// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.detail.hpp"
#include "tracker_mil_state.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

/**
 * TrackerStateEstimatorMILBoosting::TrackerMILTargetState
 */
TrackerStateEstimatorMILBoosting::TrackerMILTargetState::TrackerMILTargetState(const Point2f& position, int width, int height, bool foreground,
        const Mat& features)
{
    setTargetPosition(position);
    setTargetWidth(width);
    setTargetHeight(height);
    setTargetFg(foreground);
    setFeatures(features);
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setTargetFg(bool foreground)
{
    isTarget = foreground;
}

void TrackerStateEstimatorMILBoosting::TrackerMILTargetState::setFeatures(const Mat& features)
{
    targetFeatures = features;
}

bool TrackerStateEstimatorMILBoosting::TrackerMILTargetState::isTargetFg() const
{
    return isTarget;
}

Mat TrackerStateEstimatorMILBoosting::TrackerMILTargetState::getFeatures() const
{
    return targetFeatures;
}

TrackerStateEstimatorMILBoosting::TrackerStateEstimatorMILBoosting(int nFeatures)
{
    className = "BOOSTING";
    trained = false;
    numFeatures = nFeatures;
}

TrackerStateEstimatorMILBoosting::~TrackerStateEstimatorMILBoosting()
{
}

void TrackerStateEstimatorMILBoosting::setCurrentConfidenceMap(ConfidenceMap& confidenceMap)
{
    currentConfidenceMap.clear();
    currentConfidenceMap = confidenceMap;
}

uint TrackerStateEstimatorMILBoosting::max_idx(const std::vector<float>& v)
{
    const float* findPtr = &(*std::max_element(v.begin(), v.end()));
    const float* beginPtr = &(*v.begin());
    return (uint)(findPtr - beginPtr);
}

Ptr<TrackerTargetState> TrackerStateEstimatorMILBoosting::estimateImpl(const std::vector<ConfidenceMap>& /*confidenceMaps*/)
{
    //run ClfMilBoost classify in order to compute next location
    if (currentConfidenceMap.empty())
        return Ptr<TrackerTargetState>();

    Mat positiveStates;
    Mat negativeStates;

    prepareData(currentConfidenceMap, positiveStates, negativeStates);

    std::vector<float> prob = boostMILModel.classify(positiveStates);

    int bestind = max_idx(prob);
    //float resp = prob[bestind];

    return currentConfidenceMap.at(bestind).first;
}

void TrackerStateEstimatorMILBoosting::prepareData(const ConfidenceMap& confidenceMap, Mat& positive, Mat& negative)
{

    int posCounter = 0;
    int negCounter = 0;

    for (size_t i = 0; i < confidenceMap.size(); i++)
    {
        Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at(i).first.staticCast<TrackerMILTargetState>();
        CV_DbgAssert(currentTargetState);
        if (currentTargetState->isTargetFg())
            posCounter++;
        else
            negCounter++;
    }

    positive.create(posCounter, numFeatures, CV_32FC1);
    negative.create(negCounter, numFeatures, CV_32FC1);

    //TODO change with mat fast access
    //initialize trainData (positive and negative)

    int pc = 0;
    int nc = 0;
    for (size_t i = 0; i < confidenceMap.size(); i++)
    {
        Ptr<TrackerMILTargetState> currentTargetState = confidenceMap.at(i).first.staticCast<TrackerMILTargetState>();
        Mat stateFeatures = currentTargetState->getFeatures();

        if (currentTargetState->isTargetFg())
        {
            for (int j = 0; j < stateFeatures.rows; j++)
            {
                //fill the positive trainData with the value of the feature j for sample i
                positive.at<float>(pc, j) = stateFeatures.at<float>(j, 0);
            }
            pc++;
        }
        else
        {
            for (int j = 0; j < stateFeatures.rows; j++)
            {
                //fill the negative trainData with the value of the feature j for sample i
                negative.at<float>(nc, j) = stateFeatures.at<float>(j, 0);
            }
            nc++;
        }
    }
}

void TrackerStateEstimatorMILBoosting::updateImpl(std::vector<ConfidenceMap>& confidenceMaps)
{

    if (!trained)
    {
        //this is the first time that the classifier is built
        //init MIL
        boostMILModel.init();
        trained = true;
    }

    ConfidenceMap lastConfidenceMap = confidenceMaps.back();
    Mat positiveStates;
    Mat negativeStates;

    prepareData(lastConfidenceMap, positiveStates, negativeStates);
    //update MIL
    boostMILModel.update(positiveStates, negativeStates);
}

}}}  // namespace cv::detail::tracking
