// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.detail.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerModel::TrackerModel()
{
    stateEstimator = Ptr<TrackerStateEstimator>();
    maxCMLength = 10;
}

TrackerModel::~TrackerModel()
{
    // nothing
}

bool TrackerModel::setTrackerStateEstimator(Ptr<TrackerStateEstimator> trackerStateEstimator)
{
    if (stateEstimator.get())
    {
        return false;
    }

    stateEstimator = trackerStateEstimator;
    return true;
}

Ptr<TrackerStateEstimator> TrackerModel::getTrackerStateEstimator() const
{
    return stateEstimator;
}

void TrackerModel::modelEstimation(const std::vector<Mat>& responses)
{
    modelEstimationImpl(responses);
}

void TrackerModel::clearCurrentConfidenceMap()
{
    currentConfidenceMap.clear();
}

void TrackerModel::modelUpdate()
{
    modelUpdateImpl();

    if (maxCMLength != -1 && (int)confidenceMaps.size() >= maxCMLength - 1)
    {
        int l = maxCMLength / 2;
        confidenceMaps.erase(confidenceMaps.begin(), confidenceMaps.begin() + l);
    }
    if (maxCMLength != -1 && (int)trajectory.size() >= maxCMLength - 1)
    {
        int l = maxCMLength / 2;
        trajectory.erase(trajectory.begin(), trajectory.begin() + l);
    }
    confidenceMaps.push_back(currentConfidenceMap);
    stateEstimator->update(confidenceMaps);

    clearCurrentConfidenceMap();
}

bool TrackerModel::runStateEstimator()
{
    if (!stateEstimator)
    {
        CV_Error(-1, "Tracker state estimator is not setted");
    }
    Ptr<TrackerTargetState> targetState = stateEstimator->estimate(confidenceMaps);
    if (!targetState)
        return false;

    setLastTargetState(targetState);
    return true;
}

void TrackerModel::setLastTargetState(const Ptr<TrackerTargetState>& lastTargetState)
{
    trajectory.push_back(lastTargetState);
}

Ptr<TrackerTargetState> TrackerModel::getLastTargetState() const
{
    return trajectory.back();
}

const std::vector<ConfidenceMap>& TrackerModel::getConfidenceMaps() const
{
    return confidenceMaps;
}

const ConfidenceMap& TrackerModel::getLastConfidenceMap() const
{
    return confidenceMaps.back();
}

Point2f TrackerTargetState::getTargetPosition() const
{
    return targetPosition;
}

void TrackerTargetState::setTargetPosition(const Point2f& position)
{
    targetPosition = position;
}

int TrackerTargetState::getTargetWidth() const
{
    return targetWidth;
}

void TrackerTargetState::setTargetWidth(int width)
{
    targetWidth = width;
}
int TrackerTargetState::getTargetHeight() const
{
    return targetHeight;
}

void TrackerTargetState::setTargetHeight(int height)
{
    targetHeight = height;
}

}}}  // namespace cv::detail::tracking
