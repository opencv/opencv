// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.private.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerStateEstimator::~TrackerStateEstimator()
{
}

Ptr<TrackerTargetState> TrackerStateEstimator::estimate(const std::vector<ConfidenceMap>& confidenceMaps)
{
    if (confidenceMaps.empty())
        return Ptr<TrackerTargetState>();

    return estimateImpl(confidenceMaps);
}

void TrackerStateEstimator::update(std::vector<ConfidenceMap>& confidenceMaps)
{
    if (confidenceMaps.empty())
        return;

    return updateImpl(confidenceMaps);
}

String TrackerStateEstimator::getClassName() const
{
    return className;
}

}}}  // namespace cv::detail::tracking
