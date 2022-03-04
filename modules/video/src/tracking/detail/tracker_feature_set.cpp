// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.detail.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerFeatureSet::TrackerFeatureSet()
{
    blockAddTrackerFeature = false;
}

TrackerFeatureSet::~TrackerFeatureSet()
{
    // nothing
}

void TrackerFeatureSet::extraction(const std::vector<Mat>& images)
{
    blockAddTrackerFeature = true;

    clearResponses();
    responses.resize(features.size());

    for (size_t i = 0; i < features.size(); i++)
    {
        CV_DbgAssert(features[i]);
        features[i]->compute(images, responses[i]);
    }
}

bool TrackerFeatureSet::addTrackerFeature(const Ptr<TrackerFeature>& feature)
{
    CV_Assert(!blockAddTrackerFeature);
    CV_Assert(feature);

    features.push_back(feature);
    return true;
}

const std::vector<Ptr<TrackerFeature>>& TrackerFeatureSet::getTrackerFeatures() const
{
    return features;
}

const std::vector<Mat>& TrackerFeatureSet::getResponses() const
{
    return responses;
}

void TrackerFeatureSet::clearResponses()
{
    responses.clear();
}

}}}  // namespace cv::detail::tracking
