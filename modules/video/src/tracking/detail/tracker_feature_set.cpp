// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.private.hpp"

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
    clearResponses();
    responses.resize(features.size());

    for (size_t i = 0; i < features.size(); i++)
    {
        Mat response;
        features[i].second->compute(images, response);
        responses[i] = response;
    }

    if (!blockAddTrackerFeature)
    {
        blockAddTrackerFeature = true;
    }
}

void TrackerFeatureSet::selection()
{
}

void TrackerFeatureSet::removeOutliers()
{
}

bool TrackerFeatureSet::addTrackerFeature(String trackerFeatureType)
{
    if (blockAddTrackerFeature)
    {
        return false;
    }
    Ptr<TrackerFeature> feature = TrackerFeature::create(trackerFeatureType);

    if (!feature)
    {
        return false;
    }

    features.push_back(std::make_pair(trackerFeatureType, feature));

    return true;
}

bool TrackerFeatureSet::addTrackerFeature(Ptr<TrackerFeature>& feature)
{
    if (blockAddTrackerFeature)
    {
        return false;
    }

    String trackerFeatureType = feature->getClassName();
    features.push_back(std::make_pair(trackerFeatureType, feature));

    return true;
}

const std::vector<std::pair<String, Ptr<TrackerFeature>>>& TrackerFeatureSet::getTrackerFeature() const
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
