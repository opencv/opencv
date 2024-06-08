// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.detail.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerFeature::~TrackerFeature()
{
    // nothing
}

void TrackerFeature::compute(const std::vector<Mat>& images, Mat& response)
{
    if (images.empty())
        return;

    computeImpl(images, response);
}

}}}  // namespace cv::detail::tracking