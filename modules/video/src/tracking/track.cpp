// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

namespace cv {


Track::Track()
{
    // nothing
}


Track::Track(const cv::Rect2f& tlwh, int trackId, int classId, float score)
{
    rect = tlwh;
    classScore = score;
    classLabel = classId; // static_cast<> ()
    trackingId = trackId; // abs(tracking_id) <= (1 << 24) or tracking_id % (1 << 24)
}

Track::~Track()
{
    // nothing
}

}  // namespace cv
