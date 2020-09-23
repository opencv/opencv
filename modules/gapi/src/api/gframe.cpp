// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gframe.hpp>

#include "api/gorigin.hpp"

// cv::GFrame public implementation //////////////////////////////////////////////
cv::GFrame::GFrame()
    : m_priv(new GOrigin(GShape::GMAT, GNode::Param())) {
    // N.B.: The shape here is still GMAT as currently cv::Mat is used
    // as an underlying host type. Will be changed to GFRAME once
    // GExecutor & GStreamingExecutor & selected backends will be extended
    // to support cv::MediaFrame.
}

cv::GFrame::GFrame(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GMAT, n, out)) {
    // N.B.: GMAT is here for the same reason as above ^
}

cv::GOrigin& cv::GFrame::priv() {
    return *m_priv;
}

const cv::GOrigin& cv::GFrame::priv() const {
    return *m_priv;
}

namespace cv {
std::ostream& operator<<(std::ostream& os, const cv::GFrameDesc &) {
    return os;
}

} // namespace cv
