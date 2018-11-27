// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include "opencv2/gapi/garray.hpp"
#include "api/gapi_priv.hpp" // GOrigin

// cv::detail::GArrayU public implementation ///////////////////////////////////
cv::detail::GArrayU::GArrayU()
    : m_priv(new GOrigin(GShape::GARRAY, cv::GNode::Param()))
{
}

cv::detail::GArrayU::GArrayU(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GARRAY, n, out))
{
}

cv::GOrigin& cv::detail::GArrayU::priv()
{
    return *m_priv;
}

const cv::GOrigin& cv::detail::GArrayU::priv() const
{
    return *m_priv;
}

void cv::detail::GArrayU::setConstructFcn(ConstructVec &&cv)
{
    m_priv->ctor = std::move(cv);
}

namespace cv {
std::ostream& operator<<(std::ostream& os, const cv::GArrayDesc &)
{
    // FIXME: add type information here
    os << "(array)";
    return os;
}
}
