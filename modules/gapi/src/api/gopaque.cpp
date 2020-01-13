// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"
#include <opencv2/gapi/gopaque.hpp>
#include "api/gorigin.hpp"

// cv::detail::GOpaqueU public implementation ///////////////////////////////////
cv::detail::GOpaqueU::GOpaqueU()
    : m_priv(new GOrigin(GShape::GOPAQUE, cv::GNode::Param()))
{
}

cv::detail::GOpaqueU::GOpaqueU(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GOPAQUE, n, out))
{
}

cv::GOrigin& cv::detail::GOpaqueU::priv()
{
    return *m_priv;
}

const cv::GOrigin& cv::detail::GOpaqueU::priv() const
{
    return *m_priv;
}

void cv::detail::GOpaqueU::setConstructFcn(ConstructOpaque &&co)
{
    m_priv->ctor = std::move(co);
}

namespace cv {
std::ostream& operator<<(std::ostream& os, const cv::GOpaqueDesc &)
{
    // FIXME: add type information here
    os << "(Opaque)";
    return os;
}
}
