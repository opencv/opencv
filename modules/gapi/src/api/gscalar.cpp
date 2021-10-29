// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gscalar.hpp>
#include "api/gorigin.hpp"

// cv::GScalar public implementation ///////////////////////////////////////////
cv::GScalar::GScalar()
    : m_priv(new GOrigin(GShape::GSCALAR, cv::GNode::Param()))
{
}

cv::GScalar::GScalar(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GSCALAR, n, out))
{
}

cv::GScalar::GScalar(const cv::Scalar& s)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(s)))
{
}

cv::GScalar::GScalar(cv::Scalar&& s)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(std::move(s))))
{
}

cv::GScalar::GScalar(double v0)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(cv::Scalar(v0))))
{
}

cv::GOrigin& cv::GScalar::priv()
{
    return *m_priv;
}

const cv::GOrigin& cv::GScalar::priv() const
{
    return *m_priv;
}

//N.B. if we ever need more complicated logic for desc_of(cv::(gapi::own::)Scalar)
//dispatching should be done in the same way as for cv::(gapi::own)::Mat
cv::GScalarDesc cv::descr_of(const cv::Scalar &)
{
    return empty_scalar_desc();
}

namespace cv {
std::ostream& operator<<(std::ostream& os, const cv::GScalarDesc &)
{
    os << "(scalar)";
    return os;
}
}
