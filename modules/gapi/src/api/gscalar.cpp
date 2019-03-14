// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include "opencv2/gapi/gscalar.hpp"
#include "opencv2/gapi/own/convert.hpp"
#include "api/gapi_priv.hpp" // GOrigin

// cv::GScalar public implementation ///////////////////////////////////////////
cv::GScalar::GScalar()
    : m_priv(new GOrigin(GShape::GSCALAR, cv::GNode::Param()))
{
}

cv::GScalar::GScalar(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GSCALAR, n, out))
{
}

cv::GScalar::GScalar(const cv::gapi::own::Scalar& s)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(s)))
{
}

cv::GScalar::GScalar(cv::gapi::own::Scalar&& s)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(std::move(s))))
{
}

cv::GScalar::GScalar(double v0)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(cv::gapi::own::Scalar(v0))))
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

cv::GScalarDesc cv::descr_of(const cv::gapi::own::Scalar &)
{
    return empty_scalar_desc();
}

#if !defined(GAPI_STANDALONE)
cv::GScalar::GScalar(const cv::Scalar& s)
    : m_priv(new GOrigin(GShape::GSCALAR, cv::gimpl::ConstVal(to_own(s))))
{
}

cv::GScalarDesc cv::descr_of(const cv::Scalar& s)
{
    return cv::descr_of(to_own(s));
}
#endif // !defined(GAPI_STANDALONE)

namespace cv {
std::ostream& operator<<(std::ostream& os, const cv::GScalarDesc &)
{
    os << "(scalar)";
    return os;
}
}
