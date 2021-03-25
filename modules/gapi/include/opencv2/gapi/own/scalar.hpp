// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GAPI_OWN_SCALAR_HPP
#define OPENCV_GAPI_GAPI_OWN_SCALAR_HPP

#include <opencv2/gapi/own/exports.hpp>

namespace cv
{
namespace gapi
{
namespace own
{

class GAPI_EXPORTS Scalar
{
public:
    Scalar() = default;
    explicit Scalar(double v0) { val[0] = v0; };
    Scalar(double v0, double v1, double v2 = 0, double v3 = 0)
        : val{v0, v1, v2, v3}
    {
    }

    const double& operator[](int i) const { return val[i]; }
          double& operator[](int i)       { return val[i]; }

    static Scalar all(double v0) { return Scalar(v0, v0, v0, v0); }

    double val[4] = {0};
};

inline bool operator==(const Scalar& lhs, const Scalar& rhs)
{
    return std::equal(std::begin(lhs.val), std::end(lhs.val), std::begin(rhs.val));
}

} // namespace own
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GAPI_OWN_SCALAR_HPP
