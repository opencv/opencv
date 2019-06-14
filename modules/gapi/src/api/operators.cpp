// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/operators.hpp>

cv::GMat operator+(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::add(lhs, rhs);
}

cv::GMat operator+(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::addC(lhs, rhs);
}

cv::GMat operator+(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::addC(rhs, lhs);
}

cv::GMat operator-(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::sub(lhs, rhs);
}

cv::GMat operator-(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::subC(lhs, rhs);
}

cv::GMat operator-(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::subRC(lhs, rhs);
}

cv::GMat operator*(const cv::GMat& lhs, float rhs)
{
    return cv::gapi::mulC(lhs, static_cast<double>(rhs));
}

cv::GMat operator*(float lhs, const cv::GMat& rhs)
{
    return cv::gapi::mulC(rhs, static_cast<double>(lhs));
}

cv::GMat operator*(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::mulC(lhs, rhs);
}

cv::GMat operator*(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::mulC(rhs, lhs);
}

cv::GMat operator/(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::divC(lhs, rhs, 1.0);
}

cv::GMat operator/(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::div(lhs, rhs, 1.0);
}

cv::GMat operator/(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::divRC(lhs, rhs, 1.0);
}

cv::GMat operator&(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::bitwise_and(lhs, rhs);
}

cv::GMat operator&(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::bitwise_and(lhs, rhs);
}

cv::GMat operator&(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::bitwise_and(rhs, lhs);
}

cv::GMat operator|(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::bitwise_or(lhs, rhs);
}

cv::GMat operator|(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::bitwise_or(lhs, rhs);
}

cv::GMat operator|(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::bitwise_or(rhs, lhs);
}

cv::GMat operator^(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::bitwise_xor(lhs, rhs);
}

cv::GMat operator^(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::bitwise_xor(lhs, rhs);
}

cv::GMat operator^(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::bitwise_xor(rhs, lhs);
}

cv::GMat operator~(const cv::GMat& lhs)
{
    return cv::gapi::bitwise_not(lhs);
}

cv::GMat operator>(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpGT(lhs, rhs);
}

cv::GMat operator>=(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpGE(lhs, rhs);
}

cv::GMat operator<(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpLT(lhs, rhs);
}

cv::GMat operator<=(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpLE(lhs, rhs);
}

cv::GMat operator==(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpEQ(lhs, rhs);
}

cv::GMat operator!=(const cv::GMat& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpNE(lhs, rhs);
}

cv::GMat operator>(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::cmpGT(lhs, rhs);
}

cv::GMat operator>=(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::cmpGE(lhs, rhs);
}

cv::GMat operator<(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::cmpLT(lhs, rhs);
}

cv::GMat operator<=(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::cmpLE(lhs, rhs);
}

cv::GMat operator==(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::cmpEQ(lhs, rhs);
}

cv::GMat operator!=(const cv::GMat& lhs, const cv::GScalar& rhs)
{
    return cv::gapi::cmpNE(lhs, rhs);
}

cv::GMat operator>(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpLT(rhs, lhs);
}
cv::GMat operator>=(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpLE(rhs, lhs);
}
cv::GMat operator<(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpGT(rhs, lhs);
}
cv::GMat operator<=(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpGE(rhs, lhs);
}
cv::GMat operator==(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpEQ(rhs, lhs);
}
cv::GMat operator!=(const cv::GScalar& lhs, const cv::GMat& rhs)
{
    return cv::gapi::cmpNE(rhs, lhs);
}
