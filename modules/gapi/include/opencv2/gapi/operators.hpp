// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPERATORS_HPP
#define OPENCV_GAPI_OPERATORS_HPP

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>

GAPI_EXPORTS cv::GMat operator+(const cv::GMat&    lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator+(const cv::GMat&    lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator+(const cv::GScalar& lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator-(const cv::GMat&    lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator-(const cv::GMat&    lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator-(const cv::GScalar& lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator*(const cv::GMat&    lhs, float              rhs);
GAPI_EXPORTS cv::GMat operator*(float              lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator*(const cv::GMat&    lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator*(const cv::GScalar& lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator/(const cv::GMat&    lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator/(const cv::GScalar& lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator/(const cv::GMat&    lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator&(const cv::GMat&    lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator|(const cv::GMat&    lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator^(const cv::GMat&    lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator~(const cv::GMat&    lhs);

GAPI_EXPORTS cv::GMat operator&(const cv::GScalar& lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator|(const cv::GScalar& lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator^(const cv::GScalar& lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator&(const cv::GMat& lhs, const cv::GScalar&    rhs);
GAPI_EXPORTS cv::GMat operator|(const cv::GMat& lhs, const cv::GScalar&    rhs);
GAPI_EXPORTS cv::GMat operator^(const cv::GMat& lhs, const cv::GScalar&    rhs);

GAPI_EXPORTS cv::GMat operator>(const cv::GMat&    lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator>=(const cv::GMat&   lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator<(const cv::GMat&    lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator<=(const cv::GMat&   lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator==(const cv::GMat&   lhs, const cv::GMat&    rhs);
GAPI_EXPORTS cv::GMat operator!=(const cv::GMat&   lhs, const cv::GMat&    rhs);

GAPI_EXPORTS cv::GMat operator>(const cv::GMat&    lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator>=(const cv::GMat&   lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator<(const cv::GMat&    lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator<=(const cv::GMat&   lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator==(const cv::GMat&   lhs, const cv::GScalar& rhs);
GAPI_EXPORTS cv::GMat operator!=(const cv::GMat&   lhs, const cv::GScalar& rhs);

GAPI_EXPORTS cv::GMat operator>(const cv::GScalar&    lhs, const cv::GMat& rhs);
GAPI_EXPORTS cv::GMat operator>=(const cv::GScalar&   lhs, const cv::GMat& rhs);
GAPI_EXPORTS cv::GMat operator<(const cv::GScalar&    lhs, const cv::GMat& rhs);
GAPI_EXPORTS cv::GMat operator<=(const cv::GScalar&   lhs, const cv::GMat& rhs);
GAPI_EXPORTS cv::GMat operator==(const cv::GScalar&   lhs, const cv::GMat& rhs);
GAPI_EXPORTS cv::GMat operator!=(const cv::GScalar&   lhs, const cv::GMat& rhs);



#endif // OPENCV_GAPI_OPERATORS_HPP
