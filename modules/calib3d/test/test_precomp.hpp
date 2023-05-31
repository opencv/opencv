// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <functional>
#include <numeric>

#include "opencv2/ts.hpp"
#include "opencv2/calib3d.hpp"

namespace cvtest
{
    void Rodrigues(const Mat& src, Mat& dst, Mat* jac=0);
}

namespace opencv_test {
CVTEST_GUARD_SYMBOL(Rodrigues)
} // namespace

#endif
