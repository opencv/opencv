// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CALIB3D_TEST_ROTATION_DISTANCE_HPP
#define OPENCV_CALIB3D_TEST_ROTATION_DISTANCE_HPP

#include <cmath>
#include "opencv2/core.hpp"

namespace opencv_test
{

/**
 * Angular distance on @f$SO(3)@f$ is the principal angle of the relative
 * rotation of @f$R_1@f$ and @f$R_2@f$:
 * @f[
 * d_\angle(R_1, R_2) =
 * \left\|\log\left(R_1 R_2^\top\right)\right\|_2 \in [0, \pi].
 * @f]
 *
 * Here @f$R_1@f$ is the estimated rotation and @f$R_2@f$ is the expected
 * rotation. The implementation evaluates this norm from the sine and cosine
 * of the relative rotation angle.
 *
 * The returned value is in radians.
 */
template<typename T>
inline T angularDistance(const cv::Matx<T, 3, 3>& estimated, const cv::Matx<T, 3, 3>& expected)
{
    using std::atan2;
    using std::hypot;
    const cv::Matx<T, 3, 3> relative = estimated * expected.t();
    const T x = (relative(2, 1) - relative(1, 2)) / T(2);
    const T y = (relative(0, 2) - relative(2, 0)) / T(2);
    const T z = (relative(1, 0) - relative(0, 1)) / T(2);
#if defined(__cpp_lib_hypot) && __cpp_lib_hypot >= 201603L
    const T sine = hypot(x, y, z);
#else
    const T sine = hypot(hypot(x, y), z);
#endif
    const T cosine = (static_cast<T>(cv::trace(relative)) - T(1)) / T(2);
    // For a relative rotation angle θ ∈ [0, π],
    // acos(cos(θ)) = atan2(sin(θ), cos(θ)) = θ for θ ∈ [0, π]. Unlike acos,
    // atan2 derives the angle from the direction of the sine and cosine pair,
    // so no explicit normalization or clamping is needed after roundoff.
    return atan2(sine, cosine);
}

} // namespace opencv_test

#endif
