// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

// This code is also subject to the license terms in the LICENSE_WillowGarage.md file found in this module's directory

#ifndef __OPENCV_RGBD_UTILS_HPP__
#define __OPENCV_RGBD_UTILS_HPP__

#include "precomp.hpp"

namespace cv
{
namespace rgbd
{

/** If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
 * by 1000 to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
 * Otherwise, the image is simply converted to floats
 * @param in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), it is assumed in meters)
 * @param the desired output depth (floats or double)
 * @param out The rescaled float depth image
 */
/* void rescaleDepth(InputArray in_in, int depth, OutputArray out_out); */

template<typename T>
void
rescaleDepthTemplated(const Mat& in, Mat& out);

template<>
inline void
rescaleDepthTemplated<float>(const Mat& in, Mat& out)
{
  rescaleDepth(in, CV_32F, out);
}

template<>
inline void
rescaleDepthTemplated<double>(const Mat& in, Mat& out)
{
  rescaleDepth(in, CV_64F, out);
}

} // namespace rgbd


namespace kinfu {

// One place to turn intrinsics on and off
#define USE_INTRINSICS CV_SIMD128

typedef float depthType;

const float qnan = std::numeric_limits<float>::quiet_NaN();
const cv::Vec3f nan3(qnan, qnan, qnan);
#if USE_INTRINSICS
const cv::v_float32x4 nanv(qnan, qnan, qnan, qnan);
#endif

inline bool isNaN(cv::Point3f p)
{
    return (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z));
}

#if USE_INTRINSICS
static inline bool isNaN(const cv::v_float32x4& p)
{
    return cv::v_check_any(p != p);
}
#endif

inline size_t roundDownPow2(size_t x)
{
    size_t shift = 0;
    while(x != 0)
    {
        shift++; x >>= 1;
    }
    return (size_t)(1ULL << (shift-1));
}

} // namespace kinfu

} // namespace cv


#endif

/* End of file. */
