// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "utils.hpp"

namespace cv
{

template<typename TMat>
inline TMat getTMat(InputArray, int)
{
    return TMat();
}

template<>
Mat getTMat<Mat>(InputArray a, int i)
{
    return a.getMat(i);
}

template<>
UMat getTMat<UMat>(InputArray a, int i)
{
    return a.getUMat(i);
}

template<typename TMat>
inline TMat& getTMatRef(InputOutputArray, int)
{
    return TMat();
}

template<>
Mat& getTMatRef<Mat>(InputOutputArray a, int i)
{
    return a.getMatRef(i);
}

/** If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
 * by 1000 to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
 * Otherwise, the image is simply converted to floats
 * @param in_in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), it is assumed in meters)
 * @param depth the desired output depth (floats or double)
 * @param out_out The rescaled float depth image
 */
void rescaleDepth(InputArray in_in, int type, OutputArray out_out, double depth_factor)
{
    cv::Mat in = in_in.getMat();
    CV_Assert(in.type() == CV_64FC1 || in.type() == CV_32FC1 || in.type() == CV_16UC1 || in.type() == CV_16SC1);
    CV_Assert(type == CV_64FC1 || type == CV_32FC1);

    int in_depth = in.depth();

    out_out.create(in.size(), type);
    cv::Mat out = out_out.getMat();
    if (in_depth == CV_16U)
    {
      in.convertTo(out, type, 1 / depth_factor); //convert to float so that it is in meters
      cv::Mat valid_mask = in == std::numeric_limits<ushort>::min(); // Should we do std::numeric_limits<ushort>::max() too ?
      out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
    }
    if (in_depth == CV_16S)
    {
      in.convertTo(out, type, 1 / depth_factor); //convert to float so tha$
      cv::Mat valid_mask = (in == std::numeric_limits<short>::min()) | (in == std::numeric_limits<short>::max()); // Should we do std::numeric_limits<ushort>::max() too ?
      out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
    }
    if ((in_depth == CV_32F) || (in_depth == CV_64F))
      in.convertTo(out, type);
}
} // namespace cv
