// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "utils.hpp"

namespace cv {
namespace ccm {

inline double gammaCorrection_(const double& element, const double& gamma)
{
    return (element >= 0 ? pow(element, gamma) : -pow((-element), gamma));
}

Mat gammaCorrection(const Mat& src, const double& gamma, Mat dst)
{
    return elementWise(src, [gamma](const double& element) { return gammaCorrection_(element, gamma); }, dst);
}

Mat saturate(Mat& src, const double& low, const double& up)
{
    Mat dst = Mat::ones(src.size(), CV_8UC1);
    MatIterator_<Vec3d> it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>();
    MatIterator_<uchar> it_dst = dst.begin<uchar>();
    for (; it_src != end_src; ++it_src, ++it_dst)
    {
        for (int i = 0; i < 3; ++i)
        {
            if ((*it_src)[i] > up || (*it_src)[i] < low)
            {
                *it_dst = 0;
                break;
            }
        }
    }
    return dst;
}

Mat rgb2gray(const Mat& rgb)
{
    const Matx31d m_gray(0.2126, 0.7152, 0.0722);
    Mat dst;
    cv::transform(rgb, dst, Mat(m_gray));
    return dst;
}

}
}  // namespace cv::ccm
