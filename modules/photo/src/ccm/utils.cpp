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
    return elementWise(src, [gamma](double element) -> double { return gammaCorrection_(element, gamma); }, dst);
}

Mat maskCopyTo(const Mat& src, const Mat& mask)
{
    Mat fullMasked;
    src.copyTo(fullMasked, mask);

    std::vector<Point> nonZeroLocations;
    findNonZero(mask, nonZeroLocations);

    Mat dst(static_cast<int>(nonZeroLocations.size()), 1, src.type());

    int channels = src.channels();
    if (channels == 1)
    {
        for (size_t i = 0; i < nonZeroLocations.size(); i++)
        {
            dst.at<double>(static_cast<int>(i), 0) = fullMasked.at<double>(nonZeroLocations[i]);
        }
    }
    else if (channels == 3)
    {
        for (size_t i = 0; i < nonZeroLocations.size(); i++)
        {
            dst.at<Vec3d>(static_cast<int>(i), 0) = fullMasked.at<Vec3d>(nonZeroLocations[i]);
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Unsupported number of channels");
    }

    return dst;
}

Mat multiple(const Mat& xyz, const Mat& ccm)
{
    Mat tmp = xyz.reshape(1, xyz.rows * xyz.cols);
    Mat res = tmp * ccm;
    res = res.reshape(res.cols, xyz.rows);
    return res;
}

Mat saturate(Mat& src, const double& low, const double& up)
{
    CV_Assert(src.type() == CV_64FC3);
    Scalar lower_bound(low, low, low);
    Scalar upper_bound(up, up, up);

    Mat mask;
    inRange(src, lower_bound, upper_bound, mask);
    mask /= 255;

    return mask;
}


Mat rgb2gray(const Mat& rgb)
{
    const Matx31d m_gray(0.2126, 0.7152, 0.0722);
    return multiple(rgb, Mat(m_gray));
}

}
}  // namespace cv::ccm
