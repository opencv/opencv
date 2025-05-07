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

void gammaCorrection(InputArray _src, OutputArray _dst, double gamma)
{
    Mat src = _src.getMat();
    CV_Assert(gamma > 0);

    double  maxVal;
    int     depth = src.depth();
    switch (depth)
    {
        case CV_8U:  maxVal = 255.0;    break;
        case CV_16U: maxVal = 65535.0;  break;
        case CV_16S: maxVal = 32767.0;  break;
        case CV_32F: maxVal = 1.0;      break;
        case CV_64F: maxVal = 1.0;      break;
        default:
            CV_Error(Error::StsUnsupportedFormat,
                "gammaCorrection: unsupported image depth");
    }

    // Special‐case for uint8 with a LUT
    if (depth == CV_8U)
    {
        Mat lut(1, 256, CV_8U);
        uchar* p = lut.ptr<uchar>();
        for (int i = 0; i < 256; ++i)
        {
            double fn = std::pow(i / 255.0, gamma) * 255.0;
            p[i] = cv::saturate_cast<uchar>(fn + 0.5);
        }
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();
        cv::LUT(src, lut, dst);
        return;
    }

    Mat f;
    src.convertTo(f, CV_64F, 1.0 / maxVal);
    cv::pow(f, gamma, f);

    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    f.convertTo(dst, src.type(), maxVal);
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

Mat saturate(Mat& src, double low, double up)
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
    Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    return gray;
}
}
}  // namespace cv::ccm
