// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Longbu Wang <wanglongbu@huawei.com.com>
//         Jinheng Zhang <zhangjinheng1@huawei.com>
//         Chenqi Shan <shanchenqi@huawei.com>

#include "linearize.hpp"

namespace cv {
namespace ccm {

Polyfit::Polyfit(Mat x, Mat y, int deg_)
    : deg(deg_)
{
    int n = x.cols * x.rows * x.channels();
    x = x.reshape(1, n);
    y = y.reshape(1, n);
    Mat_<double> A = Mat_<double>::ones(n, deg + 1);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 1; j < A.cols; ++j)
        {
            A.at<double>(i, j) = x.at<double>(i) * A.at<double>(i, j - 1);
        }
    }
    Mat y_(y);
    cv::solve(A, y_, p, DECOMP_SVD);
}

Mat Polyfit::operator()(const Mat& inp)
{
    return elementWise(inp, [this](double x) -> double { return fromEW(x); });
};

double Polyfit::fromEW(double x)
{
    double res = 0;
    for (int d = 0; d <= deg; ++d)
    {
        res += pow(x, d) * p.at<double>(d, 0);
    }
    return res;
};

LogPolyfit::LogPolyfit(Mat x, Mat y, int deg_)
    : deg(deg_)
{
    Mat mask_ = (x > 0) & (y > 0);
    Mat src_, dst_, s_, d_;
    src_ = maskCopyTo(x, mask_);
    dst_ = maskCopyTo(y, mask_);
    log(src_, s_);
    log(dst_, d_);
    p = Polyfit(s_, d_, deg);
}

Mat LogPolyfit::operator()(const Mat& inp)
{
    Mat mask_ = inp >= 0;
    Mat y, y_, res;
    log(inp, y);
    y = p(y);
    exp(y, y_);
    y_.copyTo(res, mask_);
    return res;
};

Mat Linear::linearize(Mat inp)
{
    return inp;
};

Mat LinearGamma::linearize(Mat inp)
{
    return gammaCorrection(inp, gamma);
};

std::shared_ptr<Linear> getLinear(double gamma, int deg, Mat src, Color dst, Mat mask, RGBBase_ cs, LINEAR_TYPE linear_type)
{
    std::shared_ptr<Linear> p = std::make_shared<Linear>();
    switch (linear_type)
    {
    case cv::ccm::LINEARIZATION_IDENTITY:
        p.reset(new LinearIdentity());
        break;
    case cv::ccm::LINEARIZATION_GAMMA:
        p.reset(new LinearGamma(gamma));
        break;
    case cv::ccm::LINEARIZATION_COLORPOLYFIT:
        p.reset(new LinearColor<Polyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::LINEARIZATION_COLORLOGPOLYFIT:
        p.reset(new LinearColor<LogPolyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::LINEARIZATION_GRAYPOLYFIT:
        p.reset(new LinearGray<Polyfit>(deg, src, dst, mask, cs));
        break;
    case cv::ccm::LINEARIZATION_GRAYLOGPOLYFIT:
        p.reset(new LinearGray<LogPolyfit>(deg, src, dst, mask, cs));
        break;
    default:
        CV_Error(Error::StsBadArg, "Wrong linear_type!" );
        break;
    }
    return p;
};

}
}  // namespace cv::ccm