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

Polyfit::Polyfit() : deg(0) {}

void Polyfit::write(cv::FileStorage& fs) const {
    fs << "{" << "deg" << deg << "p" << p << "}";
}

void Polyfit::read(const cv::FileNode& node) {
    node["deg"] >> deg;
    node["p"] >> p;
}

// Global functions to support FileStorage for Polyfit
void write(cv::FileStorage& fs, const std::string&, const Polyfit& polyfit) {
    polyfit.write(fs);
}
void read(const cv::FileNode& node, Polyfit& polyfit, const Polyfit& defaultValue) {
    if(node.empty())
        polyfit = defaultValue;
    else
        polyfit.read(node);
}

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

// Default constructor for LogPolyfit
LogPolyfit::LogPolyfit() : deg(0) {}

void LogPolyfit::write(cv::FileStorage& fs) const {
    fs << "{" << "deg" << deg << "p" << p << "}";
}

void LogPolyfit::read(const cv::FileNode& node) {
    node["deg"] >> deg;
    node["p"] >> p;
}

// Global functions to support FileStorage for LogPolyfit
void write(cv::FileStorage& fs, const std::string&, const LogPolyfit& logpolyfit) {
    logpolyfit.write(fs);
}
void read(const cv::FileNode& node, LogPolyfit& logpolyfit, const LogPolyfit& defaultValue) {
    if(node.empty())
        logpolyfit = defaultValue;
    else
        logpolyfit.read(node);
}

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

void LinearIdentity::write(cv::FileStorage& fs) const
{
    fs << "{" << "}";
}

void LinearIdentity::read(const cv::FileNode&)
{
}

void LinearGamma::write(cv::FileStorage& fs) const
{
    fs << "{" << "gamma" << gamma << "}";
}

void LinearGamma::read(const cv::FileNode& node)
{
    node["gamma"] >> gamma;
}

template <typename T>
void LinearColor<T>::write(cv::FileStorage& fs) const
{
    fs << "{" << "deg" << deg << "pr" << pr << "pg" << pg << "pb" << pb << "}";
}

template <typename T>
void LinearColor<T>::read(const cv::FileNode& node)
{
    node["deg"] >> deg;
    node["pr"] >> pr;
    node["pg"] >> pg;
    node["pb"] >> pb;
}

template <typename T>
void LinearGray<T>::write(cv::FileStorage& fs) const
{
    fs << "{" << "deg" << deg << "p" << p << "}";
}

template <typename T>
void LinearGray<T>::read(const cv::FileNode& node)
{
    node["deg"] >> deg;
    node["p"] >> p;
}

void Linear::write(cv::FileStorage&) const
{
    CV_Error(Error::StsNotImplemented, "This is a base class, so this shouldn't be called");
}

void Linear::read(const cv::FileNode&)
{
    CV_Error(Error::StsNotImplemented, "This is a base class, so this shouldn't be called");
}

void write(cv::FileStorage& fs, const std::string&, const Linear& linear)
{
    linear.write(fs);
}

void read(const cv::FileNode& node, Linear& linear, const Linear& defaultValue)
{
    if (node.empty())
        linear = defaultValue;
    else
        linear.read(node);
}

void write(cv::FileStorage& fs, const std::string&, const LinearIdentity& linearidentity)
{
    linearidentity.write(fs);
}

void read(const cv::FileNode& node, LinearIdentity& linearidentity, const LinearIdentity& defaultValue)
{
    if (node.empty())
        linearidentity = defaultValue;
    else
        linearidentity.read(node);
}

void write(cv::FileStorage& fs, const std::string&, const LinearGamma& lineargamma)
{
    lineargamma.write(fs);
}

void read(const cv::FileNode& node, LinearGamma& lineargamma, const LinearGamma& defaultValue)
{
    if (node.empty())
        lineargamma = defaultValue;
    else
        lineargamma.read(node);
}

template <typename T>
void write(cv::FileStorage& fs, const std::string&, const LinearColor<T>& linearcolor)
{
    linearcolor.write(fs);
}

template <typename T>
void read(const cv::FileNode& node, LinearColor<T>& linearcolor, const LinearColor<T>& defaultValue)
{
    if (node.empty())
        linearcolor = defaultValue;
    else
        linearcolor.read(node);
}

template <typename T>
void write(cv::FileStorage& fs, const std::string&, const LinearGray<T>& lineargray)
{
    lineargray.write(fs);
}

template <typename T>
void read(const cv::FileNode& node, LinearGray<T>& lineargray, const LinearGray<T>& defaultValue)
{
    if (node.empty())
        lineargray = defaultValue;
    else
        lineargray.read(node);
}

Mat Linear::linearize(Mat inp)
{
    return inp;
};

Mat LinearGamma::linearize(Mat inp)
{
    return gammaCorrection(inp, gamma);
};

std::shared_ptr<Linear> getLinear(double gamma, int deg, Mat src, Color dst, Mat mask, RGBBase_ cs, LinearizationType linearizationType)
{
    std::shared_ptr<Linear> p = std::make_shared<Linear>();
    switch (linearizationType)
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
        CV_Error(Error::StsBadArg, "Wrong linearizationType!" );
        break;
    }
    return p;
};

}
}  // namespace cv::ccm