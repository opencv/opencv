/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;
using namespace testing;

int randomInt(int minVal, int maxVal)
{
    RNG& rng = TS::ptr()->get_rng();
    return rng.uniform(minVal, maxVal);
}

double randomDouble(double minVal, double maxVal)
{
    RNG& rng = TS::ptr()->get_rng();
    return rng.uniform(minVal, maxVal);
}

Size randomSize(int minVal, int maxVal)
{
    return cv::Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
}

Scalar randomScalar(double minVal, double maxVal)
{
    return Scalar(randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal));
}

Mat randomMat(Size size, int type, double minVal, double maxVal)
{
    return randomMat(TS::ptr()->get_rng(), size, type, minVal, maxVal, false);
}

cv::gpu::GpuMat createMat(cv::Size size, int type, bool useRoi)
{
    Size size0 = size;

    if (useRoi)
    {
        size0.width += randomInt(5, 15);
        size0.height += randomInt(5, 15);
    }

    GpuMat d_m(size0, type);

    if (size0 != size)
        d_m = d_m(Rect((size0.width - size.width) / 2, (size0.height - size.height) / 2, size.width, size.height));

    return d_m;
}

GpuMat loadMat(const Mat& m, bool useRoi)
{
    GpuMat d_m = createMat(m.size(), m.type(), useRoi);
    d_m.upload(m);
    return d_m;
}

bool supportFeature(const DeviceInfo& info, FeatureSet feature)
{
    return TargetArchs::builtWith(feature) && info.supports(feature);
}

const vector<DeviceInfo>& devices()
{
    static vector<DeviceInfo> devs;
    static bool first = true;

    if (first)
    {
        int deviceCount = getCudaEnabledDeviceCount();

        devs.reserve(deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            DeviceInfo info(i);
            if (info.isCompatible())
                devs.push_back(info);
        }

        first = false;
    }

    return devs;
}

vector<DeviceInfo> devices(FeatureSet feature)
{
    const vector<DeviceInfo>& d = devices();

    vector<DeviceInfo> devs_filtered;

    if (TargetArchs::builtWith(feature))
    {
        devs_filtered.reserve(d.size());

        for (size_t i = 0, size = d.size(); i < size; ++i)
        {
            const DeviceInfo& info = d[i];

            if (info.supports(feature))
                devs_filtered.push_back(info);
        }
    }

    return devs_filtered;
}

vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end)
{
    vector<MatType> v;

    v.reserve((depth_end - depth_start + 1) * (cn_end - cn_start + 1));

    for (int depth = depth_start; depth <= depth_end; ++depth)
    {
        for (int cn = cn_start; cn <= cn_end; ++cn)
        {
            v.push_back(CV_MAKETYPE(depth, cn));
        }
    }

    return v;
}

const vector<MatType>& all_types()
{
    static vector<MatType> v = types(CV_8U, CV_64F, 1, 4);

    return v;
}

Mat readImage(const string& fileName, int flags)
{
    return imread(string(cvtest::TS::ptr()->get_data_path()) + fileName, flags);
}

Mat readImageType(const string& fname, int type)
{
    Mat src = readImage(fname, CV_MAT_CN(type) == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
    if (CV_MAT_CN(type) == 4)
    {
        Mat temp;
        cvtColor(src, temp, cv::COLOR_BGR2BGRA);
        swap(src, temp);
    }
    src.convertTo(src, CV_MAT_DEPTH(type));
    return src;
}

namespace
{
    Mat getMat(InputArray arr)
    {
        if (arr.kind() == _InputArray::GPU_MAT)
        {
            Mat m;
            arr.getGpuMat().download(m);
            return m;
        }

        return arr.getMat();
    }
}

void showDiff(InputArray gold_, InputArray actual_, double eps)
{
    Mat gold = getMat(gold_);
    Mat actual = getMat(actual_);

    Mat diff;
    absdiff(gold, actual, diff);
    threshold(diff, diff, eps, 255.0, cv::THRESH_BINARY);

    namedWindow("gold", WINDOW_NORMAL);
    namedWindow("actual", WINDOW_NORMAL);
    namedWindow("diff", WINDOW_NORMAL);

    imshow("gold", gold);
    imshow("actual", actual);
    imshow("diff", diff);

    waitKey();
}

double checkNorm(InputArray m1, const InputArray m2)
{
    return norm(getMat(m1), getMat(m2), NORM_INF);
}

double checkSimilarity(InputArray m1, InputArray m2)
{
    Mat diff;
    matchTemplate(getMat(m1), getMat(m2), diff, CV_TM_CCORR_NORMED);
    return std::abs(diff.at<float>(0, 0) - 1.f);
}

void cv::gpu::PrintTo(const DeviceInfo& info, ostream* os)
{
    (*os) << info.name();
}

void PrintTo(const UseRoi& useRoi, std::ostream* os)
{
    if (useRoi)
        (*os) << "sub matrix";
    else
        (*os) << "whole matrix";
}

void PrintTo(const Inverse& inverse, std::ostream* os)
{
    if (inverse)
        (*os) << "inverse";
    else
        (*os) << "direct";
}
