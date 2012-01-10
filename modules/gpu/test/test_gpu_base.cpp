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

#include "test_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;

GpuMat loadMat(const Mat& m, bool useRoi)
{
    Size size = m.size();
    Size size0 = size;

    if (useRoi)
    {
        RNG& rng = TS::ptr()->get_rng();

        size0.width += rng.uniform(5, 15);
        size0.height += rng.uniform(5, 15);
    }
        
    GpuMat d_m(size0, m.type());
    
    if (size0 != size)
        d_m = d_m(Rect((size0.width - size.width) / 2, (size0.height - size.height) / 2, size.width, size.height));

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

double checkNorm(const Mat& m1, const Mat& m2)
{
    return norm(m1, m2, NORM_INF);
}

double checkSimilarity(const Mat& m1, const Mat& m2)
{
    Mat diff;
    matchTemplate(m1, m2, diff, CV_TM_CCORR_NORMED);
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
