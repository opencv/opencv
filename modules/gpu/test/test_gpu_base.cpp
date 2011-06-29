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

bool supportFeature(const cv::gpu::DeviceInfo& info, cv::gpu::FeatureSet feature)
{
    return cv::gpu::TargetArchs::builtWith(feature) && info.supports(feature);
}

const std::vector<cv::gpu::DeviceInfo>& devices()
{
    static std::vector<cv::gpu::DeviceInfo> devs;
    static bool first = true;

    if (first)
    {
        int deviceCount = cv::gpu::getCudaEnabledDeviceCount();

        devs.reserve(deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            cv::gpu::DeviceInfo info(i);
            if (info.isCompatible())
                devs.push_back(info);
        }

        first = false;
    }

    return devs;
}

std::vector<cv::gpu::DeviceInfo> devices(cv::gpu::FeatureSet feature)
{
    const std::vector<cv::gpu::DeviceInfo>& d = devices();
    
    std::vector<cv::gpu::DeviceInfo> devs_filtered;

    if (cv::gpu::TargetArchs::builtWith(feature))
    {
        devs_filtered.reserve(d.size());

        for (size_t i = 0, size = d.size(); i < size; ++i)
        {
            const cv::gpu::DeviceInfo& info = d[i];

            if (info.supports(feature))
                devs_filtered.push_back(info);
        }
    }

    return devs_filtered;
}

std::vector<int> types(int depth_start, int depth_end, int cn_start, int cn_end)
{
    std::vector<int> v;

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

const std::vector<int>& all_types()
{
    static std::vector<int> v = types(CV_8U, CV_64F, 1, 4);
    return v;
}

cv::Mat readImage(const std::string& fileName, int flags)
{
    return cv::imread(std::string(cvtest::TS::ptr()->get_data_path()) + fileName, flags);
}

double checkNorm(const cv::Mat& m1, const cv::Mat& m2)
{
    return cv::norm(m1, m2, cv::NORM_INF);
}

double checkSimilarity(const cv::Mat& m1, const cv::Mat& m2)
{
    cv::Mat diff;
    cv::matchTemplate(m1, m2, diff, CV_TM_CCORR_NORMED);
    return std::abs(diff.at<float>(0, 0) - 1.f);
}

namespace cv
{
    std::ostream& operator << (std::ostream& os, const Size& sz)
    {
        return os << sz.width << "x" << sz.height;
    }

    std::ostream& operator << (std::ostream& os, const Scalar& s)
    {
        return os << "[" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << "]";
    }

    namespace gpu
    {
        std::ostream& operator << (std::ostream& os, const DeviceInfo& info)
        {
            return os << info.name();
        }
    }
}
