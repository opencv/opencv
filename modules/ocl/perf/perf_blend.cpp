/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "perf_precomp.hpp"
///////////// blend ////////////////////////
template <typename T>
void blendLinearGold(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &weights1, const cv::Mat &weights2, cv::Mat &result_gold)
{
    result_gold.create(img1.size(), img1.type());

    int cn = img1.channels();

    for (int y = 0; y < img1.rows; ++y)
    {
        const float *weights1_row = weights1.ptr<float>(y);
        const float *weights2_row = weights2.ptr<float>(y);
        const T *img1_row = img1.ptr<T>(y);
        const T *img2_row = img2.ptr<T>(y);
        T *result_gold_row = result_gold.ptr<T>(y);

        for (int x = 0; x < img1.cols * cn; ++x)
        {
            float w1 = weights1_row[x / cn];
            float w2 = weights2_row[x / cn];
            result_gold_row[x] = static_cast<T>((img1_row[x] * w1 + img2_row[x] * w2) / (w1 + w2 + 1e-5f));
        }
    }
}
PERFTEST(blend)
{
    Mat src1, src2, weights1, weights2, dst, ocl_dst;
    ocl::oclMat d_src1, d_src2, d_weights1, d_weights2, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " and CV_32FC1";

            gen(src1, size, size, all_type[j], 0, 256);
            gen(src2, size, size, all_type[j], 0, 256);
            gen(weights1, size, size, CV_32FC1, 0, 1);
            gen(weights2, size, size, CV_32FC1, 0, 1);

            blendLinearGold<uchar>(src1, src2, weights1, weights2, dst);

            CPU_ON;
            blendLinearGold<uchar>(src1, src2, weights1, weights2, dst);
            CPU_OFF;

            d_src1.upload(src1);
            d_src2.upload(src2);
            d_weights1.upload(weights1);
            d_weights2.upload(weights2);

            WARMUP_ON;
            ocl::blendLinear(d_src1, d_src2, d_weights1, d_weights2, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::blendLinear(d_src1, d_src2, d_weights1, d_weights2, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src1.upload(src1);
            d_src2.upload(src2);
            d_weights1.upload(weights1);
            d_weights2.upload(weights2);
            ocl::blendLinear(d_src1, d_src2, d_weights1, d_weights2, d_dst);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 1.f);
        }
    }
}