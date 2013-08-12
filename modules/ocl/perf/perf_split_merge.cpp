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

///////////// Merge////////////////////////
PERFTEST(Merge)
{
    Mat dst, ocl_dst;
    ocl::oclMat d_dst;

    int channels = 4;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;
            Size size1 = Size(size, size);
            std::vector<Mat> src(channels);

            for (int i = 0; i < channels; ++i)
            {
                src[i] = Mat(size1, all_type[j], cv::Scalar::all(i));
            }

            merge(src, dst);

            CPU_ON;
            merge(src, dst);
            CPU_OFF;

            std::vector<ocl::oclMat> d_src(channels);

            for (int i = 0; i < channels; ++i)
            {
                d_src[i] = ocl::oclMat(size1, all_type[j], cv::Scalar::all(i));
            }

            WARMUP_ON;
            ocl::merge(d_src, d_dst);
            WARMUP_OFF;

            GPU_ON;
            ocl::merge(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            for (int i = 0; i < channels; ++i)
            {
                d_src[i] = ocl::oclMat(size1, all_type[j], cv::Scalar::all(i));
            }
            ocl::merge(d_src, d_dst);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 0.0);
        }

    }
}

///////////// Split////////////////////////
PERFTEST(Split)
{
    //int channels = 4;
    int all_type[] = {CV_8UC1, CV_32FC1};
    std::string type_name[] = {"CV_8UC1", "CV_32FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];
            Size size1 = Size(size, size);

            Mat src(size1, CV_MAKE_TYPE(all_type[j], 4), cv::Scalar(1, 2, 3, 4));

            std::vector<cv::Mat> dst, ocl_dst(4);

            split(src, dst);

            CPU_ON;
            split(src, dst);
            CPU_OFF;

            ocl::oclMat d_src(size1, CV_MAKE_TYPE(all_type[j], 4), cv::Scalar(1, 2, 3, 4));
            std::vector<cv::ocl::oclMat> d_dst;

            WARMUP_ON;
            ocl::split(d_src, d_dst);
            WARMUP_OFF;         

            GPU_ON;
            ocl::split(d_src, d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            ocl::split(d_src, d_dst);
            for(size_t i = 0; i < dst.size(); i++)
                d_dst[i].download(ocl_dst[i]);
            GPU_FULL_OFF;

            vector<double> eps(4, 0.);
            TestSystem::instance().ExpectMatsNear(dst, ocl_dst, eps);
        }

    }
}
