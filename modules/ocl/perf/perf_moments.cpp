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
///////////// Moments ////////////////////////
PERFTEST(Moments)
{
    Mat src;
    bool binaryImage = 0;

    int all_type[] = {CV_8UC1, CV_16SC1, CV_32FC1, CV_64FC1};
    std::string type_name[] = {"CV_8UC1", "CV_16SC1", "CV_32FC1", "CV_64FC1"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j];

            gen(src, size, size, all_type[j], 0, 256);

            cv::Moments CvMom = moments(src, binaryImage);

            CPU_ON;
            moments(src, binaryImage);
            CPU_OFF;

            cv::Moments oclMom;
            WARMUP_ON;
            oclMom = ocl::ocl_moments(src, binaryImage);
            WARMUP_OFF;

            Mat gpu_dst, cpu_dst;
            HuMoments(CvMom, cpu_dst);
            HuMoments(oclMom, gpu_dst);

            GPU_ON;
            ocl::ocl_moments(src, binaryImage);
            GPU_OFF;

            GPU_FULL_ON;
            ocl::ocl_moments(src, binaryImage);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(gpu_dst, cpu_dst, .5);

        }

    }
}
