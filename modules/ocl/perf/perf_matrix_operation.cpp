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

///////////// ConvertTo////////////////////////
PERFTEST(ConvertTo)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] << " to 32FC1";

            gen(src, size, size, all_type[j], 0, 256);
            //gen(dst, size, size, all_type[j], 0, 256);

            //d_dst.upload(dst);

            src.convertTo(dst, CV_32FC1);

            CPU_ON;
            src.convertTo(dst, CV_32FC1);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            d_src.convertTo(d_dst, CV_32FC1);
            WARMUP_OFF;

            GPU_ON;
            d_src.convertTo(d_dst, CV_32FC1);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_src.convertTo(d_dst, CV_32FC1);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 0.0);
        }

    }
}
///////////// copyTo////////////////////////
PERFTEST(copyTo)
{
    Mat src, dst, ocl_dst;
    ocl::oclMat d_src, d_dst;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);
            //gen(dst, size, size, all_type[j], 0, 256);

            //d_dst.upload(dst);

            src.copyTo(dst);

            CPU_ON;
            src.copyTo(dst);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            d_src.copyTo(d_dst);
            WARMUP_OFF;

            GPU_ON;
            d_src.copyTo(d_dst);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_src.copyTo(d_dst);
            d_dst.download(ocl_dst);
            GPU_FULL_OFF;

            TestSystem::instance().ExpectedMatNear(dst, ocl_dst, 0.0);
        }

    }
}
///////////// setTo////////////////////////
PERFTEST(setTo)
{
    Mat src, ocl_src;
    Scalar val(1, 2, 3, 4);
    ocl::oclMat d_src;

    int all_type[] = {CV_8UC1, CV_8UC4};
    std::string type_name[] = {"CV_8UC1", "CV_8UC4"};

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            SUBTEST << size << 'x' << size << "; " << type_name[j] ;

            gen(src, size, size, all_type[j], 0, 256);

            src.setTo(val);

            CPU_ON;
            src.setTo(val);
            CPU_OFF;

            d_src.upload(src);

            WARMUP_ON;
            d_src.setTo(val);
            WARMUP_OFF;

            d_src.download(ocl_src);
            TestSystem::instance().ExpectedMatNear(src, ocl_src, 1.0);

            GPU_ON;;
            d_src.setTo(val);
            GPU_OFF;

            GPU_FULL_ON;
            d_src.upload(src);
            d_src.setTo(val);
            GPU_FULL_OFF;
        }

    }
}