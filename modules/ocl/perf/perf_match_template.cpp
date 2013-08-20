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

/////////// matchTemplate ////////////////////////
//void InitMatchTemplate()
//{
//	Mat src; gen(src, 500, 500, CV_32F, 0, 1);
//	Mat templ; gen(templ, 500, 500, CV_32F, 0, 1);
//	ocl::oclMat d_src(src), d_templ(templ), d_dst;
//	ocl::matchTemplate(d_src, d_templ, d_dst, CV_TM_CCORR);
//}
PERFTEST(matchTemplate)
{
    //InitMatchTemplate();
    Mat src, templ, dst, ocl_dst;
    int templ_size = 5;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        int all_type[] = {CV_32FC1, CV_32FC4};
        std::string type_name[] = {"CV_32FC1", "CV_32FC4"};

        for (size_t j = 0; j < sizeof(all_type) / sizeof(int); j++)
        {
            for(templ_size = 5; templ_size <= 5; templ_size *= 5)
            {
                gen(src, size, size, all_type[j], 0, 1);

                SUBTEST << src.cols << 'x' << src.rows << "; " << type_name[j] << "; templ " << templ_size << 'x' << templ_size << "; CCORR";

                gen(templ, templ_size, templ_size, all_type[j], 0, 1);

                matchTemplate(src, templ, dst, TM_CCORR);

                CPU_ON;
                matchTemplate(src, templ, dst, TM_CCORR);
                CPU_OFF;

                ocl::oclMat d_src(src), d_templ(templ), d_dst;

                WARMUP_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, TM_CCORR);
                WARMUP_OFF;

                GPU_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, TM_CCORR);
                GPU_OFF;

                GPU_FULL_ON;
                d_src.upload(src);
                d_templ.upload(templ);
                ocl::matchTemplate(d_src, d_templ, d_dst, TM_CCORR);
                d_dst.download(ocl_dst);
                GPU_FULL_OFF;

                TestSystem::instance().ExpectedMatNear(dst, ocl_dst, templ.rows * templ.cols * 1e-1);
            }
        }

        int all_type_8U[] = {CV_8UC1};
        std::string type_name_8U[] = {"CV_8UC1"};

        for (size_t j = 0; j < sizeof(all_type_8U) / sizeof(int); j++)
        {
            for(templ_size = 5; templ_size <= 5; templ_size *= 5)
            {
                SUBTEST << src.cols << 'x' << src.rows << "; " << type_name_8U[j] << "; templ " << templ_size << 'x' << templ_size << "; CCORR_NORMED";

                gen(src, size, size, all_type_8U[j], 0, 255);

                gen(templ, templ_size, templ_size, all_type_8U[j], 0, 255);

                matchTemplate(src, templ, dst, TM_CCORR_NORMED);

                CPU_ON;
                matchTemplate(src, templ, dst, TM_CCORR_NORMED);
                CPU_OFF;

                ocl::oclMat d_src(src);
                ocl::oclMat d_templ(templ), d_dst;

                WARMUP_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, TM_CCORR_NORMED);
                WARMUP_OFF;

                GPU_ON;
                ocl::matchTemplate(d_src, d_templ, d_dst, TM_CCORR_NORMED);
                GPU_OFF;

                GPU_FULL_ON;
                d_src.upload(src);
                d_templ.upload(templ);
                ocl::matchTemplate(d_src, d_templ, d_dst, TM_CCORR_NORMED);
                d_dst.download(ocl_dst);
                GPU_FULL_OFF;

                TestSystem::instance().ExpectedMatNear(dst, ocl_dst, templ.rows * templ.cols * 1e-1);
            }
        }
    }
}
