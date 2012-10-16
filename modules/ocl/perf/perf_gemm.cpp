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


#include "precomp.hpp"
using namespace std;
#ifdef HAVE_CLAMDBLAS
////////////////////////////////////////////////////////////////////////////
// GEMM
PARAM_TEST_CASE(Gemm, int, cv::Size, int)
{
    int      type;
    cv::Size mat_size;
    int		 flags;
    vector<cv::ocl::Info> info;
    virtual void SetUp()
    {
        type     = GET_PARAM(0);
        mat_size = GET_PARAM(1);
        flags    = GET_PARAM(2);

        cv::ocl::getDevice(info);
    }
};

TEST_P(Gemm, Performance)
{
    cv::Mat a = randomMat(mat_size, type, 0.0, 10.0);
    cv::Mat b = randomMat(mat_size, type, 0.0, 10.0);
    cv::Mat c = randomMat(mat_size, type, 0.0, 10.0);
    cv::ocl::oclMat ocl_dst;

    double totalgputick = 0;
    double totalgputick_kernel = 0;
    double t1 = 0;
    double t2 = 0;

    for(int j = 0; j < LOOP_TIMES + 1; j ++)
    {

        t1 = (double)cvGetTickCount();//gpu start1

        cv::ocl::oclMat ga = cv::ocl::oclMat(a);//upload
        cv::ocl::oclMat gb = cv::ocl::oclMat(b);//upload
        cv::ocl::oclMat gc = cv::ocl::oclMat(c);//upload

        t2 = (double)cvGetTickCount(); //kernel
        cv::ocl::gemm(ga, gb, 1.0, gc, 1.0, ocl_dst, flags);
        t2 = (double)cvGetTickCount() - t2;//kernel

        cv::Mat cpu_dst;
        ocl_dst.download (cpu_dst);//download

        t1 = (double)cvGetTickCount() - t1;//gpu end

        if(j == 0)
            continue;

        totalgputick = t1 + totalgputick;
        totalgputick_kernel = t2 + totalgputick_kernel;

    }
    cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
}


INSTANTIATE_TEST_CASE_P(ocl_gemm, Gemm, testing::Combine(
                            testing::Values(CV_32FC1, CV_32FC2/* , CV_64FC1, CV_64FC2*/),
                            testing::Values(cv::Size(512, 512), cv::Size(1024, 1024)),
                            testing::Values(0, cv::GEMM_1_T, cv::GEMM_2_T, cv::GEMM_1_T + cv::GEMM_2_T)));
#endif