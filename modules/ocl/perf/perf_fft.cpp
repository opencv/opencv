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
//    Fangfangbai, fangfang@multicorewareinc.com
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

#include "precomp.hpp"
using namespace std;
#ifdef HAVE_CLAMDFFT
////////////////////////////////////////////////////////////////////////////
// Dft
PARAM_TEST_CASE(Dft, cv::Size, bool)
{
    cv::Size dft_size;
    bool	 dft_rows;
    vector<cv::ocl::Info> info;
    virtual void SetUp()
    {
        dft_size = GET_PARAM(0);
        dft_rows = GET_PARAM(1);
        cv::ocl::getDevice(info);
    }
};

TEST_P(Dft, C2C)
{
    cv::Mat a = randomMat(dft_size, CV_32FC2, 0.0, 10.0);
    int flags = 0;
    flags |= dft_rows ? cv::DFT_ROWS : 0;

    cv::ocl::oclMat d_b;

    double totalgputick = 0;
    double totalgputick_kernel = 0;
    double t1 = 0;
    double t2 = 0;

    for(int j = 0; j < LOOP_TIMES + 1; j ++)
    {

        t1 = (double)cvGetTickCount();//gpu start1

        cv::ocl::oclMat ga = cv::ocl::oclMat(a); //upload

        t2 = (double)cvGetTickCount(); //kernel
        cv::ocl::dft(ga, d_b, a.size(), flags);
        t2 = (double)cvGetTickCount() - t2;//kernel

        cv::Mat cpu_dst;
        d_b.download (cpu_dst);//download

        t1 = (double)cvGetTickCount() - t1;//gpu end1

        if(j == 0)
            continue;

        totalgputick = t1 + totalgputick;
        totalgputick_kernel = t2 + totalgputick_kernel;

    }

    cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
}



TEST_P(Dft, R2CthenC2R)
{
    cv::Mat a = randomMat(dft_size, CV_32FC1, 0.0, 10.0);

    int flags = 0;
    //flags |= dft_rows ? cv::DFT_ROWS : 0; // not supported yet

    cv::ocl::oclMat d_b, d_c;

    cv::ocl::dft(cv::ocl::oclMat(a), d_b, a.size(), flags);
    cv::ocl::dft(d_b, d_c, a.size(), flags + cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT);

    EXPECT_MAT_NEAR(a, d_c, a.size().area() * 1e-4, "");
}

//INSTANTIATE_TEST_CASE_P(ocl_DFT, Dft, testing::Combine(
//						testing::Values(cv::Size(1280, 1024), cv::Size(1920, 1080),cv::Size(1800, 1500)),
//						testing::Values(false, true)));

#endif // HAVE_CLAMDFFT
