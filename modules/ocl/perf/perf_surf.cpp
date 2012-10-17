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
//    Peng Xiao, pengxiao@multicorewareinc.com
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
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

#define FILTER_IMAGE "../../../samples/gpu/road.png"

TEST(SURF, Performance)
{
    cv::Mat img = readImage(FILTER_IMAGE, cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    ocl::SURF_OCL d_surf;
    ocl::oclMat d_keypoints;
    ocl::oclMat d_descriptors;

    double totalgputick = 0;
    double totalgputick_kernel = 0;

    double t1 = 0;
    double t2 = 0;
    for(int j = 0; j < LOOP_TIMES + 1; j ++)
    {
        t1 = (double)cvGetTickCount();//gpu start1

        ocl::oclMat d_src(img);//upload

        t2 = (double)cvGetTickCount(); //kernel
        d_surf(d_src, ocl::oclMat(), d_keypoints, d_descriptors);
        t2 = (double)cvGetTickCount() - t2;//kernel

        cv::Mat cpu_kp, cpu_dp;
        d_keypoints.download (cpu_kp);//download
        d_descriptors.download (cpu_dp);//download

        t1 = (double)cvGetTickCount() - t1;//gpu end1

        if(j == 0)
            continue;

        totalgputick = t1 + totalgputick;

        totalgputick_kernel = t2 + totalgputick_kernel;

    }

    cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;
    cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << endl;


}
#endif  //Have opencl