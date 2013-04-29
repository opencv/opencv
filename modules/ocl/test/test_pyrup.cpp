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
//    Zhang Chunpeng chunpeng@multicorewareinc.com
//    Yao Wang yao@multicorewareinc.com
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

#include "precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cvtest;
using namespace testing;
using namespace std;

PARAM_TEST_CASE(PyrUp, MatType, int)
{
    int type;
    int channels;
    //std::vector<cv::ocl::Info> oclinfo;

    virtual void SetUp()
    {
        //int devnums = cv::ocl::getDevice(oclinfo, OPENCV_DEFAULT_OPENCL_DEVICE);
        //CV_Assert(devnums > 0);
        type = GET_PARAM(0);
        channels = GET_PARAM(1);
    }
};

TEST_P(PyrUp, Accuracy)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        Size size(MWIDTH, MHEIGHT);
        Mat src = randomMat(size, CV_MAKETYPE(type, channels));
        Mat dst_gold;
        pyrUp(src, dst_gold);
        ocl::oclMat dst;
        ocl::oclMat srcMat(src);
        ocl::pyrUp(srcMat, dst);
        Mat cpu_dst;
        dst.download(cpu_dst);
        char s[100] = {0};

        EXPECT_MAT_NEAR(dst_gold, cpu_dst, (src.depth() == CV_32F ? 1e-4f : 1.0), s);
    }

}


INSTANTIATE_TEST_CASE_P(GPU_ImgProc, PyrUp, testing::Combine(
                            Values(CV_8U, CV_32F), Values(1, 3, 4)));


#endif // HAVE_OPENCL