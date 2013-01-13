///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
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
#include <iomanip>

#ifdef HAVE_OPENCL

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
using namespace std;

PARAM_TEST_CASE(PyrDown, MatType, int)
{
    int type;
    int channels;
    
    virtual void SetUp()
    {
        type = GET_PARAM(0);
        channels = GET_PARAM(1);
        
        //int devnums = getDevice(oclinfo);
        //CV_Assert(devnums > 0);
        ////if you want to use undefault device, set it here
        ////setDevice(oclinfo[0]);
    }
    
    void Cleanup()
    {
    }
    
};


TEST_P(PyrDown, Mat)
{
    for(int j = 0; j < LOOP_TIMES; j++)
    {
        cv::Size size(MWIDTH, MHEIGHT);
        cv::RNG &rng = TS::ptr()->get_rng();
        cv::Mat src = randomMat(rng, size, CV_MAKETYPE(type, channels), 0, 100, false);
        
        cv::ocl::oclMat gsrc(src), gdst;
        cv::Mat dst_cpu;
        cv::pyrDown(src, dst_cpu);
        cv::ocl::pyrDown(gsrc, gdst);
        
        cv::Mat dst;
        gdst.download(dst);
        char s[1024] = {0};
        
        EXPECT_MAT_NEAR(dst, dst_cpu, dst.depth() == CV_32F ? 1e-4f : 1.0f, s);
        
        Cleanup();
    }
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, PyrDown, Combine(
                            Values(CV_8U, CV_32F), Values(1, 3, 4)));


#endif // HAVE_OPENCL
