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
//	   Chunpeng Zhang chunpeng@multicorewareinc.com
//
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

///////////////////////////////////////////////////////////////////////////////
/// ColumnSum

#ifdef HAVE_OPENCL

////////////////////////////////////////////////////////////////////////
// ColumnSum

PARAM_TEST_CASE(ColumnSum, cv::Size, bool)
{
    cv::Size size;
    cv::Mat src;
    bool useRoi;
    //std::vector<cv::ocl::Info> oclinfo;
    
    virtual void SetUp()
    {
        size = GET_PARAM(0);
        useRoi = GET_PARAM(1);
        //int devnums = getDevice(oclinfo, OPENCV_DEFAULT_OPENCL_DEVICE);
        //CV_Assert(devnums > 0);
    }
};

TEST_P(ColumnSum, Accuracy)
{
    cv::Mat src = randomMat(size, CV_32FC1);
    cv::ocl::oclMat d_dst;
    cv::ocl::oclMat d_src(src);
    
    cv::ocl::columnSum(d_src, d_dst);
    
    cv::Mat dst(d_dst);
    
    for(int j = 0; j < src.cols; ++j)
    {
        float gold = src.at<float>(0, j);
        float res = dst.at<float>(0, j);
        ASSERT_NEAR(res, gold, 1e-5);
    }
    
    for(int i = 1; i < src.rows; ++i)
    {
        for(int j = 0; j < src.cols; ++j)
        {
            float gold = src.at<float>(i, j) += src.at<float>(i - 1, j);
            float res = dst.at<float>(i, j);
            ASSERT_NEAR(res, gold, 1e-5);
        }
    }
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, ColumnSum, testing::Combine(
                            DIFFERENT_SIZES, testing::Values(Inverse(false), Inverse(true))));


#endif
