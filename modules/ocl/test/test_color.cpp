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

#include "test_precomp.hpp"

using namespace cv;

#ifdef HAVE_OPENCL

//#define MAT_DEBUG
#ifdef MAT_DEBUG
#define MAT_DIFF(mat, mat2)\
{\
    for(int i = 0; i < mat.rows; i ++)\
    {\
        for(int j = 0; j < mat.cols; j ++)\
        {\
            cv::Vec4b s = mat.at<cv::Vec4b>(i, j);\
            cv::Vec4b s2 = mat2.at<cv::Vec4b>(i, j);\
            if(s != s2) printf("*");\
            else printf(".");\
        }\
        puts("\n");\
    }\
}
#else
#define MAT_DIFF(mat, mat2)
#endif


namespace
{

///////////////////////////////////////////////////////////////////////////////////////////////////////
// cvtColor
PARAM_TEST_CASE(CvtColor, cv::Size, MatDepth)
{
    cv::Size size;
    int depth;
    bool useRoi;

    cv::Mat img;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        depth = GET_PARAM(1);

        img = randomMat(size, CV_MAKE_TYPE(depth, 3), 0.0, depth == CV_32F ? 1.0 : 255.0);
    }
};

#define CVTCODE(name) cv::COLOR_ ## name
#define TEST_P_CVTCOLOR(name) TEST_P(CvtColor, name)\
{\
    cv::Mat src = img;\
    cv::ocl::oclMat ocl_img, dst;\
    ocl_img.upload(img);\
    cv::ocl::cvtColor(ocl_img, dst, CVTCODE(name));\
    cv::Mat dst_gold;\
    cv::cvtColor(src, dst_gold, CVTCODE(name));\
    cv::Mat dst_mat;\
    dst.download(dst_mat);\
    EXPECT_MAT_NEAR(dst_gold, dst_mat, 1e-5);\
}

//add new ones here using macro
TEST_P_CVTCOLOR(RGB2GRAY)
TEST_P_CVTCOLOR(BGR2GRAY)
TEST_P_CVTCOLOR(RGBA2GRAY)
TEST_P_CVTCOLOR(BGRA2GRAY)

TEST_P_CVTCOLOR(RGB2YUV)
TEST_P_CVTCOLOR(BGR2YUV)
TEST_P_CVTCOLOR(YUV2RGB)
TEST_P_CVTCOLOR(YUV2BGR)
TEST_P_CVTCOLOR(RGB2YCrCb)
TEST_P_CVTCOLOR(BGR2YCrCb)

PARAM_TEST_CASE(CvtColor_Gray2RGB, cv::Size, MatDepth, int)
{
    cv::Size size;
    int code;
    int depth;
    cv::Mat img;

    virtual void SetUp()
    {
        size  = GET_PARAM(0);
        depth = GET_PARAM(1);
        code  = GET_PARAM(2);
        img   = randomMat(size, CV_MAKETYPE(depth, 1), 0.0, depth == CV_32F ? 1.0 : 255.0);
    }
};
TEST_P(CvtColor_Gray2RGB, Accuracy)
{
    cv::Mat src = img;
    cv::ocl::oclMat ocl_img, dst;
    ocl_img.upload(src);
    cv::ocl::cvtColor(ocl_img, dst, code);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, code);
    cv::Mat dst_mat;
    dst.download(dst_mat);
    EXPECT_MAT_NEAR(dst_gold, dst_mat, 1e-5);
}


PARAM_TEST_CASE(CvtColor_YUV420, cv::Size, int)
{
    cv::Size size;
    int code;

    cv::Mat img;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        code = GET_PARAM(1);
        img  = randomMat(size, CV_8UC1, 0.0, 255.0);
    }
};

TEST_P(CvtColor_YUV420, Accuracy)
{
    cv::Mat src = img;
    cv::ocl::oclMat ocl_img, dst;
    ocl_img.upload(src);
    cv::ocl::cvtColor(ocl_img, dst, code);
    cv::Mat dst_gold;
    cv::cvtColor(src, dst_gold, code);
    cv::Mat dst_mat;
    dst.download(dst_mat);
    MAT_DIFF(dst_mat, dst_gold);
    EXPECT_MAT_NEAR(dst_gold, dst_mat, 1e-5);
}

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, CvtColor, testing::Combine(
                            DIFFERENT_SIZES,
                            testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F))
                        ));

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, CvtColor_YUV420, testing::Combine(
                            testing::Values(cv::Size(128, 45), cv::Size(46, 132), cv::Size(1024, 1023)),
                            testing::Values((int)COLOR_YUV2RGBA_NV12, (int)COLOR_YUV2BGRA_NV12, (int)COLOR_YUV2RGB_NV12, (int)COLOR_YUV2BGR_NV12)
                        ));

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, CvtColor_Gray2RGB, testing::Combine(
                            DIFFERENT_SIZES,
                            testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_32F)),
                            testing::Values((int)COLOR_GRAY2BGR, (int)COLOR_GRAY2BGRA, (int)COLOR_GRAY2RGB, (int)COLOR_GRAY2RGBA)
                        ));
}
#endif
