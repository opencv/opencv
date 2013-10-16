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
//     Peng Xiao, pengxiao@outlook.com
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

#include "test_precomp.hpp"
#include <iomanip>

using namespace cv;

#ifdef HAVE_OPENCL

PARAM_TEST_CASE(StereoMatchBM, int, int)
{
    int n_disp;
    int winSize;

    virtual void SetUp()
    {
        n_disp  = GET_PARAM(0);
        winSize = GET_PARAM(1);
    }
};

OCL_TEST_P(StereoMatchBM, Regression)
{

    Mat left_image  = readImage("gpu/stereobm/aloe-L.png", IMREAD_GRAYSCALE);
    Mat right_image = readImage("gpu/stereobm/aloe-R.png", IMREAD_GRAYSCALE);
    Mat disp_gold   = readImage("gpu/stereobm/aloe-disp.png", IMREAD_GRAYSCALE);
    ocl::oclMat d_left, d_right;
    ocl::oclMat d_disp(left_image.size(), CV_8U);
    Mat  disp;

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());
    d_left.upload(left_image);
    d_right.upload(right_image);

    ocl::StereoBM_OCL bm(0, n_disp, winSize);


    bm(d_left, d_right, d_disp);
    d_disp.download(disp);

    EXPECT_MAT_SIMILAR(disp_gold, disp, 1e-3);
}

INSTANTIATE_TEST_CASE_P(OCL_Calib3D, StereoMatchBM, testing::Combine(testing::Values(128),
                                       testing::Values(19)));

PARAM_TEST_CASE(StereoMatchBP, int, int, int, float, float, float, float)
{
    int ndisp_;
    int iters_;
    int levels_;
    float max_data_term_;
    float data_weight_;
    float max_disc_term_;
    float disc_single_jump_;
    virtual void SetUp()
    {
        ndisp_          = GET_PARAM(0);
        iters_          = GET_PARAM(1);
        levels_         = GET_PARAM(2);
        max_data_term_  = GET_PARAM(3);
        data_weight_    = GET_PARAM(4);
        max_disc_term_     = GET_PARAM(5);
        disc_single_jump_  = GET_PARAM(6);
    }
};
OCL_TEST_P(StereoMatchBP, Regression)
{
    Mat left_image  = readImage("gpu/stereobp/aloe-L.png");
    Mat right_image = readImage("gpu/stereobp/aloe-R.png");
    Mat disp_gold   = readImage("gpu/stereobp/aloe-disp.png", IMREAD_GRAYSCALE);
    ocl::oclMat d_left, d_right;
    ocl::oclMat d_disp;
    Mat  disp;
    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());
    d_left.upload(left_image);
    d_right.upload(right_image);
    ocl::StereoBeliefPropagation bp(ndisp_, iters_, levels_, max_data_term_, data_weight_,
        max_disc_term_, disc_single_jump_, CV_16S);
    bp(d_left, d_right, d_disp);
    d_disp.download(disp);
    disp.convertTo(disp, disp_gold.depth());
    EXPECT_MAT_NEAR(disp_gold, disp, 0.0);
}
INSTANTIATE_TEST_CASE_P(OCL_Calib3D, StereoMatchBP, testing::Combine(testing::Values(64),
    testing::Values(8),testing::Values(2),testing::Values(25.0f),
    testing::Values(0.1f),testing::Values(15.0f),testing::Values(1.0f)));

//////////////////////////////////////////////////////////////////////////
//  ConstSpaceBeliefPropagation
PARAM_TEST_CASE(StereoMatchConstSpaceBP, int, int, int, int, float, float, float, float, int, int)
{
    int ndisp_;
    int iters_;
    int levels_;
    int nr_plane_;
    float max_data_term_;
    float data_weight_;
    float max_disc_term_;
    float disc_single_jump_;
    int min_disp_th_;
    int msg_type_;

    virtual void SetUp()
    {
        ndisp_          = GET_PARAM(0);
        iters_          = GET_PARAM(1);
        levels_         = GET_PARAM(2);
        nr_plane_ = GET_PARAM(3);
        max_data_term_  = GET_PARAM(4);
        data_weight_    = GET_PARAM(5);
        max_disc_term_     = GET_PARAM(6);
        disc_single_jump_  = GET_PARAM(7);
        min_disp_th_ = GET_PARAM(8);
        msg_type_  = GET_PARAM(9);
    }
};
OCL_TEST_P(StereoMatchConstSpaceBP, Regression)
{
    Mat left_image  = readImage("gpu/csstereobp/aloe-L.png");
    Mat right_image = readImage("gpu/csstereobp/aloe-R.png");
    Mat disp_gold   = readImage("gpu/csstereobp/aloe-disp.png", IMREAD_GRAYSCALE);

    ocl::oclMat d_left, d_right;
    ocl::oclMat d_disp;

    Mat  disp;
    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    d_left.upload(left_image);
    d_right.upload(right_image);

    ocl::StereoConstantSpaceBP bp(ndisp_, iters_, levels_, nr_plane_, max_data_term_, data_weight_,
        max_disc_term_, disc_single_jump_, 0, CV_32F);
    bp(d_left, d_right, d_disp);
    d_disp.download(disp);
    disp.convertTo(disp, disp_gold.depth());

    EXPECT_MAT_SIMILAR(disp_gold, disp, 1e-4);
    //EXPECT_MAT_NEAR(disp_gold, disp, 1.0, "");
}
INSTANTIATE_TEST_CASE_P(OCL_Calib3D, StereoMatchConstSpaceBP, testing::Combine(testing::Values(128),
    testing::Values(16),testing::Values(4), testing::Values(4), testing::Values(30.0f),
    testing::Values(1.0f),testing::Values(160.0f),
    testing::Values(10.0f), testing::Values(0), testing::Values(CV_32F)));
#endif // HAVE_OPENCL
