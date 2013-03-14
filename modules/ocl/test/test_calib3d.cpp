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

extern std::string workdir;
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

TEST_P(StereoMatchBM, Accuracy)
{

    Mat left_image  = readImage(workdir + "../ocl/aloe-L.png", IMREAD_GRAYSCALE);
    Mat right_image = readImage(workdir + "../ocl/aloe-R.png", IMREAD_GRAYSCALE);
    Mat disp_gold   = readImage(workdir + "../ocl/aloe-disp.png", IMREAD_GRAYSCALE);
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

INSTANTIATE_TEST_CASE_P(GPU_Calib3D, StereoMatchBM, testing::Combine(testing::Values(128),
	                                   testing::Values(19)));

#endif // HAVE_OPENCL
