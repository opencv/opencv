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
//     and/or other materials provided with the distribution.
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
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

////////////////////////////////////////////////////////////////////////////
// Dft

PARAM_TEST_CASE(Dft, cv::Size, MatDepth, bool, bool, bool, bool)
{
    cv::Size dft_size;
    int	dft_flags, depth;
    bool inplace;

    TEST_DECLARE_INPUT_PARAMETER(src)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        dft_size = GET_PARAM(0);
        depth = GET_PARAM(1);
        inplace = GET_PARAM(2);

        dft_flags = 0;
        if (GET_PARAM(3))
            dft_flags |= cv::DFT_ROWS;
        if (GET_PARAM(4))
            dft_flags |= cv::DFT_SCALE;
        if (GET_PARAM(5))
            dft_flags |= cv::DFT_INVERSE;
    }

    void generateTestData(int cn = 2)
    {
        src = randomMat(dft_size, CV_MAKE_TYPE(depth, cn), 0.0, 100.0);
        usrc = src.getUMat(ACCESS_READ);

        if (inplace)
            dst = src, udst = usrc;
    }
};

OCL_TEST_P(Dft, C2C)
{
    generateTestData();

    OCL_OFF(cv::dft(src, dst, dft_flags | cv::DFT_COMPLEX_OUTPUT));
    OCL_ON(cv::dft(usrc, udst, dft_flags | cv::DFT_COMPLEX_OUTPUT));

    double eps = src.size().area() * 1e-4;
    EXPECT_MAT_NEAR(dst, udst, eps);
}

////////////////////////////////////////////////////////////////////////////
// MulSpectrums

PARAM_TEST_CASE(MulSpectrums, bool, bool)
{
    bool ccorr, useRoi;

    TEST_DECLARE_INPUT_PARAMETER(src1)
    TEST_DECLARE_INPUT_PARAMETER(src2)
    TEST_DECLARE_OUTPUT_PARAMETER(dst)

    virtual void SetUp()
    {
        ccorr = GET_PARAM(0);
        useRoi = GET_PARAM(1);
    }

    void generateTestData()
    {
        Size srcRoiSize = randomSize(1, MAX_VALUE);
        Border src1Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src1, src1_roi, srcRoiSize, src1Border, CV_32FC2, -11, 11);


        Border src2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, srcRoiSize, src2Border, CV_32FC2, -11, 11);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, srcRoiSize, dstBorder, CV_32FC2, 5, 16);

        UMAT_UPLOAD_INPUT_PARAMETER(src1)
        UMAT_UPLOAD_INPUT_PARAMETER(src2)
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst)
    }
};

OCL_TEST_P(MulSpectrums, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        generateTestData();

        OCL_OFF(cv::mulSpectrums(src1_roi, src2_roi, dst_roi, 0, ccorr));
        OCL_ON(cv::mulSpectrums(usrc1_roi, usrc2_roi, udst_roi, 0, ccorr));

        OCL_EXPECT_MATS_NEAR_RELATIVE(dst, 1e-6);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(OCL_ImgProc, MulSpectrums, testing::Combine(Bool(), Bool()));

OCL_INSTANTIATE_TEST_CASE_P(Core, Dft, Combine(Values(cv::Size(2, 3), cv::Size(5, 4), cv::Size(25, 20),
                                                       cv::Size(512, 1), cv::Size(1024, 768)),
                                               Values(CV_32F, CV_64F),
                                               Bool(), // inplace
                                               Bool(), // DFT_ROWS
                                               Bool(), // DFT_SCALE
                                               Bool()) // DFT_INVERSE
                            );

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
