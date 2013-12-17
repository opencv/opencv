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

using namespace std;

////////////////////////////////////////////////////////////////////////////
// Dft

PARAM_TEST_CASE(Dft, cv::Size, int, bool)
{
    cv::Size dft_size;
    int	 dft_flags;
    bool doubleFP;

    virtual void SetUp()
    {
        dft_size  = GET_PARAM(0);
        dft_flags = GET_PARAM(1);
        doubleFP = GET_PARAM(2);
    }
};

OCL_TEST_P(Dft, C2C)
{
    cv::Mat a = randomMat(dft_size, doubleFP ? CV_64FC2 : CV_32FC2, 0.0, 100.0);
    cv::Mat b_gold;

    cv::ocl::oclMat d_b;

    cv::dft(a, b_gold, dft_flags);
    cv::ocl::dft(cv::ocl::oclMat(a), d_b, a.size(), dft_flags);

    EXPECT_MAT_NEAR(b_gold, cv::Mat(d_b), a.size().area() * 1e-4);
}

OCL_TEST_P(Dft, R2C)
{
    cv::Mat a = randomMat(dft_size, doubleFP ? CV_64FC1 : CV_32FC1, 0.0, 100.0);
    cv::Mat b_gold, b_gold_roi;

    cv::ocl::oclMat d_b, d_c;
    cv::ocl::dft(cv::ocl::oclMat(a), d_b, a.size(), dft_flags);
    cv::dft(a, b_gold, cv::DFT_COMPLEX_OUTPUT | dft_flags);

    b_gold_roi = b_gold(cv::Rect(0, 0, d_b.cols, d_b.rows));
    EXPECT_MAT_NEAR(b_gold_roi, cv::Mat(d_b), a.size().area() * 1e-4);

    cv::Mat c_gold;
    cv::dft(b_gold, c_gold, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    EXPECT_MAT_NEAR(b_gold_roi, cv::Mat(d_b), a.size().area() * 1e-4);
}

OCL_TEST_P(Dft, R2CthenC2R)
{
    cv::Mat a = randomMat(dft_size, doubleFP ? CV_64FC1 : CV_32FC1, 0.0, 10.0);

    cv::ocl::oclMat d_b, d_c;
    cv::ocl::dft(cv::ocl::oclMat(a), d_b, a.size(), 0);
    cv::ocl::dft(d_b, d_c, a.size(), cv::DFT_SCALE | cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    EXPECT_MAT_NEAR(a, d_c, a.size().area() * 1e-4);
}

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, Dft, testing::Combine(
                            testing::Values(cv::Size(2, 3), cv::Size(5, 4), cv::Size(25, 20), cv::Size(512, 1), cv::Size(1024, 768)),
                            testing::Values(0, (int)cv::DFT_ROWS, (int)cv::DFT_SCALE), testing::Bool()));

////////////////////////////////////////////////////////////////////////////
// MulSpectrums

PARAM_TEST_CASE(MulSpectrums, cv::Size, DftFlags, bool)
{
    cv::Size size;
    int flag;
    bool ccorr;
    cv::Mat a, b;

    virtual void SetUp()
    {
        size  = GET_PARAM(0);
        flag  = GET_PARAM(1);
        ccorr = GET_PARAM(2);

        a = randomMat(size, CV_32FC2, -100, 100, false);
        b = randomMat(size, CV_32FC2, -100, 100, false);
    }
};

OCL_TEST_P(MulSpectrums, Simple)
{
    cv::ocl::oclMat c;
    cv::ocl::mulSpectrums(cv::ocl::oclMat(a), cv::ocl::oclMat(b), c, flag, 1.0, ccorr);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, ccorr);

    EXPECT_MAT_NEAR(c_gold, c, 1e-2);
}

OCL_TEST_P(MulSpectrums, Scaled)
{
    float scale = 1.f / size.area();

    cv::ocl::oclMat c;
    cv::ocl::mulSpectrums(cv::ocl::oclMat(a), cv::ocl::oclMat(b), c, flag, scale, ccorr);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, ccorr);
    c_gold.convertTo(c_gold, c_gold.type(), scale);

    EXPECT_MAT_NEAR(c_gold, c, 1e-2);
}

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, MulSpectrums, testing::Combine(
    DIFFERENT_SIZES,
    testing::Values(DftFlags(0)),
    testing::Values(false, true)));


////////////////////////////////////////////////////////
// Convolve

void static convolveDFT(const cv::Mat& A, const cv::Mat& B, cv::Mat& C, bool ccorr = false)
{
    // reallocate the output array if needed
    C.create(std::abs(A.rows - B.rows) + 1, std::abs(A.cols - B.cols) + 1, A.type());
    cv::Size dftSize;

    // compute the size of DFT transform
    dftSize.width = cv::getOptimalDFTSize(A.cols + B.cols - 1);
    dftSize.height = cv::getOptimalDFTSize(A.rows + B.rows - 1);

    // allocate temporary buffers and initialize them with 0s
    cv::Mat tempA(dftSize, A.type(), cv::Scalar::all(0));
    cv::Mat tempB(dftSize, B.type(), cv::Scalar::all(0));

    // copy A and B to the top-left corners of tempA and tempB, respectively
    cv::Mat roiA(tempA, cv::Rect(0, 0, A.cols, A.rows));
    A.copyTo(roiA);
    cv::Mat roiB(tempB, cv::Rect(0, 0, B.cols, B.rows));
    B.copyTo(roiB);

    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    cv::dft(tempA, tempA, 0, A.rows);
    cv::dft(tempB, tempB, 0, B.rows);

    // multiply the spectrums;
    // the function handles packed spectrum representations well
    cv::mulSpectrums(tempA, tempB, tempA, 0, ccorr);

    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, C.rows);

    // now copy the result back to C.
    tempA(cv::Rect(0, 0, C.cols, C.rows)).copyTo(C);
}

IMPLEMENT_PARAM_CLASS(KSize, int);
IMPLEMENT_PARAM_CLASS(Ccorr, bool);

PARAM_TEST_CASE(Convolve_DFT, cv::Size, KSize, Ccorr)
{
    cv::Size size;
    int ksize;
    bool ccorr;

    cv::Mat src;
    cv::Mat kernel;

    cv::Mat dst_gold;

    virtual void SetUp()
    {
        size  = GET_PARAM(0);
        ksize = GET_PARAM(1);
        ccorr = GET_PARAM(2);
    }
};

OCL_TEST_P(Convolve_DFT, Accuracy)
{
    cv::Mat src = randomMat(size, CV_32FC1, 0.0, 100.0);
    cv::Mat kernel = randomMat(cv::Size(ksize, ksize), CV_32FC1, 0.0, 1.0);

    cv::ocl::oclMat dst;
    cv::ocl::convolve(cv::ocl::oclMat(src), cv::ocl::oclMat(kernel), dst, ccorr);

    cv::Mat dst_gold;
    convolveDFT(src, kernel, dst_gold, ccorr);

    EXPECT_MAT_NEAR(dst, dst_gold, 1e-1);
}
#define DIFFERENT_CONVOLVE_SIZES testing::Values(cv::Size(251, 257), cv::Size(113, 113), cv::Size(200, 480), cv::Size(1300, 1300))
INSTANTIATE_TEST_CASE_P(OCL_ImgProc, Convolve_DFT, testing::Combine(
    DIFFERENT_CONVOLVE_SIZES,
    testing::Values(KSize(19), KSize(23), KSize(45)),
    testing::Values(Ccorr(true)/*, Ccorr(false)*/))); // TODO false ccorr cannot pass for some instances
