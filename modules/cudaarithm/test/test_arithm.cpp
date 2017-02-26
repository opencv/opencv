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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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

#ifdef HAVE_CUDA

using namespace cvtest;

namespace {

//////////////////////////////////////////////////////////////////////////////
// GEMM

#ifdef HAVE_CUBLAS

CV_FLAGS(GemmFlags, 0, cv::GEMM_1_T, cv::GEMM_2_T, cv::GEMM_3_T);
#define ALL_GEMM_FLAGS testing::Values(GemmFlags(0), GemmFlags(cv::GEMM_1_T), GemmFlags(cv::GEMM_2_T), GemmFlags(cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_3_T), GemmFlags(cv::GEMM_1_T | cv::GEMM_2_T | cv::GEMM_3_T))

PARAM_TEST_CASE(GEMM, cv::cuda::DeviceInfo, cv::Size, MatType, GemmFlags, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int flags;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        flags = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(GEMM, Accuracy)
{
    cv::Mat src1 = randomMat(size, type, -10.0, 10.0);
    cv::Mat src2 = randomMat(size, type, -10.0, 10.0);
    cv::Mat src3 = randomMat(size, type, -10.0, 10.0);
    double alpha = randomDouble(-10.0, 10.0);
    double beta = randomDouble(-10.0, 10.0);

    if (CV_MAT_DEPTH(type) == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::gemm(loadMat(src1), loadMat(src2), alpha, loadMat(src3), beta, dst, flags);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else if (type == CV_64FC2 && flags != 0)
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::gemm(loadMat(src1), loadMat(src2), alpha, loadMat(src3), beta, dst, flags);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsNotImplemented, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(size, type, useRoi);
        cv::cuda::gemm(loadMat(src1, useRoi), loadMat(src2, useRoi), alpha, loadMat(src3, useRoi), beta, dst, flags);

        cv::Mat dst_gold;
        cv::gemm(src1, src2, alpha, src3, beta, dst_gold, flags);

        EXPECT_MAT_NEAR(dst_gold, dst, CV_MAT_DEPTH(type) == CV_32F ? 1e-1 : 1e-10);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, GEMM, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_32FC1), MatType(CV_32FC2), MatType(CV_64FC1), MatType(CV_64FC2)),
    ALL_GEMM_FLAGS,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////
// MulSpectrums

CV_FLAGS(DftFlags, 0, cv::DFT_INVERSE, cv::DFT_SCALE, cv::DFT_ROWS, cv::DFT_COMPLEX_OUTPUT, cv::DFT_REAL_OUTPUT)

PARAM_TEST_CASE(MulSpectrums, cv::cuda::DeviceInfo, cv::Size, DftFlags)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int flag;

    cv::Mat a, b;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        flag = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());

        a = randomMat(size, CV_32FC2);
        b = randomMat(size, CV_32FC2);
    }
};

CUDA_TEST_P(MulSpectrums, Simple)
{
    cv::cuda::GpuMat c;
    cv::cuda::mulSpectrums(loadMat(a), loadMat(b), c, flag, false);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);

    EXPECT_MAT_NEAR(c_gold, c, 1e-2);
}

CUDA_TEST_P(MulSpectrums, Scaled)
{
    float scale = 1.f / size.area();

    cv::cuda::GpuMat c;
    cv::cuda::mulAndScaleSpectrums(loadMat(a), loadMat(b), c, flag, scale, false);

    cv::Mat c_gold;
    cv::mulSpectrums(a, b, c_gold, flag, false);
    c_gold.convertTo(c_gold, c_gold.type(), scale);

    EXPECT_MAT_NEAR(c_gold, c, 1e-2);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, MulSpectrums, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(DftFlags(0), DftFlags(cv::DFT_ROWS))));

////////////////////////////////////////////////////////////////////////////
// Dft

struct Dft : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

namespace
{
    void testC2C(const std::string& hint, int cols, int rows, int flags, bool inplace)
    {
        SCOPED_TRACE(hint);

        cv::Mat a = randomMat(cv::Size(cols, rows), CV_32FC2, 0.0, 10.0);

        cv::Mat b_gold;
        cv::dft(a, b_gold, flags);

        cv::cuda::GpuMat d_b;
        cv::cuda::GpuMat d_b_data;
        if (inplace)
        {
            d_b_data.create(1, a.size().area(), CV_32FC2);
            d_b = cv::cuda::GpuMat(a.rows, a.cols, CV_32FC2, d_b_data.ptr(), a.cols * d_b_data.elemSize());
        }
        cv::cuda::dft(loadMat(a), d_b, cv::Size(cols, rows), flags);

        EXPECT_TRUE(!inplace || d_b.ptr() == d_b_data.ptr());
        ASSERT_EQ(CV_32F, d_b.depth());
        ASSERT_EQ(2, d_b.channels());
        EXPECT_MAT_NEAR(b_gold, cv::Mat(d_b), rows * cols * 1e-4);
    }
}

CUDA_TEST_P(Dft, C2C)
{
    int cols = randomInt(2, 100);
    int rows = randomInt(2, 100);

    for (int i = 0; i < 2; ++i)
    {
        bool inplace = i != 0;

        testC2C("no flags", cols, rows, 0, inplace);
        testC2C("no flags 0 1", cols, rows + 1, 0, inplace);
        testC2C("no flags 1 0", cols, rows + 1, 0, inplace);
        testC2C("no flags 1 1", cols + 1, rows, 0, inplace);
        testC2C("DFT_INVERSE", cols, rows, cv::DFT_INVERSE, inplace);
        testC2C("DFT_ROWS", cols, rows, cv::DFT_ROWS, inplace);
        testC2C("single col", 1, rows, 0, inplace);
        testC2C("single row", cols, 1, 0, inplace);
        testC2C("single col inversed", 1, rows, cv::DFT_INVERSE, inplace);
        testC2C("single row inversed", cols, 1, cv::DFT_INVERSE, inplace);
        testC2C("single row DFT_ROWS", cols, 1, cv::DFT_ROWS, inplace);
        testC2C("size 1 2", 1, 2, 0, inplace);
        testC2C("size 2 1", 2, 1, 0, inplace);
    }
}

namespace
{
    void testR2CThenC2R(const std::string& hint, int cols, int rows, bool inplace)
    {
        SCOPED_TRACE(hint);

        cv::Mat a = randomMat(cv::Size(cols, rows), CV_32FC1, 0.0, 10.0);

        cv::cuda::GpuMat d_b, d_c;
        cv::cuda::GpuMat d_b_data, d_c_data;
        if (inplace)
        {
            if (a.cols == 1)
            {
                d_b_data.create(1, (a.rows / 2 + 1) * a.cols, CV_32FC2);
                d_b = cv::cuda::GpuMat(a.rows / 2 + 1, a.cols, CV_32FC2, d_b_data.ptr(), a.cols * d_b_data.elemSize());
            }
            else
            {
                d_b_data.create(1, a.rows * (a.cols / 2 + 1), CV_32FC2);
                d_b = cv::cuda::GpuMat(a.rows, a.cols / 2 + 1, CV_32FC2, d_b_data.ptr(), (a.cols / 2 + 1) * d_b_data.elemSize());
            }
            d_c_data.create(1, a.size().area(), CV_32F);
            d_c = cv::cuda::GpuMat(a.rows, a.cols, CV_32F, d_c_data.ptr(), a.cols * d_c_data.elemSize());
        }

        cv::cuda::dft(loadMat(a), d_b, cv::Size(cols, rows), 0);
        cv::cuda::dft(d_b, d_c, cv::Size(cols, rows), cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        EXPECT_TRUE(!inplace || d_b.ptr() == d_b_data.ptr());
        EXPECT_TRUE(!inplace || d_c.ptr() == d_c_data.ptr());
        ASSERT_EQ(CV_32F, d_c.depth());
        ASSERT_EQ(1, d_c.channels());

        cv::Mat c(d_c);
        EXPECT_MAT_NEAR(a, c, rows * cols * 1e-5);
    }
}

CUDA_TEST_P(Dft, R2CThenC2R)
{
    int cols = randomInt(2, 100);
    int rows = randomInt(2, 100);

    testR2CThenC2R("sanity", cols, rows, false);
    testR2CThenC2R("sanity 0 1", cols, rows + 1, false);
    testR2CThenC2R("sanity 1 0", cols + 1, rows, false);
    testR2CThenC2R("sanity 1 1", cols + 1, rows + 1, false);
    testR2CThenC2R("single col", 1, rows, false);
    testR2CThenC2R("single col 1", 1, rows + 1, false);
    testR2CThenC2R("single row", cols, 1, false);
    testR2CThenC2R("single row 1", cols + 1, 1, false);

    testR2CThenC2R("sanity", cols, rows, true);
    testR2CThenC2R("sanity 0 1", cols, rows + 1, true);
    testR2CThenC2R("sanity 1 0", cols + 1, rows, true);
    testR2CThenC2R("sanity 1 1", cols + 1, rows + 1, true);
    testR2CThenC2R("single row", cols, 1, true);
    testR2CThenC2R("single row 1", cols + 1, 1, true);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Dft, ALL_DEVICES);

////////////////////////////////////////////////////////
// Convolve

namespace
{
    void convolveDFT(const cv::Mat& A, const cv::Mat& B, cv::Mat& C, bool ccorr = false)
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

    IMPLEMENT_PARAM_CLASS(KSize, int)
    IMPLEMENT_PARAM_CLASS(Ccorr, bool)
}

PARAM_TEST_CASE(Convolve, cv::cuda::DeviceInfo, cv::Size, KSize, Ccorr)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int ksize;
    bool ccorr;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        ksize = GET_PARAM(2);
        ccorr = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Convolve, Accuracy)
{
    cv::Mat src = randomMat(size, CV_32FC1, 0.0, 100.0);
    cv::Mat kernel = randomMat(cv::Size(ksize, ksize), CV_32FC1, 0.0, 1.0);

    cv::Ptr<cv::cuda::Convolution> conv = cv::cuda::createConvolution();

    cv::cuda::GpuMat dst;
    conv->convolve(loadMat(src), loadMat(kernel), dst, ccorr);

    cv::Mat dst_gold;
    convolveDFT(src, kernel, dst_gold, ccorr);

    EXPECT_MAT_NEAR(dst, dst_gold, 1e-1);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Convolve, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(KSize(3), KSize(7), KSize(11), KSize(17), KSize(19), KSize(23), KSize(45)),
    testing::Values(Ccorr(false), Ccorr(true))));

#endif // HAVE_CUBLAS

} // namespace

#endif // HAVE_CUDA
