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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "test_precomp.hpp"

namespace opencv_test { namespace {

enum HALFunc
{
    HAL_EXP = 0,
    HAL_LOG = 1,
    HAL_SQRT = 2,
    HAL_INV_SQRT = 3,
    HAL_LU = 4,
    HAL_CHOL = 5,
};

void PrintTo(const HALFunc& v, std::ostream* os)
{
    switch (v) {
    case HAL_EXP: *os << "HAL_EXP"; return;
    case HAL_LOG: *os << "HAL_LOG"; return;
    case HAL_SQRT: *os << "HAL_SQRT"; return;
    case HAL_INV_SQRT: *os << "HAL_INV_SQRT"; return;
    case HAL_LU: *os << "LU"; return;
    case HAL_CHOL: *os << "Cholesky"; return;
    } // don't use "default:" to emit compiler warnings
}

typedef testing::TestWithParam<std::tuple<int, HALFunc> > mathfuncs;
TEST_P(mathfuncs, accuracy)
{
    const int depth = std::get<0>(GetParam());
    const int nfunc = std::get<1>(GetParam());

    double eps = depth == CV_32F ? 1e-5 : 1e-10;
    int n = 100;

    Mat src(1, n, depth), dst(1, n, depth), dst0(1, n, depth);
    randu(src, 1, 10);

    switch (nfunc)
    {
    case HAL_EXP:
        if( depth == CV_32F )
            hal::exp32f(src.ptr<float>(), dst.ptr<float>(), n);
        else
            hal::exp64f(src.ptr<double>(), dst.ptr<double>(), n);
        break;
    case HAL_LOG:
        if( depth == CV_32F )
            hal::log32f(src.ptr<float>(), dst.ptr<float>(), n);
        else
            hal::log64f(src.ptr<double>(), dst.ptr<double>(), n);
        break;
    case HAL_SQRT:
        if( depth == CV_32F )
            hal::sqrt32f(src.ptr<float>(), dst.ptr<float>(), n);
        else
            hal::sqrt64f(src.ptr<double>(), dst.ptr<double>(), n);
        break;
    case HAL_INV_SQRT:
        if( depth == CV_32F )
            hal::invSqrt32f(src.ptr<float>(), dst.ptr<float>(), n);
        else
            hal::invSqrt64f(src.ptr<double>(), dst.ptr<double>(), n);
        break;

    default:
        CV_Error(Error::StsBadArg, "unknown function");
    }

    src.copyTo(dst0);
    switch (nfunc)
    {
    case HAL_EXP:
        if( depth == CV_32F )
            dst0.forEach<float>([](float& v, const int*) -> void { v = std::exp(v); });
        else
            dst0.forEach<double>([](double& v, const int*) -> void { v = std::exp(v); });
        break;
    case HAL_LOG:
        if( depth == CV_32F )
            dst0.forEach<float>([](float& v, const int*) -> void { v = std::log(v); });
        else
            dst0.forEach<double>([](double& v, const int*) -> void { v = std::log(v); });
        break;
    case HAL_SQRT:
        if( depth == CV_32F )
            dst0.forEach<float>([](float& v, const int*) -> void { v = std::sqrt(v); });
        else
            dst0.forEach<double>([](double& v, const int*) -> void { v = std::sqrt(v); });
        break;
    case HAL_INV_SQRT:
        if( depth == CV_32F )
            dst0.forEach<float>([](float& v, const int*) -> void { v = std::pow(v, -0.5f); });
        else
            dst0.forEach<double>([](double& v, const int*) -> void { v = std::pow(v, -0.5); });
        break;
    default:
        CV_Error(Error::StsBadArg, "unknown function");
    }
    EXPECT_LE(cvtest::norm(dst, dst0, NORM_INF | NORM_RELATIVE), eps);
}
INSTANTIATE_TEST_CASE_P(Core_HAL, mathfuncs,
    testing::Combine(
        testing::Values(CV_32F, CV_64F),
        testing::Values(HAL_EXP, HAL_LOG, HAL_SQRT, HAL_INV_SQRT)
    )
);

typedef testing::TestWithParam<std::tuple<int, HALFunc, int> > mat_decomp;
TEST_P(mat_decomp, accuracy)
{
    const int depth = std::get<0>(GetParam());
    const int nfunc = std::get<1>(GetParam());
    const int size = std::get<2>(GetParam());

#if CV_LASX
    double eps = depth == CV_32F ? 1e-5 : 2e-10;
#else
    double eps = depth == CV_32F ? 1e-5 : 1e-10;
#endif

    Mat a0(size, size, depth), x0(size, 1, depth);
    randu(a0, -1, 1);
    a0 = a0*a0.t();
    randu(x0, -1, 1);
    Mat b = a0 * x0;
    Mat x = b.clone();
    Mat a = a0.clone();

    int solveStatus;
    switch (nfunc)
    {
    case HAL_LU:
        if( depth == CV_32F )
            solveStatus = hal::LU32f(a.ptr<float>(), a.step, size, x.ptr<float>(), x.step, 1);
        else
            solveStatus = hal::LU64f(a.ptr<double>(), a.step, size, x.ptr<double>(), x.step, 1);
        break;
    case HAL_CHOL:
        if( depth == CV_32F )
            solveStatus = hal::Cholesky32f(a.ptr<float>(), a.step, size, x.ptr<float>(), x.step, 1);
        else
            solveStatus = hal::Cholesky64f(a.ptr<double>(), a.step, size, x.ptr<double>(), x.step, 1);
        break;
    default:
        CV_Error(Error::StsBadArg, "unknown function");
    }
    EXPECT_NE(0, solveStatus);
    EXPECT_LE(cvtest::norm(a0 * x, b, NORM_INF | NORM_RELATIVE), eps)
        << "x:  " << Mat(x.t())
        << "\nx0: " << Mat(x0.t())
        << "\na0: " << a0
        << "\nb: " << b;
}

INSTANTIATE_TEST_CASE_P(Core_HAL, mat_decomp,
    testing::Combine(
        testing::Values(CV_32F, CV_64F),
        testing::Values(HAL_LU, HAL_CHOL),
        testing::Values(3, 4, 6, 15)
    )
);

}} // namespace
