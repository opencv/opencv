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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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

#ifndef __OPENCV_TS_OCL_TEST_HPP__
#define __OPENCV_TS_OCL_TEST_HPP__

#include "cvconfig.h" // to get definition of HAVE_OPENCL
#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCL

#include "opencv2/ts.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/core/ocl.hpp"

namespace cvtest {
namespace ocl {

using namespace cv;
using namespace testing;

namespace traits {

template <typename T>
struct GetMatForRead
{
};
template <>
struct GetMatForRead<Mat>
{
    static const Mat get(const Mat& m) { return m; }
};
template <>
struct GetMatForRead<UMat>
{
    static const Mat get(const UMat& m) { return m.getMat(ACCESS_READ); }
};

} // namespace traits

template <typename T>
const Mat getMatForRead(const T& mat)
{
    return traits::GetMatForRead<T>::get(mat);
}

extern int test_loop_times;

#define MAX_VALUE 357

#define EXPECT_MAT_NORM(mat, eps) \
{ \
    EXPECT_LE(checkNorm(mat), eps) \
}

#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(checkNorm(mat1, mat2), eps) \
       << cv::format("Size: %d x %d", mat1.size().width, mat1.size().height) << std::endl; \
}

#define EXPECT_MAT_NEAR_RELATIVE(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(checkNormRelative(mat1, mat2), eps) \
       << cv::format("Size: %d x %d", mat1.size().width, mat1.size().height) << std::endl; \
}

#define OCL_EXPECT_MATS_NEAR(name, eps) \
{ \
    EXPECT_MAT_NEAR(name ## _roi, u ## name ## _roi, eps); \
    int nextValue = rng.next(); \
    RNG dataRng1(nextValue), dataRng2(nextValue); \
    dataRng1.fill(name ## _roi, RNG::UNIFORM, Scalar::all(-MAX_VALUE), Scalar::all(MAX_VALUE)); \
    dataRng2.fill(u ## name ## _roi, RNG::UNIFORM, Scalar::all(-MAX_VALUE), Scalar::all(MAX_VALUE)); \
    EXPECT_MAT_NEAR(name, u ## name, 0/*FLT_EPSILON*/); \
}

#define OCL_EXPECT_MATS_NEAR_RELATIVE(name, eps) \
{ \
    EXPECT_MAT_NEAR_RELATIVE(name ## _roi, u ## name ## _roi, eps); \
    int nextValue = rng.next(); \
    RNG dataRng1(nextValue), dataRng2(nextValue); \
    dataRng1.fill(name ## _roi, RNG::UNIFORM, Scalar::all(-MAX_VALUE), Scalar::all(MAX_VALUE)); \
    dataRng2.fill(u ## name ## _roi, RNG::UNIFORM, Scalar::all(-MAX_VALUE), Scalar::all(MAX_VALUE)); \
    EXPECT_MAT_NEAR_RELATIVE(name, u ## name, 0/*FLT_EPSILON*/); \
}

#define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
{ \
    ASSERT_EQ(mat1.type(), mat2.type()); \
    ASSERT_EQ(mat1.size(), mat2.size()); \
    EXPECT_LE(checkSimilarity(mat1, mat2), eps); \
        << cv::format("Size: %d x %d", mat1.size().width, mat1.size().height) << std::endl; \
}

using perf::MatDepth;
using perf::MatType;

#define OCL_RNG_SEED 123456

struct CV_EXPORTS TestUtils
{
    cv::RNG rng;

    TestUtils()
    {
        rng = cv::RNG(OCL_RNG_SEED);
    }

    int randomInt(int minVal, int maxVal)
    {
        return rng.uniform(minVal, maxVal);
    }

    double randomDouble(double minVal, double maxVal)
    {
        return rng.uniform(minVal, maxVal);
    }

    double randomDoubleLog(double minVal, double maxVal)
    {
        double logMin = log((double)minVal + 1);
        double logMax = log((double)maxVal + 1);
        double pow = rng.uniform(logMin, logMax);
        double v = exp(pow) - 1;
        CV_Assert(v >= minVal && (v < maxVal || (v == minVal && v == maxVal)));
        return v;
    }

    Size randomSize(int minVal, int maxVal)
    {
#if 1
        return cv::Size((int)randomDoubleLog(minVal, maxVal), (int)randomDoubleLog(minVal, maxVal));
#else
        return cv::Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
#endif
    }

    Size randomSize(int minValX, int maxValX, int minValY, int maxValY)
    {
#if 1
        return cv::Size((int)randomDoubleLog(minValX, maxValX), (int)randomDoubleLog(minValY, maxValY));
#else
        return cv::Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
#endif
    }

    Scalar randomScalar(double minVal, double maxVal)
    {
        return Scalar(randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal));
    }

    Mat randomMat(Size size, int type, double minVal, double maxVal, bool useRoi = false)
    {
        RNG dataRng(rng.next());
        return cvtest::randomMat(dataRng, size, type, minVal, maxVal, useRoi);
    }

    struct Border
    {
        int top, bot, lef, rig;
    };

    Border randomBorder(int minValue = 0, int maxValue = MAX_VALUE)
    {
        Border border = {
                (int)randomDoubleLog(minValue, maxValue),
                (int)randomDoubleLog(minValue, maxValue),
                (int)randomDoubleLog(minValue, maxValue),
                (int)randomDoubleLog(minValue, maxValue)
        };
        return border;
    }

    void randomSubMat(Mat& whole, Mat& subMat, const Size& roiSize, const Border& border, int type, double minVal, double maxVal)
    {
        Size wholeSize = Size(roiSize.width + border.lef + border.rig, roiSize.height + border.top + border.bot);
        whole = randomMat(wholeSize, type, minVal, maxVal, false);
        subMat = whole(Rect(border.lef, border.top, roiSize.width, roiSize.height));
    }

    // If the two vectors are not equal, it will return the difference in vector size
    // Else it will return (total diff of each 1 and 2 rects covered pixels)/(total 1 rects covered pixels)
    // The smaller, the better matched
    static double checkRectSimilarity(cv::Size sz, std::vector<cv::Rect>& ob1, std::vector<cv::Rect>& ob2);

    //! read image from testdata folder.

    static cv::Mat readImage(const String &fileName, int flags = cv::IMREAD_COLOR);
    static cv::Mat readImageType(const String &fname, int type);

    static double checkNorm(const cv::Mat &m);
    static double checkNorm(const cv::Mat &m1, const cv::Mat &m2);
    static double checkSimilarity(const cv::Mat &m1, const cv::Mat &m2);
    static inline double checkNormRelative(const Mat &m1, const Mat &m2)
    {
        return cv::norm(m1, m2, cv::NORM_INF) /
                std::max((double)std::numeric_limits<float>::epsilon(),
                         (double)std::max(cv::norm(m1, cv::NORM_INF), norm(m2, cv::NORM_INF)));
    }
    static void showDiff(const Mat& src, const Mat& gold, const Mat& actual, double eps, bool alwaysShow = false);

    template <typename T1>
    static double checkNorm(const T1& m)
    {
        return checkNorm(getMatForRead(m));
    }
    template <typename T1, typename T2>
    static double checkNorm(const T1& m1, const T2& m2)
    {
        return checkNorm(getMatForRead(m1), getMatForRead(m2));
    }
    template <typename T1, typename T2>
    static double checkSimilarity(const T1& m1, const T2& m2)
    {
        return checkSimilarity(getMatForRead(m1), getMatForRead(m2));
    }
    template <typename T1, typename T2>
    static inline double checkNormRelative(const T1& m1, const T2& m2)
    {
        const Mat _m1 = getMatForRead(m1);
        const Mat _m2 = getMatForRead(m2);
        return checkNormRelative(_m1, _m2);
    }

    template <typename T1, typename T2, typename T3>
    static void showDiff(const T1& src, const T2& gold, const T3& actual, double eps, bool alwaysShow = false)
    {
        const Mat _src = getMatForRead(src);
        const Mat _gold = getMatForRead(gold);
        const Mat _actual = getMatForRead(actual);
        showDiff(_src, _gold, _actual, eps, alwaysShow);
    }
};

#define TEST_DECLARE_INPUT_PARAMETER(name) Mat name, name ## _roi; UMat u ## name, u ## name ## _roi;
#define TEST_DECLARE_OUTPUT_PARAMETER(name) TEST_DECLARE_INPUT_PARAMETER(name)

#define UMAT_UPLOAD_INPUT_PARAMETER(name) \
{ \
    name.copyTo(u ## name); \
    Size _wholeSize; Point ofs; name ## _roi.locateROI(_wholeSize, ofs); \
    u ## name ## _roi = u ## name(Rect(ofs.x, ofs.y, name ## _roi.size().width, name ## _roi.size().height)); \
}
#define UMAT_UPLOAD_OUTPUT_PARAMETER(name) UMAT_UPLOAD_INPUT_PARAMETER(name)

template <typename T>
struct CV_EXPORTS TSTestWithParam : public TestUtils, public ::testing::TestWithParam<T>
{

};

#define PARAM_TEST_CASE(name, ...) struct name : public TSTestWithParam< std::tr1::tuple< __VA_ARGS__ > >

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#ifndef IMPLEMENT_PARAM_CLASS
#define IMPLEMENT_PARAM_CLASS(name, type) \
    class name \
    { \
    public: \
        name ( type arg = type ()) : val_(arg) {} \
        operator type () const {return val_;} \
    private: \
        type val_; \
    }; \
    inline void PrintTo( name param, std::ostream* os) \
    { \
        *os << #name <<  "(" << testing::PrintToString(static_cast< type >(param)) << ")"; \
    }

IMPLEMENT_PARAM_CLASS(Channels, int)
#endif // IMPLEMENT_PARAM_CLASS

#define OCL_TEST_P TEST_P
#define OCL_TEST_F(name, ...) typedef name OCL_##name; TEST_F(OCL_##name, __VA_ARGS__)
#define OCL_TEST(name, ...) TEST(OCL_##name, __VA_ARGS__)

#define OCL_OFF(fn) cv::ocl::setUseOpenCL(false); fn
#define OCL_ON(fn) cv::ocl::setUseOpenCL(true); fn

#define OCL_ALL_DEPTHS Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
#define OCL_ALL_CHANNELS Values(1, 2, 3, 4)

CV_ENUM(Interpolation, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA)
CV_ENUM(ThreshOp, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)
CV_ENUM(BorderType, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101)

#define OCL_INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_CASE_P(OCL_ ## prefix, test_case_name, generator)

}} // namespace cvtest::ocl

#endif // HAVE_OPENCL

#endif // __OPENCV_TS_OCL_TEST_HPP__
