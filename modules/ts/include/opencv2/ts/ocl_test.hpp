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

#ifndef OPENCV_TS_OCL_TEST_HPP
#define OPENCV_TS_OCL_TEST_HPP

#include "opencv2/ts.hpp"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/ocl.hpp"

namespace cvtest {
namespace ocl {

using namespace cv;
using namespace testing;

inline std::vector<UMat> ToUMat(const std::vector<Mat>& src)
{
    std::vector<UMat> dst;
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        src[i].copyTo(dst[i]);
    }
    return dst;
}

inline UMat ToUMat(const Mat& src)
{
    UMat dst;
    src.copyTo(dst);
    return dst;
}

inline UMat ToUMat(InputArray src)
{
    UMat dst;
    src.getMat().copyTo(dst);
    return dst;
}

extern int test_loop_times;

#define MAX_VALUE 357

#define EXPECT_MAT_NORM(mat, eps) \
do \
{ \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNorm1(mat), eps) \
} while ((void)0, 0)

#undef EXPECT_MAT_NEAR
#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
do \
{ \
    ASSERT_EQ(mat1.type(), mat2.type()); \
    ASSERT_EQ(mat1.size(), mat2.size()); \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNorm2(mat1, mat2), eps) \
        << "Size: " << mat1.size() << std::endl; \
} while ((void)0, 0)

#define EXPECT_MAT_NEAR_RELATIVE(mat1, mat2, eps) \
do \
{ \
    ASSERT_EQ((mat1).type(), (mat2).type()); \
    ASSERT_EQ((mat1).size(), (mat2).size()); \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNormRelative((mat1), (mat2)), eps) \
        << "Size: " << (mat1).size() << std::endl; \
} while ((void)0, 0)

#define EXPECT_MAT_N_DIFF(mat1, mat2, num) \
do \
{ \
    ASSERT_EQ(mat1.type(), mat2.type()); \
    ASSERT_EQ(mat1.size(), mat2.size()); \
    Mat diff; \
    absdiff(mat1, mat2, diff); \
    EXPECT_LE(countNonZero(diff.reshape(1)), num) \
    << "Size: " << mat1.size() << std::endl; \
} while ((void)0, 0)

#define OCL_EXPECT_MAT_N_DIFF(name, eps) \
do \
{ \
    ASSERT_EQ(name ## _roi.type(), u ## name ## _roi.type()); \
    ASSERT_EQ(name ## _roi.size(), u ## name ## _roi.size()); \
    Mat diff, binary, binary_8; \
    absdiff(name ## _roi, u ## name ## _roi, diff); \
    Mat mask(diff.size(), CV_8UC(dst.channels()), cv::Scalar::all(255)); \
    if (mask.cols > 2 && mask.rows > 2) \
        mask(cv::Rect(1, 1, mask.cols - 2, mask.rows - 2)).setTo(0); \
    cv::threshold(diff, binary, (double)eps, 255, cv::THRESH_BINARY); \
    EXPECT_LE(countNonZero(binary.reshape(1)), (int)(binary.cols*binary.rows*5/1000)) \
        << "Size: " << name ## _roi.size() << std::endl; \
    binary.convertTo(binary_8, mask.type()); \
    binary_8 = binary_8 & mask; \
    EXPECT_LE(countNonZero(binary_8.reshape(1)), (int)((binary_8.cols+binary_8.rows)/100)) \
        << "Size: " << name ## _roi.size() << std::endl; \
} while ((void)0, 0)

#define OCL_EXPECT_MATS_NEAR(name, eps) \
do \
{ \
    ASSERT_EQ(name ## _roi.type(), u ## name ## _roi.type()); \
    ASSERT_EQ(name ## _roi.size(), u ## name ## _roi.size()); \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNorm2(name ## _roi, u ## name ## _roi), eps) \
        << "Size: " << name ## _roi.size() << std::endl; \
    Point _offset; \
    Size _wholeSize; \
    u ## name ## _roi.locateROI(_wholeSize, _offset); \
    Mat _mask(name.size(), CV_8UC1, Scalar::all(255)); \
    _mask(Rect(_offset, name ## _roi.size())).setTo(Scalar::all(0)); \
    ASSERT_EQ(name.type(), u ## name.type()); \
    ASSERT_EQ(name.size(), u ## name.size()); \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNorm2(name, u ## name, _mask), eps) \
        << "Size: " << name ## _roi.size() << std::endl; \
} while ((void)0, 0)

#define OCL_EXPECT_MATS_NEAR_RELATIVE(name, eps) \
do \
{ \
    ASSERT_EQ(name ## _roi.type(), u ## name ## _roi.type()); \
    ASSERT_EQ(name ## _roi.size(), u ## name ## _roi.size()); \
    EXPECT_LE(TestUtils::checkNormRelative(name ## _roi, u ## name ## _roi), eps) \
        << "Size: " << name ## _roi.size() << std::endl; \
    Point _offset; \
    Size _wholeSize; \
    name ## _roi.locateROI(_wholeSize, _offset); \
    Mat _mask(name.size(), CV_8UC1, Scalar::all(255)); \
    _mask(Rect(_offset, name ## _roi.size())).setTo(Scalar::all(0)); \
    ASSERT_EQ(name.type(), u ## name.type()); \
    ASSERT_EQ(name.size(), u ## name.size()); \
    EXPECT_LE(TestUtils::checkNormRelative(name, u ## name, _mask), eps) \
        << "Size: " << name ## _roi.size() << std::endl; \
} while ((void)0, 0)

//for sparse matrix
#define OCL_EXPECT_MATS_NEAR_RELATIVE_SPARSE(name, eps) \
do \
{ \
    ASSERT_EQ(name ## _roi.type(), u ## name ## _roi.type()); \
    ASSERT_EQ(name ## _roi.size(), u ## name ## _roi.size()); \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNormRelativeSparse(name ## _roi, u ## name ## _roi), eps) \
        << "Size: " << name ## _roi.size() << std::endl; \
    Point _offset; \
    Size _wholeSize; \
    name ## _roi.locateROI(_wholeSize, _offset); \
    Mat _mask(name.size(), CV_8UC1, Scalar::all(255)); \
    _mask(Rect(_offset, name ## _roi.size())).setTo(Scalar::all(0)); \
    ASSERT_EQ(name.type(), u ## name.type()); \
    ASSERT_EQ(name.size(), u ## name.size()); \
    EXPECT_LE(cvtest::ocl::TestUtils::checkNormRelativeSparse(name, u ## name, _mask), eps) \
        << "Size: " << name ## _roi.size() << std::endl; \
} while ((void)0, 0)

#undef EXPECT_MAT_SIMILAR
#define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
do \
{ \
    ASSERT_EQ(mat1.type(), mat2.type()); \
    ASSERT_EQ(mat1.size(), mat2.size()); \
    EXPECT_LE(checkSimilarity(mat1, mat2), eps) \
        << "Size: " << mat1.size() << std::endl; \
} while ((void)0, 0)

using perf::MatDepth;
using perf::MatType;

#define OCL_RNG_SEED 123456

struct TestUtils
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
    static double checkRectSimilarity(const cv::Size & sz, std::vector<cv::Rect>& ob1, std::vector<cv::Rect>& ob2);

    //! read image from testdata folder.
    static cv::Mat readImage(const String &fileName, int flags = cv::IMREAD_COLOR);
    static cv::Mat readImageType(const String &fname, int type);

    static double checkNorm1(InputArray m, InputArray mask = noArray());
    static double checkNorm2(InputArray m1, InputArray m2, InputArray mask = noArray());
    static double checkSimilarity(InputArray m1, InputArray m2);
    static void showDiff(InputArray _src, InputArray _gold, InputArray _actual, double eps, bool alwaysShow);

    static inline double checkNormRelative(InputArray m1, InputArray m2, InputArray mask = noArray())
    {
        return cvtest::norm(m1.getMat(), m2.getMat(), cv::NORM_INF, mask) /
                std::max((double)std::numeric_limits<float>::epsilon(),
                         (double)std::max(cvtest::norm(m1.getMat(), cv::NORM_INF), cvtest::norm(m2.getMat(), cv::NORM_INF)));
    }

    static inline double checkNormRelativeSparse(InputArray m1, InputArray m2, InputArray mask = noArray())
    {
        double norm_inf = cvtest::norm(m1.getMat(), m2.getMat(), cv::NORM_INF, mask);
        double norm_rel = norm_inf /
                std::max((double)std::numeric_limits<float>::epsilon(),
                         (double)std::max(cvtest::norm(m1.getMat(), cv::NORM_INF), cvtest::norm(m2.getMat(), cv::NORM_INF)));
        return std::min(norm_inf, norm_rel);
    }

};

#define TEST_DECLARE_INPUT_PARAMETER(name) Mat name, name ## _roi; UMat u ## name, u ## name ## _roi
#define TEST_DECLARE_OUTPUT_PARAMETER(name) TEST_DECLARE_INPUT_PARAMETER(name)

#define UMAT_UPLOAD_INPUT_PARAMETER(name) \
do \
{ \
    name.copyTo(u ## name); \
    Size _wholeSize; Point ofs; name ## _roi.locateROI(_wholeSize, ofs); \
    u ## name ## _roi = u ## name(Rect(ofs.x, ofs.y, name ## _roi.size().width, name ## _roi.size().height)); \
} while ((void)0, 0)

#define UMAT_UPLOAD_OUTPUT_PARAMETER(name) UMAT_UPLOAD_INPUT_PARAMETER(name)

template <typename T>
struct TSTestWithParam : public TestUtils, public ::testing::TestWithParam<T>
{

};

#undef PARAM_TEST_CASE
#define PARAM_TEST_CASE(name, ...) struct name : public ::cvtest::ocl::TSTestWithParam< testing::tuple< __VA_ARGS__ > >

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

#define OCL_OFF(...) cv::ocl::setUseOpenCL(false); __VA_ARGS__ ;
#define OCL_ON(...) cv::ocl::setUseOpenCL(true); __VA_ARGS__ ;

#define OCL_ALL_DEPTHS Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)
#define OCL_ABSOLUTELY_ALL_DEPTHS Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32U, CV_64S, CV_64U, CV_16F, CV_16BF, CV_32F, CV_64F)
//, CV_16F, CV_16BF, CV_64U, CV_64S, CV_32U)
#define OCL_ALL_DEPTHS_16F Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, CV_16F)
#define OCL_ALL_CHANNELS Values(1, 2, 3, 4)

CV_ENUM(Interpolation, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LINEAR_EXACT)
CV_ENUM(ThreshOp, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)
CV_ENUM(BorderType, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101)

#define OCL_INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator) \
    INSTANTIATE_TEST_CASE_P(OCL_ ## prefix, test_case_name, generator)

} } // namespace cvtest::ocl

namespace opencv_test {
namespace ocl {
using namespace cvtest::ocl;
}} // namespace

#endif // OPENCV_TS_OCL_TEST_HPP
