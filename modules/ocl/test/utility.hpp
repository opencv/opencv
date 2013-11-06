/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_TEST_UTILITY_HPP__
#define __OPENCV_TEST_UTILITY_HPP__
#include "opencv2/core.hpp"


extern int LOOP_TIMES;

#define MWIDTH 256
#define MHEIGHT 256

#define MIN_VALUE 171
#define MAX_VALUE 357

namespace cvtest {

void showDiff(const Mat& gold, const Mat& actual, double eps, bool alwaysShow = false);

cv::ocl::oclMat createMat_ocl(cv::RNG& rng, Size size, int type, bool useRoi);
cv::ocl::oclMat loadMat_ocl(cv::RNG& rng, const Mat& m, bool useRoi);

// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else it will return (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
// The smaller, the better matched
double checkRectSimilarity(cv::Size sz, std::vector<cv::Rect>& ob1, std::vector<cv::Rect>& ob2);


//! read image from testdata folder.
cv::Mat readImage(const std::string &fileName, int flags = cv::IMREAD_COLOR);
cv::Mat readImageType(const std::string &fname, int type);

double checkNorm(const cv::Mat &m);
double checkNorm(const cv::Mat &m1, const cv::Mat &m2);
double checkSimilarity(const cv::Mat &m1, const cv::Mat &m2);

inline double checkNormRelative(const Mat &m1, const Mat &m2)
{
    return cv::norm(m1, m2, cv::NORM_INF) /
            std::max((double)std::numeric_limits<float>::epsilon(),
                     (double)std::max(cv::norm(m1, cv::NORM_INF), norm(m2, cv::NORM_INF)));
}

#define EXPECT_MAT_NORM(mat, eps) \
{ \
    EXPECT_LE(checkNorm(cv::Mat(mat)), eps) \
}

#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(checkNorm(cv::Mat(mat1), cv::Mat(mat2)), eps) \
       << cv::format("Size: %d x %d", mat1.cols, mat1.rows) << std::endl; \
}

#define EXPECT_MAT_NEAR_RELATIVE(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(checkNormRelative(cv::Mat(mat1), cv::Mat(mat2)), eps) \
       << cv::format("Size: %d x %d", mat1.cols, mat1.rows) << std::endl; \
}

#define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
{ \
    ASSERT_EQ(mat1.type(), mat2.type()); \
    ASSERT_EQ(mat1.size(), mat2.size()); \
    EXPECT_LE(checkSimilarity(cv::Mat(mat1), cv::Mat(mat2)), eps); \
}


using perf::MatDepth;
using perf::MatType;

//! return vector with types from specified range.
std::vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end);

//! return vector with all types (depth: CV_8U-CV_64F, channels: 1-4).
const std::vector<MatType> &all_types();

class Inverse
{
public:
    inline Inverse(bool val = false) : val_(val) {}

    inline operator bool() const
    {
        return val_;
    }

private:
    bool val_;
};

void PrintTo(const Inverse &useRoi, std::ostream *os);

#define OCL_RNG_SEED 123456

template <typename T>
struct TSTestWithParam : public ::testing::TestWithParam<T>
{
    cv::RNG rng;

    TSTestWithParam()
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
        return cv::Size(randomDoubleLog(minValX, maxValX), randomDoubleLog(minValY, maxValY));
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

    void generateOclMat(cv::ocl::oclMat& whole, cv::ocl::oclMat& subMat, const Mat& wholeMat, const Size& roiSize, const Border& border)
    {
        whole = wholeMat;
        subMat = whole(Rect(border.lef, border.top, roiSize.width, roiSize.height));
    }
};

#define PARAM_TEST_CASE(name, ...) struct name : public TSTestWithParam< std::tr1::tuple< __VA_ARGS__ > >

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#define ALL_TYPES testing::ValuesIn(all_types())
#define TYPES(depth_start, depth_end, cn_start, cn_end) testing::ValuesIn(types(depth_start, depth_end, cn_start, cn_end))

#define DIFFERENT_SIZES testing::Values(cv::Size(128, 128), cv::Size(113, 113), cv::Size(1300, 1300))

#define IMAGE_CHANNELS testing::Values(Channels(1), Channels(3), Channels(4))
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

} // namespace cvtest

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)

CV_ENUM(CmpCode, CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE)
CV_ENUM(NormCode, NORM_INF, NORM_L1, NORM_L2, NORM_TYPE_MASK, NORM_RELATIVE, NORM_MINMAX)
CV_ENUM(ReduceOp, REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_MIN)
CV_ENUM(MorphOp, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT)
CV_ENUM(ThreshOp, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)
CV_ENUM(Interpolation, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC)
CV_ENUM(Border, BORDER_REFLECT101, BORDER_REPLICATE, BORDER_CONSTANT, BORDER_REFLECT, BORDER_WRAP)
CV_ENUM(TemplateMethod, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

CV_FLAGS(GemmFlags, GEMM_1_T, GEMM_2_T, GEMM_3_T);
CV_FLAGS(WarpFlags, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, WARP_INVERSE_MAP)
CV_FLAGS(DftFlags, DFT_INVERSE, DFT_SCALE, DFT_ROWS, DFT_COMPLEX_OUTPUT, DFT_REAL_OUTPUT)

# define OCL_TEST_P(test_case_name, test_name) \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : \
        public test_case_name { \
    public: \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() { } \
        virtual void TestBody(); \
        void OCLTestBody(); \
    private: \
        static int AddToRegistry() \
        { \
            ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
              GetTestCasePatternHolder<test_case_name>(\
                  #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                      #test_case_name, \
                      #test_name, \
                      new ::testing::internal::TestMetaFactory< \
                          GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
            return 0; \
        } \
    \
        static int gtest_registering_dummy_; \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
    }; \
    \
    int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
    \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() \
    { \
        try \
        { \
            OCLTestBody(); \
        } \
        catch (const cv::Exception & ex) \
        { \
            if (ex.code == cv::Error::OpenCLDoubleNotSupported)\
                std::cout << "Test skipped (selected device does not support double)" << std::endl; \
            else if (ex.code == cv::Error::OpenCLNoAMDBlasFft) \
                std::cout << "Test skipped (AMD Blas / Fft libraries are not available)" << std::endl; \
            else \
                throw; \
        } \
    } \
    \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::OCLTestBody()

#endif // __OPENCV_TEST_UTILITY_HPP__
