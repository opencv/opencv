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


#define LOOP_TIMES 1

#define MWIDTH 256
#define MHEIGHT 256

#define MIN_VALUE 171
#define MAX_VALUE 357

//#define RANDOMROI
int randomInt(int minVal, int maxVal);
double randomDouble(double minVal, double maxVal);
//std::string generateVarList(int first,...);
std::string generateVarList(int &p1, int &p2);
cv::Size randomSize(int minVal, int maxVal);
cv::Scalar randomScalar(double minVal, double maxVal);
cv::Mat randomMat(cv::Size size, int type, double minVal = 0.0, double maxVal = 255.0);

void showDiff(cv::InputArray gold, cv::InputArray actual, double eps);

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

//oclMat create
cv::ocl::oclMat createMat_ocl(cv::Size size, int type, bool useRoi = false);
cv::ocl::oclMat loadMat_ocl(const cv::Mat& m, bool useRoi = false);

#define EXPECT_MAT_NORM(mat, eps) \
{ \
    EXPECT_LE(checkNorm(cv::Mat(mat)), eps) \
}

#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
{ \
   ASSERT_EQ(mat1.type(), mat2.type()); \
   ASSERT_EQ(mat1.size(), mat2.size()); \
   EXPECT_LE(checkNorm(cv::Mat(mat1), cv::Mat(mat2)), eps); \
}

#define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
{ \
    ASSERT_EQ(mat1.type(), mat2.type()); \
    ASSERT_EQ(mat1.size(), mat2.size()); \
    EXPECT_LE(checkSimilarity(cv::Mat(mat1), cv::Mat(mat2)), eps); \
}

namespace cv
{
    namespace ocl
    {
        // void PrintTo(const DeviceInfo& info, std::ostream* os);
    }
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

void  run_perf_test();

#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#define ALL_DEVICES testing::ValuesIn(devices())
#define DEVICES(feature) testing::ValuesIn(devices(feature))

#define ALL_TYPES testing::ValuesIn(all_types())
#define TYPES(depth_start, depth_end, cn_start, cn_end) testing::ValuesIn(types(depth_start, depth_end, cn_start, cn_end))

#define DIFFERENT_SIZES testing::Values(cv::Size(128, 128), cv::Size(113, 113), cv::Size(1300, 1300))

#define DIRECT_INVERSE testing::Values(Inverse(false), Inverse(true))

#ifndef ALL_DEPTH
#define ALL_DEPTH testing::Values(MatDepth(CV_8U), MatDepth(CV_8S), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32S), MatDepth(CV_32F), MatDepth(CV_64F))
#endif
#define REPEAT   1000
#define COUNT_U  0 // count the uploading execution time for ocl mat structures
#define COUNT_D  0
// the following macro section tests the target function (kernel) performance
// upload is the code snippet for converting cv::mat to cv::ocl::oclMat
// downloading is the code snippet for converting cv::ocl::oclMat back to cv::mat
// change COUNT_U and COUNT_D to take downloading and uploading time into account
#define P_TEST_FULL( upload, kernel_call, download ) \
{ \
    std::cout<< "\n" #kernel_call "\n----------------------"; \
    {upload;} \
    R_TEST( kernel_call, 2 ); \
    double t = (double)cvGetTickCount(); \
    R_T( { \
            if( COUNT_U ) {upload;} \
            kernel_call; \
            if( COUNT_D ) {download;} \
            } ); \
    t = (double)cvGetTickCount() - t; \
    std::cout << "runtime is  " << t/((double)cvGetTickFrequency()* 1000.) << "ms" << std::endl; \
}

#define R_T2( test ) \
{ \
    std::cout<< "\n" #test "\n----------------------"; \
    R_TEST( test, 15 ) \
    clock_t st = clock(); \
    R_T( test ) \
    std::cout<< clock() - st << "ms\n"; \
}
#define R_T( test ) \
    R_TEST( test, REPEAT )
#define R_TEST( test, repeat ) \
    try{ \
        for( int i = 0; i < repeat; i ++ ) { test; } \
    } catch( ... ) { std::cout << "||||| Exception catched! |||||\n"; return; }

//////// Utility

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

#endif // __OPENCV_TEST_UTILITY_HPP__
