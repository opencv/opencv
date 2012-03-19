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

int randomInt(int minVal, int maxVal);
double randomDouble(double minVal, double maxVal);
cv::Size randomSize(int minVal, int maxVal);
cv::Scalar randomScalar(double minVal, double maxVal);
cv::Mat randomMat(cv::Size size, int type, double minVal = 0.0, double maxVal = 255.0);

cv::gpu::GpuMat createMat(cv::Size size, int type, bool useRoi = false);
cv::gpu::GpuMat loadMat(const cv::Mat& m, bool useRoi = false);

void showDiff(cv::InputArray gold, cv::InputArray actual, double eps);

//! return true if device supports specified feature and gpu module was built with support the feature.
bool supportFeature(const cv::gpu::DeviceInfo& info, cv::gpu::FeatureSet feature);

//! return all devices compatible with current gpu module build.
const std::vector<cv::gpu::DeviceInfo>& devices();
//! return all devices compatible with current gpu module build which support specified feature.
std::vector<cv::gpu::DeviceInfo> devices(cv::gpu::FeatureSet feature);

//! read image from testdata folder.
cv::Mat readImage(const std::string& fileName, int flags = cv::IMREAD_COLOR);
cv::Mat readImageType(const std::string& fname, int type);

double checkNorm(const cv::Mat& m);
double checkNorm(const cv::Mat& m1, const cv::Mat& m2);
double checkSimilarity(const cv::Mat& m1, const cv::Mat& m2);

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

namespace cv { namespace gpu
{
    void PrintTo(const DeviceInfo& info, std::ostream* os);
}}

using perf::MatDepth;
using perf::MatType;

//! return vector with types from specified range.
std::vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end);

//! return vector with all types (depth: CV_8U-CV_64F, channels: 1-4).
const std::vector<MatType>& all_types();

class UseRoi
{
public:
    inline UseRoi(bool val = false) : val_(val) {}

    inline operator bool() const { return val_; }

private:
    bool val_;
};

void PrintTo(const UseRoi& useRoi, std::ostream* os);

class Inverse
{
public:
    inline Inverse(bool val = false) : val_(val) {}

    inline operator bool() const { return val_; }

private:
    bool val_;
};

void PrintTo(const Inverse& useRoi, std::ostream* os);

CV_ENUM(CmpCode, cv::CMP_EQ, cv::CMP_GT, cv::CMP_GE, cv::CMP_LT, cv::CMP_LE, cv::CMP_NE)

CV_ENUM(NormCode, cv::NORM_INF, cv::NORM_L1, cv::NORM_L2, cv::NORM_TYPE_MASK, cv::NORM_RELATIVE, cv::NORM_MINMAX)

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)

CV_ENUM(ReduceOp, CV_REDUCE_SUM, CV_REDUCE_AVG, CV_REDUCE_MAX, CV_REDUCE_MIN)

CV_FLAGS(GemmFlags, cv::GEMM_1_T, cv::GEMM_2_T, cv::GEMM_3_T);

CV_ENUM(DistType, cv::gpu::BruteForceMatcher_GPU_base::L1Dist, cv::gpu::BruteForceMatcher_GPU_base::L2Dist)

CV_ENUM(MorphOp, cv::MORPH_OPEN, cv::MORPH_CLOSE, cv::MORPH_GRADIENT, cv::MORPH_TOPHAT, cv::MORPH_BLACKHAT)

CV_ENUM(ThreshOp, cv::THRESH_BINARY, cv::THRESH_BINARY_INV, cv::THRESH_TRUNC, cv::THRESH_TOZERO, cv::THRESH_TOZERO_INV)

CV_ENUM(Interpolation, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC)

CV_ENUM(Border, cv::BORDER_REFLECT101, cv::BORDER_REPLICATE, cv::BORDER_CONSTANT, cv::BORDER_REFLECT, cv::BORDER_WRAP)

CV_FLAGS(WarpFlags, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::WARP_INVERSE_MAP)

CV_ENUM(TemplateMethod, cv::TM_SQDIFF, cv::TM_SQDIFF_NORMED, cv::TM_CCORR, cv::TM_CCORR_NORMED, cv::TM_CCOEFF, cv::TM_CCOEFF_NORMED)

CV_FLAGS(DftFlags, cv::DFT_INVERSE, cv::DFT_SCALE, cv::DFT_ROWS, cv::DFT_COMPLEX_OUTPUT, cv::DFT_REAL_OUTPUT)

#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#define ALL_DEVICES testing::ValuesIn(devices())
#define DEVICES(feature) testing::ValuesIn(devices(feature))

#define ALL_TYPES testing::ValuesIn(all_types())
#define TYPES(depth_start, depth_end, cn_start, cn_end) testing::ValuesIn(types(depth_start, depth_end, cn_start, cn_end))

#define DIFFERENT_SIZES testing::Values(cv::Size(128, 128), cv::Size(113, 113))

#define WHOLE testing::Values(UseRoi(false))
#define SUBMAT testing::Values(UseRoi(true))
#define WHOLE_SUBMAT testing::Values(UseRoi(false), UseRoi(true))

#define DIRECT_INVERSE testing::Values(Inverse(false), Inverse(true))

#endif // __OPENCV_TEST_UTILITY_HPP__
