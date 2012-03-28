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

#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/ts/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"

//////////////////////////////////////////////////////////////////////
// random generators

int randomInt(int minVal, int maxVal);
double randomDouble(double minVal, double maxVal);
cv::Size randomSize(int minVal, int maxVal);
cv::Scalar randomScalar(double minVal, double maxVal);
cv::Mat randomMat(cv::Size size, int type, double minVal = 0.0, double maxVal = 255.0);

//////////////////////////////////////////////////////////////////////
// GpuMat create

cv::gpu::GpuMat createMat(cv::Size size, int type, bool useRoi = false);
cv::gpu::GpuMat loadMat(const cv::Mat& m, bool useRoi = false);

//////////////////////////////////////////////////////////////////////
// Image load

//! read image from testdata folder
cv::Mat readImage(const std::string& fileName, int flags = cv::IMREAD_COLOR);

//! read image from testdata folder and convert it to specified type
cv::Mat readImageType(const std::string& fname, int type);

//////////////////////////////////////////////////////////////////////
// Gpu devices

//! return true if device supports specified feature and gpu module was built with support the feature.
bool supportFeature(const cv::gpu::DeviceInfo& info, cv::gpu::FeatureSet feature);

//! return all devices compatible with current gpu module build.
const std::vector<cv::gpu::DeviceInfo>& devices();

//! return all devices compatible with current gpu module build which support specified feature.
std::vector<cv::gpu::DeviceInfo> devices(cv::gpu::FeatureSet feature);

#define ALL_DEVICES testing::ValuesIn(devices())
#define DEVICES(feature) testing::ValuesIn(devices(feature))

//////////////////////////////////////////////////////////////////////
// Additional assertion

cv::Mat getMat(cv::InputArray arr);

double checkNorm(cv::InputArray m1, cv::InputArray m2);

void minMaxLocGold(const cv::Mat& src, double* minVal_, double* maxVal_ = 0, cv::Point* minLoc_ = 0, cv::Point* maxLoc_ = 0, const cv::Mat& mask = cv::Mat());

testing::AssertionResult assertMatNear(const char* expr1, const char* expr2, const char* eps_expr, cv::InputArray m1, cv::InputArray m2, double eps);

#define EXPECT_MAT_NEAR(m1, m2, eps) EXPECT_PRED_FORMAT3(assertMatNear, m1, m2, eps)
#define ASSERT_MAT_NEAR(m1, m2, eps) ASSERT_PRED_FORMAT3(assertMatNear, m1, m2, eps)

#define EXPECT_SCALAR_NEAR(s1, s2, eps) \
    { \
        EXPECT_NEAR(s1[0], s2[0], eps); \
        EXPECT_NEAR(s1[1], s2[1], eps); \
        EXPECT_NEAR(s1[2], s2[2], eps); \
        EXPECT_NEAR(s1[3], s2[3], eps); \
    }
#define ASSERT_SCALAR_NEAR(s1, s2, eps) \
    { \
        ASSERT_NEAR(s1[0], s2[0], eps); \
        ASSERT_NEAR(s1[1], s2[1], eps); \
        ASSERT_NEAR(s1[2], s2[2], eps); \
        ASSERT_NEAR(s1[3], s2[3], eps); \
    }

#define EXPECT_POINT2_NEAR(p1, p2, eps) \
    { \
        EXPECT_NEAR(p1.x, p2.x, eps); \
        EXPECT_NEAR(p1.y, p2.y, eps); \
    }
#define ASSERT_POINT2_NEAR(p1, p2, eps) \
    { \
        ASSERT_NEAR(p1.x, p2.x, eps); \
        ASSERT_NEAR(p1.y, p2.y, eps); \
    }

#define EXPECT_POINT3_NEAR(p1, p2, eps) \
    { \
        EXPECT_NEAR(p1.x, p2.x, eps); \
        EXPECT_NEAR(p1.y, p2.y, eps); \
        EXPECT_NEAR(p1.z, p2.z, eps); \
    }
#define ASSERT_POINT3_NEAR(p1, p2, eps) \
    { \
        ASSERT_NEAR(p1.x, p2.x, eps); \
        ASSERT_NEAR(p1.y, p2.y, eps); \
        ASSERT_NEAR(p1.z, p2.z, eps); \
    }

double checkSimilarity(cv::InputArray m1, cv::InputArray m2);

#define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
    { \
        ASSERT_EQ(mat1.type(), mat2.type()); \
        ASSERT_EQ(mat1.size(), mat2.size()); \
        EXPECT_LE(checkSimilarity(mat1, mat2), eps); \
    }
#define ASSERT_MAT_SIMILAR(mat1, mat2, eps) \
    { \
        ASSERT_EQ(mat1.type(), mat2.type()); \
        ASSERT_EQ(mat1.size(), mat2.size()); \
        ASSERT_LE(checkSimilarity(mat1, mat2), eps); \
    }

//////////////////////////////////////////////////////////////////////
// Helper structs for value-parameterized tests

#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >
#define GET_PARAM(k) std::tr1::get< k >(GetParam())

namespace cv { namespace gpu
{
    void PrintTo(const DeviceInfo& info, std::ostream* os);
}}

#define DIFFERENT_SIZES testing::Values(cv::Size(128, 128), cv::Size(113, 113))

// Depth

using perf::MatDepth;

//! return vector with depths from specified range.
std::vector<MatDepth> depths(int depth_start, int depth_end);

#define ALL_DEPTH testing::Values(MatDepth(CV_8U), MatDepth(CV_8S), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32S), MatDepth(CV_32F), MatDepth(CV_64F))
#define DEPTHS(depth_start, depth_end) testing::ValuesIn(depths(depth_start, depth_end))
#define DEPTH_PAIRS testing::Values(std::make_pair(MatDepth(CV_8U), MatDepth(CV_8U)),   \
                                    std::make_pair(MatDepth(CV_8U), MatDepth(CV_16U)),  \
                                    std::make_pair(MatDepth(CV_8U), MatDepth(CV_16S)),  \
                                    std::make_pair(MatDepth(CV_8U), MatDepth(CV_32S)),  \
                                    std::make_pair(MatDepth(CV_8U), MatDepth(CV_32F)),  \
                                    std::make_pair(MatDepth(CV_8U), MatDepth(CV_64F)),  \
                                                                                        \
                                    std::make_pair(MatDepth(CV_16U), MatDepth(CV_16U)), \
                                    std::make_pair(MatDepth(CV_16U), MatDepth(CV_32S)), \
                                    std::make_pair(MatDepth(CV_16U), MatDepth(CV_32F)), \
                                    std::make_pair(MatDepth(CV_16U), MatDepth(CV_64F)), \
                                                                                        \
                                    std::make_pair(MatDepth(CV_16S), MatDepth(CV_16S)), \
                                    std::make_pair(MatDepth(CV_16S), MatDepth(CV_32S)), \
                                    std::make_pair(MatDepth(CV_16S), MatDepth(CV_32F)), \
                                    std::make_pair(MatDepth(CV_16S), MatDepth(CV_64F)), \
                                                                                        \
                                    std::make_pair(MatDepth(CV_32S), MatDepth(CV_32S)), \
                                    std::make_pair(MatDepth(CV_32S), MatDepth(CV_32F)), \
                                    std::make_pair(MatDepth(CV_32S), MatDepth(CV_64F)), \
                                                                                        \
                                    std::make_pair(MatDepth(CV_32F), MatDepth(CV_32F)), \
                                    std::make_pair(MatDepth(CV_32F), MatDepth(CV_64F)), \
                                                                                        \
                                    std::make_pair(MatDepth(CV_64F), MatDepth(CV_64F)))

// Type

using perf::MatType;

//! return vector with types from specified range.
std::vector<MatType> types(int depth_start, int depth_end, int cn_start, int cn_end);

//! return vector with all types (depth: CV_8U-CV_64F, channels: 1-4).
const std::vector<MatType>& all_types();

#define ALL_TYPES testing::ValuesIn(all_types())
#define TYPES(depth_start, depth_end, cn_start, cn_end) testing::ValuesIn(types(depth_start, depth_end, cn_start, cn_end))

// ROI

class UseRoi
{
public:
    inline UseRoi(bool val = false) : val_(val) {}

    inline operator bool() const { return val_; }

private:
    bool val_;
};

void PrintTo(const UseRoi& useRoi, std::ostream* os);

#define WHOLE testing::Values(UseRoi(false))
#define SUBMAT testing::Values(UseRoi(true))
#define WHOLE_SUBMAT testing::Values(UseRoi(false), UseRoi(true))

// Direct/Inverse

class Inverse
{
public:
    inline Inverse(bool val = false) : val_(val) {}

    inline operator bool() const { return val_; }

private:
    bool val_;
};
void PrintTo(const Inverse& useRoi, std::ostream* os);
#define DIRECT_INVERSE testing::Values(Inverse(false), Inverse(true))

// Param class

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

#define ALL_CHANNELS testing::Values(Channels(1), Channels(2), Channels(3), Channels(4))
#define IMAGE_CHANNELS testing::Values(Channels(1), Channels(3), Channels(4))

// Flags and enums

CV_ENUM(NormCode, cv::NORM_INF, cv::NORM_L1, cv::NORM_L2, cv::NORM_TYPE_MASK, cv::NORM_RELATIVE, cv::NORM_MINMAX)

CV_ENUM(Interpolation, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC)

CV_ENUM(BorderType, cv::BORDER_REFLECT101, cv::BORDER_REPLICATE, cv::BORDER_CONSTANT, cv::BORDER_REFLECT, cv::BORDER_WRAP)
#define ALL_BORDER_TYPES testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT), BorderType(cv::BORDER_WRAP))

CV_FLAGS(WarpFlags, cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::WARP_INVERSE_MAP)

//////////////////////////////////////////////////////////////////////
// Other

void showDiff(cv::InputArray gold, cv::InputArray actual, double eps);

#endif // __OPENCV_TEST_UTILITY_HPP__
