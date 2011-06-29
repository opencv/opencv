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

#ifndef __OPENCV_TEST_GPU_BASE_HPP__
#define __OPENCV_TEST_GPU_BASE_HPP__

//! return true if device supports specified feature and gpu module was built with support the feature.
bool supportFeature(const cv::gpu::DeviceInfo& info, cv::gpu::FeatureSet feature);

//! return all devices compatible with current gpu module build.
const std::vector<cv::gpu::DeviceInfo>& devices();
//! return all devices compatible with current gpu module build which support specified feature.
std::vector<cv::gpu::DeviceInfo> devices(cv::gpu::FeatureSet feature);

//! return vector with types from specified range.
std::vector<int> types(int depth_start, int depth_end, int cn_start, int cn_end);

//! return vector with all types (depth: CV_8U-CV_64F, channels: 1-4).
const std::vector<int>& all_types();

//! read image from testdata folder.
cv::Mat readImage(const std::string& fileName, int flags = CV_LOAD_IMAGE_COLOR);

double checkNorm(const cv::Mat& m1, const cv::Mat& m2);
double checkSimilarity(const cv::Mat& m1, const cv::Mat& m2);

#define OSTR_NAME(suf) ostr_ ## suf

#define PRINT_PARAM(name) \
        std::ostringstream OSTR_NAME(name); \
        OSTR_NAME(name) << # name << ": " << name; \
        SCOPED_TRACE(OSTR_NAME(name).str());

#define PRINT_TYPE(type) \
        std::ostringstream OSTR_NAME(type); \
        OSTR_NAME(type) << # type << ": " << cvtest::getTypeName(type) << "c" << CV_MAT_CN(type); \
        SCOPED_TRACE(OSTR_NAME(type).str());

#define EXPECT_MAT_NEAR(mat1, mat2, eps) \
    { \
        ASSERT_EQ(mat1.type(), mat2.type()); \
        ASSERT_EQ(mat1.size(), mat2.size()); \
        EXPECT_LE(checkNorm(mat1, mat2), eps); \
    }

#define EXPECT_MAT_SIMILAR(mat1, mat2, eps) \
    { \
        ASSERT_EQ(mat1.type(), mat2.type()); \
        ASSERT_EQ(mat1.size(), mat2.size()); \
        EXPECT_LE(checkSimilarity(mat1, mat2), eps); \
    }


//! for gtest ASSERT
namespace cv
{
    std::ostream& operator << (std::ostream& os, const Size& sz);
    std::ostream& operator << (std::ostream& os, const Scalar& s);
    namespace gpu
    {
        std::ostream& operator << (std::ostream& os, const DeviceInfo& info);
    }
}

#endif // __OPENCV_TEST_GPU_BASE_HPP__
