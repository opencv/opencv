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

#include "precomp.hpp"

#ifdef HAVE_CUDA

using namespace cvtest;
using namespace testing;

enum OutputLevel
{
    OutputLevelNone,
    OutputLevelCompact,
    OutputLevelFull
};

bool nvidia_NPPST_Integral_Image(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Squared_Integral_Image(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_RectStdDev(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Resize(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Vector_Operations(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NPPST_Transpose(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Vector_Operations(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Haar_Cascade_Loader(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Haar_Cascade_Application(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Hypotheses_Filtration(const std::string& test_data_path, OutputLevel outputLevel);
bool nvidia_NCV_Visualization(const std::string& test_data_path, OutputLevel outputLevel);

struct NVidiaTest : TestWithParam<cv::gpu::DeviceInfo>
{
    cv::gpu::DeviceInfo devInfo;

    std::string path;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::gpu::setDevice(devInfo.deviceID());

        path = std::string(TS::ptr()->get_data_path()) + "haarcascade/";
    }
};

struct NPPST : NVidiaTest {};
struct NCV : NVidiaTest {};

OutputLevel nvidiaTestOutputLevel = OutputLevelCompact;

TEST_P(NPPST, Integral)
{
    bool res = nvidia_NPPST_Integral_Image(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NPPST, SquaredIntegral)
{
    bool res = nvidia_NPPST_Squared_Integral_Image(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NPPST, RectStdDev)
{
    bool res = nvidia_NPPST_RectStdDev(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NPPST, Resize)
{
    bool res = nvidia_NPPST_Resize(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NPPST, VectorOperations)
{
    bool res = nvidia_NPPST_Vector_Operations(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NPPST, Transpose)
{
    bool res = nvidia_NPPST_Transpose(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NCV, VectorOperations)
{
    bool res = nvidia_NCV_Vector_Operations(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NCV, HaarCascadeLoader)
{
    bool res = nvidia_NCV_Haar_Cascade_Loader(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NCV, HaarCascadeApplication)
{
    bool res = nvidia_NCV_Haar_Cascade_Application(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NCV, HypothesesFiltration)
{
    bool res = nvidia_NCV_Hypotheses_Filtration(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

TEST_P(NCV, Visualization)
{
    // this functionality doesn't used in gpu module
    bool res = nvidia_NCV_Visualization(path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

INSTANTIATE_TEST_CASE_P(GPU_NVidia, NPPST, ALL_DEVICES);
INSTANTIATE_TEST_CASE_P(GPU_NVidia, NCV, ALL_DEVICES);

#endif // HAVE_CUDA
