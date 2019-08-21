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

OutputLevel nvidiaTestOutputLevel = OutputLevelNone;

namespace opencv_test { namespace {

struct NVidiaTest : TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    std::string _path;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
        _path = TS::ptr()->get_data_path().c_str();
        _path = _path + "haarcascade/";
    }
};

struct NPPST : NVidiaTest {};
struct NCV : NVidiaTest {};

CUDA_TEST_P(NPPST, Integral)
{
    bool res = nvidia_NPPST_Integral_Image(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NPPST, SquaredIntegral)
{
    bool res = nvidia_NPPST_Squared_Integral_Image(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NPPST, RectStdDev)
{
    bool res = nvidia_NPPST_RectStdDev(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NPPST, Resize)
{
    bool res = nvidia_NPPST_Resize(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NPPST, VectorOperations)
{
    bool res = nvidia_NPPST_Vector_Operations(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NPPST, Transpose)
{
    bool res = nvidia_NPPST_Transpose(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NCV, VectorOperations)
{
    bool res = nvidia_NCV_Vector_Operations(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NCV, HaarCascadeLoader)
{
    bool res = nvidia_NCV_Haar_Cascade_Loader(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NCV, HaarCascadeApplication)
{
    bool res = nvidia_NCV_Haar_Cascade_Application(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NCV, HypothesesFiltration)
{
    bool res = nvidia_NCV_Hypotheses_Filtration(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

CUDA_TEST_P(NCV, Visualization)
{
    bool res = nvidia_NCV_Visualization(_path, nvidiaTestOutputLevel);

    ASSERT_TRUE(res);
}

INSTANTIATE_TEST_CASE_P(CUDA_Legacy, NPPST, ALL_DEVICES);
INSTANTIATE_TEST_CASE_P(CUDA_Legacy, NCV, ALL_DEVICES);


}} // namespace
#endif // HAVE_CUDA
