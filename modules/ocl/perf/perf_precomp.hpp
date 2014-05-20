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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

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
// This software is provided by the copyright holders and contributors as is and
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

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>

#include "cvconfig.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/ts/ts.hpp"

// TODO remove it

#define OCL_SIZE_1000 Size(1000, 1000)
#define OCL_SIZE_2000 Size(2000, 2000)
#define OCL_SIZE_4000 Size(4000, 4000)

#define OCL_TYPICAL_MAT_SIZES ::testing::Values(OCL_SIZE_1000, OCL_SIZE_2000, OCL_SIZE_4000)

using namespace std;
using namespace cv;

#define OCL_SIZE_1 szVGA
#define OCL_SIZE_2 sz720p
#define OCL_SIZE_3 sz1080p
#define OCL_SIZE_4 sz2160p

#define OCL_TEST_SIZES ::testing::Values(OCL_SIZE_1, OCL_SIZE_2, OCL_SIZE_3, OCL_SIZE_4)
#define OCL_TEST_TYPES ::testing::Values(CV_8UC1, CV_32FC1, CV_8UC4, CV_32FC4)
#define OCL_TEST_TYPES_14 OCL_TEST_TYPES
#define OCL_TEST_TYPES_134 ::testing::Values(CV_8UC1, CV_32FC1, CV_8UC3, CV_32FC3, CV_8UC4, CV_32FC4)

#define OCL_PERF_ENUM(type, ...) ::testing::Values(type, ## __VA_ARGS__ )

#define IMPL_OCL "ocl"
#define IMPL_GPU "gpu"
#define IMPL_PLAIN "plain"

#define RUN_OCL_IMPL (IMPL_OCL == getSelectedImpl())
#define RUN_PLAIN_IMPL (IMPL_PLAIN == getSelectedImpl())

#ifdef HAVE_OPENCV_GPU
# define RUN_GPU_IMPL (IMPL_GPU == getSelectedImpl())
#endif

#ifdef HAVE_OPENCV_GPU
#define OCL_PERF_ELSE               \
        if (RUN_GPU_IMPL)           \
            CV_TEST_FAIL_NO_IMPL(); \
        else                        \
            CV_TEST_FAIL_NO_IMPL();
#else
#define OCL_PERF_ELSE               \
            CV_TEST_FAIL_NO_IMPL();
#endif

#define OCL_PERF_TEST(fixture, name) \
    class OCL##_##fixture##_##name : \
        public ::perf::TestBase \
    { \
    public: \
        OCL##_##fixture##_##name() { } \
    protected: \
        virtual void PerfTestBody(); \
    }; \
    TEST_F(OCL##_##fixture##_##name, name) { RunPerfTestBody(); } \
    void OCL##_##fixture##_##name::PerfTestBody()

#define OCL_PERF_TEST_P(fixture, name, params) \
    class OCL##_##fixture##_##name : \
        public fixture \
    { \
    public: \
        OCL##_##fixture##_##name() { } \
    protected: \
        virtual void PerfTestBody(); \
    }; \
    TEST_P(OCL##_##fixture##_##name, name) { RunPerfTestBody(); } \
    INSTANTIATE_TEST_CASE_P(/*none*/, OCL##_##fixture##_##name, params); \
    void OCL##_##fixture##_##name::PerfTestBody()

#define OCL_TEST_CYCLE_N(n) for(declare.iterations(n); startTimer(), next(); cv::ocl::finish(), stopTimer())
#define OCL_TEST_CYCLE() for(; startTimer(), next(); cv::ocl::finish(), stopTimer())
#define OCL_TEST_CYCLE_MULTIRUN(runsNum) for(declare.runs(runsNum); startTimer(), next(); stopTimer()) for(int r = 0; r < runsNum; cv::ocl::finish(), ++r)

namespace cvtest {
namespace ocl {
inline void checkDeviceMaxMemoryAllocSize(const Size& size, int type, int factor = 1)
{
    assert(factor > 0);
    if (!(IMPL_OCL == perf::TestBase::getSelectedImpl()))
        return; // OpenCL devices are not used
    int cn = CV_MAT_CN(type);
    int cn_ocl = cn == 3 ? 4 : cn;
    int type_ocl = CV_MAKE_TYPE(CV_MAT_DEPTH(type), cn_ocl);
    size_t memSize = size.area() * CV_ELEM_SIZE(type_ocl);
    const cv::ocl::DeviceInfo& devInfo = cv::ocl::Context::getContext()->getDeviceInfo();
    if (memSize * factor >= devInfo.maxMemAllocSize)
    {
        throw perf::TestBase::PerfSkipTestException();
    }
}
} // namespace cvtest::ocl
} // namespace cvtest

using namespace cvtest::ocl;

#endif
