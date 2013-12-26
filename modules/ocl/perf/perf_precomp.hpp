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

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#define CV_BUILD_OCL_MODULE

#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>

#include "cvconfig.h"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ocl.hpp"
#include "opencv2/ts.hpp"

using namespace std;
using namespace cv;

#define OCL_SIZE_1000 Size(1000, 1000)
#define OCL_SIZE_2000 Size(2000, 2000)
#define OCL_SIZE_4000 Size(4000, 4000)

#define OCL_TYPICAL_MAT_SIZES ::testing::Values(OCL_SIZE_1000, OCL_SIZE_2000, OCL_SIZE_4000)

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

#define OCL_TEST_CYCLE_N(n) for(declare.iterations(n); startTimer(), next(); cv::ocl::finish(), stopTimer())
#define OCL_TEST_CYCLE() for(; startTimer(), next(); cv::ocl::finish(), stopTimer())
#define OCL_TEST_CYCLE_MULTIRUN(runsNum) for(declare.runs(runsNum); startTimer(), next(); stopTimer()) for(int r = 0; r < runsNum; cv::ocl::finish(), ++r)

// TODO: Move to the ts module
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

struct KeypointIdxCompare
{
    std::vector<cv::KeyPoint>* keypoints;

    explicit KeypointIdxCompare(std::vector<cv::KeyPoint>* _keypoints) : keypoints(_keypoints) {}

    bool operator ()(size_t i1, size_t i2) const
    {
        cv::KeyPoint kp1 = (*keypoints)[i1];
        cv::KeyPoint kp2 = (*keypoints)[i2];
        if (kp1.pt.x != kp2.pt.x)
            return kp1.pt.x < kp2.pt.x;
        if (kp1.pt.y != kp2.pt.y)
            return kp1.pt.y < kp2.pt.y;
        if (kp1.response != kp2.response)
            return kp1.response < kp2.response;
        return kp1.octave < kp2.octave;
    }
};

inline void sortKeyPoints(std::vector<cv::KeyPoint>& keypoints, cv::InputOutputArray _descriptors = cv::noArray())
{
    std::vector<size_t> indexies(keypoints.size());
    for (size_t i = 0; i < indexies.size(); ++i)
        indexies[i] = i;

    std::sort(indexies.begin(), indexies.end(), KeypointIdxCompare(&keypoints));

    std::vector<cv::KeyPoint> new_keypoints;
    cv::Mat new_descriptors;

    new_keypoints.resize(keypoints.size());

    cv::Mat descriptors;
    if (_descriptors.needed())
    {
        descriptors = _descriptors.getMat();
        new_descriptors.create(descriptors.size(), descriptors.type());
    }

    for (size_t i = 0; i < indexies.size(); ++i)
    {
        size_t new_idx = indexies[i];
        new_keypoints[i] = keypoints[new_idx];
        if (!new_descriptors.empty())
            descriptors.row((int) new_idx).copyTo(new_descriptors.row((int) i));
    }

    keypoints.swap(new_keypoints);
    if (_descriptors.needed())
        new_descriptors.copyTo(_descriptors);
}

} // namespace cvtest::ocl
} // namespace cvtest

using namespace cvtest::ocl;

#endif
