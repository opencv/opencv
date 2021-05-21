// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_PERF_TESTS_HPP
#define OPENCV_GAPI_VIDEO_PERF_TESTS_HPP

#include "../../test/common/gapi_video_tests_common.hpp"

namespace opencv_test
{

using namespace perf;

//------------------------------------------------------------------------------

class BuildOptFlowPyramidPerfTest : public TestPerfParams<tuple<std::string,int,int,bool,int,int,
                                                                bool,GCompileArgs>> {};
class OptFlowLKPerfTest : public TestPerfParams<tuple<std::string,int,tuple<int,int>,int,
                                                      cv::TermCriteria,cv::GCompileArgs>> {};
class OptFlowLKForPyrPerfTest : public TestPerfParams<tuple<std::string,int,tuple<int,int>,int,
                                                            cv::TermCriteria,bool,
                                                            cv::GCompileArgs>> {};
class BuildPyr_CalcOptFlow_PipelinePerfTest : public TestPerfParams<tuple<std::string,int,int,bool,
                                                                          cv::GCompileArgs>> {};

class BackgroundSubtractorPerfTest:
    public TestPerfParams<tuple<cv::gapi::video::BackgroundSubtractorType, std::string,
                                bool, double, std::size_t, cv::GCompileArgs, CompareMats>> {};

class KalmanFilterControlPerfTest   :
    public TestPerfParams<tuple<MatType2, int, int, size_t, bool, cv::GCompileArgs>> {};
class KalmanFilterNoControlPerfTest :
    public TestPerfParams<tuple<MatType2, int, int, size_t, bool, cv::GCompileArgs>> {};

} // opencv_test

#endif // OPENCV_GAPI_VIDEO_PERF_TESTS_HPP
