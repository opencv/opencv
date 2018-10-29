// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP
#define OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP



#include "../../test/common/gapi_tests_common.hpp"
#include "opencv2/gapi/imgproc.hpp"

namespace opencv_test
{

  using namespace perf;

  //------------------------------------------------------------------------------

class SepFilterPerfTest : public TestPerfParams<tuple<compare_f, MatType, int, cv::Size, int, cv::GCompileArgs>> {};
class Filter2DPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int, cv::GCompileArgs>> {};
class BoxFilterPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int, cv::GCompileArgs>> {};
class BlurPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class GaussianBlurPerfTest : public TestPerfParams<tuple<compare_f, MatType, int, cv::Size, cv::GCompileArgs>> {};
class MedianBlurPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size, cv::GCompileArgs>> {};
class ErodePerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class Erode3x3PerfTest : public TestPerfParams<tuple<compare_f, MatType, cv::Size, int, cv::GCompileArgs>> {};
class DilatePerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class Dilate3x3PerfTest : public TestPerfParams<tuple<compare_f, MatType,cv::Size,int, cv::GCompileArgs>> {};
class SobelPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int,int, cv::GCompileArgs>> {};
class CannyPerfTest : public TestPerfParams<tuple<compare_f, MatType,cv::Size,double,double,int,bool, cv::GCompileArgs>> {};
class EqHistPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs >> {};
class RGB2GrayPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs >> {};
class BGR2GrayPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs >> {};
class RGB2YUVPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs >> {};
class YUV2RGBPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs >> {};
class RGB2LabPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2LUVPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class LUV2BGRPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2YUVPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class YUV2BGRPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
}
#endif //OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP
