// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_HPP

#include <iostream>

#include "gapi_tests_common.hpp"

namespace opencv_test
{

struct Filter2DTest : public TestParams <std::tuple<compare_f, MatType,int,cv::Size,int,int,bool,cv::GCompileArgs>> {};
struct BoxFilterTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,int,int,bool,cv::GCompileArgs>> {};
struct SepFilterTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,int,bool,cv::GCompileArgs>> {};
struct BlurTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,int,bool,cv::GCompileArgs>> {};
struct GaussianBlurTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,bool,cv::GCompileArgs>> {};
struct MedianBlurTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,bool,cv::GCompileArgs>> {};
struct ErodeTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,int,bool,cv::GCompileArgs>> {};
struct Erode3x3Test : public TestParams <std::tuple<compare_f,MatType,cv::Size,bool,int,cv::GCompileArgs>> {};
struct DilateTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,int,bool,cv::GCompileArgs>> {};
struct Dilate3x3Test : public TestParams <std::tuple<compare_f,MatType,cv::Size,bool,int,cv::GCompileArgs>> {};
struct SobelTest : public TestParams <std::tuple<compare_f,MatType,int,cv::Size,int,int,int,bool,cv::GCompileArgs>> {};
struct EqHistTest : public TestParams <std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct CannyTest : public TestParams <std::tuple<compare_f,MatType,cv::Size,double,double,int,bool,bool,cv::GCompileArgs>> {};
struct RGB2GrayTest : public  TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct BGR2GrayTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct RGB2YUVTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct YUV2RGBTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct RGB2LabTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct BGR2LUVTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct LUV2BGRTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct BGR2YUVTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
struct YUV2BGRTest : public TestParams<std::tuple<compare_f,cv::Size,bool,cv::GCompileArgs>> {};
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_HPP
