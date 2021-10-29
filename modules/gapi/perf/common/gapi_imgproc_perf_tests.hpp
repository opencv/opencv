// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP
#define OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP



#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/imgproc.hpp>

namespace opencv_test
{

  using namespace perf;

  //------------------------------------------------------------------------------

class SepFilterPerfTest       : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class Filter2DPerfTest        : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int, cv::GCompileArgs>> {};
class BoxFilterPerfTest       : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int, cv::GCompileArgs>> {};
class BlurPerfTest            : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class GaussianBlurPerfTest    : public TestPerfParams<tuple<compare_f, MatType,int, cv::Size, cv::GCompileArgs>> {};
class MedianBlurPerfTest      : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size, cv::GCompileArgs>> {};
class ErodePerfTest           : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class Erode3x3PerfTest        : public TestPerfParams<tuple<compare_f, MatType,cv::Size,int, cv::GCompileArgs>> {};
class DilatePerfTest          : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, cv::GCompileArgs>> {};
class Dilate3x3PerfTest       : public TestPerfParams<tuple<compare_f, MatType,cv::Size,int, cv::GCompileArgs>> {};
class MorphologyExPerfTest    : public TestPerfParams<tuple<compare_f,MatType,cv::Size,
                                                            cv::MorphTypes,cv::GCompileArgs>> {};
class SobelPerfTest           : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int,int, cv::GCompileArgs>> {};
class SobelXYPerfTest         : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,int, cv::GCompileArgs>> {};
class LaplacianPerfTest       : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int,
                                                            cv::GCompileArgs>> {};
class BilateralFilterPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,cv::Size,int, double,double,
                                                            cv::GCompileArgs>> {};
class CannyPerfTest           : public TestPerfParams<tuple<compare_f, MatType,cv::Size,double,double,int,bool,
                                                            cv::GCompileArgs>> {};
class GoodFeaturesPerfTest    : public TestPerfParams<tuple<compare_vector_f<cv::Point2f>, std::string,
                                                            int,int,double,double,int,bool,
                                                            cv::GCompileArgs>> {};
class FindContoursPerfTest    : public TestPerfParams<tuple<CompareMats, MatType,cv::Size,
                                                            cv::RetrievalModes,
                                                            cv::ContourApproximationModes,
                                                            cv::GCompileArgs>> {};
class FindContoursHPerfTest   : public TestPerfParams<tuple<CompareMats, MatType,cv::Size,
                                                            cv::RetrievalModes,
                                                            cv::ContourApproximationModes,
                                                            cv::GCompileArgs>> {};
class BoundingRectMatPerfTest       :
    public TestPerfParams<tuple<CompareRects, MatType,cv::Size,bool, cv::GCompileArgs>> {};
class BoundingRectVector32SPerfTest :
    public TestPerfParams<tuple<CompareRects, cv::Size, cv::GCompileArgs>> {};
class BoundingRectVector32FPerfTest :
    public TestPerfParams<tuple<CompareRects, cv::Size, cv::GCompileArgs>> {};
class FitLine2DMatVectorPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               MatType,cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine2DVector32SPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine2DVector32FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine2DVector64FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine3DMatVectorPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               MatType,cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine3DVector32SPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine3DVector32FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class FitLine3DVector64FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               cv::Size,cv::DistanceTypes,
                                                               cv::GCompileArgs>> {};
class EqHistPerfTest      : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2RGBPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class RGB2GrayPerfTest    : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2GrayPerfTest    : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class RGB2YUVPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class YUV2RGBPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2I420PerfTest    : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class RGB2I420PerfTest    : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class I4202BGRPerfTest    : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class I4202RGBPerfTest    : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class RGB2LabPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2LUVPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class LUV2BGRPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BGR2YUVPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class YUV2BGRPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class RGB2HSVPerfTest     : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class BayerGR2RGBPerfTest : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
class RGB2YUV422PerfTest  : public TestPerfParams<tuple<compare_f, cv::Size, cv::GCompileArgs>> {};
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP
