

#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_HPP


#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/imgproc.hpp>

namespace opencv_test
{

  using namespace perf;

  //------------------------------------------------------------------------------

class RenderTestTexts : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
class RenderTestRects : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
class RenderTestCircles : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
class RenderTestLines : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
class RenderTestMosaics : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
class RenderTestImages : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};
class RenderTestPolylines : public TestPerfParams<tuple<compare_f, MatType, cv::Size, cv::GCompileArgs>> {};

}

#endif // OPENCV_GAPI_RENDER_PERF_TESTS_HPP
