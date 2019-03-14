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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_TEST_COMMON_HPP__
#define __OPENCV_TEST_COMMON_HPP__

#ifdef HAVE_OPENCL
#include "opencv2/core/ocl.hpp"
#endif

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
static inline void PrintTo(const cv::dnn::Backend& v, std::ostream* os)
{
    switch (v) {
    case DNN_BACKEND_DEFAULT: *os << "DEFAULT"; return;
    case DNN_BACKEND_HALIDE: *os << "HALIDE"; return;
    case DNN_BACKEND_INFERENCE_ENGINE: *os << "DLIE"; return;
    case DNN_BACKEND_OPENCV: *os << "OCV"; return;
    case DNN_BACKEND_VKCOM: *os << "VKCOM"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_BACKEND_UNKNOWN(" << (int)v << ")";
}

static inline void PrintTo(const cv::dnn::Target& v, std::ostream* os)
{
    switch (v) {
    case DNN_TARGET_CPU: *os << "CPU"; return;
    case DNN_TARGET_OPENCL: *os << "OCL"; return;
    case DNN_TARGET_OPENCL_FP16: *os << "OCL_FP16"; return;
    case DNN_TARGET_MYRIAD: *os << "MYRIAD"; return;
    case DNN_TARGET_VULKAN: *os << "VULKAN"; return;
    case DNN_TARGET_FPGA: *os << "FPGA"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_TARGET_UNKNOWN(" << (int)v << ")";
}

using opencv_test::tuple;
using opencv_test::get;
static inline void PrintTo(const tuple<cv::dnn::Backend, cv::dnn::Target> v, std::ostream* os)
{
    PrintTo(get<0>(v), os);
    *os << "/";
    PrintTo(get<1>(v), os);
}

CV__DNN_INLINE_NS_END
}} // namespace


static inline const std::string &getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

static inline void normAssert(cv::InputArray ref, cv::InputArray test, const char *comment = "",
                       double l1 = 0.00001, double lInf = 0.0001)
{
    double normL1 = cvtest::norm(ref, test, cv::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment;

    double normInf = cvtest::norm(ref, test, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment;
}

static std::vector<cv::Rect2d> matToBoxes(const cv::Mat& m)
{
    EXPECT_EQ(m.type(), CV_32FC1);
    EXPECT_EQ(m.dims, 2);
    EXPECT_EQ(m.cols, 4);

    std::vector<cv::Rect2d> boxes(m.rows);
    for (int i = 0; i < m.rows; ++i)
    {
        CV_Assert(m.row(i).isContinuous());
        const float* data = m.ptr<float>(i);
        double l = data[0], t = data[1], r = data[2], b = data[3];
        boxes[i] = cv::Rect2d(l, t, r - l, b - t);
    }
    return boxes;
}

static inline void normAssertDetections(const std::vector<int>& refClassIds,
                                 const std::vector<float>& refScores,
                                 const std::vector<cv::Rect2d>& refBoxes,
                                 const std::vector<int>& testClassIds,
                                 const std::vector<float>& testScores,
                                 const std::vector<cv::Rect2d>& testBoxes,
                                 const char *comment = "", double confThreshold = 0.0,
                                 double scores_diff = 1e-5, double boxes_iou_diff = 1e-4)
{
    std::vector<bool> matchedRefBoxes(refBoxes.size(), false);
    for (int i = 0; i < testBoxes.size(); ++i)
    {
        double testScore = testScores[i];
        if (testScore < confThreshold)
            continue;

        int testClassId = testClassIds[i];
        const cv::Rect2d& testBox = testBoxes[i];
        bool matched = false;
        for (int j = 0; j < refBoxes.size() && !matched; ++j)
        {
            if (!matchedRefBoxes[j] && testClassId == refClassIds[j] &&
                std::abs(testScore - refScores[j]) < scores_diff)
            {
                double interArea = (testBox & refBoxes[j]).area();
                double iou = interArea / (testBox.area() + refBoxes[j].area() - interArea);
                if (std::abs(iou - 1.0) < boxes_iou_diff)
                {
                    matched = true;
                    matchedRefBoxes[j] = true;
                }
            }
        }
        if (!matched)
            std::cout << cv::format("Unmatched prediction: class %d score %f box ",
                                    testClassId, testScore) << testBox << std::endl;
        EXPECT_TRUE(matched) << comment;
    }

    // Check unmatched reference detections.
    for (int i = 0; i < refBoxes.size(); ++i)
    {
        if (!matchedRefBoxes[i] && refScores[i] > confThreshold)
        {
            std::cout << cv::format("Unmatched reference: class %d score %f box ",
                                    refClassIds[i], refScores[i]) << refBoxes[i] << std::endl;
            EXPECT_LE(refScores[i], confThreshold) << comment;
        }
    }
}

// For SSD-based object detection networks which produce output of shape 1x1xNx7
// where N is a number of detections and an every detection is represented by
// a vector [batchId, classId, confidence, left, top, right, bottom].
static inline void normAssertDetections(cv::Mat ref, cv::Mat out, const char *comment = "",
                                 double confThreshold = 0.0, double scores_diff = 1e-5,
                                 double boxes_iou_diff = 1e-4)
{
    CV_Assert(ref.total() % 7 == 0);
    CV_Assert(out.total() % 7 == 0);
    ref = ref.reshape(1, ref.total() / 7);
    out = out.reshape(1, out.total() / 7);

    cv::Mat refClassIds, testClassIds;
    ref.col(1).convertTo(refClassIds, CV_32SC1);
    out.col(1).convertTo(testClassIds, CV_32SC1);
    std::vector<float> refScores(ref.col(2)), testScores(out.col(2));
    std::vector<cv::Rect2d> refBoxes = matToBoxes(ref.colRange(3, 7));
    std::vector<cv::Rect2d> testBoxes = matToBoxes(out.colRange(3, 7));
    normAssertDetections(refClassIds, refScores, refBoxes, testClassIds, testScores,
                         testBoxes, comment, confThreshold, scores_diff, boxes_iou_diff);
}

static inline bool readFileInMemory(const std::string& filename, std::string& content)
{
    std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(filename.c_str(), mode);
    if (!ifs.is_open())
        return false;

    content.clear();

    ifs.seekg(0, std::ios::end);
    content.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    content.assign((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());

    return true;
}

namespace opencv_test {

using namespace cv::dnn;

static inline
testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargets(
        bool withInferenceEngine = true,
        bool withHalide = false,
        bool withCpuOCV = true,
        bool withVkCom = true
)
{
    std::vector< tuple<Backend, Target> > targets;
    std::vector< Target > available;
    if (withHalide)
    {
        available = getAvailableTargets(DNN_BACKEND_HALIDE);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
            targets.push_back(make_tuple(DNN_BACKEND_HALIDE, *i));
    }
    if (withInferenceEngine)
    {
        available = getAvailableTargets(DNN_BACKEND_INFERENCE_ENGINE);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
            targets.push_back(make_tuple(DNN_BACKEND_INFERENCE_ENGINE, *i));
    }
    if (withVkCom)
    {
        available = getAvailableTargets(DNN_BACKEND_VKCOM);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
            targets.push_back(make_tuple(DNN_BACKEND_VKCOM, *i));
    }
    {
        available = getAvailableTargets(DNN_BACKEND_OPENCV);
        for (std::vector< Target >::const_iterator i = available.begin(); i != available.end(); ++i)
        {
            if (!withCpuOCV && *i == DNN_TARGET_CPU)
                continue;
            targets.push_back(make_tuple(DNN_BACKEND_OPENCV, *i));
        }
    }
    if (targets.empty())  // validate at least CPU mode
        targets.push_back(make_tuple(DNN_BACKEND_OPENCV, DNN_TARGET_CPU));
    return testing::ValuesIn(targets);
}

} // namespace


namespace opencv_test {
using namespace cv::dnn;

class DNNTestLayer : public TestWithParam<tuple<Backend, Target> >
{
public:
    dnn::Backend backend;
    dnn::Target target;
    double default_l1, default_lInf;

    DNNTestLayer()
    {
        backend = (dnn::Backend)(int)get<0>(GetParam());
        target = (dnn::Target)(int)get<1>(GetParam());
        getDefaultThresholds(backend, target, &default_l1, &default_lInf);
    }

   static void getDefaultThresholds(int backend, int target, double* l1, double* lInf)
   {
       if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
       {
           *l1 = 4e-3;
           *lInf = 2e-2;
       }
       else
       {
           *l1 = 1e-5;
           *lInf = 1e-4;
       }
   }

    static void checkBackend(int backend, int target, Mat* inp = 0, Mat* ref = 0)
    {
       if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
       {
           if (inp && ref && inp->dims == 4 && ref->dims == 4 &&
               inp->size[0] != 1 && inp->size[0] != ref->size[0])
               throw SkipTestException("Inconsistent batch size of input and output blobs for Myriad plugin");
       }
   }

protected:
    void checkBackend(Mat* inp = 0, Mat* ref = 0)
    {
        checkBackend(backend, target, inp, ref);
    }
};

} // namespace

#endif
