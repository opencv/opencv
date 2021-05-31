// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

//#define DEBUG_TEST
#ifdef DEBUG_TEST
#include <opencv2/highgui.hpp>
#endif

namespace opencv_test { namespace {
//using namespace cv::tracking;

#define TESTSET_NAMES testing::Values("david", "dudek", "faceocc2")

const string TRACKING_DIR = "tracking";
const string FOLDER_IMG = "data";
const string FOLDER_OMIT_INIT = "initOmit";

#include "test_trackers.impl.hpp"

//[TESTDATA]
PARAM_TEST_CASE(DistanceAndOverlap, string)
{
    string dataset;
    virtual void SetUp()
    {
        dataset = GET_PARAM(0);
    }
};

TEST_P(DistanceAndOverlap, MIL)
{
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .65f, NoTransform);
    test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_MIL)
{
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .6f, CenterShiftLeft);
    test.run();
}

/***************************************************************************************/
//Tests with scaled initial window

TEST_P(DistanceAndOverlap, Scaled_Data_MIL)
{
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .7f, Scale_1_1);
    test.run();
}

TEST_P(DistanceAndOverlap, GOTURN)
{
    std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
    std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
    cv::TrackerGOTURN::Params params;
    params.modelTxt = model;
    params.modelBin = weights;
    TrackerTest<Tracker, Rect> test(TrackerGOTURN::create(params), dataset, 35, .35f, NoTransform);
    test.run();
}

INSTANTIATE_TEST_CASE_P(Tracking, DistanceAndOverlap, TESTSET_NAMES);

TEST(GOTURN, memory_usage)
{
    cv::Rect roi(145, 70, 85, 85);

    std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
    std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
    cv::TrackerGOTURN::Params params;
    params.modelTxt = model;
    params.modelBin = weights;
    cv::Ptr<Tracker> tracker = TrackerGOTURN::create(params);

    string inputVideo = cvtest::findDataFile("tracking/david/data/david.webm");
    cv::VideoCapture video(inputVideo);
    ASSERT_TRUE(video.isOpened()) << inputVideo;

    cv::Mat frame;
    video >> frame;
    ASSERT_FALSE(frame.empty()) << inputVideo;
    tracker->init(frame, roi);
    string ground_truth_bb;
    for (int nframes = 0; nframes < 15; ++nframes)
    {
        std::cout << "Frame: " << nframes << std::endl;
        video >> frame;
        bool res = tracker->update(frame, roi);
        ASSERT_TRUE(res);
        std::cout << "Predicted ROI: " << roi << std::endl;
    }
}

TEST(DaSiamRPN, memory_usage)
{
    cv::Rect roi(145, 70, 85, 85);

    std::string model = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_model.onnx", false);
    std::string kernel_r1 = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_kernel_r1.onnx", false);
    std::string kernel_cls1 = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_kernel_cls1.onnx", false);
    cv::TrackerDaSiamRPN::Params params;
    params.model = model;
    params.kernel_r1 = kernel_r1;
    params.kernel_cls1 = kernel_cls1;
    cv::Ptr<Tracker> tracker = TrackerDaSiamRPN::create(params);

    string inputVideo = cvtest::findDataFile("tracking/david/data/david.webm");
    cv::VideoCapture video(inputVideo);
    ASSERT_TRUE(video.isOpened()) << inputVideo;

    cv::Mat frame;
    video >> frame;
    ASSERT_FALSE(frame.empty()) << inputVideo;
    tracker->init(frame, roi);
    string ground_truth_bb;
    for (int nframes = 0; nframes < 15; ++nframes)
    {
        std::cout << "Frame: " << nframes << std::endl;
        video >> frame;
        bool res = tracker->update(frame, roi);
        ASSERT_TRUE(res);
        std::cout << "Predicted ROI: " << roi << std::endl;
    }
}

}}  // namespace opencv_test::
