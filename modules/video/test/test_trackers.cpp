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

static bool checkIOU(const Rect& r0, const Rect& r1, double threshold)
{
    int interArea = (r0 & r1).area();
    double iouVal = (interArea * 1.0 )/ (r0.area() + r1.area() - interArea);;

    if (iouVal > threshold)
        return true;
    else
    {
        std::cout <<"Unmatched IOU:  expect IOU val ("<<iouVal <<") > the IOU threadhold ("<<threshold<<")! Box 0 is "
                                << r0 <<", and Box 1 is "<<r1<< std::endl;
        return false;
    }
}

static void checkTrackingAccuracy(cv::Ptr<Tracker>& tracker, double iouThreshold = 0.7)
{
    // Template image
    Mat img0 = imread(findDataFile("tracking/bag/00000001.jpg"), 1);

    // Tracking image sequence.
    std::vector<Mat> imgs;
    imgs.push_back(imread(findDataFile("tracking/bag/00000002.jpg"), 1));
    imgs.push_back(imread(findDataFile("tracking/bag/00000003.jpg"), 1));
    imgs.push_back(imread(findDataFile("tracking/bag/00000004.jpg"), 1));
    imgs.push_back(imread(findDataFile("tracking/bag/00000005.jpg"), 1));
    imgs.push_back(imread(findDataFile("tracking/bag/00000006.jpg"), 1));

    cv::Rect roi(325, 164, 100, 100);
    std::vector<Rect> targetRois;
    targetRois.push_back(cv::Rect(278, 133, 99, 104));
    targetRois.push_back(cv::Rect(293, 88, 93, 110));
    targetRois.push_back(cv::Rect(287, 76, 89, 116));
    targetRois.push_back(cv::Rect(297, 74, 82, 122));
    targetRois.push_back(cv::Rect(311, 83, 78, 125));

    tracker->init(img0, roi);
    CV_Assert(targetRois.size() == imgs.size());

    for (int i = 0; i < (int)imgs.size(); i++)
    {
        bool res = tracker->update(imgs[i], roi);
        ASSERT_TRUE(res);
        ASSERT_TRUE(checkIOU(roi, targetRois[i], iouThreshold)) << cv::format("Fail at img %d.",i);
    }
}

TEST(GOTURN, accuracy)
{
    std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
    std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
    cv::TrackerGOTURN::Params params;
    params.modelTxt = model;
    params.modelBin = weights;
    cv::Ptr<Tracker> tracker = TrackerGOTURN::create(params);
    // TODO! GOTURN have low accuracy. Try to remove this api at 5.x.
    checkTrackingAccuracy(tracker, 0.08);
}

TEST(DaSiamRPN, accuracy)
{
    std::string model = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_model.onnx", false);
    std::string kernel_r1 = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_kernel_r1.onnx", false);
    std::string kernel_cls1 = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_kernel_cls1.onnx", false);
    cv::TrackerDaSiamRPN::Params params;
    params.model = model;
    params.kernel_r1 = kernel_r1;
    params.kernel_cls1 = kernel_cls1;
    cv::Ptr<Tracker> tracker = TrackerDaSiamRPN::create(params);
    checkTrackingAccuracy(tracker, 0.7);
}

TEST(NanoTrack, accuracy_NanoTrack_V1)
{
    std::string backbonePath = cvtest::findDataFile("dnn/onnx/models/nanotrack_backbone_sim.onnx", false);
    std::string neckheadPath = cvtest::findDataFile("dnn/onnx/models/nanotrack_head_sim.onnx", false);

    cv::TrackerNano::Params params;
    params.backbone = backbonePath;
    params.neckhead = neckheadPath;
    cv::Ptr<Tracker> tracker = TrackerNano::create(params);
    checkTrackingAccuracy(tracker);
}

TEST(NanoTrack, accuracy_NanoTrack_V2)
{
    std::string backbonePath = cvtest::findDataFile("dnn/onnx/models/nanotrack_backbone_sim_v2.onnx", false);
    std::string neckheadPath = cvtest::findDataFile("dnn/onnx/models/nanotrack_head_sim_v2.onnx", false);

    cv::TrackerNano::Params params;
    params.backbone = backbonePath;
    params.neckhead = neckheadPath;
    cv::Ptr<Tracker> tracker = TrackerNano::create(params);
    checkTrackingAccuracy(tracker, 0.69);
}

TEST(vittrack, accuracy_vittrack)
{
    std::string model = cvtest::findDataFile("dnn/onnx/models/vitTracker.onnx");
    cv::TrackerVit::Params params;
    params.net = model;
    cv::Ptr<Tracker> tracker = TrackerVit::create(params);
    checkTrackingAccuracy(tracker, 0.64);
}

}}  // namespace opencv_test::
