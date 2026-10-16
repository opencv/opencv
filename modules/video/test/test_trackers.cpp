// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
#endif

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
PARAM_TEST_CASE(DistanceAndOverlap, string, int)
{
    string dataset;
    int numFramesLimit;
    virtual void SetUp()
    {
        dataset = GET_PARAM(0);
        numFramesLimit = GET_PARAM(1);
    }
};

TEST_P(DistanceAndOverlap, MIL)
{
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .65f, NoTransform);
    test.run(numFramesLimit);
}

TEST_P(DistanceAndOverlap, Shifted_Data_MIL)
{
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .6f, CenterShiftLeft);
    test.run(numFramesLimit);
}

/***************************************************************************************/
//Tests with scaled initial window

TEST_P(DistanceAndOverlap, Scaled_Data_MIL)
{
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .7f, Scale_1_1);
    test.run(numFramesLimit);
}

INSTANTIATE_TEST_CASE_P(Tracking, DistanceAndOverlap,
    testing::Combine(
        TESTSET_NAMES,
        testing::Values(0)
    )
);

INSTANTIATE_TEST_CASE_P(Tracking5Frames, DistanceAndOverlap,
    testing::Combine(
        TESTSET_NAMES,
        testing::Values(5)
    )
);


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

#ifdef HAVE_OPENCV_DNN
// The tracker normalizes with (img/255 - mean) / std, so scalefactor has to be 1/(255*std);
// the old `1.0 / Scalar` was Scalar's quaternion inverse, which negated the last channels.
// Pin that by comparing the network the tracker feeds against the documented normalization.
TEST(vittrack, preprocessing_matches_documented_normalization)
{
    const std::string model = cvtest::findDataFile("dnn/onnx/models/vitTracker.onnx");
    const Scalar meanValue(0.485, 0.456, 0.406);
    const Scalar stdValue(0.229, 0.224, 0.225);
    const Rect roi(325, 164, 100, 100);

    Mat img = imread(findDataFile("tracking/bag/00000001.jpg"), IMREAD_COLOR);
    ASSERT_FALSE(img.empty()) << "Can't load the tracking test image";

    // crop_image() with factor 2 crops a square of side ceil(sqrt(w*h)*2) around the ROI.
    // This frame is large enough for that square to stay inside it, so no border is added.
    const int cropSide = cvCeil(std::sqrt((double)roi.width * roi.height) * 2);
    const Rect cropRect(roi.x + (roi.width - cropSide) / 2, roi.y + (roi.height - cropSide) / 2,
                        cropSide, cropSide);
    ASSERT_GE(cropRect.x, 0);
    ASSERT_GE(cropRect.y, 0);
    ASSERT_LE(cropRect.br().x, img.cols);
    ASSERT_LE(cropRect.br().y, img.rows);

    dnn::Image2BlobParams params;
    params.mean = meanValue * 255.0;
    params.scalefactor = Scalar(1.0 / (255.0 * stdValue[0]),
                                1.0 / (255.0 * stdValue[1]),
                                1.0 / (255.0 * stdValue[2]));

    Mat templateCrop, searchCrop;
    resize(img(cropRect), templateCrop, Size(128, 128));
    resize(img, searchCrop, Size(256, 256));
    Mat expectedTemplate = dnn::blobFromImageWithParams(templateCrop, params);
    Mat search = dnn::blobFromImageWithParams(searchCrop, params);

    // Net is a handle to a shared implementation, so the template blob the tracker builds in
    // init() ends up in the net that was passed in and can be forwarded from here.
    dnn::Net trackerNet = dnn::readNet(model);
    dnn::Net referenceNet = dnn::readNet(model);
    Ptr<Tracker> tracker = TrackerVit::create(trackerNet, meanValue, stdValue, 0.20f);
    tracker->init(img, roi);

    const std::vector<String> outNames = {"output1", "output2", "output3"};
    std::vector<Mat> trackerOut;
    trackerNet.setInput(search, "search");
    trackerNet.forward(trackerOut, outNames);

    std::vector<Mat> referenceOut;
    referenceNet.setInput(expectedTemplate, "template");
    referenceNet.setInput(search, "search");
    referenceNet.forward(referenceOut, outNames);

    ASSERT_EQ(referenceOut.size(), trackerOut.size());
    for (size_t i = 0; i < trackerOut.size(); i++)
        EXPECT_LE(cv::norm(trackerOut[i], referenceOut[i], NORM_INF), 1e-5f)
            << "network output " << i << " does not match the documented normalization";
}
#endif

}}  // namespace opencv_test::
