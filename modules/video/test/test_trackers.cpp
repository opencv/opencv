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
    TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .65f, Scale_1_1);
    test.run(numFramesLimit);
}

TEST_P(DistanceAndOverlap, GOTURN)
{
    std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
    std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
    cv::TrackerGOTURN::Params params;
    params.modelTxt = model;
    params.modelBin = weights;
    TrackerTest<Tracker, Rect> test(TrackerGOTURN::create(params), dataset, 35, .35f, NoTransform);
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

// A frame passed as a ROI must be tracked like a standalone copy of it: pixels of the parent image
// around the ROI must not leak into a search window that crosses the frame border.
template <typename CreateTracker>
static void checkRoiImageMatchesClone(CreateTracker createTracker)
{
    Mat parent(240, 240, CV_8UC3, Scalar(255, 0, 0));
    Mat image = parent(Rect(32, 32, 160, 160));
    theRNG().fill(image, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    Mat clone = image.clone();
    Rect initial(4, 4, 24, 24);

    Ptr<Tracker> roiTracker = createTracker(), cloneTracker = createTracker();
    roiTracker->init(image, initial);
    cloneTracker->init(clone, initial);
    for (int i = 0; i < 3; i++)
    {
        Rect roiResult, cloneResult;
        roiTracker->update(image, roiResult);
        cloneTracker->update(clone, cloneResult);
        EXPECT_EQ(roiResult, cloneResult) << "update " << i;
    }
}

TEST(DaSiamRPN, roi_image_matches_clone)
{
    cv::TrackerDaSiamRPN::Params params;
    params.model = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_model.onnx", false);
    params.kernel_r1 = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_kernel_r1.onnx", false);
    params.kernel_cls1 = cvtest::findDataFile("dnn/onnx/models/dasiamrpn_kernel_cls1.onnx", false);
    checkRoiImageMatchesClone([&]() { return TrackerDaSiamRPN::create(params); });
}

TEST(NanoTrack, roi_image_matches_clone)
{
    cv::TrackerNano::Params params;
    params.backbone = cvtest::findDataFile("dnn/onnx/models/nanotrack_backbone_sim.onnx", false);
    params.neckhead = cvtest::findDataFile("dnn/onnx/models/nanotrack_head_sim.onnx", false);
    checkRoiImageMatchesClone([&]() { return TrackerNano::create(params); });
}

TEST(vittrack, accuracy_vittrack)
{
    std::string model = cvtest::findDataFile("dnn/onnx/models/vitTracker.onnx");
    cv::TrackerVit::Params params;
    params.net = model;
    cv::Ptr<Tracker> tracker = TrackerVit::create(params);
    checkTrackingAccuracy(tracker, 0.64);
}

TEST(vittrack, bbox_inside_image)
{
    std::string model = cvtest::findDataFile("dnn/onnx/models/vitTracker.onnx");
    cv::TrackerVit::Params params;
    params.net = model;
    cv::Ptr<Tracker> tracker = TrackerVit::create(params);

    Mat img0 = imread(findDataFile("tracking/bag/00000001.jpg"), 1);
    cv::Rect imgRect(0, 0, img0.cols, img0.rows);
    cv::Rect roi = imgRect;
    tracker->init(img0, roi);

    for (int i = 2; i <= 6; i++)
    {
        Mat img = imread(findDataFile(cv::format("tracking/bag/%08d.jpg", i)), 1);
        bool res = tracker->update(img, roi);
        ASSERT_TRUE(res);
        ASSERT_EQ(roi & imgRect, roi) << cv::format("Fail at img %d.", i);
    }
}

TEST(vittrack, bbox_inside_tiny_image)
{
    std::string model = cvtest::findDataFile("dnn/onnx/models/vitTracker.onnx");
    cv::TrackerVit::Params params;
    params.net = model;
    params.tracking_score_threshold = 0.f;
    cv::Ptr<Tracker> tracker = TrackerVit::create(params);

    Mat img;
    resize(imread(findDataFile("tracking/bag/00000001.jpg"), 1), img, Size(5, 5));
    cv::Rect imgRect(0, 0, img.cols, img.rows);
    cv::Rect roi = imgRect;
    tracker->init(img, roi);

    bool res = tracker->update(img, roi);
    ASSERT_TRUE(res);
    ASSERT_EQ(roi & imgRect, roi);
}

TEST(vittrack, roi_image_matches_clone)
{
    std::string model = cvtest::findDataFile("dnn/onnx/models/vitTracker.onnx");
    cv::TrackerVit::Params params;
    params.net = model;
    params.tracking_score_threshold = 0.f;

    Mat parent(160, 160, CV_8UC3, Scalar(255, 0, 0));
    Mat image = parent(Rect(16, 16, 128, 128));
    image.setTo(Scalar::all(0));
    Mat clone = image.clone();
    Rect initial(100, 100, 20, 20);

    Ptr<TrackerVit> roiTracker = TrackerVit::create(params);
    Ptr<TrackerVit> cloneTracker = TrackerVit::create(params);
    roiTracker->init(image, initial);
    cloneTracker->init(clone, initial);

    Rect roiResult, cloneResult;
    ASSERT_TRUE(roiTracker->update(image, roiResult));
    ASSERT_TRUE(cloneTracker->update(clone, cloneResult));
    EXPECT_EQ(roiResult, cloneResult);
    EXPECT_NEAR(roiTracker->getTrackingScore(), cloneTracker->getTrackingScore(), 1e-6f);
}

TEST(TrackerMIL, gray_and_bgr_channel0_equivalence)
{
    RNG& rng = theRNG();
    Mat gray1(120, 160, CV_8UC1, Scalar(60));
    rectangle(gray1, Rect(40, 35, 30, 30), Scalar(210), FILLED);
    Mat gray2(120, 160, CV_8UC1, Scalar(60));
    rectangle(gray2, Rect(43, 37, 30, 30), Scalar(210), FILLED);

    Mat ch1(120, 160, CV_8UC1), ch2(120, 160, CV_8UC1);
    rng.fill(ch1, RNG::UNIFORM, 0, 255);
    rng.fill(ch2, RNG::UNIFORM, 0, 255);
    Mat bgr1, bgr2;
    merge(std::vector<Mat>{gray1, ch1, ch2}, bgr1);
    merge(std::vector<Mat>{gray2, ch2, ch1}, bgr2);

    Rect initBox(40, 35, 30, 30);
    uint64 seed = rng.state;
    std::srand(42);
    Ptr<TrackerMIL> trGray = TrackerMIL::create();
    trGray->init(gray1, initBox);
    Rect outGray;
    ASSERT_TRUE(trGray->update(gray2, outGray));

    rng.state = seed;
    std::srand(42);
    Ptr<TrackerMIL> trBgr = TrackerMIL::create();
    trBgr->init(bgr1, initBox);
    Rect outBgr;
    ASSERT_TRUE(trBgr->update(bgr2, outBgr));

    EXPECT_EQ(outGray, outBgr);
}

#ifdef HAVE_OPENCV_DNN
TEST(GOTURN, synthetic_net_boundary_patch_parity)
{
    RNG& rng = theRNG();
    dnn::Net net;
    dnn::LayerParams lpConcat, lpPool, lpFc;
    lpConcat.name = "concat";
    lpConcat.type = "Concat";
    lpConcat.set("axis", 1);
    int idConcat = net.addLayer(lpConcat.name, lpConcat.type, lpConcat);
    net.setInputsNames({"data1", "data2"});
    net.connect(0, 0, idConcat, 0);
    net.connect(0, 1, idConcat, 1);

    lpPool.name = "pool";
    lpPool.type = "Pooling";
    lpPool.set("pool", "AVE");
    lpPool.set("global_pooling", true);
    int idPool = net.addLayerToPrev(lpPool.name, lpPool.type, lpPool);
    CV_UNUSED(idPool);

    Mat W(4, 6, CV_32F, Scalar(0.01f));
    Mat b = (Mat_<float>(1, 4) << 56.f, 56.f, 170.f, 170.f);
    lpFc.name = "scale";
    lpFc.type = "InnerProduct";
    lpFc.set("num_output", 4);
    lpFc.set("bias_term", true);
    lpFc.blobs = {W, b};
    net.addLayerToPrev(lpFc.name, lpFc.type, lpFc);

    Mat f0(180, 240, CV_8UC3), f1(180, 240, CV_8UC3), f2(180, 240, CV_8UC3);
    rng.fill(f0, RNG::UNIFORM, 0, 255);
    rng.fill(f1, RNG::UNIFORM, 0, 255);
    rng.fill(f2, RNG::UNIFORM, 0, 255);

    // Near-corner box forces negative unpadded coordinates and BORDER_REPLICATE padding
    Rect box(4, 6, 40, 36);
    Ptr<TrackerGOTURN> tracker = TrackerGOTURN::create(net);
    tracker->init(f0, box);
    ASSERT_TRUE(tracker->update(f1, box));
    EXPECT_GT(box.width, 0);
    EXPECT_GT(box.height, 0);
    ASSERT_TRUE(tracker->update(f2, box));
    EXPECT_GT(box.width, 0);
    EXPECT_GT(box.height, 0);
}
#endif

}}  // namespace opencv_test::
