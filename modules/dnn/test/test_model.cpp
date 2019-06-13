// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    String rootFolder = "dnn/";
    return findDataFile(rootFolder + filename, false);
}


class Test_Model : public DNNTestLayer
{
public:
    void testDetectModel(const std::string& weights, const std::string& cfg,
                         const std::string& imgPath, const std::vector<int>& refClassIds,
                         const std::vector<float>& refConfidences,
                         const std::vector<Rect2d>& refBoxes,
                         double scoreDiff, double iouDiff,
                         double confThreshold = 0.24, double nmsThreshold = 0.0,
                         const Size& size = {-1, -1}, Scalar mean = Scalar(),
                         float scale = 1.0, bool swapRB = true, bool crop = false,
                         bool absoluteCoords = true)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        Model model(weights, cfg, size, mean, scale, swapRB, crop);

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect2d> boxes;

        model.detect(frame, classIds, confidences, boxes, confThreshold,
                     nmsThreshold, absoluteCoords);
        normAssertDetections(refClassIds, refConfidences, refBoxes, classIds,
                         confidences, boxes, "",
                         confThreshold, scoreDiff, iouDiff);
    }

    void testClassifyModel(const std::string& weights, const std::string& cfg,
                    const std::string& imgPath, std::pair<int, float> ref, float norm,
                    const Size& size = {-1, -1}, Scalar mean = Scalar(),
                    float scale = 1.0, bool swapRB = true, bool crop = false)
    {
        checkBackend();

        Mat frame = imread(imgPath);
        Model model(weights, cfg, size, mean, scale, swapRB, crop);

        std::pair<int, float> prediction = model.classify(frame);
        EXPECT_EQ(prediction.first, ref.first);
        ASSERT_NEAR(prediction.second, ref.second, norm);
    }
};

TEST_P(Test_Model, Classify)
{
    std::pair<int, float> ref(652, 0.641789);

    std::string img_path = _tf("grace_hopper_227.png");
    std::string config_file = _tf("bvlc_alexnet.prototxt");
    std::string weights_file = _tf("bvlc_alexnet.caffemodel");

    float scale = 1.0;
    Size size{227, 227};
    bool swapRB = false;

    float norm = 1e-4;

    testClassifyModel(weights_file, config_file, img_path, ref, norm,
                      size, Scalar(), scale, swapRB);
}

TEST_P(Test_Model, DetectRegion)
{
    std::vector<int> refClassIds = {6, 1, 11};
    std::vector<float> refConfidences = {0.750469f, 0.780879f, 0.901615f};
    std::vector<Rect2d> refBoxes = {Rect2d(0.577374f, 0.127391f, 0.325575f, 0.173418f),
                                    Rect2d(0.270762f, 0.264102f, 0.461713f, 0.48131f),
                                    Rect2d(0.1386f, 0.338509f, 0.282737f, 0.60028f)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("yolo-voc.weights");
    std::string config_file = _tf("yolo-voc.cfg");

    float scale = 1.0 / 255.0;
    Size size{416, 416};
    bool crop = false;
    bool swapRB = true;
    bool absoluteCoords = false;

    double confThreshold = 0.24;
    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-2 : 8e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.018 : 3e-4;
    double nmsThreshold = (target == DNN_TARGET_MYRIAD) ? 0.397 : 0.4;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences,
                    refBoxes, scoreDiff, iouDiff, confThreshold, nmsThreshold, size,
                    Scalar(), scale, swapRB, crop, absoluteCoords);
}

TEST_P(Test_Model, DetectionOutput)
{
    std::vector<int> refClassIds = {7, 12};
    std::vector<float> refConfidences = {0.991359f, 0.94786f};
    std::vector<Rect2d> refBoxes = {Rect2d(491.822, 81.1668, 211.751, 98.0672),
                                    Rect2d(132.093, 223.903, 206.984, 343.257)};

    std::string img_path = _tf("dog416.png");
    std::string weights_file = _tf("resnet50_rfcn_final.caffemodel");
    std::string config_file = _tf("rfcn_pascal_voc_resnet50.prototxt");

    Scalar mean = Scalar(102.9801, 115.9465, 122.7717);
    float scale = 1.0;
    Size size{800, 600};
    bool swapRB = false;

    double scoreDiff = (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) ?
                        4e-3 : default_l1;
    double iouDiff = 0.011;
    float confThreshold = 0.8;
    float nmsThreshold = 0;

    testDetectModel(weights_file, config_file, img_path, refClassIds, refConfidences, refBoxes,
                    scoreDiff, iouDiff, confThreshold, nmsThreshold, size, mean, scale, swapRB);
}


INSTANTIATE_TEST_CASE_P(/**/, Test_Model, dnnBackendsAndTargets());

}} // namespace
