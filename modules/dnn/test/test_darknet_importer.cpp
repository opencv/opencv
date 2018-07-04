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
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_Darknet, read_tiny_yolo_voc)
{
    Net net = readNetFromDarknet(_tf("tiny-yolo-voc.cfg"));
    ASSERT_FALSE(net.empty());
}

TEST(Test_Darknet, read_yolo_voc)
{
    Net net = readNetFromDarknet(_tf("yolo-voc.cfg"));
    ASSERT_FALSE(net.empty());
}

// Test object detection network from Darknet framework.
static void testDarknetModel(const std::string& cfg, const std::string& weights,
                             const std::vector<cv::String>& outNames,
                             const std::vector<int>& refClassIds,
                             const std::vector<float>& refConfidences,
                             const std::vector<Rect2d>& refBoxes,
                             int backendId, int targetId, float scoreDiff = 0.0,
                             float iouDiff = 0.0, float confThreshold = 0.24)
{
    if (backendId == DNN_BACKEND_OPENCV && targetId == DNN_TARGET_OPENCL)
    {
  #ifdef HAVE_OPENCL
        if (!cv::ocl::useOpenCL())
  #endif
        {
            throw SkipTestException("OpenCL is not available/disabled in OpenCV");
        }
    }
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE && targetId == DNN_TARGET_MYRIAD)
    {
        if (!checkMyriadTarget())
        {
            throw SkipTestException("Myriad is not available/disabled in OpenCV");
        }
    }
    Mat sample = imread(_tf("dog416.png"));
    Mat inp = blobFromImage(sample, 1.0/255, Size(416, 416), Scalar(), true, false);

    Net net = readNet(findDataFile("dnn/" + cfg, false),
                      findDataFile("dnn/" + weights, false));
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    net.setInput(inp);
    std::vector<Mat> outs;
    net.forward(outs, outNames);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;
    for (int i = 0; i < outs.size(); ++i)
    {
        Mat& out = outs[i];
        for (int j = 0; j < out.rows; ++j)
        {
            Mat scores = out.row(j).colRange(5, out.cols);
            double confidence;
            Point maxLoc;
            minMaxLoc(scores, 0, &confidence, 0, &maxLoc);

            float* detection = out.ptr<float>(j);
            double centerX = detection[0];
            double centerY = detection[1];
            double width = detection[2];
            double height = detection[3];
            boxes.push_back(Rect2d(centerX - 0.5 * width, centerY - 0.5 * height,
                                   width, height));
            confidences.push_back(confidence);
            classIds.push_back(maxLoc.x);
        }
    }
    normAssertDetections(refClassIds, refConfidences, refBoxes, classIds,
                         confidences, boxes, "", confThreshold, scoreDiff, iouDiff);
}

typedef testing::TestWithParam<tuple<DNNBackend, DNNTarget> > Test_Darknet_nets;

TEST_P(Test_Darknet_nets, YoloVoc)
{
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());
    std::vector<cv::String> outNames(1, "detection_out");

    std::vector<int> classIds(3);
    std::vector<float> confidences(3);
    std::vector<Rect2d> boxes(3);
    classIds[0] = 6;  confidences[0] = 0.750469f; boxes[0] = Rect2d(0.577374, 0.127391, 0.325575, 0.173418);  // a car
    classIds[1] = 1;  confidences[1] = 0.780879f; boxes[1] = Rect2d(0.270762, 0.264102, 0.461713, 0.48131); // a bicycle
    classIds[2] = 11; confidences[2] = 0.901615f; boxes[2] = Rect2d(0.1386, 0.338509, 0.282737, 0.60028);  // a dog
    double scoreDiff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 1e-2 : 8e-5;
    double iouDiff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 0.013 : 3e-5;
    testDarknetModel("yolo-voc.cfg", "yolo-voc.weights", outNames,
                     classIds, confidences, boxes, backendId, targetId, scoreDiff, iouDiff);
}

TEST_P(Test_Darknet_nets, TinyYoloVoc)
{
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());
    std::vector<cv::String> outNames(1, "detection_out");
    std::vector<int> classIds(2);
    std::vector<float> confidences(2);
    std::vector<Rect2d> boxes(2);
    classIds[0] = 6;  confidences[0] = 0.761967f; boxes[0] = Rect2d(0.579042, 0.159161, 0.31544, 0.160779);  // a car
    classIds[1] = 11; confidences[1] = 0.780595f; boxes[1] = Rect2d(0.129696, 0.386467, 0.315579, 0.534527);  // a dog
    double scoreDiff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 8e-3 : 8e-5;
    double iouDiff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 8e-3 : 3e-5;
    testDarknetModel("tiny-yolo-voc.cfg", "tiny-yolo-voc.weights", outNames,
                     classIds, confidences, boxes, backendId, targetId, scoreDiff, iouDiff);
}

TEST_P(Test_Darknet_nets, YOLOv3)
{
    int backendId = get<0>(GetParam());
    int targetId = get<1>(GetParam());
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE && targetId == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
    std::vector<cv::String> outNames(3);
    outNames[0] = "yolo_82";
    outNames[1] = "yolo_94";
    outNames[2] = "yolo_106";

    std::vector<int> classIds(3);
    std::vector<float> confidences(3);
    std::vector<Rect2d> boxes(3);
    classIds[0] = 7;  confidences[0] = 0.952983f; boxes[0] = Rect2d(0.614622, 0.150257, 0.286747, 0.138994);  // a truck
    classIds[1] = 1; confidences[1] = 0.987908f; boxes[1] = Rect2d(0.150913, 0.221933, 0.591342, 0.524327);  // a bicycle
    classIds[2] = 16; confidences[2] = 0.998836f; boxes[2] = Rect2d(0.160024, 0.389964, 0.257861, 0.553752);  // a dog (COCO)
    double scoreDiff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 4e-3 : 8e-5;
    double iouDiff = (targetId == DNN_TARGET_OPENCL_FP16 || targetId == DNN_TARGET_MYRIAD) ? 0.011 : 3e-5;
    testDarknetModel("yolov3.cfg", "yolov3.weights", outNames,
                     classIds, confidences, boxes, backendId, targetId, scoreDiff, iouDiff);
}

const tuple<DNNBackend, DNNTarget> testCases[] = {
#ifdef HAVE_INF_ENGINE
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_CPU),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_OPENCL),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_OPENCL_FP16),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_MYRIAD),
#endif
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_OPENCV, DNN_TARGET_CPU),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_OPENCV, DNN_TARGET_OPENCL_FP16)
};

INSTANTIATE_TEST_CASE_P(/**/, Test_Darknet_nets, testing::ValuesIn(testCases));

static void testDarknetLayer(const std::string& name, bool hasWeights = false)
{
    std::string cfg = findDataFile("dnn/darknet/" + name + ".cfg", false);
    std::string model = "";
    if (hasWeights)
        model = findDataFile("dnn/darknet/" + name + ".weights", false);
    Mat inp = blobFromNPY(findDataFile("dnn/darknet/" + name + "_in.npy", false));
    Mat ref = blobFromNPY(findDataFile("dnn/darknet/" + name + "_out.npy", false));

    Net net = readNet(cfg, model);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setInput(inp);
    Mat out = net.forward();
    normAssert(out, ref);
}

TEST(Test_Darknet, shortcut)
{
    testDarknetLayer("shortcut");
}

TEST(Test_Darknet, upsample)
{
    testDarknetLayer("upsample");
}

TEST(Test_Darknet, avgpool_softmax)
{
    testDarknetLayer("avgpool_softmax");
}

}} // namespace
