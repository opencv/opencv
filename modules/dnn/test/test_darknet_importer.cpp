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

TEST(Test_Darknet, read_yolo_voc_stream)
{
    Mat ref;
    Mat sample = imread(_tf("dog416.png"));
    Mat inp = blobFromImage(sample, 1.0/255, Size(416, 416), Scalar(), true, false);
    const std::string cfgFile = findDataFile("dnn/yolo-voc.cfg", false);
    const std::string weightsFile = findDataFile("dnn/yolo-voc.weights", false);
    // Import by paths.
    {
        Net net = readNetFromDarknet(cfgFile, weightsFile);
        net.setInput(inp);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        ref = net.forward();
    }
    // Import from bytes array.
    {
        std::string cfg, weights;
        readFileInMemory(cfgFile, cfg);
        readFileInMemory(weightsFile, weights);

        Net net = readNetFromDarknet(&cfg[0], cfg.size(), &weights[0], weights.size());
        net.setInput(inp);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        Mat out = net.forward();
        normAssert(ref, out);
    }
}

class Test_Darknet_layers : public DNNTestLayer
{
public:
    void testDarknetLayer(const std::string& name, bool hasWeights = false)
    {
        std::string cfg = findDataFile("dnn/darknet/" + name + ".cfg", false);
        std::string model = "";
        if (hasWeights)
            model = findDataFile("dnn/darknet/" + name + ".weights", false);
        Mat inp = blobFromNPY(findDataFile("dnn/darknet/" + name + "_in.npy", false));
        Mat ref = blobFromNPY(findDataFile("dnn/darknet/" + name + "_out.npy", false));

        checkBackend(&inp, &ref);

        Net net = readNet(cfg, model);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(inp);
        Mat out = net.forward();
        normAssert(out, ref, "", default_l1, default_lInf);
    }
};

class Test_Darknet_nets : public DNNTestLayer
{
public:
    // Test object detection network from Darknet framework.
    void testDarknetModel(const std::string& cfg, const std::string& weights,
                          const std::vector<cv::String>& outNames,
                          const std::vector<std::vector<int> >& refClassIds,
                          const std::vector<std::vector<float> >& refConfidences,
                          const std::vector<std::vector<Rect2d> >& refBoxes,
                          double scoreDiff, double iouDiff, float confThreshold = 0.24)
    {
        checkBackend();

        Mat img1 = imread(_tf("dog416.png"));
        Mat img2 = imread(_tf("street.png"));
        std::vector<Mat> samples {img1, img2};
        Mat inp = blobFromImages(samples, 1.0/255, Size(416, 416), Scalar(), true, false);

        Net net = readNet(findDataFile("dnn/" + cfg, false),
                          findDataFile("dnn/" + weights, false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(inp);
        std::vector<Mat> outs;
        net.forward(outs, outNames);

        int batch_size = refClassIds.size();

        for (int b = 0; b < batch_size; ++b)
        {
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect2d> boxes;
            for (int i = 0; i < outs.size(); ++i)
            {
                Mat out;
                if (batch_size < 2){
                    out = outs[i];
                }else{
                    // get the sample slice from 3D matrix (batch, box, classes+5)
                    Range ranges[3] = {Range(b, b+1), Range::all(), Range::all()};
                    out = outs[i](ranges).reshape(1, outs[i].size[1]);
                }
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
            normAssertDetections(refClassIds[b], refConfidences[b], refBoxes[b], classIds,
                             confidences, boxes, format("batch size %d\n", batch_size).c_str(), confThreshold, scoreDiff, iouDiff);
        }
    }

    void testDarknetModel(const std::string& cfg, const std::string& weights,
                          const std::vector<cv::String>& outNames,
                          const std::vector<int>& refClassIds,
                          const std::vector<float>& refConfidences,
                          const std::vector<Rect2d>& refBoxes,
                          double scoreDiff, double iouDiff, float confThreshold = 0.24)
    {
        testDarknetModel(cfg, weights, outNames,
                         std::vector<std::vector<int> >(1, refClassIds),
                         std::vector<std::vector<float> >(1, refConfidences),
                         std::vector<std::vector<Rect2d> >(1, refBoxes),
                         scoreDiff, iouDiff, confThreshold = 0.24);
    }
};

TEST_P(Test_Darknet_nets, YoloVoc)
{
    std::vector<cv::String> outNames(1, "detection_out");

    std::vector<std::vector<int> > classIds(2);
    std::vector<std::vector<float> > confidences(2);
    std::vector<std::vector<Rect2d> > boxes(2);
    classIds[0].resize(3);
    confidences[0].resize(3);
    boxes[0].resize(3);

    // detections dog416.png
    classIds[0][0] = 6;  confidences[0][0] = 0.750469f; boxes[0][0] = Rect2d(0.577374, 0.127391, 0.325575, 0.173418);  // a car
    classIds[0][1] = 1;  confidences[0][1] = 0.780879f; boxes[0][1] = Rect2d(0.270762, 0.264102, 0.461713, 0.48131); // a bicycle
    classIds[0][2] = 11; confidences[0][2] = 0.901615f; boxes[0][2] = Rect2d(0.1386, 0.338509, 0.282737, 0.60028);  // a dog

    classIds[1].resize(4);
    confidences[1].resize(4);
    boxes[1].resize(4);

    // detections street.png
    classIds[1][0] = 14;  confidences[1][0] = 0.606976f; boxes[1][0] = Rect2d(0.256026f, 0.486801f, 0.080365f, 0.261640f);  // a person
    classIds[1][1] = 14;  confidences[1][1] = 0.610263f; boxes[1][1] = Rect2d(0.224827f, 0.470590f, 0.079679f, 0.299220f);  // a person
    classIds[1][2] = 6;  confidences[1][2] = 0.666661f; boxes[1][2] = Rect2d(0.549544f, 0.558468f, 0.065900f, 0.080578f);  // a car
    classIds[1][3] = 6;  confidences[1][3] = 0.846023f; boxes[1][3] = Rect2d(0.784135f, 0.566773f, 0.235595f, 0.250844f);  // a car

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 1e-2 : 8e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.013 : 3e-5;

    std::string config_file = "yolo-voc.cfg";
    std::string weights_file = "yolo-voc.weights";

    // batch size 1
    testDarknetModel(config_file, weights_file, outNames,
                     classIds[0], confidences[0], boxes[0], scoreDiff, iouDiff);

    // batch size 2
    testDarknetModel(config_file, weights_file, outNames,
                     classIds, confidences, boxes, scoreDiff, iouDiff);
}

TEST_P(Test_Darknet_nets, TinyYoloVoc)
{
    std::vector<cv::String> outNames(1, "detection_out");

    std::vector<std::vector<int> > classIds(2);
    std::vector<std::vector<float> > confidences(2);
    std::vector<std::vector<Rect2d> > boxes(2);

    classIds[0].resize(2);
    confidences[0].resize(2);
    boxes[0].resize(2);

    // dog416.png
    classIds[0][0] = 6;  confidences[0][0] = 0.761967f; boxes[0][0] = Rect2d(0.579042, 0.159161, 0.31544, 0.160779);  // a car
    classIds[0][1] = 11; confidences[0][1] = 0.780595f; boxes[0][1] = Rect2d(0.129696, 0.386467, 0.315579, 0.534527);  // a dog

    classIds[1].resize(2);
    confidences[1].resize(2);
    boxes[1].resize(2);

    // street.png
    classIds[1][0] = 6;  confidences[1][0] = 0.654414f; boxes[1][0] = Rect2d(0.566171f, 0.563864f, 0.077476f, 0.094536f);  // a car
    classIds[1][1] = 6;  confidences[1][1] = 0.927104f; boxes[1][1] = Rect2d(0.800025f, 0.569567f, 0.213016f, 0.237674f);  // a car

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 8e-3 : 8e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 8e-3 : 3e-5;

    std::string config_file = "tiny-yolo-voc.cfg";
    std::string weights_file = "tiny-yolo-voc.weights";

    // batch size 1
    testDarknetModel(config_file, weights_file, outNames,
                     classIds[0], confidences[0], boxes[0], scoreDiff, iouDiff);

    // batch size 2
    testDarknetModel(config_file, weights_file, outNames,
                     classIds, confidences, boxes, scoreDiff, iouDiff);
}

TEST_P(Test_Darknet_nets, YOLOv3)
{
    std::vector<cv::String> outNames(3);
    outNames[0] = "yolo_82";
    outNames[1] = "yolo_94";
    outNames[2] = "yolo_106";

    std::vector<std::vector<int> > classIds(2);
    std::vector<std::vector<float> > confidences(2);
    std::vector<std::vector<Rect2d> > boxes(2);

    classIds[0].resize(3);
    confidences[0].resize(3);
    boxes[0].resize(3);

    classIds[0][0] = 7;  confidences[0][0] = 0.952983f; boxes[0][0] = Rect2d(0.614622, 0.150257, 0.286747, 0.138994);  // a truck
    classIds[0][1] = 1; confidences[0][1] = 0.987908f; boxes[0][1] = Rect2d(0.150913, 0.221933, 0.591342, 0.524327);  // a bicycle
    classIds[0][2] = 16; confidences[0][2] = 0.998836f; boxes[0][2] = Rect2d(0.160024, 0.389964, 0.257861, 0.553752);  // a dog (COCO)

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 4e-3 : 8e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.011 : 3e-5;

    std::string config_file = "yolov3.cfg";
    std::string weights_file = "yolov3.weights";

    // batch size 1
    testDarknetModel(config_file, weights_file, outNames,
                     classIds[0], confidences[0], boxes[0], scoreDiff, iouDiff);

    if (backend != DNN_BACKEND_INFERENCE_ENGINE || target != DNN_TARGET_MYRIAD)
    {
        // batch size 2
        testDarknetModel(config_file, weights_file, outNames,
                        classIds, confidences, boxes, scoreDiff, iouDiff);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Darknet_nets, dnnBackendsAndTargets());

TEST_P(Test_Darknet_layers, shortcut)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_CPU)
        throw SkipTestException("");
    testDarknetLayer("shortcut");
}

TEST_P(Test_Darknet_layers, upsample)
{
    testDarknetLayer("upsample");
}

TEST_P(Test_Darknet_layers, avgpool_softmax)
{
    testDarknetLayer("avgpool_softmax");
}

TEST_P(Test_Darknet_layers, region)
{
    testDarknetLayer("region");
}

TEST_P(Test_Darknet_layers, reorg)
{
    testDarknetLayer("reorg");
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Darknet_layers, dnnBackendsAndTargets());

}} // namespace
