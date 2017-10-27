// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Test for Tensorflow models loading
*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_TensorFlow, read_inception)
{
    Net net;
    {
        const string model = findDataFile("dnn/tensorflow_inception_graph.pb", false);
        net = readNetFromTensorflow(model);
        ASSERT_FALSE(net.empty());
    }

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());
    Mat input;
    resize(sample, input, Size(224, 224));
    input -= 128; // mean sub

    Mat inputBlob = blobFromImage(input);

    net.setInput(inputBlob, "input");
    Mat out = net.forward("softmax2");

    std::cout << out.dims << std::endl;
}

TEST(Test_TensorFlow, inception_accuracy)
{
    Net net;
    {
        const string model = findDataFile("dnn/tensorflow_inception_graph.pb", false);
        net = readNetFromTensorflow(model);
        ASSERT_FALSE(net.empty());
    }

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());
    resize(sample, sample, Size(224, 224));
    Mat inputBlob = blobFromImage(sample);

    net.setInput(inputBlob, "input");
    Mat out = net.forward("softmax2");

    Mat ref = blobFromNPY(_tf("tf_inception_prob.npy"));

    normAssert(ref, out);
}

static std::string path(const std::string& file)
{
    return findDataFile("dnn/tensorflow/" + file, false);
}

static void runTensorFlowNet(const std::string& prefix, bool hasText = false,
                             double l1 = 1e-5, double lInf = 1e-4,
                             bool memoryLoad = false)
{
    std::string netPath = path(prefix + "_net.pb");
    std::string netConfig = (hasText ? path(prefix + "_net.pbtxt") : "");
    std::string inpPath = path(prefix + "_in.npy");
    std::string outPath = path(prefix + "_out.npy");

    Net net;
    if (memoryLoad)
    {
        // Load files into a memory buffers
        string dataModel;
        ASSERT_TRUE(readFileInMemory(netPath, dataModel));

        string dataConfig;
        if (hasText)
            ASSERT_TRUE(readFileInMemory(netConfig, dataConfig));

        net = readNetFromTensorflow(dataModel.c_str(), dataModel.size(),
                                    dataConfig.c_str(), dataConfig.size());
    }
    else
        net = readNetFromTensorflow(netPath, netConfig);

    ASSERT_FALSE(net.empty());

    cv::Mat input = blobFromNPY(inpPath);
    cv::Mat target = blobFromNPY(outPath);

    net.setInput(input);
    cv::Mat output = net.forward();
    normAssert(target, output, "", l1, lInf);
}

TEST(Test_TensorFlow, conv)
{
    runTensorFlowNet("single_conv");
    runTensorFlowNet("atrous_conv2d_valid");
    runTensorFlowNet("atrous_conv2d_same");
    runTensorFlowNet("depthwise_conv2d");
}

TEST(Test_TensorFlow, padding)
{
    runTensorFlowNet("padding_same");
    runTensorFlowNet("padding_valid");
    runTensorFlowNet("spatial_padding");
}

TEST(Test_TensorFlow, eltwise_add_mul)
{
    runTensorFlowNet("eltwise_add_mul");
}

TEST(Test_TensorFlow, pad_and_concat)
{
    runTensorFlowNet("pad_and_concat");
}

TEST(Test_TensorFlow, batch_norm)
{
    runTensorFlowNet("batch_norm");
    runTensorFlowNet("fused_batch_norm");
    runTensorFlowNet("batch_norm_text", true);
}

TEST(Test_TensorFlow, pooling)
{
    runTensorFlowNet("max_pool_even");
    runTensorFlowNet("max_pool_odd_valid");
    runTensorFlowNet("max_pool_odd_same");
}

TEST(Test_TensorFlow, deconvolution)
{
    runTensorFlowNet("deconvolution");
}

TEST(Test_TensorFlow, matmul)
{
    runTensorFlowNet("matmul");
}

TEST(Test_TensorFlow, defun)
{
    runTensorFlowNet("defun_dropout");
}

TEST(Test_TensorFlow, reshape)
{
    runTensorFlowNet("shift_reshape_no_reorder");
    runTensorFlowNet("reshape_reduce");
    runTensorFlowNet("flatten", true);
}

TEST(Test_TensorFlow, fp16)
{
    const float l1 = 1e-3;
    const float lInf = 1e-2;
    runTensorFlowNet("fp16_single_conv", false, l1, lInf);
    runTensorFlowNet("fp16_deconvolution", false, l1, lInf);
    runTensorFlowNet("fp16_max_pool_odd_same", false, l1, lInf);
    runTensorFlowNet("fp16_padding_valid", false, l1, lInf);
    runTensorFlowNet("fp16_eltwise_add_mul", false, l1, lInf);
    runTensorFlowNet("fp16_max_pool_odd_valid", false, l1, lInf);
    runTensorFlowNet("fp16_pad_and_concat", false, l1, lInf);
    runTensorFlowNet("fp16_max_pool_even", false, l1, lInf);
    runTensorFlowNet("fp16_padding_same", false, l1, lInf);
}

TEST(Test_TensorFlow, MobileNet_SSD)
{
    std::string netPath = findDataFile("dnn/ssd_mobilenet_v1_coco.pb", false);
    std::string netConfig = findDataFile("dnn/ssd_mobilenet_v1_coco.pbtxt", false);
    std::string imgPath = findDataFile("dnn/street.png", false);

    Mat inp;
    resize(imread(imgPath), inp, Size(300, 300));
    inp = blobFromImage(inp, 1.0f / 127.5, Size(), Scalar(127.5, 127.5, 127.5), true);

    std::vector<String> outNames(3);
    outNames[0] = "concat";
    outNames[1] = "concat_1";
    outNames[2] = "detection_out";

    std::vector<Mat> target(outNames.size());
    for (int i = 0; i < outNames.size(); ++i)
    {
        std::string path = findDataFile("dnn/tensorflow/ssd_mobilenet_v1_coco." + outNames[i] + ".npy", false);
        target[i] = blobFromNPY(path);
    }

    Net net = readNetFromTensorflow(netPath, netConfig);
    net.setInput(inp);

    std::vector<Mat> output;
    net.forward(output, outNames);

    normAssert(target[0].reshape(1, 1), output[0].reshape(1, 1));
    normAssert(target[1].reshape(1, 1), output[1].reshape(1, 1), "", 1e-5, 2e-4);
    normAssert(target[2].reshape(1, 1), output[2].reshape(1, 1), "", 4e-5, 1e-2);
}

TEST(Test_TensorFlow, lstm)
{
    runTensorFlowNet("lstm", true);
}

TEST(Test_TensorFlow, split)
{
    runTensorFlowNet("split_equals");
}

TEST(Test_TensorFlow, resize_nearest_neighbor)
{
    runTensorFlowNet("resize_nearest_neighbor");
}

TEST(Test_TensorFlow, memory_read)
{
    double l1 = 1e-5;
    double lInf = 1e-4;
    runTensorFlowNet("lstm", true, l1, lInf, true);

    runTensorFlowNet("batch_norm", false, l1, lInf, true);
    runTensorFlowNet("fused_batch_norm", false, l1, lInf, true);
    runTensorFlowNet("batch_norm_text", true, l1, lInf, true);
}

}
