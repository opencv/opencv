// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
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

static void runTensorFlowNet(const std::string& prefix,
                             double l1 = 1e-5, double lInf = 1e-4)
{
    std::string netPath = path(prefix + "_net.pb");
    std::string inpPath = path(prefix + "_in.npy");
    std::string outPath = path(prefix + "_out.npy");

    Net net = readNetFromTensorflow(netPath);

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

TEST(Test_TensorFlow, fp16)
{
    const float l1 = 1e-3;
    const float lInf = 1e-2;
    runTensorFlowNet("fp16_single_conv", l1, lInf);
    runTensorFlowNet("fp16_deconvolution", l1, lInf);
    runTensorFlowNet("fp16_max_pool_odd_same", l1, lInf);
    runTensorFlowNet("fp16_padding_valid", l1, lInf);
    runTensorFlowNet("fp16_eltwise_add_mul", l1, lInf);
    runTensorFlowNet("fp16_max_pool_odd_valid", l1, lInf);
    runTensorFlowNet("fp16_pad_and_concat", l1, lInf);
    runTensorFlowNet("fp16_max_pool_even", l1, lInf);
    runTensorFlowNet("fp16_padding_same", l1, lInf);
}

}
