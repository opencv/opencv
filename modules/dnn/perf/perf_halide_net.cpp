// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

namespace cvtest
{

#ifdef HAVE_HALIDE
using namespace cv;
using namespace dnn;

static void loadNet(std::string weights, std::string proto, std::string scheduler,
                    int inWidth, int inHeight, const std::string& outputLayer,
                    const std::string& framework, int targetId, Net* net)
{
    Mat input(inHeight, inWidth, CV_32FC3);
    randu(input, 0.0f, 1.0f);

    weights = findDataFile(weights, false);
    if (!proto.empty())
        proto = findDataFile(proto, false);
    if (!scheduler.empty())
        scheduler = findDataFile(scheduler, false);
    if (framework == "caffe")
    {
        *net = cv::dnn::readNetFromCaffe(proto, weights);
    }
    else if (framework == "torch")
    {
        *net = cv::dnn::readNetFromTorch(weights);
    }
    else if (framework == "tensorflow")
    {
        *net = cv::dnn::readNetFromTensorflow(weights);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown framework " + framework);

    net->setInput(blobFromImage(input, 1.0, Size(), Scalar(), false));
    net->setPreferableBackend(DNN_BACKEND_HALIDE);
    net->setPreferableTarget(targetId);
    net->setHalideScheduler(scheduler);
    net->forward(outputLayer);
}

////////////////////////////////////////////////////////////////////////////////
// CPU target
////////////////////////////////////////////////////////////////////////////////
PERF_TEST(GoogLeNet, HalidePerfTest)
{
    Net net;
    loadNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt",
            "", 227, 227, "prob", "caffe", DNN_TARGET_CPU, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(AlexNet, HalidePerfTest)
{
    Net net;
    loadNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt",
            "dnn/halide_scheduler_alexnet.yml", 227, 227, "prob", "caffe",
            DNN_TARGET_CPU, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(ResNet50, HalidePerfTest)
{
    Net net;
    loadNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt",
            "dnn/halide_scheduler_resnet_50.yml", 224, 224, "prob", "caffe",
            DNN_TARGET_CPU, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(SqueezeNet_v1_1, HalidePerfTest)
{
    Net net;
    loadNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt",
            "dnn/halide_scheduler_squeezenet_v1_1.yml", 227, 227, "prob",
            "caffe", DNN_TARGET_CPU, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Inception_5h, HalidePerfTest)
{
    Net net;
    loadNet("dnn/tensorflow_inception_graph.pb", "",
            "dnn/halide_scheduler_inception_5h.yml",
            224, 224, "softmax2", "tensorflow", DNN_TARGET_CPU, &net);
    TEST_CYCLE() net.forward("softmax2");
    SANITY_CHECK_NOTHING();
}

PERF_TEST(ENet, HalidePerfTest)
{
    Net net;
    loadNet("dnn/Enet-model-best.net", "", "dnn/halide_scheduler_enet.yml",
            512, 256, "l367_Deconvolution", "torch", DNN_TARGET_CPU, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}
////////////////////////////////////////////////////////////////////////////////
// OpenCL target
////////////////////////////////////////////////////////////////////////////////
PERF_TEST(GoogLeNet_opencl, HalidePerfTest)
{
    Net net;
    loadNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt",
            "", 227, 227, "prob", "caffe", DNN_TARGET_OPENCL, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(AlexNet_opencl, HalidePerfTest)
{
    Net net;
    loadNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt",
            "dnn/halide_scheduler_opencl_alexnet.yml", 227, 227, "prob", "caffe",
            DNN_TARGET_OPENCL, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(ResNet50_opencl, HalidePerfTest)
{
    Net net;
    loadNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt",
            "dnn/halide_scheduler_opencl_resnet_50.yml", 224, 224, "prob", "caffe",
            DNN_TARGET_OPENCL, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}


PERF_TEST(SqueezeNet_v1_1_opencl, HalidePerfTest)
{
    Net net;
    loadNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt",
            "dnn/halide_scheduler_opencl_squeezenet_v1_1.yml", 227, 227, "prob",
            "caffe", DNN_TARGET_OPENCL, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Inception_5h_opencl, HalidePerfTest)
{
    Net net;
    loadNet("dnn/tensorflow_inception_graph.pb", "",
            "dnn/halide_scheduler_opencl_inception_5h.yml",
            224, 224, "softmax2", "tensorflow", DNN_TARGET_OPENCL, &net);
    TEST_CYCLE() net.forward("softmax2");
    SANITY_CHECK_NOTHING();
}

PERF_TEST(ENet_opencl, HalidePerfTest)
{
    Net net;
    loadNet("dnn/Enet-model-best.net", "", "dnn/halide_scheduler_opencl_enet.yml",
            512, 256, "l367_Deconvolution", "torch", DNN_TARGET_OPENCL, &net);
    TEST_CYCLE() net.forward();
    SANITY_CHECK_NOTHING();
}
#endif  // HAVE_HALIDE

}  // namespace cvtest
