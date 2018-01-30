// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/ocl.hpp"

namespace cvtest {

using namespace cv;
using namespace dnn;
using namespace testing;

CV_ENUM(DNNBackend, DNN_BACKEND_DEFAULT, DNN_BACKEND_HALIDE)
CV_ENUM(DNNTarget, DNN_TARGET_CPU, DNN_TARGET_OPENCL)

static void loadNet(const std::string& weights, const std::string& proto,
                    const std::string& framework, Net* net)
{
    if (framework == "caffe")
        *net = cv::dnn::readNetFromCaffe(proto, weights);
    else if (framework == "torch")
        *net = cv::dnn::readNetFromTorch(weights);
    else if (framework == "tensorflow")
        *net = cv::dnn::readNetFromTensorflow(weights, proto);
    else
        CV_Error(Error::StsNotImplemented, "Unknown framework " + framework);
}

class DNNTestNetwork : public TestWithParam <tuple<DNNBackend, DNNTarget> >
{
public:
    dnn::Backend backend;
    dnn::Target target;

    DNNTestNetwork()
    {
        backend = (dnn::Backend)(int)get<0>(GetParam());
        target = (dnn::Target)(int)get<1>(GetParam());
    }

    void processNet(const std::string& weights, const std::string& proto,
                    Size inpSize, const std::string& outputLayer,
                    const std::string& framework, const std::string& halideScheduler = "",
                    double l1 = 1e-5, double lInf = 1e-4)
    {
        // Create a common input blob.
        int blobSize[] = {1, 3, inpSize.height, inpSize.width};
        Mat inp(4, blobSize, CV_32FC1);
        randu(inp, 0.0f, 1.0f);

        processNet(weights, proto, inp, outputLayer, framework, halideScheduler, l1, lInf);
    }

    void processNet(std::string weights, std::string proto,
                    Mat inp, const std::string& outputLayer,
                    const std::string& framework, std::string halideScheduler = "",
                    double l1 = 1e-5, double lInf = 1e-4)
    {
        if (backend == DNN_BACKEND_DEFAULT && target == DNN_TARGET_OPENCL)
        {
#ifdef HAVE_OPENCL
            if (!cv::ocl::useOpenCL())
#endif
            {
                throw SkipTestException("OpenCL is not available/disabled in OpenCV");
            }
        }
        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto, false);

        // Create two networks - with default backend and target and a tested one.
        Net netDefault, net;
        loadNet(weights, proto, framework, &netDefault);
        loadNet(weights, proto, framework, &net);

        netDefault.setInput(inp);
        Mat outDefault = netDefault.forward(outputLayer).clone();

        net.setInput(inp);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        if (backend == DNN_BACKEND_HALIDE && !halideScheduler.empty())
        {
            halideScheduler = findDataFile(halideScheduler, false);
            net.setHalideScheduler(halideScheduler);
        }
        Mat out = net.forward(outputLayer).clone();

        if (outputLayer == "detection_out")
            checkDetections(outDefault, out, "First run", l1, lInf);
        else
            normAssert(outDefault, out, "First run", l1, lInf);

        // Test 2: change input.
        inp *= 0.1f;
        netDefault.setInput(inp);
        net.setInput(inp);
        outDefault = netDefault.forward(outputLayer).clone();
        out = net.forward(outputLayer).clone();

        if (outputLayer == "detection_out")
            checkDetections(outDefault, out, "Second run", l1, lInf);
        else
            normAssert(outDefault, out, "Second run", l1, lInf);
    }

    void checkDetections(const Mat& out, const Mat& ref, const std::string& msg,
                         float l1, float lInf, int top = 5)
    {
        top = std::min(std::min(top, out.size[2]), out.size[3]);
        std::vector<cv::Range> range(4, cv::Range::all());
        range[2] = cv::Range(0, top);
        normAssert(out(range), ref(range));
    }
};

TEST_P(DNNTestNetwork, AlexNet)
{
    processNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt",
               Size(227, 227), "prob", "caffe",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_alexnet.yml" :
                                             "dnn/halide_scheduler_alexnet.yml");
}

TEST_P(DNNTestNetwork, ResNet_50)
{
    processNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt",
               Size(224, 224), "prob", "caffe",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_resnet_50.yml" :
                                             "dnn/halide_scheduler_resnet_50.yml");
}

TEST_P(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt",
               Size(227, 227), "prob", "caffe",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_squeezenet_v1_1.yml" :
                                             "dnn/halide_scheduler_squeezenet_v1_1.yml");
}

TEST_P(DNNTestNetwork, GoogLeNet)
{
    processNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt",
               Size(224, 224), "prob", "caffe");
}

TEST_P(DNNTestNetwork, Inception_5h)
{
    processNet("dnn/tensorflow_inception_graph.pb", "", Size(224, 224), "softmax2", "tensorflow",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_inception_5h.yml" :
                                             "dnn/halide_scheduler_inception_5h.yml");
}

TEST_P(DNNTestNetwork, ENet)
{
    processNet("dnn/Enet-model-best.net", "", Size(512, 512), "l367_Deconvolution", "torch",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_enet.yml" :
                                             "dnn/halide_scheduler_enet.yml",
               2e-5, 0.15);
}

TEST_P(DNNTestNetwork, MobileNetSSD)
{
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);

    processNet("dnn/MobileNetSSD_deploy.caffemodel", "dnn/MobileNetSSD_deploy.prototxt",
               inp, "detection_out", "caffe");
}

TEST_P(DNNTestNetwork, SSD_VGG16)
{
    if (backend == DNN_BACKEND_DEFAULT && target == DNN_TARGET_OPENCL ||
        backend == DNN_BACKEND_HALIDE && target == DNN_TARGET_CPU)
        throw SkipTestException("");
    processNet("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel",
               "dnn/ssd_vgg16.prototxt", Size(300, 300), "detection_out", "caffe");
}

const tuple<DNNBackend, DNNTarget> testCases[] = {
#ifdef HAVE_HALIDE
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_HALIDE, DNN_TARGET_CPU),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_HALIDE, DNN_TARGET_OPENCL),
#endif
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_DEFAULT, DNN_TARGET_OPENCL)
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, ValuesIn(testCases));

}  // namespace cvtest
