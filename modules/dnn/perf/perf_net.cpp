// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/dnn/shape_utils.hpp"

namespace
{

#ifdef HAVE_HALIDE
#define TEST_DNN_BACKEND DNN_BACKEND_DEFAULT, DNN_BACKEND_HALIDE
#else
#define TEST_DNN_BACKEND DNN_BACKEND_DEFAULT
#endif
#define TEST_DNN_TARGET DNN_TARGET_CPU, DNN_TARGET_OPENCL

CV_ENUM(DNNBackend, DNN_BACKEND_DEFAULT, DNN_BACKEND_HALIDE)
CV_ENUM(DNNTarget, DNN_TARGET_CPU, DNN_TARGET_OPENCL)

class DNNTestNetwork : public ::perf::TestBaseWithParam< tuple<DNNBackend, DNNTarget> >
{
public:
    dnn::Backend backend;
    dnn::Target target;

    dnn::Net net;

    void processNet(std::string weights, std::string proto, std::string halide_scheduler,
                        int inWidth, int inHeight, const std::string& outputLayer,
                        const std::string& framework)
    {
        backend = (dnn::Backend)(int)get<0>(GetParam());
        target = (dnn::Target)(int)get<1>(GetParam());

        if (backend == DNN_BACKEND_DEFAULT && target == DNN_TARGET_OPENCL)
        {
#if defined(HAVE_OPENCL)
            if (!cv::ocl::useOpenCL())
#endif
            {
                throw ::SkipTestException("OpenCL is not available/disabled in OpenCV");
            }
        }

        Mat input(inHeight, inWidth, CV_32FC3);
        randu(input, 0.0f, 1.0f);


        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto, false);
        if (!halide_scheduler.empty() && backend == DNN_BACKEND_HALIDE)
            halide_scheduler = findDataFile(std::string("dnn/halide_scheduler_") + (target == DNN_TARGET_OPENCL ? "opencl_" : "") + halide_scheduler, true);
        if (framework == "caffe")
        {
            net = cv::dnn::readNetFromCaffe(proto, weights);
        }
        else if (framework == "torch")
        {
            net = cv::dnn::readNetFromTorch(weights);
        }
        else if (framework == "tensorflow")
        {
            net = cv::dnn::readNetFromTensorflow(weights);
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown framework " + framework);

        net.setInput(blobFromImage(input, 1.0, Size(), Scalar(), false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        if (backend == DNN_BACKEND_HALIDE)
        {
            net.setHalideScheduler(halide_scheduler);
        }

        MatShape netInputShape = shape(1, 3, inHeight, inWidth);
        size_t weightsMemory = 0, blobsMemory = 0;
        net.getMemoryConsumption(netInputShape, weightsMemory, blobsMemory);
        int64 flops = net.getFLOPS(netInputShape);

        net.forward(outputLayer); // warmup

        std::cout << "Memory consumption:" << std::endl;
        std::cout << "    Weights(parameters): " << divUp(weightsMemory, 1u<<20) << " Mb" << std::endl;
        std::cout << "    Blobs: " << divUp(blobsMemory, 1u<<20) << " Mb" << std::endl;
        std::cout << "Calculation complexity: " << flops * 1e-9 << " GFlops" << std::endl;

        PERF_SAMPLE_BEGIN()
            net.forward();
        PERF_SAMPLE_END()

        SANITY_CHECK_NOTHING();
    }
};


PERF_TEST_P_(DNNTestNetwork, AlexNet)
{
    processNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt",
            "alexnet.yml", 227, 227, "prob", "caffe");
}

PERF_TEST_P_(DNNTestNetwork, GoogLeNet)
{
    processNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt",
            "", 224, 224, "prob", "caffe");
}

PERF_TEST_P_(DNNTestNetwork, ResNet50)
{
    processNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt",
            "resnet_50.yml", 224, 224, "prob", "caffe");
}

PERF_TEST_P_(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt",
            "squeezenet_v1_1.yml", 227, 227, "prob", "caffe");
}

PERF_TEST_P_(DNNTestNetwork, Inception_5h)
{
    processNet("dnn/tensorflow_inception_graph.pb", "",
            "inception_5h.yml",
            224, 224, "softmax2", "tensorflow");
}

PERF_TEST_P_(DNNTestNetwork, ENet)
{
    processNet("dnn/Enet-model-best.net", "", "enet.yml",
            512, 256, "l367_Deconvolution", "torch");
}


INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork,
    testing::Combine(
        ::testing::Values(TEST_DNN_BACKEND),
        DNNTarget::all()
    )
);

} // namespace
