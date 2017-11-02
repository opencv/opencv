// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

namespace cvtest
{

#ifdef HAVE_HALIDE
using namespace cv;
using namespace dnn;

static void loadNet(const std::string& weights, const std::string& proto,
                    const std::string& framework, Net* net)
{
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
}

static void test(const std::string& weights, const std::string& proto,
                 const std::string& scheduler, int inWidth, int inHeight,
                 const std::string& outputLayer, const std::string& framework,
                 int targetId, double l1 = 1e-5, double lInf = 1e-4)
{
    Mat input(inHeight, inWidth, CV_32FC3), outputDefault, outputHalide;
    randu(input, 0.0f, 1.0f);

    Net netDefault, netHalide;
    loadNet(weights, proto, framework, &netDefault);
    loadNet(weights, proto, framework, &netHalide);

    netDefault.setInput(blobFromImage(input.clone(), 1.0f, Size(), Scalar(), false));
    outputDefault = netDefault.forward(outputLayer).clone();

    netHalide.setInput(blobFromImage(input.clone(), 1.0f, Size(), Scalar(), false));
    netHalide.setPreferableBackend(DNN_BACKEND_HALIDE);
    netHalide.setPreferableTarget(targetId);
    netHalide.setHalideScheduler(scheduler);
    outputHalide = netHalide.forward(outputLayer).clone();

    normAssert(outputDefault, outputHalide, "First run", l1, lInf);

    // An extra test: change input.
    input *= 0.1f;
    netDefault.setInput(blobFromImage(input.clone(), 1.0, Size(), Scalar(), false));
    netHalide.setInput(blobFromImage(input.clone(), 1.0, Size(), Scalar(), false));

    normAssert(outputDefault, outputHalide, "Second run", l1, lInf);
    std::cout << "." << std::endl;

    // Swap backends.
    netHalide.setPreferableBackend(DNN_BACKEND_DEFAULT);
    netHalide.setPreferableTarget(DNN_TARGET_CPU);
    outputDefault = netHalide.forward(outputLayer).clone();

    netDefault.setPreferableBackend(DNN_BACKEND_HALIDE);
    netDefault.setPreferableTarget(targetId);
    netDefault.setHalideScheduler(scheduler);
    outputHalide = netDefault.forward(outputLayer).clone();

    normAssert(outputDefault, outputHalide, "Swap backends", l1, lInf);
}

////////////////////////////////////////////////////////////////////////////////
// CPU target
////////////////////////////////////////////////////////////////////////////////
TEST(Reproducibility_MobileNetSSD_Halide, Accuracy)
{
    test(findDataFile("dnn/MobileNetSSD_deploy.caffemodel", false),
         findDataFile("dnn/MobileNetSSD_deploy.prototxt", false),
         "", 300, 300, "detection_out", "caffe", DNN_TARGET_CPU);
};

// TODO: Segmentation fault from time to time.
// TEST(Reproducibility_SSD_Halide, Accuracy)
// {
//     test(findDataFile("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", false),
//          findDataFile("dnn/ssd_vgg16.prototxt", false),
//          "", 300, 300, "detection_out", "caffe", DNN_TARGET_CPU);
// };

TEST(Reproducibility_GoogLeNet_Halide, Accuracy)
{
    test(findDataFile("dnn/bvlc_googlenet.caffemodel", false),
         findDataFile("dnn/bvlc_googlenet.prototxt", false),
         "", 224, 224, "prob", "caffe", DNN_TARGET_CPU);
};

TEST(Reproducibility_AlexNet_Halide, Accuracy)
{
    test(findDataFile("dnn/bvlc_alexnet.caffemodel", false),
         findDataFile("dnn/bvlc_alexnet.prototxt", false),
         findDataFile("dnn/halide_scheduler_alexnet.yml", false),
         227, 227, "prob", "caffe", DNN_TARGET_CPU);
};

TEST(Reproducibility_ResNet_50_Halide, Accuracy)
{
    test(findDataFile("dnn/ResNet-50-model.caffemodel", false),
         findDataFile("dnn/ResNet-50-deploy.prototxt", false),
         findDataFile("dnn/halide_scheduler_resnet_50.yml", false),
         224, 224, "prob", "caffe", DNN_TARGET_CPU);
};

TEST(Reproducibility_SqueezeNet_v1_1_Halide, Accuracy)
{
    test(findDataFile("dnn/squeezenet_v1.1.caffemodel", false),
         findDataFile("dnn/squeezenet_v1.1.prototxt", false),
         findDataFile("dnn/halide_scheduler_squeezenet_v1_1.yml", false),
         227, 227, "prob", "caffe", DNN_TARGET_CPU);
};

TEST(Reproducibility_Inception_5h_Halide, Accuracy)
{
    test(findDataFile("dnn/tensorflow_inception_graph.pb", false), "",
         findDataFile("dnn/halide_scheduler_inception_5h.yml", false),
         224, 224, "softmax2", "tensorflow", DNN_TARGET_CPU);
};

TEST(Reproducibility_ENet_Halide, Accuracy)
{
    test(findDataFile("dnn/Enet-model-best.net", false), "",
         findDataFile("dnn/halide_scheduler_enet.yml", false),
         512, 512, "l367_Deconvolution", "torch", DNN_TARGET_CPU, 2e-5, 0.15);
};
////////////////////////////////////////////////////////////////////////////////
// OpenCL target
////////////////////////////////////////////////////////////////////////////////
TEST(Reproducibility_MobileNetSSD_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/MobileNetSSD_deploy.caffemodel", false),
         findDataFile("dnn/MobileNetSSD_deploy.prototxt", false),
         "", 300, 300, "detection_out", "caffe", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_SSD_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", false),
         findDataFile("dnn/ssd_vgg16.prototxt", false),
         "", 300, 300, "detection_out", "caffe", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_GoogLeNet_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/bvlc_googlenet.caffemodel", false),
         findDataFile("dnn/bvlc_googlenet.prototxt", false),
         "", 227, 227, "prob", "caffe", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_AlexNet_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/bvlc_alexnet.caffemodel", false),
         findDataFile("dnn/bvlc_alexnet.prototxt", false),
         findDataFile("dnn/halide_scheduler_opencl_alexnet.yml", false),
         227, 227, "prob", "caffe", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_ResNet_50_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/ResNet-50-model.caffemodel", false),
         findDataFile("dnn/ResNet-50-deploy.prototxt", false),
         findDataFile("dnn/halide_scheduler_opencl_resnet_50.yml", false),
         224, 224, "prob", "caffe", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_SqueezeNet_v1_1_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/squeezenet_v1.1.caffemodel", false),
         findDataFile("dnn/squeezenet_v1.1.prototxt", false),
         findDataFile("dnn/halide_scheduler_opencl_squeezenet_v1_1.yml", false),
         227, 227, "prob", "caffe", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_Inception_5h_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/tensorflow_inception_graph.pb", false), "",
         findDataFile("dnn/halide_scheduler_opencl_inception_5h.yml", false),
         224, 224, "softmax2", "tensorflow", DNN_TARGET_OPENCL);
};

TEST(Reproducibility_ENet_Halide_opencl, Accuracy)
{
    test(findDataFile("dnn/Enet-model-best.net", false), "",
         findDataFile("dnn/halide_scheduler_opencl_enet.yml", false),
         512, 512, "l367_Deconvolution", "torch", DNN_TARGET_OPENCL, 2e-5, 0.14);
};
#endif  // HAVE_HALIDE

}  // namespace cvtest
