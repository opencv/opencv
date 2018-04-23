// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/ocl.hpp"

namespace opencv_test { namespace {

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
                    Size inpSize, const std::string& outputLayer = "",
                    const std::string& halideScheduler = "",
                    double l1 = 0.0, double lInf = 0.0)
    {
        // Create a common input blob.
        int blobSize[] = {1, 3, inpSize.height, inpSize.width};
        Mat inp(4, blobSize, CV_32FC1);
        randu(inp, 0.0f, 1.0f);

        processNet(weights, proto, inp, outputLayer, halideScheduler, l1, lInf);
    }

    void processNet(std::string weights, std::string proto,
                    Mat inp, const std::string& outputLayer = "",
                    std::string halideScheduler = "",
                    double l1 = 0.0, double lInf = 0.0)
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
        if (target == DNN_TARGET_OPENCL_FP16)
        {
            l1 = l1 == 0.0 ? 4e-3 : l1;
            lInf = lInf == 0.0 ? 2e-2 : lInf;
        }
        else
        {
            l1 = l1 == 0.0 ? 1e-5 : l1;
            lInf = lInf == 0.0 ? 1e-4 : lInf;
        }
        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto, false);

        // Create two networks - with default backend and target and a tested one.
        Net netDefault = readNet(weights, proto);
        Net net = readNet(weights, proto);

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
            normAssertDetections(outDefault, out, "First run", 0.2, l1, lInf);
        else
            normAssert(outDefault, out, "First run", l1, lInf);

        // Test 2: change input.
        float* inpData = (float*)inp.data;
        for (int i = 0; i < inp.size[0] * inp.size[1]; ++i)
        {
            Mat slice(inp.size[2], inp.size[3], CV_32F, inpData);
            cv::flip(slice, slice, 1);
            inpData += slice.total();
        }
        netDefault.setInput(inp);
        net.setInput(inp);
        outDefault = netDefault.forward(outputLayer).clone();
        out = net.forward(outputLayer).clone();

        if (outputLayer == "detection_out")
            normAssertDetections(outDefault, out, "Second run", 0.2, l1, lInf);
        else
            normAssert(outDefault, out, "Second run", l1, lInf);
    }
};

TEST_P(DNNTestNetwork, AlexNet)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    processNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt",
               Size(227, 227), "prob",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_alexnet.yml" :
                                             "dnn/halide_scheduler_alexnet.yml");
}

TEST_P(DNNTestNetwork, ResNet_50)
{
    processNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt",
               Size(224, 224), "prob",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_resnet_50.yml" :
                                             "dnn/halide_scheduler_resnet_50.yml");
}

TEST_P(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt",
               Size(227, 227), "prob",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_squeezenet_v1_1.yml" :
                                             "dnn/halide_scheduler_squeezenet_v1_1.yml");
}

TEST_P(DNNTestNetwork, GoogLeNet)
{
    processNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt",
               Size(224, 224), "prob");
}

TEST_P(DNNTestNetwork, Inception_5h)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE) throw SkipTestException("");
    processNet("dnn/tensorflow_inception_graph.pb", "", Size(224, 224), "softmax2",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_inception_5h.yml" :
                                             "dnn/halide_scheduler_inception_5h.yml");
}

TEST_P(DNNTestNetwork, ENet)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE) throw SkipTestException("");
    processNet("dnn/Enet-model-best.net", "", Size(512, 512), "l367_Deconvolution",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_enet.yml" :
                                             "dnn/halide_scheduler_enet.yml",
               2e-5, 0.15);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_Caffe)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);

    processNet("dnn/MobileNetSSD_deploy.caffemodel", "dnn/MobileNetSSD_deploy.prototxt",
               inp, "detection_out");
}

TEST_P(DNNTestNetwork, MobileNet_SSD_TensorFlow)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    processNet("dnn/ssd_mobilenet_v1_coco.pb", "dnn/ssd_mobilenet_v1_coco.pbtxt",
               inp, "detection_out");
}

TEST_P(DNNTestNetwork, SSD_VGG16)
{
    if (backend == DNN_BACKEND_DEFAULT && target == DNN_TARGET_OPENCL ||
        backend == DNN_BACKEND_HALIDE && target == DNN_TARGET_CPU ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    processNet("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel",
               "dnn/ssd_vgg16.prototxt", Size(300, 300), "detection_out");
}

TEST_P(DNNTestNetwork, OpenPose_pose_coco)
{
    if (backend == DNN_BACKEND_HALIDE) throw SkipTestException("");
    processNet("dnn/openpose_pose_coco.caffemodel", "dnn/openpose_pose_coco.prototxt",
               Size(368, 368));
}

TEST_P(DNNTestNetwork, OpenPose_pose_mpi)
{
    if (backend == DNN_BACKEND_HALIDE) throw SkipTestException("");
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi.prototxt",
               Size(368, 368));
}

TEST_P(DNNTestNetwork, OpenPose_pose_mpi_faster_4_stages)
{
    if (backend == DNN_BACKEND_HALIDE) throw SkipTestException("");
    // The same .caffemodel but modified .prototxt
    // See https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi_faster_4_stages.prototxt",
               Size(368, 368));
}

TEST_P(DNNTestNetwork, OpenFace)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    processNet("dnn/openface_nn4.small2.v1.t7", "", Size(96, 96), "");
}

TEST_P(DNNTestNetwork, opencv_face_detector)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    Mat img = imread(findDataFile("gpu/lbpcascade/er.png", false));
    Mat inp = blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);
    processNet("dnn/opencv_face_detector.caffemodel", "dnn/opencv_face_detector.prototxt",
               inp, "detection_out");
}

TEST_P(DNNTestNetwork, Inception_v2_SSD_TensorFlow)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target != DNN_TARGET_CPU)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    processNet("dnn/ssd_inception_v2_coco_2017_11_17.pb", "dnn/ssd_inception_v2_coco_2017_11_17.pbtxt",
               inp, "detection_out");
}

TEST_P(DNNTestNetwork, DenseNet_121)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    processNet("dnn/DenseNet_121.caffemodel", "dnn/DenseNet_121.prototxt", Size(224, 224), "", "caffe");
}

const tuple<DNNBackend, DNNTarget> testCases[] = {
#ifdef HAVE_HALIDE
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_HALIDE, DNN_TARGET_CPU),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_HALIDE, DNN_TARGET_OPENCL),
#endif
#ifdef HAVE_INF_ENGINE
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_CPU),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_OPENCL),
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_INFERENCE_ENGINE, DNN_TARGET_OPENCL_FP16),
#endif
    tuple<DNNBackend, DNNTarget>(DNN_BACKEND_DEFAULT, DNN_TARGET_OPENCL)
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, testing::ValuesIn(testCases));

}} // namespace
