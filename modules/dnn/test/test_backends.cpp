// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/ocl.hpp"

namespace opencv_test { namespace {

class DNNTestNetwork : public DNNTestLayer
{
public:
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
                    double l1 = 0.0, double lInf = 0.0, double detectionConfThresh = 0.2)
    {
        checkBackend();
        l1 = l1 ? l1 : default_l1;
        lInf = lInf ? lInf : default_lInf;

        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto, false);

        // Create two networks - with default backend and target and a tested one.
        Net netDefault = readNet(weights, proto);
        netDefault.setPreferableBackend(DNN_BACKEND_OPENCV);
        netDefault.setInput(inp);
        Mat outDefault = netDefault.forward(outputLayer).clone();

        Net net = readNet(weights, proto);
        net.setInput(inp);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        if (backend == DNN_BACKEND_HALIDE && !halideScheduler.empty())
        {
            halideScheduler = findDataFile(halideScheduler, false);
            net.setHalideScheduler(halideScheduler);
        }
        Mat out = net.forward(outputLayer).clone();

        check(outDefault, out, outputLayer, l1, lInf, detectionConfThresh, "First run");

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
        check(outDefault, out, outputLayer, l1, lInf, detectionConfThresh, "Second run");
    }

    void check(Mat& ref, Mat& out, const std::string& outputLayer, double l1, double lInf,
               double detectionConfThresh, const char* msg)
    {
        if (outputLayer == "detection_out")
        {
            if (backend == DNN_BACKEND_INFERENCE_ENGINE)
            {
                // Inference Engine produces detections terminated by a row which starts from -1.
                out = out.reshape(1, out.total() / 7);
                int numDetections = 0;
                while (numDetections < out.rows && out.at<float>(numDetections, 0) != -1)
                {
                    numDetections += 1;
                }
                out = out.rowRange(0, numDetections);
            }
            normAssertDetections(ref, out, msg, detectionConfThresh, l1, lInf);
        }
        else
            normAssert(ref, out, msg, l1, lInf);
    }
};

TEST_P(DNNTestNetwork, AlexNet)
{
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
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE) ||
        (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16))
        throw SkipTestException("");
    processNet("dnn/Enet-model-best.net", "", Size(512, 512), "l367_Deconvolution",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_enet.yml" :
                                             "dnn/halide_scheduler_enet.yml",
               2e-5, 0.15);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_Caffe)
{
    if (backend == DNN_BACKEND_HALIDE)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    float diffScores = (target == DNN_TARGET_OPENCL_FP16) ? 6e-3 : 0.0;
    processNet("dnn/MobileNetSSD_deploy.caffemodel", "dnn/MobileNetSSD_deploy.prototxt",
               inp, "detection_out", "", diffScores);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_v1_TensorFlow)
{
    if (backend == DNN_BACKEND_HALIDE)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    float l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.011 : 0.0;
    float lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.06 : 0.0;
    processNet("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", "dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt",
               inp, "detection_out", "", l1, lInf);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_v2_TensorFlow)
{
    if (backend == DNN_BACKEND_HALIDE)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    float l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.011 : 0.0;
    float lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.062 : 0.0;
    processNet("dnn/ssd_mobilenet_v2_coco_2018_03_29.pb", "dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
               inp, "detection_out", "", l1, lInf, 0.25);
}

TEST_P(DNNTestNetwork, SSD_VGG16)
{
    if (backend == DNN_BACKEND_HALIDE && target == DNN_TARGET_CPU)
        throw SkipTestException("");
    double scoreThreshold = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.0252 : 0.0;
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    processNet("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel",
               "dnn/ssd_vgg16.prototxt", inp, "detection_out", "", scoreThreshold);
}

TEST_P(DNNTestNetwork, OpenPose_pose_coco)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
    processNet("dnn/openpose_pose_coco.caffemodel", "dnn/openpose_pose_coco.prototxt",
               Size(368, 368));
}

TEST_P(DNNTestNetwork, OpenPose_pose_mpi)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi.prototxt",
               Size(368, 368));
}

TEST_P(DNNTestNetwork, OpenPose_pose_mpi_faster_4_stages)
{
    if (backend == DNN_BACKEND_HALIDE ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
    // The same .caffemodel but modified .prototxt
    // See https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi_faster_4_stages.prototxt",
               Size(368, 368));
}

TEST_P(DNNTestNetwork, OpenFace)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_RELEASE < 2018030000
    if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
        throw SkipTestException("Test is enabled starts from OpenVINO 2018R3");
#endif
    if (backend == DNN_BACKEND_HALIDE ||
        (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16))
        throw SkipTestException("");
    processNet("dnn/openface_nn4.small2.v1.t7", "", Size(96, 96), "");
}

TEST_P(DNNTestNetwork, opencv_face_detector)
{
    if (backend == DNN_BACKEND_HALIDE)
        throw SkipTestException("");
    Mat img = imread(findDataFile("gpu/lbpcascade/er.png", false));
    Mat inp = blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);
    processNet("dnn/opencv_face_detector.caffemodel", "dnn/opencv_face_detector.prototxt",
               inp, "detection_out");
}

TEST_P(DNNTestNetwork, Inception_v2_SSD_TensorFlow)
{
    if (backend == DNN_BACKEND_HALIDE)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/street.png", false));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    float l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.015 : 0.0;
    float lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.0731 : 0.0;
    processNet("dnn/ssd_inception_v2_coco_2017_11_17.pb", "dnn/ssd_inception_v2_coco_2017_11_17.pbtxt",
               inp, "detection_out", "", l1, lInf);
}

TEST_P(DNNTestNetwork, DenseNet_121)
{
    if (backend == DNN_BACKEND_HALIDE)
        throw SkipTestException("");

    float l1 = 0.0, lInf = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 9e-3; lInf = 5e-2;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        l1 = 6e-2; lInf = 0.27;
    }
    processNet("dnn/DenseNet_121.caffemodel", "dnn/DenseNet_121.prototxt", Size(224, 224), "", "", l1, lInf);
}

TEST_P(DNNTestNetwork, FastNeuralStyle_eccv16)
{
    if (backend == DNN_BACKEND_HALIDE ||
        (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) ||
        (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD))
        throw SkipTestException("");
    Mat img = imread(findDataFile("dnn/googlenet_1.png", false));
    Mat inp = blobFromImage(img, 1.0, Size(320, 240), Scalar(103.939, 116.779, 123.68), false, false);
    // Output image has values in range [-143.526, 148.539].
    float l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.3 : 4e-5;
    float lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 7.0 : 2e-3;
    processNet("dnn/fast_neural_style_eccv16_starry_night.t7", "", inp, "", "", l1, lInf);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, dnnBackendsAndTargets(true, true, false, true));

}} // namespace
