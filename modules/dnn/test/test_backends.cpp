// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
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
                    double l1 = 0.0, double lInf = 0.0, double detectionConfThresh = 0.2, bool useWinograd = true)
    {
        checkBackend();
        l1 = l1 ? l1 : default_l1;
        lInf = lInf ? lInf : default_lInf;

        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto);

        // Create two networks - with default backend and target and a tested one.
        Net netDefault = readNet(weights, proto);
        netDefault.setPreferableBackend(DNN_BACKEND_OPENCV);
        netDefault.setInput(inp);
        Mat outDefault = netDefault.forward(outputLayer).clone();

        net = readNet(weights, proto);
        net.setInput(inp);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.enableWinograd(useWinograd);
        if (backend == DNN_BACKEND_HALIDE && !halideScheduler.empty())
        {
            halideScheduler = findDataFile(halideScheduler);
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
            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
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

    Net net;
};

TEST_P(DNNTestNetwork, AlexNet)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB);
    processNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt",
               Size(227, 227), "prob",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_alexnet.yml" :
                                             "dnn/halide_scheduler_alexnet.yml");
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, ResNet_50)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
        CV_TEST_TAG_DEBUG_LONG
    );

    processNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt",
               Size(224, 224), "prob",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_resnet_50.yml" :
                                             "dnn/halide_scheduler_resnet_50.yml");
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt",
               Size(227, 227), "prob",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_squeezenet_v1_1.yml" :
                                             "dnn/halide_scheduler_squeezenet_v1_1.yml");
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, GoogLeNet)
{
    applyTestTag(target == DNN_TARGET_CPU ? "" : CV_TEST_TAG_MEMORY_512MB);

    processNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt",
               Size(224, 224), "prob");
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, Inception_5h)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    double l1 = default_l1, lInf = default_lInf;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_CPU || target == DNN_TARGET_OPENCL))
    {
        l1 = 1.72e-5;
        lInf = 8e-4;
    }
    processNet("dnn/tensorflow_inception_graph.pb", "", Size(224, 224), "softmax2",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_inception_5h.yml" :
                                             "dnn/halide_scheduler_inception_5h.yml",
               l1, lInf);
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, ENet)
{
    applyTestTag(target == DNN_TARGET_CPU ? "" : CV_TEST_TAG_MEMORY_512MB);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    processNet("dnn/Enet-model-best.net", "", Size(512, 512), "l367_Deconvolution",
               target == DNN_TARGET_OPENCL ? "dnn/halide_scheduler_opencl_enet.yml" :
                                             "dnn/halide_scheduler_enet.yml",
               2e-5, 0.15);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_Caffe)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
    float scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16) ? 1.5e-2 : 0.0;
    float iouDiff = (target == DNN_TARGET_MYRIAD) ? 0.063  : 0.0;
    float detectionConfThresh = (target == DNN_TARGET_MYRIAD) ? 0.262  : FLT_MIN;
         processNet("dnn/MobileNetSSD_deploy.caffemodel", "dnn/MobileNetSSD_deploy.prototxt",
                    inp, "detection_out", "", scoreDiff, iouDiff, detectionConfThresh);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_Caffe_Different_Width_Height)
{
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    // May hang on some configurations
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
#elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
    // IE exception: Ngraph operation Transpose with name conv15_2_mbox_conf_perm has dynamic output shape on 0 port, but CPU plug-in supports only static shape
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
            CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
        );
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) &&
        target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#elif defined(INF_ENGINE_RELEASE)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) &&
        target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f / 127.5, Size(300, 560), Scalar(127.5, 127.5, 127.5), false);
    float scoreDiff = 0.0, iouDiff = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.029;
        iouDiff = 0.09;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.03;
        iouDiff = 0.08;
    }
    processNet("dnn/MobileNetSSD_deploy.caffemodel", "dnn/MobileNetSSD_deploy.prototxt",
                inp, "detection_out", "", scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_v1_TensorFlow)
{
    applyTestTag((target == DNN_TARGET_CPU || target == DNN_TARGET_CPU_FP16) ? "" : CV_TEST_TAG_MEMORY_512MB);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);

    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    float detectionConfThresh = (target == DNN_TARGET_MYRIAD) ? 0.216 : 0.2;
    float scoreDiff = 0.0, iouDiff = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.095;
        iouDiff = 0.09;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.007;
        iouDiff = 0.08;
    }
    processNet("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", "dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt",
               inp, "detection_out", "", scoreDiff, iouDiff, detectionConfThresh);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_v1_TensorFlow_Different_Width_Height)
{
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) &&
        target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2019020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 560), Scalar(), false);
    float scoreDiff = 0.0, iouDiff = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.013;
        iouDiff = 0.06;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.007;
        iouDiff = 0.06;
    }
    processNet("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", "dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt",
               inp, "detection_out", "", scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, MobileNet_SSD_v2_TensorFlow)
{
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);

    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    float scoreDiff = 2e-5, iouDiff = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.013;
        iouDiff = 0.062;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.02;
        iouDiff = 0.07;
    }
    processNet("dnn/ssd_mobilenet_v2_coco_2018_03_29.pb", "dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
               inp, "detection_out", "", scoreDiff, iouDiff, 0.25);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, SSD_VGG16)
{
    applyTestTag(CV_TEST_TAG_LONG, (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
                 CV_TEST_TAG_DEBUG_VERYLONG);
    if (backend == DNN_BACKEND_HALIDE && target == DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);  // TODO HALIDE_CPU

    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);

    float scoreDiff = 0.0, iouDiff = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.04;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.0325;
        iouDiff = 0.032;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.03;
        iouDiff = 0.13;
    }

    processNet("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel",
               "dnn/ssd_vgg16.prototxt", inp, "detection_out", "", scoreDiff,
               iouDiff, 0.2, false);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, OpenPose_pose_coco)
{
    applyTestTag(CV_TEST_TAG_LONG, (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
                 CV_TEST_TAG_DEBUG_LONG);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LE(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    const float l1 = (target == DNN_TARGET_MYRIAD) ? 0.009 : 0.0;
    const float lInf = (target == DNN_TARGET_MYRIAD) ? 0.09 : 0.0;
    processNet("dnn/openpose_pose_coco.caffemodel", "dnn/openpose_pose_coco.prototxt",
               Size(46, 46), "", "", l1, lInf);
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, OpenPose_pose_mpi)
{
    applyTestTag(CV_TEST_TAG_LONG, (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
                 CV_TEST_TAG_DEBUG_VERYLONG);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LE(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    // output range: [-0.001, 0.97]
    const float l1 = (target == DNN_TARGET_MYRIAD) ? 0.02 : 0.0;
    const float lInf = (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16) ? 0.2 : 0.0;
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi.prototxt",
               Size(46, 46), "", "", l1, lInf);
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, OpenPose_pose_mpi_faster_4_stages)
{
    applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_MEMORY_1GB);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LE(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    // The same .caffemodel but modified .prototxt
    // See https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi_faster_4_stages.prototxt",
               Size(46, 46));
    expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, OpenFace)
{
#if defined(INF_ENGINE_RELEASE)
#if INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
#endif
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
    const float l1 = (target == DNN_TARGET_MYRIAD) ? 0.0024 : 0.0;
    const float lInf = (target == DNN_TARGET_MYRIAD) ? 0.0071 : 0.0;
    processNet("dnn/openface_nn4.small2.v1.t7", "", Size(96, 96), "", "", l1, lInf);

    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, opencv_face_detector)
{
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
    Mat img = imread(findDataFile("gpu/lbpcascade/er.png"));
    Mat inp = blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);
    processNet("dnn/opencv_face_detector.caffemodel", "dnn/opencv_face_detector.prototxt",
               inp, "detection_out");
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, Inception_v2_SSD_TensorFlow)
{
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
        CV_TEST_TAG_DEBUG_LONG
    );
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X);
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2019020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
    float scoreDiff = 0.0, iouDiff = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        scoreDiff = 0.02;
        iouDiff = 0.1;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.015;
        iouDiff = 0.08;
    }
    processNet("dnn/ssd_inception_v2_coco_2017_11_17.pb", "dnn/ssd_inception_v2_coco_2017_11_17.pbtxt",
               inp, "detection_out", "", scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

TEST_P(DNNTestNetwork, DenseNet_121)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
    // Reference output values are in range [-3.807, 4.605]
    float l1 = 0.0, lInf = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 2e-2;
        lInf = 9e-2;
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            lInf = 0.1f;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        l1 = 0.1;
        lInf = 0.6;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.008;
        lInf = 0.06;
    }
    processNet("dnn/DenseNet_121.caffemodel", "dnn/DenseNet_121.prototxt", Size(224, 224), "", "", l1, lInf);
    if (target != DNN_TARGET_MYRIAD || getInferenceEngineVPUType() != CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        expectNoFallbacksFromIE(net);
    expectNoFallbacksFromCUDA(net);
}

TEST_P(DNNTestNetwork, FastNeuralStyle_eccv16)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB, CV_TEST_TAG_DEBUG_VERYLONG);

    if (backend == DNN_BACKEND_HALIDE)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

#if defined(INF_ENGINE_RELEASE)
#if INF_ENGINE_VER_MAJOR_LE(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
#endif

    Mat img = imread(findDataFile("dnn/googlenet_1.png"));
    Mat inp = blobFromImage(img, 1.0, Size(320, 240), Scalar(103.939, 116.779, 123.68), false, false);
    // Output image has values in range [-143.526, 148.539].
    float l1 = 2e-4, lInf = 2e-3;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        l1 = 0.4;
        lInf = 7.46;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.3;
        lInf = 7.6;
    }
    else if (target == DNN_TARGET_CPU_FP16)
    {
        l1 = 0.4;
        lInf = 19.;
    }
    else if (target == DNN_TARGET_VULKAN)
    {
        l1 = 0.4;
        lInf = 7.46;
    }

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
    {
        l1 = 5e-3;
        lInf = 5e-3;
    }
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
    {
        lInf = 25;
    }
#endif


    processNet("dnn/fast_neural_style_eccv16_starry_night.t7", "", inp, "", "", l1, lInf);
#if defined(HAVE_INF_ENGINE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    expectNoFallbacksFromIE(net);
#endif
    expectNoFallbacksFromCUDA(net);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, dnnBackendsAndTargets(true, true, false, true, true));

}} // namespace
