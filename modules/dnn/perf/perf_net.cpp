// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/dnn/shape_utils.hpp"

#include "../test/test_common.hpp"

namespace opencv_test {

class DNNTestNetwork : public ::perf::TestBaseWithParam< tuple<Backend, Target> >
{
public:
    dnn::Backend backend;
    dnn::Target target;

    dnn::Net net;

    DNNTestNetwork()
    {
        backend = (dnn::Backend)(int)get<0>(GetParam());
        target = (dnn::Target)(int)get<1>(GetParam());
    }

    void processNet(std::string weights, std::string proto, const Mat& input, const std::string& outputLayer = "")
    {
        randu(input, 0.0f, 1.0f);

        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto);
        net = readNet(proto, weights);
        net.setInput(blobFromImage(input, 1.0, Size(), Scalar(), false));
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        MatShape netInputShape = shape(1, 3, input.rows, input.cols);
        size_t weightsMemory = 0, blobsMemory = 0;
        net.getMemoryConsumption(netInputShape, weightsMemory, blobsMemory);
        int64 flops = net.getFLOPS(netInputShape);
        CV_Assert(flops > 0);

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
    processNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt", Mat(cv::Size(227, 227), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, GoogLeNet)
{
    processNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt", Mat(cv::Size(224, 224), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, ResNet_50)
{
    processNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt", Mat(cv::Size(224, 224), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt", Mat(cv::Size(227, 227), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, Inception_5h)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019) throw SkipTestException("");
    processNet("dnn/tensorflow_inception_graph.pb", "", Mat(cv::Size(224, 224), CV_32FC3), "softmax2");
}

PERF_TEST_P_(DNNTestNetwork, ENet)
{
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU) ||
        (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16))
        throw SkipTestException("");
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        throw SkipTestException("");
#endif
    processNet("dnn/Enet-model-best.net", "", Mat(cv::Size(512, 256), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, SSD)
{
    processNet("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", "dnn/ssd_vgg16.prototxt", Mat(cv::Size(300, 300), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, OpenFace)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_HDDL))
        throw SkipTestException("");
#endif
    processNet("dnn/openface_nn4.small2.v1.t7", "", Mat(cv::Size(96, 96), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_Caffe)
{
    processNet("dnn/MobileNetSSD_deploy_19e3ec3.caffemodel", "dnn/MobileNetSSD_deploy_19e3ec3.prototxt", Mat(cv::Size(300, 300), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_v1_TensorFlow)
{
    processNet("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", "ssd_mobilenet_v1_coco_2017_11_17.pbtxt", Mat(cv::Size(300, 300), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_v2_TensorFlow)
{
    processNet("dnn/ssd_mobilenet_v2_coco_2018_03_29.pb", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt", Mat(cv::Size(300, 300), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, DenseNet_121)
{
    processNet("dnn/DenseNet_121.caffemodel", "dnn/DenseNet_121.prototxt", Mat(cv::Size(224, 224), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, OpenPose_pose_mpi_faster_4_stages)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_HDDL))
        throw SkipTestException("");
    // The same .caffemodel but modified .prototxt
    // See https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi_faster_4_stages.prototxt", Mat(cv::Size(368, 368), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, opencv_face_detector)
{
    processNet("dnn/opencv_face_detector.caffemodel", "dnn/opencv_face_detector.prototxt", Mat(cv::Size(300, 300), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, Inception_v2_SSD_TensorFlow)
{
    processNet("dnn/ssd_inception_v2_coco_2017_11_17.pb", "ssd_inception_v2_coco_2017_11_17.pbtxt", Mat(cv::Size(300, 300), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, YOLOv3)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        throw SkipTestException("Test is disabled in OpenVINO 2020.4");
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("Test is disabled in OpenVINO 2020.4");
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021010000)  // nGraph compilation failure
    if (target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
#endif

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    cvtColor(sample, sample, COLOR_BGR2RGB);
    Mat inp;
    sample.convertTo(inp, CV_32FC3, 1.0f / 255, 0);
    processNet("dnn/yolov3.weights", "dnn/yolov3.cfg", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv4)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
    if (target == DNN_TARGET_MYRIAD)  // not enough resources
        throw SkipTestException("");
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        throw SkipTestException("Test is disabled in OpenVINO 2020.4");
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("Test is disabled in OpenVINO 2020.4");
#endif
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    cvtColor(sample, sample, COLOR_BGR2RGB);
    Mat inp;
    sample.convertTo(inp, CV_32FC3, 1.0f / 255, 0);
    processNet("dnn/yolov4.weights", "dnn/yolov4.cfg", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv4_tiny)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021010000)  // nGraph compilation failure
    if (target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
#endif
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    cvtColor(sample, sample, COLOR_BGR2RGB);
    Mat inp;
    sample.convertTo(inp, CV_32FC3, 1.0f / 255, 0);
    processNet("dnn/yolov4-tiny-2020-12.weights", "dnn/yolov4-tiny-2020-12.cfg", inp);
}

PERF_TEST_P_(DNNTestNetwork, EAST_text_detection)
{
    processNet("dnn/frozen_east_text_detection.pb", "", Mat(cv::Size(320, 320), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, FastNeuralStyle_eccv16)
{
    processNet("dnn/fast_neural_style_eccv16_starry_night.t7", "", Mat(cv::Size(320, 240), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, Inception_v2_Faster_RCNN)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        throw SkipTestException("Test is disabled in OpenVINO 2019R1");
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2019020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        throw SkipTestException("Test is disabled in OpenVINO 2019R2");
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021010000)
    if (target == DNN_TARGET_MYRIAD)
        throw SkipTestException("Test is disabled in OpenVINO 2021.1+ / MYRIAD");
#endif
    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU) ||
        (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16))
        throw SkipTestException("");
    processNet("dnn/faster_rcnn_inception_v2_coco_2018_01_28.pb",
               "dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt", Mat(cv::Size(800, 600), CV_32FC3));
}

PERF_TEST_P_(DNNTestNetwork, EfficientDet)
{
    if (target != DNN_TARGET_CPU)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    resize(sample, sample, Size(512, 512));
    Mat inp;
    sample.convertTo(inp, CV_32FC3, 1.0/255);
    processNet("dnn/efficientdet-d0.pb", "dnn/efficientdet-d0.pbtxt", inp);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, dnnBackendsAndTargets());

} // namespace
