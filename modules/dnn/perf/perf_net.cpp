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

    void processNet(std::string weights, std::string proto,
                    const std::vector<std::tuple<Mat, std::string>>& inputs, const std::string& outputLayer = ""){
        weights = findDataFile(weights, false);
        if (!proto.empty())
            proto = findDataFile(proto);
        net = readNet(weights, proto);
        // Set multiple inputs
        for(auto &inp: inputs){
            net.setInput(std::get<0>(inp), std::get<1>(inp));
        }

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        // Calculate multiple inputs memory consumption
        std::vector<MatShape> netMatShapes;
        for(auto &inp: inputs){
            netMatShapes.push_back(shape(std::get<0>(inp)));
        }

        bool fp16 = false;
#ifdef HAVE_OPENCL
        fp16 = ocl::Device::getDefault().isExtensionSupported("cl_khr_fp16");
#endif
        std::vector<cv::dnn::MatType> netMatTypes;
        for (auto& inp : inputs) {
            cv::dnn::MatType t = std::get<0>(inp).depth();
            if (t == CV_32F && fp16 && target == DNN_TARGET_OPENCL_FP16)
                t = CV_16F;
            netMatTypes.push_back(t);
        }

        net.forward(outputLayer); // warmup

        size_t weightsMemory = 0, blobsMemory = 0;
        net.getMemoryConsumption(netMatShapes, netMatTypes, weightsMemory, blobsMemory);
        int64 flops = net.getFLOPS(netMatShapes, netMatTypes);
        // [TODO] implement getFLOPS in the new engine
        // Issue: https://github.com/opencv/opencv/issues/26199
        CV_Assert(flops > 0 || net.getMainGraph());
        std::cout << "Memory consumption:" << std::endl;
        std::cout << "    Weights(parameters): " << divUp(weightsMemory, 1u<<20) << " Mb" << std::endl;
        std::cout << "    Blobs: " << divUp(blobsMemory, 1u<<20) << " Mb" << std::endl;
        std::cout << "Calculation complexity: " << flops * 1e-9 << " GFlops" << std::endl;

        PERF_SAMPLE_BEGIN()
            net.forward();
        PERF_SAMPLE_END()

        SANITY_CHECK_NOTHING();
    }

    void processNet(std::string weights, std::string proto,
                    Mat &input, const std::string& outputLayer = "")
    {
        processNet(weights, proto, {std::make_tuple(input, "")}, outputLayer);
    }

    void processNet(std::string weights, std::string proto,
                    Size inpSize, const std::string& outputLayer = "")
    {
        Mat input_data(inpSize, CV_32FC3);
        randu(input_data, 0.0f, 1.0f);
        Mat input = blobFromImage(input_data, 1.0, Size(), Scalar(), false);
        processNet(weights, proto, input, outputLayer);
    }
};

PERF_TEST_P_(DNNTestNetwork, AlexNet)
{
    processNet("dnn/bvlc_alexnet.caffemodel", "dnn/bvlc_alexnet.prototxt", cv::Size(227, 227));
}

PERF_TEST_P_(DNNTestNetwork, GoogLeNet)
{
    processNet("dnn/bvlc_googlenet.caffemodel", "dnn/bvlc_googlenet.prototxt", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, ResNet_50)
{
    processNet("dnn/ResNet-50-model.caffemodel", "dnn/ResNet-50-deploy.prototxt", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt", cv::Size(227, 227));
}

PERF_TEST_P_(DNNTestNetwork, Inception_5h)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019) throw SkipTestException("");
    processNet("dnn/tensorflow_inception_graph.pb", "", cv::Size(224, 224), "softmax2");
}

PERF_TEST_P_(DNNTestNetwork, SSD)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    processNet("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", "dnn/ssd_vgg16.prototxt", cv::Size(300, 300));
}

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_Caffe)
{
    processNet("dnn/MobileNetSSD_deploy_19e3ec3.caffemodel", "dnn/MobileNetSSD_deploy_19e3ec3.prototxt", cv::Size(300, 300));
}

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_v1_TensorFlow)
{
    processNet("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", "ssd_mobilenet_v1_coco_2017_11_17.pbtxt", cv::Size(300, 300));
}

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_v2_TensorFlow)
{
    processNet("dnn/ssd_mobilenet_v2_coco_2018_03_29.pb", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt", cv::Size(300, 300));
}

PERF_TEST_P_(DNNTestNetwork, DenseNet_121)
{
    processNet("dnn/DenseNet_121.caffemodel", "dnn/DenseNet_121.prototxt", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, OpenPose_pose_mpi_faster_4_stages)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_MYRIAD || target == DNN_TARGET_HDDL))
        throw SkipTestException("");
    // The same .caffemodel but modified .prototxt
    // See https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp
    processNet("dnn/openpose_pose_mpi.caffemodel", "dnn/openpose_pose_mpi_faster_4_stages.prototxt", cv::Size(368, 368));
}

PERF_TEST_P_(DNNTestNetwork, Inception_v2_SSD_TensorFlow)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    processNet("dnn/ssd_inception_v2_coco_2017_11_17.pb", "ssd_inception_v2_coco_2017_11_17.pbtxt", cv::Size(300, 300));
}

PERF_TEST_P_(DNNTestNetwork, YOLOv3)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );
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
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(), Scalar(), true);
    processNet("dnn/yolov3.weights", "dnn/yolov3.cfg", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv4)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_2GB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );
    if (target == DNN_TARGET_MYRIAD)  // not enough resources
        throw SkipTestException("");
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020040000)  // nGraph compilation failure
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        throw SkipTestException("Test is disabled in OpenVINO 2020.4");
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("Test is disabled in OpenVINO 2020.4");
#endif
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(), Scalar(), true);
    processNet("dnn/yolov4.weights", "dnn/yolov4.cfg", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv4_tiny)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021010000)  // nGraph compilation failure
    if (target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
#endif
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(), Scalar(), true);
    processNet("dnn/yolov4-tiny-2020-12.weights", "dnn/yolov4-tiny-2020-12.cfg", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv5) {
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(640, 640), Scalar(), true);
    processNet("dnn/yolov5n.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv8)
{
    applyTestTag(
        CV_TEST_TAG_MEMORY_512MB,
        CV_TEST_TAG_DEBUG_LONG
    );

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(640, 640), Scalar(), true);
    processNet("dnn/yolov8n.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOX) {
    applyTestTag(
        CV_TEST_TAG_MEMORY_512MB,
        CV_TEST_TAG_DEBUG_VERYLONG
    );
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(640, 640), Scalar(), true);
    processNet("dnn/yolox_s.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, EAST_text_detection)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    processNet("dnn/frozen_east_text_detection.pb", "", cv::Size(320, 320));
}

PERF_TEST_P_(DNNTestNetwork, FastNeuralStyle_eccv16)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    processNet("dnn/mosaic-9.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, Inception_v2_Faster_RCNN)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

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
               "dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt",
               cv::Size(800, 600));
}

PERF_TEST_P_(DNNTestNetwork, EfficientDet)
{
    if (target != DNN_TARGET_CPU)
        throw SkipTestException("");
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(512, 512), Scalar(), true);
    processNet("dnn/efficientdet-d0.pb", "dnn/efficientdet-d0.pbtxt", inp);
}

PERF_TEST_P_(DNNTestNetwork, EfficientNet)
{
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(224, 224), Scalar(), true);
    transposeND(inp, {0, 2, 3, 1}, inp);
    processNet("dnn/efficientnet-lite4.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, YuNet) {
    processNet("dnn/onnx/models/yunet-202303.onnx", "", cv::Size(640, 640));
}

PERF_TEST_P_(DNNTestNetwork, SFace) {
    processNet("dnn/face_recognition_sface_2021dec.onnx", "", cv::Size(112, 112));
}

PERF_TEST_P_(DNNTestNetwork, MPPalm) {
    Mat inp(cv::Size(192, 192), CV_32FC3);
    randu(inp, 0.0f, 1.0f);
    inp = blobFromImage(inp, 1.0, Size(), Scalar(), false);
    transposeND(inp, {0, 2, 3, 1}, inp);
    processNet("dnn/palm_detection_mediapipe_2023feb.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, MPHand) {
    Mat inp(cv::Size(224, 224), CV_32FC3);
    randu(inp, 0.0f, 1.0f);
    inp = blobFromImage(inp, 1.0, Size(), Scalar(), false);
    transposeND(inp, {0, 2, 3, 1}, inp);
    processNet("dnn/handpose_estimation_mediapipe_2023feb.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, MPPose) {
    Mat inp(cv::Size(256, 256), CV_32FC3);
    randu(inp, 0.0f, 1.0f);
    inp = blobFromImage(inp, 1.0, Size(), Scalar(), false);
    transposeND(inp, {0, 2, 3, 1}, inp);
    processNet("dnn/pose_estimation_mediapipe_2023mar.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, PPOCRv3) {
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    processNet("dnn/onnx/models/PP_OCRv3_DB_text_det.onnx", "", cv::Size(736, 736));
}

PERF_TEST_P_(DNNTestNetwork, PPHumanSeg) {
    processNet("dnn/human_segmentation_pphumanseg_2023mar.onnx", "", cv::Size(192, 192));
}

PERF_TEST_P_(DNNTestNetwork, CRNN) {
    Mat inp(cv::Size(100, 32), CV_32FC1);
    randu(inp, 0.0f, 1.0f);
    inp = blobFromImage(inp, 1.0, Size(), Scalar(), false);
    processNet("dnn/text_recognition_CRNN_EN_2021sep.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, VitTrack) {
    Mat inp1(cv::Size(128, 128), CV_32FC3);
    Mat inp2(cv::Size(256, 256), CV_32FC3);
    randu(inp1, 0.0f, 1.0f);
    randu(inp2, 0.0f, 1.0f);
    inp1 = blobFromImage(inp1, 1.0, Size(), Scalar(), false);
    inp2 = blobFromImage(inp2, 1.0, Size(), Scalar(), false);
    processNet("dnn/onnx/models/object_tracking_vittrack_2023sep.onnx", "", {std::make_tuple(inp1, "template"), std::make_tuple(inp2, "search")});
}

PERF_TEST_P_(DNNTestNetwork, EfficientDet_int8)
{
    if (target != DNN_TARGET_CPU || (backend != DNN_BACKEND_OPENCV &&
        backend != DNN_BACKEND_TIMVX && backend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)) {
        throw SkipTestException("");
    }
    Mat inp = imread(findDataFile("dnn/dog416.png"));
    inp = blobFromImage(inp, 1.0 / 255.0, Size(320, 320), Scalar(), true);
    processNet("dnn/tflite/coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, VIT_B_32)
{
    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);

    processNet("dnn/onnx/models/vit_b_32.onnx", "", cv::Size(224, 224));
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, dnnBackendsAndTargets());

} // namespace
