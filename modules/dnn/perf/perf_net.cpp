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
        CV_Assert(flops > 0);
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

PERF_TEST_P_(DNNTestNetwork, ResNet_18_v1_ONNX)
{
    processNet("dnn/onnx/models/resnet18v1.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, ResNet_50_v1_ONNX)
{
    processNet("dnn/onnx/models/resnet50v1.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, MobileNetv2_ONNX)
{
    processNet("dnn/onnx/models/mobilenetv2.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, ResNet50_QDQ_ONNX)
{
    processNet("dnn/onnx/models/resnet50-v1-12-qdq.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, SqueezeNet_v1_1)
{
    processNet("dnn/squeezenet_v1.1.caffemodel", "dnn/squeezenet_v1.1.prototxt", cv::Size(227, 227));
}

PERF_TEST_P_(DNNTestNetwork, Inception_5h)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019) throw SkipTestException("");
    processNet("dnn/tensorflow_inception_graph.pb", "", cv::Size(224, 224));
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

PERF_TEST_P_(DNNTestNetwork, MobileNet_SSD_v1_ONNX)
{
    Mat image(cv::Size(300, 300), CV_8UC3);
    randu(image, 0, 255);
    int imsize[] = {1, image.rows, image.cols, 3};
    Mat input(4, imsize, CV_8U, image.data);
    processNet("dnn/onnx/models/ssd_mobilenet_v1_12.onnx", "", input);
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
    cv::resize(sample, sample, Size(640, 640));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(), Scalar(), true);
    processNet("dnn/yolov3.onnx", "", inp);
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
    cv::resize(sample, sample, Size(608, 608));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(), Scalar(), true);
    processNet("dnn/yolov4.onnx", "", inp);
}

PERF_TEST_P_(DNNTestNetwork, YOLOv4_tiny)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2021010000)  // nGraph compilation failure
    if (target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
#endif
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(), Scalar(), true);
    processNet("dnn/yolov4-tiny.onnx", "", inp);
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

PERF_TEST_P_(DNNTestNetwork, BERT)
{
    const int seq_len = 9;
    int64_t input_ids_data[seq_len] = {101, 1996, 103, 2938, 2006, 1996, 13523, 1012, 102};
    int64_t attention_mask_data[seq_len] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int64_t token_type_ids_data[seq_len] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int shp[2] = {1, seq_len};
    Mat input_ids(2, shp, CV_64S, input_ids_data);
    Mat attention_mask(2, shp, CV_64S, attention_mask_data);
    Mat token_type_ids(2, shp, CV_64S, token_type_ids_data);
    processNet("dnn/onnx/models/bert.onnx", "",
               {std::make_tuple(input_ids, "input_ids"),
                std::make_tuple(attention_mask, "attention_mask"),
                std::make_tuple(token_type_ids, "token_type_ids")});
}

PERF_TEST_P_(DNNTestNetwork, VIT_Base_Patch16_224)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    processNet("dnn/vit_base_patch16_224_Opset16.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, DeiT_Tiny_Patch16_224)
{
    processNet("dnn/deit_tiny_patch16_224_Opset16.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, MobileViT_XS)
{
    processNet("dnn/mobilevit_xs_Opset16.onnx", "", cv::Size(256, 256));
}

PERF_TEST_P_(DNNTestNetwork, MobileViTv2_100_ONNX)
{
    processNet("dnn/mobilevitv2_100_Opset16.onnx", "", cv::Size(256, 256));
}

PERF_TEST_P_(DNNTestNetwork, BEiT_Base_Patch16_224)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
    processNet("dnn/beit_base_patch16_224_Opset16.onnx", "", cv::Size(224, 224));
}

PERF_TEST_P_(DNNTestNetwork, BlazeFace)
{
    Mat input(cv::Size(128, 128), CV_32FC3);
    randu(input, 0.0f, 1.0f);
    input = blobFromImage(input, 1.0 / 255.0, Size(128, 128));

    const int oneDim[] = {1};
    Mat conf(1, oneDim, CV_32F); conf.ptr<float>()[0] = 0.20f;
    Mat iou(1, oneDim, CV_32F); iou.ptr<float>()[0] = 0.30f;
    Mat maxDet(1, oneDim, CV_64S); maxDet.ptr<int64_t>()[0] = 25;

    processNet("dnn/onnx/models/blazeface.onnx", "",
               {std::make_tuple(input, "image"),
                std::make_tuple(conf, "conf_threshold"),
                std::make_tuple(iou, "iou_threshold"),
                std::make_tuple(maxDet, "max_detections")});
}

PERF_TEST_P_(DNNTestNetwork, FacePaint)
{
    processNet("dnn/onnx/models/face_paint_512_v2_0.onnx", "", cv::Size(512, 512));
}

// Model: https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/blob/main/sam2_hiera_large.encoder.onnx
PERF_TEST_P_(DNNTestNetwork, SAM2_Encoder)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_VERYLONG);

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(1024, 1024), Scalar(), true);
    processNet("dnn/onnx/models/sam2_hiera_large.encoder.onnx", "", inp);
}

// Model: https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/blob/main/sam2_hiera_large.decoder.onnx
PERF_TEST_P_(DNNTestNetwork, SAM2_Decoder)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB, CV_TEST_TAG_VERYLONG);

    // Synthetic encoder outputs used as decoder inputs
    int shp_embed[4]  = {1, 256, 64, 64};
    int shp_feat0[4]  = {1, 32, 256, 256};
    int shp_feat1[4]  = {1, 64, 128, 128};

    Mat image_embed(4, shp_embed, CV_32F);
    Mat high_res_feats_0(4, shp_feat0, CV_32F);
    Mat high_res_feats_1(4, shp_feat1, CV_32F);
    randu(image_embed, 0.0f, 1.0f);
    randu(high_res_feats_0, 0.0f, 1.0f);
    randu(high_res_feats_1, 0.0f, 1.0f);

    // Single point prompt at center of image, label=1 (foreground)
    int shp_pts[3]   = {1, 1, 2};
    int shp_lbl[2]   = {1, 1};
    int shp_mask[4]  = {1, 1, 256, 256};
    int shp_hasmask[1] = {1};
    float point_coords_data[2]  = {512.0f, 512.0f};
    float point_labels_data[1]  = {1.0f};
    float has_mask_input_data[1]= {0.0f};
    Mat point_coords(3, shp_pts, CV_32F, point_coords_data);
    Mat point_labels(2, shp_lbl, CV_32F, point_labels_data);
    Mat mask_input(4, shp_mask, CV_32F, Scalar(0));
    Mat has_mask_input(1, shp_hasmask, CV_32F, has_mask_input_data);

    processNet("dnn/onnx/models/sam2_hiera_large.decoder.onnx", "",
               {std::make_tuple(image_embed,      "image_embed"),
                std::make_tuple(high_res_feats_0, "high_res_feats_0"),
                std::make_tuple(high_res_feats_1, "high_res_feats_1"),
                std::make_tuple(point_coords,     "point_coords"),
                std::make_tuple(point_labels,     "point_labels"),
                std::make_tuple(mask_input,       "mask_input"),
                std::make_tuple(has_mask_input,   "has_mask_input")});
}

// Model: https://github.com/opencv/opencv_zoo/tree/main/models/optical_flow_estimation_raft
PERF_TEST_P_(DNNTestNetwork, RAFT)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_VERYLONG);

    // RAFT takes two consecutive frames to estimate optical flow between them
    Mat frame0 = imread(findDataFile("gpu/opticalflow/frame0.png"));
    Mat frame1 = imread(findDataFile("gpu/opticalflow/frame1.png"));
    Mat blob0 = blobFromImage(frame0, 1.0, Size(480, 360), Scalar(), true);
    Mat blob1 = blobFromImage(frame1, 1.0, Size(480, 360), Scalar(), true);

    processNet("dnn/onnx/models/optical_flow_estimation_raft_2023aug.onnx", "",
               {std::make_tuple(blob0, "0"),
                std::make_tuple(blob1, "1")});
}

// Model: https://huggingface.co/onnx-community/owlv2-base-patch16-finetuned-ONNX
PERF_TEST_P_(DNNTestNetwork, OWLv2)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB, CV_TEST_TAG_VERYLONG);

    // Image input: [1, 3, 960, 960] (60x60 patches x 16 = 960)
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat pixel_values = blobFromImage(sample, 1.0 / 255.0, Size(960, 960), Scalar(), true);

    // Text query tokens: "a dog" with CLIP tokenizer, seq_len=16
    // [BOS=49406, "a"=320, "dog"=1929, EOS=49407, pad=0, ...]
    const int seq_len = 16;
    int shp[2] = {1, seq_len};
    int64_t input_ids_data[seq_len]     = {49406, 320, 1929, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int64_t attention_mask_data[seq_len]= {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Mat input_ids(2, shp, CV_64S, input_ids_data);
    Mat attention_mask(2, shp, CV_64S, attention_mask_data);

    processNet("dnn/onnx/models/owlv2_base_patch_16.onnx", "",
               {std::make_tuple(input_ids,      "input_ids"),
                std::make_tuple(pixel_values,   "pixel_values"),
                std::make_tuple(attention_mask, "attention_mask")});
}

// Model: https://drive.google.com/file/d/1IU7iktOUbvNPFnDJb_ivl3LxYIdpEp3f/view?usp=drive_link
PERF_TEST_P_(DNNTestNetwork, YOLO26m_Seg)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB, CV_TEST_TAG_VERYLONG);

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(640, 640), Scalar(), true);
    processNet("dnn/onnx/models/yolo26m-seg.onnx", "", inp);
}

// Model: https://drive.google.com/file/d/17OWMXSiefFMmj46CT42Fd2q5kl_jHRBC/view?usp=drive_link
PERF_TEST_P_(DNNTestNetwork, YOLO26n)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(640, 640), Scalar(), true);
    processNet("dnn/onnx/models/yolo26n.onnx", "", inp);
}

// Model: https://huggingface.co/Xenova/segformer_b2_clothes/blob/main/onnx/model.onnx
PERF_TEST_P_(DNNTestNetwork, SegFormer_B2_Clothes)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB, CV_TEST_TAG_VERYLONG);

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(512, 512), Scalar(), true);
    processNet("dnn/onnx/models/segformer_b2_clothes.onnx", "", inp);
}

// Model: https://huggingface.co/Xenova/siglip-base-patch16-224/blob/main/onnx/model.onnx
PERF_TEST_P_(DNNTestNetwork, SigLIP)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB, CV_TEST_TAG_VERYLONG);

    // Image input: [1, 3, 224, 224] normalized to [-1, 1]
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat pixel_values = blobFromImage(sample, 1.0 / 255.0, Size(224, 224), Scalar(0.5, 0.5, 0.5), true);
    pixel_values = (pixel_values - 0.5f) / 0.5f;

    // Text input: dummy token IDs for "a photo of a dog", seq_len=64
    const int seq_len = 64;
    int shp[2] = {1, seq_len};
    Mat input_ids(2, shp, CV_64S, Scalar(0));
    // BOS=1, "a photo of a dog"=some tokens, EOS=2
    int64_t* ids = input_ids.ptr<int64_t>();
    ids[0] = 1; ids[1] = 263; ids[2] = 2514; ids[3] = 275; ids[4] = 262; ids[5] = 3914; ids[6] = 2;

    processNet("dnn/onnx/models/siglip_base_patch16_224.onnx", "",
               {std::make_tuple(input_ids,    "input_ids"),
                std::make_tuple(pixel_values, "pixel_values")});
}

// Model: https://huggingface.co/onnx-community/depth-anything-v2-small/blob/main/onnx/model.onnx
PERF_TEST_P_(DNNTestNetwork, Depth_Anything_V2)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB, CV_TEST_TAG_VERYLONG);

    Mat sample = imread(findDataFile("dnn/street.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(518, 518), Scalar(), true);
    processNet("dnn/onnx/models/depth_anything_v2_small.onnx", "", inp);
}

// Model: https://drive.google.com/file/d/1G2begS7rrEmWnI-xj2K5UL3PQ7H_0svc/view?usp=drive_link
PERF_TEST_P_(DNNTestNetwork, RetinaFace)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    processNet("dnn/onnx/models/retinaface_10g.onnx", "", cv::Size(640, 640));
}

// Model: https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX
PERF_TEST_P_(DNNTestNetwork, Grounding_DINO)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB, CV_TEST_TAG_VERYLONG);

    // Image input
    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat img = blobFromImage(sample, 1.0 / 255.0, Size(800, 800), Scalar(), true);

    // Text token inputs (dummy tokens for "dog ." as query text, seq_len=7)
    const int seq_len = 7;
    int64_t input_ids_data[seq_len]      = {101, 3899, 1012, 102, 0, 0, 0};
    int64_t attention_mask_data[seq_len] = {1, 1, 1, 1, 0, 0, 0};
    int64_t token_type_ids_data[seq_len] = {0, 0, 0, 0, 0, 0, 0};
    int64_t position_ids_data[seq_len]   = {0, 1, 2, 3, 0, 0, 0};
    uint8_t text_token_mask_data[seq_len]= {1, 1, 1, 1, 0, 0, 0};

    int shp[2] = {1, seq_len};
    Mat input_ids(2, shp, CV_64S, input_ids_data);
    Mat attention_mask(2, shp, CV_64S, attention_mask_data);
    Mat token_type_ids(2, shp, CV_64S, token_type_ids_data);
    Mat position_ids(2, shp, CV_64S, position_ids_data);
    Mat text_token_mask(2, shp, CV_8U, text_token_mask_data);

    processNet("dnn/onnx/models/groundingdino_swint_ogc.onnx", "",
               {std::make_tuple(img,             "img"),
                std::make_tuple(input_ids,       "input_ids"),
                std::make_tuple(attention_mask,  "attention_mask"),
                std::make_tuple(token_type_ids,  "token_type_ids"),
                std::make_tuple(position_ids,    "position_ids"),
                std::make_tuple(text_token_mask, "text_token_mask")});
}

// Model: https://drive.google.com/file/d/1P6a7oS_dV5y09FsCA4XDZK1-WcdZbWFh/view?usp=drive_link
PERF_TEST_P_(DNNTestNetwork, RF_DETR)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB, CV_TEST_TAG_VERYLONG);

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(560, 560), Scalar(), true);
    processNet("dnn/onnx/models/rfdetr.onnx", "", inp);
}

// Model: https://drive.google.com/file/d/1OrSmlXURayVQgW8nrrxjggzPMN7xPRGJ/view?usp=sharing
PERF_TEST_P_(DNNTestNetwork, RT_DETR_L)
{
    applyTestTag(CV_TEST_TAG_MEMORY_1GB, CV_TEST_TAG_VERYLONG);

    Mat sample = imread(findDataFile("dnn/dog416.png"));
    Mat inp = blobFromImage(sample, 1.0 / 255.0, Size(640, 640), Scalar(), true);
    processNet("dnn/onnx/models/rtdetr-l.onnx", "", inp);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, DNNTestNetwork, dnnBackendsAndTargets());

} // namespace
