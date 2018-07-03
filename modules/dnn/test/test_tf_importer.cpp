// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Test for Tensorflow models loading
*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

namespace opencv_test
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_TensorFlow, read_inception)
{
    Net net;
    {
        const string model = findDataFile("dnn/tensorflow_inception_graph.pb", false);
        net = readNetFromTensorflow(model);
        ASSERT_FALSE(net.empty());
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());
    Mat input;
    resize(sample, input, Size(224, 224));
    input -= 128; // mean sub

    Mat inputBlob = blobFromImage(input);

    net.setInput(inputBlob, "input");
    Mat out = net.forward("softmax2");

    std::cout << out.dims << std::endl;
}

TEST(Test_TensorFlow, inception_accuracy)
{
    Net net;
    {
        const string model = findDataFile("dnn/tensorflow_inception_graph.pb", false);
        net = readNetFromTensorflow(model);
        ASSERT_FALSE(net.empty());
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());
    resize(sample, sample, Size(224, 224));
    Mat inputBlob = blobFromImage(sample);

    net.setInput(inputBlob, "input");
    Mat out = net.forward("softmax2");

    Mat ref = blobFromNPY(_tf("tf_inception_prob.npy"));

    normAssert(ref, out);
}

static std::string path(const std::string& file)
{
    return findDataFile("dnn/tensorflow/" + file, false);
}

static void runTensorFlowNet(const std::string& prefix, int targetId = DNN_TARGET_CPU, bool hasText = false,
                             double l1 = 1e-5, double lInf = 1e-4,
                             bool memoryLoad = false)
{
    std::string netPath = path(prefix + "_net.pb");
    std::string netConfig = (hasText ? path(prefix + "_net.pbtxt") : "");
    std::string inpPath = path(prefix + "_in.npy");
    std::string outPath = path(prefix + "_out.npy");

    Net net;
    if (memoryLoad)
    {
        // Load files into a memory buffers
        string dataModel;
        ASSERT_TRUE(readFileInMemory(netPath, dataModel));

        string dataConfig;
        if (hasText)
            ASSERT_TRUE(readFileInMemory(netConfig, dataConfig));

        net = readNetFromTensorflow(dataModel.c_str(), dataModel.size(),
                                    dataConfig.c_str(), dataConfig.size());
    }
    else
        net = readNetFromTensorflow(netPath, netConfig);

    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    cv::Mat input = blobFromNPY(inpPath);
    cv::Mat target = blobFromNPY(outPath);

    net.setInput(input);
    cv::Mat output = net.forward();
    normAssert(target, output, "", l1, lInf);
}

typedef testing::TestWithParam<DNNTarget> Test_TensorFlow_layers;

TEST_P(Test_TensorFlow_layers, conv)
{
    int targetId = GetParam();
    runTensorFlowNet("single_conv", targetId);
    runTensorFlowNet("atrous_conv2d_valid", targetId);
    runTensorFlowNet("atrous_conv2d_same", targetId);
    runTensorFlowNet("depthwise_conv2d", targetId);
    runTensorFlowNet("keras_atrous_conv2d_same", targetId);
    runTensorFlowNet("conv_pool_nchw", targetId);
}

TEST_P(Test_TensorFlow_layers, padding)
{
    int targetId = GetParam();
    runTensorFlowNet("padding_same", targetId);
    runTensorFlowNet("padding_valid", targetId);
    runTensorFlowNet("spatial_padding", targetId);
}

TEST_P(Test_TensorFlow_layers, eltwise_add_mul)
{
    runTensorFlowNet("eltwise_add_mul", GetParam());
}

TEST_P(Test_TensorFlow_layers, concat)
{
    runTensorFlowNet("pad_and_concat", GetParam());
    runTensorFlowNet("concat_axis_1", GetParam());
}

TEST_P(Test_TensorFlow_layers, batch_norm)
{
    int targetId = GetParam();
    runTensorFlowNet("batch_norm", targetId);
    runTensorFlowNet("fused_batch_norm", targetId);
    runTensorFlowNet("batch_norm_text", targetId, true);
    runTensorFlowNet("mvn_batch_norm", targetId);
    runTensorFlowNet("mvn_batch_norm_1x1", targetId);
    runTensorFlowNet("unfused_batch_norm", targetId);
    runTensorFlowNet("fused_batch_norm_no_gamma", targetId);
    runTensorFlowNet("unfused_batch_norm_no_gamma", targetId);
}

TEST_P(Test_TensorFlow_layers, pooling)
{
    int targetId = GetParam();
    cv::ocl::Device d = cv::ocl::Device::getDefault();
    bool loosenFlag = targetId == DNN_TARGET_OPENCL && d.isIntel() && d.type() == cv::ocl::Device::TYPE_CPU;
    runTensorFlowNet("max_pool_even", targetId);
    runTensorFlowNet("max_pool_odd_valid", targetId);
    runTensorFlowNet("ave_pool_same", targetId);
    runTensorFlowNet("max_pool_odd_same", targetId, false, loosenFlag ? 3e-5 : 1e-5, loosenFlag ? 3e-4 : 1e-4);
    runTensorFlowNet("reduce_mean", targetId);  // an average pooling over all spatial dimensions.
}

TEST_P(Test_TensorFlow_layers, deconvolution)
{
    int targetId = GetParam();
    runTensorFlowNet("deconvolution", targetId);
    runTensorFlowNet("deconvolution_same", targetId);
    runTensorFlowNet("deconvolution_stride_2_same", targetId);
    runTensorFlowNet("deconvolution_adj_pad_valid", targetId);
    runTensorFlowNet("deconvolution_adj_pad_same", targetId);
    runTensorFlowNet("keras_deconv_valid", targetId);
    runTensorFlowNet("keras_deconv_same", targetId);
}

TEST_P(Test_TensorFlow_layers, matmul)
{
    int targetId = GetParam();
    runTensorFlowNet("matmul", targetId);
    runTensorFlowNet("nhwc_reshape_matmul", targetId);
    runTensorFlowNet("nhwc_transpose_reshape_matmul", targetId);
}

TEST_P(Test_TensorFlow_layers, reshape)
{
    int targetId = GetParam();
    runTensorFlowNet("shift_reshape_no_reorder", targetId);
    runTensorFlowNet("reshape_no_reorder", targetId);
    runTensorFlowNet("reshape_reduce", targetId);
    runTensorFlowNet("flatten", targetId, true);
    runTensorFlowNet("unfused_flatten", targetId);
    runTensorFlowNet("unfused_flatten_unknown_batch", targetId);
}

TEST_P(Test_TensorFlow_layers, l2_normalize)
{
    int targetId = GetParam();
    runTensorFlowNet("l2_normalize", targetId);
    runTensorFlowNet("l2_normalize_3d", targetId);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TensorFlow_layers, availableDnnTargets());

typedef testing::TestWithParam<DNNTarget> Test_TensorFlow_nets;

TEST_P(Test_TensorFlow_nets, MobileNet_SSD)
{
    std::string netPath = findDataFile("dnn/ssd_mobilenet_v1_coco.pb", false);
    std::string netConfig = findDataFile("dnn/ssd_mobilenet_v1_coco.pbtxt", false);
    std::string imgPath = findDataFile("dnn/street.png", false);

    Mat inp;
    resize(imread(imgPath), inp, Size(300, 300));
    inp = blobFromImage(inp, 1.0f / 127.5, Size(), Scalar(127.5, 127.5, 127.5), true);

    std::vector<String> outNames(3);
    outNames[0] = "concat";
    outNames[1] = "concat_1";
    outNames[2] = "detection_out";

    std::vector<Mat> target(outNames.size());
    for (int i = 0; i < outNames.size(); ++i)
    {
        std::string path = findDataFile("dnn/tensorflow/ssd_mobilenet_v1_coco." + outNames[i] + ".npy", false);
        target[i] = blobFromNPY(path);
    }

    Net net = readNetFromTensorflow(netPath, netConfig);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(GetParam());

    net.setInput(inp);

    std::vector<Mat> output;
    net.forward(output, outNames);

    normAssert(target[0].reshape(1, 1), output[0].reshape(1, 1), "", 1e-5, 1.5e-4);
    normAssert(target[1].reshape(1, 1), output[1].reshape(1, 1), "", 1e-5, 3e-4);
    normAssertDetections(target[2], output[2], "", 0.2);
}

TEST_P(Test_TensorFlow_nets, Inception_v2_SSD)
{
    std::string proto = findDataFile("dnn/ssd_inception_v2_coco_2017_11_17.pbtxt", false);
    std::string model = findDataFile("dnn/ssd_inception_v2_coco_2017_11_17.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat img = imread(findDataFile("dnn/street.png", false));
    Mat blob = blobFromImage(img, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(GetParam());

    net.setInput(blob);
    // Output has shape 1x1xNx7 where N - number of detections.
    // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
    Mat out = net.forward();
    Mat ref = (Mat_<float>(5, 7) << 0, 1, 0.90176028, 0.19872092, 0.36311883, 0.26461923, 0.63498729,
                                    0, 3, 0.93569964, 0.64865261, 0.45906419, 0.80675775, 0.65708131,
                                    0, 3, 0.75838411, 0.44668293, 0.45907149, 0.49459291, 0.52197015,
                                    0, 10, 0.95932811, 0.38349164, 0.32528657, 0.40387636, 0.39165527,
                                    0, 10, 0.93973452, 0.66561931, 0.37841269, 0.68074018, 0.42907384);
    normAssertDetections(ref, out, "", 0.5);
}

TEST_P(Test_TensorFlow_nets, Inception_v2_Faster_RCNN)
{
    std::string proto = findDataFile("dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt", false);
    std::string model = findDataFile("dnn/faster_rcnn_inception_v2_coco_2018_01_28.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    Mat img = imread(findDataFile("dnn/dog416.png", false));
    Mat blob = blobFromImage(img, 1.0f / 127.5, Size(800, 600), Scalar(127.5, 127.5, 127.5), true, false);

    net.setInput(blob);
    Mat out = net.forward();

    Mat ref = blobFromNPY(findDataFile("dnn/tensorflow/faster_rcnn_inception_v2_coco_2018_01_28.detection_out.npy"));
    normAssertDetections(ref, out, "", 0.3);
}

TEST_P(Test_TensorFlow_nets, opencv_face_detector_uint8)
{
    std::string proto = findDataFile("dnn/opencv_face_detector.pbtxt", false);
    std::string model = findDataFile("dnn/opencv_face_detector_uint8.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat img = imread(findDataFile("gpu/lbpcascade/er.png", false));
    Mat blob = blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(GetParam());

    net.setInput(blob);
    // Output has shape 1x1xNx7 where N - number of detections.
    // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
    Mat out = net.forward();

    // References are from test for Caffe model.
    Mat ref = (Mat_<float>(6, 7) << 0, 1, 0.99520785, 0.80997437, 0.16379407, 0.87996572, 0.26685631,
                                    0, 1, 0.9934696, 0.2831718, 0.50738752, 0.345781, 0.5985168,
                                    0, 1, 0.99096733, 0.13629119, 0.24892329, 0.19756334, 0.3310290,
                                    0, 1, 0.98977017, 0.23901358, 0.09084064, 0.29902688, 0.1769477,
                                    0, 1, 0.97203469, 0.67965847, 0.06876482, 0.73999709, 0.1513494,
                                    0, 1, 0.95097077, 0.51901293, 0.45863652, 0.5777427, 0.5347801);
    normAssertDetections(ref, out, "", 0.9, 3.4e-3, 1e-2);
}

// inp = cv.imread('opencv_extra/testdata/cv/ximgproc/sources/08.png')
// inp = inp[:,:,[2, 1, 0]].astype(np.float32).reshape(1, 512, 512, 3)
// outs = sess.run([sess.graph.get_tensor_by_name('feature_fusion/Conv_7/Sigmoid:0'),
//                  sess.graph.get_tensor_by_name('feature_fusion/concat_3:0')],
//                 feed_dict={'input_images:0': inp})
// scores = np.ascontiguousarray(outs[0].transpose(0, 3, 1, 2))
// geometry = np.ascontiguousarray(outs[1].transpose(0, 3, 1, 2))
// np.save('east_text_detection.scores.npy', scores)
// np.save('east_text_detection.geometry.npy', geometry)
TEST_P(Test_TensorFlow_nets, EAST_text_detection)
{
    std::string netPath = findDataFile("dnn/frozen_east_text_detection.pb", false);
    std::string imgPath = findDataFile("cv/ximgproc/sources/08.png", false);
    std::string refScoresPath = findDataFile("dnn/east_text_detection.scores.npy", false);
    std::string refGeometryPath = findDataFile("dnn/east_text_detection.geometry.npy", false);

    Net net = readNet(findDataFile("dnn/frozen_east_text_detection.pb", false));

    net.setPreferableTarget(GetParam());

    Mat img = imread(imgPath);
    Mat inp = blobFromImage(img, 1.0, Size(), Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(inp);

    std::vector<Mat> outs;
    std::vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";
    net.forward(outs, outNames);

    Mat scores = outs[0];
    Mat geometry = outs[1];

    normAssert(scores, blobFromNPY(refScoresPath), "scores");
    normAssert(geometry, blobFromNPY(refGeometryPath), "geometry", 1e-4, 3e-3);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TensorFlow_nets, availableDnnTargets());

typedef testing::TestWithParam<DNNTarget> Test_TensorFlow_fp16;

TEST_P(Test_TensorFlow_fp16, tests)
{
    int targetId = GetParam();
    const float l1 = 7e-4;
    const float lInf = 1e-2;
    runTensorFlowNet("fp16_single_conv", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_deconvolution", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_max_pool_odd_same", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_padding_valid", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_eltwise_add_mul", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_max_pool_odd_valid", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_pad_and_concat", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_max_pool_even", targetId, false, l1, lInf);
    runTensorFlowNet("fp16_padding_same", targetId, false, l1, lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TensorFlow_fp16,
                        Values(DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16));

TEST(Test_TensorFlow, defun)
{
    runTensorFlowNet("defun_dropout");
}

TEST(Test_TensorFlow, quantized)
{
    runTensorFlowNet("uint8_single_conv");
}

TEST(Test_TensorFlow, lstm)
{
    runTensorFlowNet("lstm", DNN_TARGET_CPU, true);
}

TEST(Test_TensorFlow, split)
{
    runTensorFlowNet("split_equals");
}

TEST(Test_TensorFlow, resize_nearest_neighbor)
{
    runTensorFlowNet("resize_nearest_neighbor");
}

TEST(Test_TensorFlow, slice)
{
    runTensorFlowNet("slice_4d");
}

TEST(Test_TensorFlow, softmax)
{
    runTensorFlowNet("keras_softmax");
}

TEST(Test_TensorFlow, relu6)
{
    runTensorFlowNet("keras_relu6");
    runTensorFlowNet("keras_relu6", DNN_TARGET_CPU, /*hasText*/ true);
}

TEST(Test_TensorFlow, keras_mobilenet_head)
{
    runTensorFlowNet("keras_mobilenet_head");
}

TEST(Test_TensorFlow, memory_read)
{
    double l1 = 1e-5;
    double lInf = 1e-4;
    runTensorFlowNet("lstm", DNN_TARGET_CPU, true, l1, lInf, true);

    runTensorFlowNet("batch_norm", DNN_TARGET_CPU, false, l1, lInf, true);
    runTensorFlowNet("fused_batch_norm", DNN_TARGET_CPU, false, l1, lInf, true);
    runTensorFlowNet("batch_norm_text", DNN_TARGET_CPU, true, l1, lInf, true);
}

TEST(Test_TensorFlow, resize_bilinear)
{
    runTensorFlowNet("resize_bilinear");
    runTensorFlowNet("resize_bilinear_factor");
}

TEST(Test_TensorFlow, two_inputs)
{
    Net net = readNet(path("two_inputs_net.pbtxt"));
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    Mat firstInput(2, 3, CV_32FC1), secondInput(2, 3, CV_32FC1);
    randu(firstInput, -1, 1);
    randu(secondInput, -1, 1);

    net.setInput(firstInput, "first_input");
    net.setInput(secondInput, "second_input");
    Mat out = net.forward();

    normAssert(out, firstInput + secondInput);
}

}
