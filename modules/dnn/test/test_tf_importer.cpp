// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017-2019, Intel Corporation, all rights reserved.
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
    input -= Scalar::all(117); // mean sub

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
    Mat inputBlob = blobFromImage(sample, 1.0, Size(224, 224), Scalar(), /*swapRB*/true);

    net.setInput(inputBlob, "input");
    Mat out = net.forward("softmax2");

    Mat ref = blobFromNPY(_tf("tf_inception_prob.npy"));

    normAssert(ref, out);
}

static std::string path(const std::string& file)
{
    return findDataFile("dnn/tensorflow/" + file);
}

class Test_TensorFlow_layers : public DNNTestLayer
{
public:
    void runTensorFlowNet(const std::string& prefix, bool hasText = false,
                          double l1 = 0.0, double lInf = 0.0, bool memoryLoad = false)
    {
        std::string netPath = path(prefix + "_net.pb");
        std::string netConfig = (hasText ? path(prefix + "_net.pbtxt") : "");
        std::string inpPath = path(prefix + "_in.npy");
        std::string outPath = path(prefix + "_out.npy");

        cv::Mat input = blobFromNPY(inpPath);
        cv::Mat ref = blobFromNPY(outPath);
        checkBackend(&input, &ref);

        Net net;
        if (memoryLoad)
        {
            // Load files into a memory buffers
            std::vector<char> dataModel;
            readFileContent(netPath, dataModel);

            std::vector<char> dataConfig;
            if (hasText)
            {
                readFileContent(netConfig, dataConfig);
            }

            net = readNetFromTensorflow(dataModel.data(), dataModel.size(),
                                        dataConfig.data(), dataConfig.size());
        }
        else
            net = readNetFromTensorflow(netPath, netConfig);

        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(input);
        cv::Mat output = net.forward();
        normAssert(ref, output, "", l1 ? l1 : default_l1, lInf ? lInf : default_lInf);
    }
};

TEST_P(Test_TensorFlow_layers, reduce_mean)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    runTensorFlowNet("global_pool_by_axis");
}

TEST_P(Test_TensorFlow_layers, conv_single_conv)
{
    runTensorFlowNet("single_conv");
}
TEST_P(Test_TensorFlow_layers, conv_atrous_conv2d_valid)
{
    runTensorFlowNet("atrous_conv2d_valid");
}
TEST_P(Test_TensorFlow_layers, conv_atrous_conv2d_same)
{
    runTensorFlowNet("atrous_conv2d_same");
}
TEST_P(Test_TensorFlow_layers, conv_depthwise_conv2d)
{
    runTensorFlowNet("depthwise_conv2d");
}
TEST_P(Test_TensorFlow_layers, conv_keras_atrous_conv2d_same)
{
    runTensorFlowNet("keras_atrous_conv2d_same");
}
TEST_P(Test_TensorFlow_layers, conv_pool_nchw)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTensorFlowNet("conv_pool_nchw");
}

TEST_P(Test_TensorFlow_layers, Convolution3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2019010000)
    applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    if (backend == DNN_BACKEND_CUDA)
    {
        // ok
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
    else if (target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");

    runTensorFlowNet("conv3d");
}

TEST_P(Test_TensorFlow_layers, padding)
{
    runTensorFlowNet("padding_valid");
    runTensorFlowNet("spatial_padding");
    runTensorFlowNet("mirror_pad");
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019020000)
    if (target == DNN_TARGET_MYRIAD)
    {
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif
    runTensorFlowNet("keras_pad_concat");
}

TEST_P(Test_TensorFlow_layers, padding_same)
{
    // Reference output values are in range [0.0006, 2.798]
    runTensorFlowNet("padding_same");
}

TEST_P(Test_TensorFlow_layers, eltwise)
{
    runTensorFlowNet("eltwise_add_mul");
    runTensorFlowNet("eltwise_sub");
}

TEST_P(Test_TensorFlow_layers, channel_broadcast)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    runTensorFlowNet("channel_broadcast");
}

TEST_P(Test_TensorFlow_layers, pad_and_concat)
{
    runTensorFlowNet("pad_and_concat");
}

TEST_P(Test_TensorFlow_layers, concat_axis_1)
{
    runTensorFlowNet("concat_axis_1");
}

TEST_P(Test_TensorFlow_layers, concat_3d)
{
    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL);
    }

    if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
         backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019) && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    runTensorFlowNet("concat_3d");
}

TEST_P(Test_TensorFlow_layers, batch_norm_1)
{
    runTensorFlowNet("batch_norm");
}
TEST_P(Test_TensorFlow_layers, batch_norm_2)
{
    runTensorFlowNet("batch_norm", false, 0.0, 0.0, true);
}
TEST_P(Test_TensorFlow_layers, batch_norm_3)
{
    runTensorFlowNet("fused_batch_norm");
}
TEST_P(Test_TensorFlow_layers, batch_norm_4)
{
    runTensorFlowNet("fused_batch_norm", false, 0.0, 0.0, true);
}
TEST_P(Test_TensorFlow_layers, batch_norm_5)
{
    runTensorFlowNet("batch_norm_text", true);
}
TEST_P(Test_TensorFlow_layers, batch_norm_6)
{
    runTensorFlowNet("batch_norm_text", true, 0.0, 0.0, true);
}
TEST_P(Test_TensorFlow_layers, batch_norm_7)
{
    runTensorFlowNet("unfused_batch_norm");
}
TEST_P(Test_TensorFlow_layers, batch_norm_8)
{
    runTensorFlowNet("fused_batch_norm_no_gamma");
}
TEST_P(Test_TensorFlow_layers, batch_norm_9)
{
    runTensorFlowNet("unfused_batch_norm_no_gamma");
}
TEST_P(Test_TensorFlow_layers, batch_norm_10)
{
    runTensorFlowNet("mvn_batch_norm");
}
TEST_P(Test_TensorFlow_layers, batch_norm_11)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    runTensorFlowNet("mvn_batch_norm_1x1");
}
TEST_P(Test_TensorFlow_layers, batch_norm_12)
{
    runTensorFlowNet("switch_identity");
}
TEST_P(Test_TensorFlow_layers, batch_norm_13)
{
    runTensorFlowNet("keras_batch_norm_training");
}

TEST_P(Test_TensorFlow_layers, batch_norm3D)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        throw SkipTestException("");
    }
    runTensorFlowNet("batch_norm3d");
}

TEST_P(Test_TensorFlow_layers, slim_batch_norm)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    // Output values range: [-40.0597, 207.827]
    double l1 =  default_l1, lInf = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        l1 = 0.041;
        lInf = 0.33;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.005;
        lInf = 0.33;
    }
    runTensorFlowNet("slim_batch_norm", false, l1, lInf);
}

TEST_P(Test_TensorFlow_layers, pooling_max_pool_even)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTensorFlowNet("max_pool_even");
}
TEST_P(Test_TensorFlow_layers, pooling_max_pool_odd_valid)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTensorFlowNet("max_pool_odd_valid");
}
TEST_P(Test_TensorFlow_layers, pooling_max_pool_odd_same)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTensorFlowNet("max_pool_odd_same");
}
TEST_P(Test_TensorFlow_layers, pooling_reduce_mean)
{
    runTensorFlowNet("reduce_mean");  // an average pooling over all spatial dimensions.
}

TEST_P(Test_TensorFlow_layers, max_pool_grad)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    runTensorFlowNet("max_pool_grad");
}

// TODO: fix tests and replace to pooling
TEST_P(Test_TensorFlow_layers, ave_pool_same)
{
    // Reference output values are in range [-0.519531, 0.112976]
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
    {
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif
    runTensorFlowNet("ave_pool_same");
}

TEST_P(Test_TensorFlow_layers, MaxPooling3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2019010000)
    applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    if (backend == DNN_BACKEND_CUDA)
    {
        // ok
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
    else if (target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");

    runTensorFlowNet("max_pool3d");
}

TEST_P(Test_TensorFlow_layers, AvePooling3D)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2019010000)
    applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    if (backend == DNN_BACKEND_CUDA)
    {
        // ok
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);  // Only CPU on DLIE backend is supported
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // Only CPU on DLIE backend is supported
    else if (target != DNN_TARGET_CPU)
        throw SkipTestException("Only CPU is supported");

    runTensorFlowNet("ave_pool3d");
}

TEST_P(Test_TensorFlow_layers, deconvolution)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);
    runTensorFlowNet("deconvolution");
    runTensorFlowNet("deconvolution_same");
    runTensorFlowNet("deconvolution_stride_2_same");
    runTensorFlowNet("deconvolution_adj_pad_valid");
    runTensorFlowNet("deconvolution_adj_pad_same");
    runTensorFlowNet("keras_deconv_valid");
    runTensorFlowNet("keras_deconv_same");
    runTensorFlowNet("keras_deconv_same_v2");
}

TEST_P(Test_TensorFlow_layers, matmul)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    runTensorFlowNet("matmul");
    runTensorFlowNet("nhwc_transpose_reshape_matmul");
    // Reference output values are in range [-5.688, 4.484]
    double l1 = target == DNN_TARGET_MYRIAD ? 6.1e-3 : default_l1;
    runTensorFlowNet("nhwc_reshape_matmul", false, l1);
    runTensorFlowNet("matmul_layout");
}

TEST_P(Test_TensorFlow_layers, reshape)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    runTensorFlowNet("shift_reshape_no_reorder");
    runTensorFlowNet("reshape_no_reorder");
    runTensorFlowNet("reshape_reduce");
    runTensorFlowNet("reshape_as_shape");
}

TEST_P(Test_TensorFlow_layers, flatten)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_2, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif

    runTensorFlowNet("flatten", true);
}

TEST_P(Test_TensorFlow_layers, unfused_flatten)
{
    runTensorFlowNet("unfused_flatten");
    runTensorFlowNet("unfused_flatten_unknown_batch");
}

TEST_P(Test_TensorFlow_layers, leaky_relu)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTensorFlowNet("leaky_relu_order1");
    runTensorFlowNet("leaky_relu_order2");
    runTensorFlowNet("leaky_relu_order3");
}

TEST_P(Test_TensorFlow_layers, l2_normalize)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    runTensorFlowNet("l2_normalize");
}

// TODO: fix it and add to l2_normalize
TEST_P(Test_TensorFlow_layers, l2_normalize_3d)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019
            && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16)
    )
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    runTensorFlowNet("l2_normalize_3d");
}

class Test_TensorFlow_nets : public DNNTestLayer {};

TEST_P(Test_TensorFlow_nets, MobileNet_SSD)
{
#if defined(INF_ENGINE_RELEASE)
    if (target == DNN_TARGET_MYRIAD)
    {
#if INF_ENGINE_VER_MAJOR_GE(2019020000)
        if (getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X,
                         backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ?
                             CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER :
                             CV_TEST_TAG_DNN_SKIP_IE_NGRAPH,
                         CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    }
#endif

    checkBackend();
    std::string imgPath = findDataFile("dnn/street.png");
    std::string netConfig = findDataFile("dnn/ssd_mobilenet_v1_coco.pbtxt");
    std::string netPath = findDataFile("dnn/ssd_mobilenet_v1_coco.pb", false);

    Mat inp;
    resize(imread(imgPath), inp, Size(300, 300));
    inp = blobFromImage(inp, 1.0f / 127.5, Size(), Scalar(127.5, 127.5, 127.5), true);

    Mat ref = blobFromNPY(findDataFile("dnn/tensorflow/ssd_mobilenet_v1_coco.detection_out.npy"));

    Net net = readNetFromTensorflow(netPath, netConfig);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(inp);
    Mat out = net.forward();

    double scoreDiff = default_l1, iouDiff = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.0043;
        iouDiff = 0.037;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        iouDiff = 0.04;
    }
    normAssertDetections(ref, out, "", 0.2, scoreDiff, iouDiff);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_RELEASE >= 2019010000
    expectNoFallbacksFromIE(net);
#endif
}

TEST_P(Test_TensorFlow_nets, Inception_v2_SSD)
{
    applyTestTag(target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LE(2019010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD &&
        getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    checkBackend();
    Mat img = imread(findDataFile("dnn/street.png"));
    std::string proto = findDataFile("dnn/ssd_inception_v2_coco_2017_11_17.pbtxt");
    std::string model = findDataFile("dnn/ssd_inception_v2_coco_2017_11_17.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat blob = blobFromImage(img, 1.0f, Size(300, 300), Scalar(), true, false);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(blob);
    // Output has shape 1x1xNx7 where N - number of detections.
    // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
    Mat out = net.forward();
    Mat ref = (Mat_<float>(5, 7) << 0, 1, 0.90176028, 0.19872092, 0.36311883, 0.26461923, 0.63498729,
                                    0, 3, 0.93569964, 0.64865261, 0.45906419, 0.80675775, 0.65708131,
                                    0, 3, 0.75838411, 0.44668293, 0.45907149, 0.49459291, 0.52197015,
                                    0, 10, 0.95932811, 0.38349164, 0.32528657, 0.40387636, 0.39165527,
                                    0, 10, 0.93973452, 0.66561931, 0.37841269, 0.68074018, 0.42907384);

    double scoreDiff = default_l1, iouDiff = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.0097;
        iouDiff = 0.09;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 6e-3;
        iouDiff = 0.05;
    }
    normAssertDetections(ref, out, "", 0.5, scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

TEST_P(Test_TensorFlow_nets, MobileNet_v1_SSD)
{
    checkBackend();
    std::string proto = findDataFile("dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt");
    std::string model = findDataFile("dnn/ssd_mobilenet_v1_coco_2017_11_17.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat img = imread(findDataFile("dnn/dog416.png"));
    Mat blob = blobFromImage(img, 1.0f, Size(300, 300), Scalar(), true, false);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(blob);
    Mat out = net.forward();

    Mat ref = blobFromNPY(findDataFile("dnn/tensorflow/ssd_mobilenet_v1_coco_2017_11_17.detection_out.npy"));
    float scoreDiff = 1.5e-5, iouDiff = 1e-3;
    float detectionConfThresh = (target == DNN_TARGET_MYRIAD) ? 0.35 : 0.3;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.011;
        iouDiff = 0.012;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.006;
        iouDiff = 0.01;
    }
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD &&
        getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
    {
        scoreDiff = 0.061;
        iouDiff = 0.12;
        detectionConfThresh = 0.36;
    }
#endif
    normAssertDetections(ref, out, "", detectionConfThresh, scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

TEST_P(Test_TensorFlow_nets, Faster_RCNN)
{
    // FIXIT split test
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_1GB : CV_TEST_TAG_MEMORY_2GB),
        CV_TEST_TAG_LONG,
        CV_TEST_TAG_DEBUG_VERYLONG
    );
    static std::string names[] = {"faster_rcnn_inception_v2_coco_2018_01_28",
                                  "faster_rcnn_resnet50_coco_2018_01_28"};

#ifdef INF_ENGINE_RELEASE
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 &&
        (INF_ENGINE_VER_MAJOR_LT(2019020000) || target != DNN_TARGET_CPU))
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

    if (INF_ENGINE_VER_MAJOR_GT(2019030000) &&
        backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    // segfault: inference-engine/thirdparty/clDNN/src/gpu/detection_output_cpu.cpp:111:
    // Assertion `prior_height > 0' failed.
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);

    checkBackend();

    double scoresDiff = backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ? 2.9e-5 : 1e-5;
    for (int i = 0; i < 2; ++i)
    {
        std::string proto = findDataFile("dnn/" + names[i] + ".pbtxt");
        std::string model = findDataFile("dnn/" + names[i] + ".pb", false);

        Net net = readNetFromTensorflow(model, proto);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        Mat img = imread(findDataFile("dnn/dog416.png"));
        Mat blob = blobFromImage(img, 1.0f, Size(800, 600), Scalar(), true, false);

        net.setInput(blob);
        Mat out = net.forward();

        Mat ref = blobFromNPY(findDataFile("dnn/tensorflow/" + names[i] + ".detection_out.npy"));
        normAssertDetections(ref, out, names[i].c_str(), 0.3, scoresDiff);
    }
}

TEST_P(Test_TensorFlow_nets, MobileNet_v1_SSD_PPN)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    checkBackend();
    std::string proto = findDataFile("dnn/ssd_mobilenet_v1_ppn_coco.pbtxt");
    std::string model = findDataFile("dnn/ssd_mobilenet_v1_ppn_coco.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat img = imread(findDataFile("dnn/dog416.png"));
    Mat ref = blobFromNPY(findDataFile("dnn/tensorflow/ssd_mobilenet_v1_ppn_coco.detection_out.npy"));
    Mat blob = blobFromImage(img, 1.0f, Size(300, 300), Scalar(), true, false);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(blob);
    Mat out = net.forward();

    double scoreDiff = 1.1e-5, iouDiff = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 0.048;
        iouDiff = 0.058;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.006;
        iouDiff = 0.05;
    }
    normAssertDetections(ref, out, "", 0.45, scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

TEST_P(Test_TensorFlow_nets, opencv_face_detector_uint8)
{
    checkBackend();
    std::string proto = findDataFile("dnn/opencv_face_detector.pbtxt");
    std::string model = findDataFile("dnn/opencv_face_detector_uint8.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat img = imread(findDataFile("gpu/lbpcascade/er.png"));
    Mat blob = blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
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
    double scoreDiff = 3.4e-3, iouDiff = 1e-2;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        scoreDiff = 4e-3;
        iouDiff = 0.024;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 4e-3;
        iouDiff = 0.02;
    }
    normAssertDetections(ref, out, "", 0.9, scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
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
    applyTestTag(
        (target == DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
        CV_TEST_TAG_DEBUG_LONG
    );

#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_OPENCL_FP16 &&
        (INF_ENGINE_VER_MAJOR_EQ(2019020000) || INF_ENGINE_VER_MAJOR_GE(2020010000))
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    checkBackend();

    std::string netPath = findDataFile("dnn/frozen_east_text_detection.pb", false);
    std::string imgPath = findDataFile("cv/ximgproc/sources/08.png");
    std::string refScoresPath = findDataFile("dnn/east_text_detection.scores.npy");
    std::string refGeometryPath = findDataFile("dnn/east_text_detection.geometry.npy");

    Net net = readNet(netPath);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

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

    // Scores are in range [0, 1]. Geometry values are in range [-0.23, 290]
    double l1_scores = default_l1, lInf_scores = default_lInf;
    double l1_geometry = default_l1, lInf_geometry = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        lInf_scores = backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ? 0.16 : 0.11;
        l1_geometry = 0.28; lInf_geometry = 5.94;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        lInf_scores = 0.41;
        l1_geometry = 0.28; lInf_geometry = 5.94;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        lInf_scores = 0.1;
        l1_geometry = 0.3; lInf_geometry = 7;
    }
    else
    {
        l1_geometry = 1e-4, lInf_geometry = 3e-3;
    }
    normAssert(scores, blobFromNPY(refScoresPath), "scores", l1_scores, lInf_scores);
    normAssert(geometry, blobFromNPY(refGeometryPath), "geometry", l1_geometry, lInf_geometry);
    expectNoFallbacksFromIE(net);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TensorFlow_nets, dnnBackendsAndTargets());


TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_single_conv)
{
    float l1 = 0.00078, lInf = 0.012;
    runTensorFlowNet("fp16_single_conv", false, l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_max_pool_odd_same)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    float l1 = 0.00078, lInf = 0.012;
    runTensorFlowNet("fp16_max_pool_odd_same", false, l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_eltwise_add_mul)
{
    float l1 = 0.00078, lInf = 0.012;
    runTensorFlowNet("fp16_eltwise_add_mul", false, l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_pad_and_concat)
{
    float l1 = 0.00078, lInf = 0.012;
    runTensorFlowNet("fp16_pad_and_concat", false, l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_padding_valid)
{
    float l1 = 0.00078, lInf = 0.012;
    runTensorFlowNet("fp16_padding_valid", false, l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_max_pool_even)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    float l1 = 0.00078, lInf = 0.012;
    // Reference output values are in range [0.0889, 1.651]
    runTensorFlowNet("fp16_max_pool_even", false, (target == DNN_TARGET_MYRIAD) ? 0.003 : l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_deconvolution)
{
    float l1 = 0.00078, lInf = 0.012;
    if (target == DNN_TARGET_MYRIAD) {
        l1 = 0.0041;
        lInf = 0.024;
    }
    // Reference output values are in range [0, 10.75]
    runTensorFlowNet("fp16_deconvolution", false, l1, lInf);
}
TEST_P(Test_TensorFlow_layers, fp16_weights_fp16_max_pool_odd_valid)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2020020000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    float l1 = 0.00078, lInf = 0.012;
    if (target == DNN_TARGET_MYRIAD) {
        l1 = 0.0041;
        lInf = 0.024;
    }
    // Reference output values are in range [0.418, 2.297]
    runTensorFlowNet("fp16_max_pool_odd_valid", false, l1, lInf);
}

TEST_P(Test_TensorFlow_layers, fp16_padding_same)
{
    // Reference output values are in range [-3.504, -0.002]
    runTensorFlowNet("fp16_padding_same", false, 7e-4, 4e-3);
}

TEST_P(Test_TensorFlow_layers, defun)
{
    runTensorFlowNet("defun_dropout");
}

TEST_P(Test_TensorFlow_layers, quantized)
{
    runTensorFlowNet("uint8_single_conv");
}

TEST_P(Test_TensorFlow_layers, lstm)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); /* not supported */
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    runTensorFlowNet("lstm", true);
    runTensorFlowNet("lstm", true, 0.0, 0.0, true);
}

TEST_P(Test_TensorFlow_layers, split)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    runTensorFlowNet("split");
}

TEST_P(Test_TensorFlow_layers, split_equals)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    runTensorFlowNet("split_equals");
}

TEST_P(Test_TensorFlow_layers, resize_nearest_neighbor)
{
    runTensorFlowNet("resize_nearest_neighbor");
    runTensorFlowNet("keras_upsampling2d");
}

TEST_P(Test_TensorFlow_layers, fused_resize_conv)
{
    runTensorFlowNet("fused_resize_conv");
}

TEST_P(Test_TensorFlow_layers, slice)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 &&
        (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    double l1 = target == DNN_TARGET_MYRIAD ? 4.9e-3 : default_l1;
    runTensorFlowNet("crop2d", false, l1);
    runTensorFlowNet("slice_4d");
    runTensorFlowNet("strided_slice");
}

TEST_P(Test_TensorFlow_layers, softmax)
{
    runTensorFlowNet("keras_softmax");
    runTensorFlowNet("slim_softmax");
}

TEST_P(Test_TensorFlow_layers, slim_softmax_v2)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD &&
        getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_2, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    runTensorFlowNet("slim_softmax_v2");
}

TEST_P(Test_TensorFlow_layers, relu6)
{
    runTensorFlowNet("keras_relu6");
    runTensorFlowNet("keras_relu6", /*hasText*/ true);
}

TEST_P(Test_TensorFlow_layers, subpixel)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    runTensorFlowNet("subpixel");
}

TEST_P(Test_TensorFlow_layers, keras_mobilenet_head)
{
    runTensorFlowNet("keras_mobilenet_head");
    runTensorFlowNet("keras_learning_phase");
}

TEST_P(Test_TensorFlow_layers, resize_bilinear)
{
    runTensorFlowNet("resize_bilinear");
    runTensorFlowNet("resize_bilinear_factor");
    runTensorFlowNet("resize_bilinear_down");
}

TEST_P(Test_TensorFlow_layers, tf2_dense)
{
    runTensorFlowNet("tf2_dense");
}

TEST_P(Test_TensorFlow_layers, clip_by_value)
{
    runTensorFlowNet("clip_by_value");
}

TEST_P(Test_TensorFlow_layers, tf2_prelu)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); // not supported; only across channels is supported
    runTensorFlowNet("tf2_prelu");
}

TEST_P(Test_TensorFlow_layers, tf2_permute_nhwc_ncwh)
{
    runTensorFlowNet("tf2_permute_nhwc_ncwh");
}

TEST_P(Test_TensorFlow_layers, squeeze)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_2, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    int inpShapes[][4] = {{1, 3, 4, 2}, {1, 3, 1, 2}, {1, 3, 4, 1}, {1, 3, 4, 1}};  // TensorFlow's shape (NHWC)
    int outShapes[][3] = {{3, 4, 2}, {1, 3, 2}, {1, 3, 4}, {1, 3, 4}};
    int squeeze_dims[] = {0, 2, 3, -1};
    for (int i = 0; i < 4; ++i)
    {
        SCOPED_TRACE(format("i=%d", i));
        std::string pbtxt =
            "node { name: \"input\" op: \"Placeholder\""
            "attr { key: \"data_format\" value { s: \"NHWC\" } } }"
            "node { name: \"squeeze\" op: \"Squeeze\" input: \"input\""
              "attr { key: \"squeeze_dims\" value { list { i:" + format("%d", squeeze_dims[i]) + "}}}}";
        Net net = readNetFromTensorflow(0, 0, pbtxt.c_str(), pbtxt.size());
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        Mat tfInp(4, &inpShapes[i][0], CV_32F);
        randu(tfInp, -1, 1);

        // NHWC to NCHW
        CV_Assert(inpShapes[i][0] == 1);
        std::swap(inpShapes[i][2], inpShapes[i][3]);
        std::swap(inpShapes[i][1], inpShapes[i][2]);
        Mat cvInp = tfInp.reshape(1, tfInp.total() / inpShapes[i][1]).t();
        cvInp = cvInp.reshape(1, 4, &inpShapes[i][0]);

        net.setInput(cvInp);
        Mat out = net.forward();
        normAssert(tfInp.reshape(1, 3, &outShapes[i][0]), out, "", default_l1, default_lInf);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TensorFlow_layers, dnnBackendsAndTargets());

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

TEST_P(Test_TensorFlow_nets, Mask_RCNN)
{
    static const double kMaskThreshold = 0.5;

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);

    if (target == DNN_TARGET_MYRIAD && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);

    if (target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);

    applyTestTag(CV_TEST_TAG_MEMORY_1GB, CV_TEST_TAG_DEBUG_VERYLONG);
    Mat img = imread(findDataFile("dnn/street.png"));
    std::string proto = findDataFile("dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
    std::string model = findDataFile("dnn/mask_rcnn_inception_v2_coco_2018_01_28.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat refDetections = blobFromNPY(path("mask_rcnn_inception_v2_coco_2018_01_28.detection_out.npy"));
    Mat refMasks = blobFromNPY(path("mask_rcnn_inception_v2_coco_2018_01_28.detection_masks.npy"));
    Mat blob = blobFromImage(img, 1.0f, Size(800, 800), Scalar(), true, false);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    net.setInput(blob);

    // Mask-RCNN predicts bounding boxes and segmentation masks.
    std::vector<String> outNames(2);
    outNames[0] = "detection_out_final";
    outNames[1] = "detection_masks";

    std::vector<Mat> outs;
    net.forward(outs, outNames);

    Mat outDetections = outs[0];
    Mat outMasks = outs[1];

    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.019 : 2e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.018 : default_lInf;
    normAssertDetections(refDetections, outDetections, "", /*threshold for zero confidence*/1e-5, scoreDiff, iouDiff);

    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];

    int masksSize[] = {1, numDetections, outMasks.size[2], outMasks.size[3]};
    Mat masks(4, &masksSize[0], CV_32F);

    std::vector<cv::Range> srcRanges(4, cv::Range::all());
    std::vector<cv::Range> dstRanges(4, cv::Range::all());

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; ++i)
    {
        // Get a class id for this bounding box and copy mask only for that class.
        int classId = static_cast<int>(outDetections.at<float>(i, 1));
        srcRanges[0] = dstRanges[1] = cv::Range(i, i + 1);
        srcRanges[1] = cv::Range(classId, classId + 1);
        outMasks(srcRanges).copyTo(masks(dstRanges));
    }
    cv::Range topRefMasks[] = {Range::all(), Range(0, numDetections), Range::all(), Range::all()};
    refMasks = refMasks(&topRefMasks[0]);

    // make binary masks
    cv::threshold(masks.reshape(1, 1), masks, kMaskThreshold, 1, THRESH_BINARY);
    cv::threshold(refMasks.reshape(1, 1), refMasks, kMaskThreshold, 1, THRESH_BINARY);

    double inter = cv::countNonZero(masks & refMasks);
    double area = cv::countNonZero(masks | refMasks);
    EXPECT_GE(inter / area, 0.99);

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        expectNoFallbacks(net);
}

TEST_P(Test_TensorFlow_nets, EfficientDet)
{
    if (target != DNN_TARGET_CPU)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
    }
    checkBackend();
    std::string proto = findDataFile("dnn/efficientdet-d0.pbtxt");
    std::string model = findDataFile("dnn/efficientdet-d0.pb", false);

    Net net = readNetFromTensorflow(model, proto);
    Mat img = imread(findDataFile("dnn/dog416.png"));
    Mat blob = blobFromImage(img, 1.0/255, Size(512, 512), Scalar(123.675, 116.28, 103.53));

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.setInput(blob);
    // Output has shape 1x1xNx7 where N - number of detections.
    // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
    Mat out = net.forward();

    // References are from test for TensorFlow model.
    Mat ref = (Mat_<float>(3, 7) << 0, 1, 0.8437444, 0.153996080160141, 0.20534580945968628, 0.7463544607162476, 0.7414066195487976,
                                    0, 17, 0.8245924, 0.16657517850399017, 0.3996818959712982, 0.4111558794975281, 0.9306337833404541,
                                    0, 7, 0.8039304, 0.6118435263633728, 0.13175517320632935, 0.9065558314323425, 0.2943994700908661);
    double scoreDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 4e-3 : 1e-5;
    double iouDiff = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 2e-3 : 1e-4;
    if (target == DNN_TARGET_CUDA_FP16)
    {
        scoreDiff = 0.002;
        iouDiff = 0.003;
    }
    normAssertDetections(ref, out, "", 0.5, scoreDiff, iouDiff);
    expectNoFallbacksFromIE(net);
}

}
