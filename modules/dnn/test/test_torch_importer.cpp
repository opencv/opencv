/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

namespace opencv_test
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

template<typename TStr>
static std::string _tf(TStr filename, bool inTorchDir = true, bool required = true)
{
    String path = "dnn/";
    if (inTorchDir)
        path += "torch/";
    path += filename;
    return findDataFile(path, required);
}

TEST(Torch_Importer, simple_read)
{
    Net net;
    ASSERT_NO_THROW(net = readNetFromTorch(_tf("net_simple_net.txt"), false));
    ASSERT_FALSE(net.empty());
}

class Test_Torch_layers : public DNNTestLayer
{
public:
    void runTorchNet(const String& prefix, String outLayerName = "",
                     bool check2ndBlob = false, bool isBinary = false, bool evaluate = true,
                     double l1 = 0.0, double lInf = 0.0)
    {
        String suffix = (isBinary) ? ".dat" : ".txt";

        Mat inp, outRef;
        ASSERT_NO_THROW( inp = readTorchBlob(_tf(prefix + "_input" + suffix), isBinary) );
        ASSERT_NO_THROW( outRef = readTorchBlob(_tf(prefix + "_output" + suffix), isBinary) );

        checkBackend(backend, target, &inp, &outRef);

        Net net = readNetFromTorch(_tf(prefix + "_net" + suffix), isBinary, evaluate);
        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        if (outLayerName.empty())
            outLayerName = net.getLayerNames().back();

        net.setInput(inp);
        std::vector<Mat> outBlobs;
        net.forward(outBlobs, outLayerName);
        l1 = l1 ? l1 : default_l1;
        lInf = lInf ? lInf : default_lInf;
        normAssert(outRef, outBlobs[0], "", l1, lInf);

        if (check2ndBlob && backend == DNN_BACKEND_OPENCV)
        {
            Mat out2 = outBlobs[1];
            Mat ref2 = readTorchBlob(_tf(prefix + "_output_2" + suffix), isBinary);
            normAssert(out2, ref2, "", l1, lInf);
        }
    }
};

TEST_P(Test_Torch_layers, run_convolution)
{
    // Output reference values are in range [23.4018, 72.0181]
    double l1 = default_l1, lInf = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 0.08;
        lInf = 0.43;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.08;
        lInf = 0.5;
    }
    runTorchNet("net_conv", "", false, true, true, l1, lInf);
}

TEST_P(Test_Torch_layers, run_pool_max)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    double l1 = 0.0, lInf = 0.0;
    runTorchNet("net_pool_max", "", true, false, true, l1, lInf);
}

TEST_P(Test_Torch_layers, run_pool_ave)
{
    runTorchNet("net_pool_ave");
}

TEST_P(Test_Torch_layers, run_reshape_change_batch_size)
{
    runTorchNet("net_reshape");
}

TEST_P(Test_Torch_layers, run_reshape)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    runTorchNet("net_reshape_batch");
    runTorchNet("net_reshape_channels", "", false, true);
}

TEST_P(Test_Torch_layers, run_reshape_single_sample)
{
    // Reference output values in range [14.4586, 18.4492].
    double l1 = default_l1, lInf = default_lInf;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 0.033;
        lInf = 0.05;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.02;
        lInf = 0.04;
    }
    runTorchNet("net_reshape_single_sample", "", false, false, true, l1, lInf);
}

TEST_P(Test_Torch_layers, run_linear)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
    runTorchNet("net_linear_2d");
}

TEST_P(Test_Torch_layers, run_concat)
{
    runTorchNet("net_concat", "l5_torchMerge");
}

TEST_P(Test_Torch_layers, run_depth_concat)
{
    double lInf = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        lInf = 0.032;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        lInf = 0.03;
    }
    runTorchNet("net_depth_concat", "", false, true, true, 0.0, lInf);
}

TEST_P(Test_Torch_layers, run_deconv)
{
    runTorchNet("net_deconv");
}

TEST_P(Test_Torch_layers, run_batch_norm)
{
    runTorchNet("net_batch_norm", "", false, true);
    runTorchNet("net_batch_norm_train", "", false, true, false);
}

TEST_P(Test_Torch_layers, net_prelu)
{
    runTorchNet("net_prelu");
}

TEST_P(Test_Torch_layers, net_cadd_table)
{
    runTorchNet("net_cadd_table");
}

TEST_P(Test_Torch_layers, net_softmax)
{
    runTorchNet("net_softmax");
    runTorchNet("net_softmax_spatial");
}

TEST_P(Test_Torch_layers, net_logsoftmax)
{
    runTorchNet("net_logsoftmax");
    runTorchNet("net_logsoftmax_spatial");
}

TEST_P(Test_Torch_layers, net_lp_pooling_square)
{
    runTorchNet("net_lp_pooling_square", "", false, true);
}
TEST_P(Test_Torch_layers, net_lp_pooling_power)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTorchNet("net_lp_pooling_power", "", false, true);
}

TEST_P(Test_Torch_layers, net_conv_gemm_lrn)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    double l1 = 0.0, lInf = 0.0;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 0.046;
        lInf = 0.023;
    }
    else if (target == DNN_TARGET_MYRIAD)
    {
        l1 = 0.02;
        lInf = 0.05;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.0042;
        lInf = 0.021;
    }
    // The OpenCL kernels use the native_ math functions which have
    // implementation defined accuracy, so we use relaxed thresholds. See
    // https://github.com/opencv/opencv/issues/9821 for more details.
    else if (target == DNN_TARGET_OPENCL)
    {
        l1 = 0.02;
        lInf = 0.02;
    }
    runTorchNet("net_conv_gemm_lrn", "", false, true, true, l1, lInf);
}

TEST_P(Test_Torch_layers, net_inception_block)
{
    runTorchNet("net_inception_block", "", false, true);
}

TEST_P(Test_Torch_layers, net_normalize)
{
    if(backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA); /* only L1 and L2 norms are supported */

    runTorchNet("net_normalize", "", false, true);
}

TEST_P(Test_Torch_layers, net_padding)
{
    runTorchNet("net_padding", "", false, true);
    runTorchNet("net_spatial_zero_padding", "", false, true);
    runTorchNet("net_spatial_reflection_padding", "", false, true);
}

TEST_P(Test_Torch_layers, net_non_spatial)
{
#if defined(INF_ENGINE_RELEASE) && ( \
    INF_ENGINE_VER_MAJOR_EQ(2021030000) || \
    INF_ENGINE_VER_MAJOR_EQ(2021040000) \
)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        // 2021.3: crash
        // 2021.4: [ GENERAL_ERROR ]  AssertionFailed: !out.networkInputs.empty()
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // exception
#endif

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 &&
        (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
    runTorchNet("net_non_spatial", "", false, true);
}

TEST_P(Test_Torch_layers, run_paralel)
{
    if (backend != DNN_BACKEND_OPENCV || target != DNN_TARGET_CPU)
        throw SkipTestException("");  // TODO: Check this
    runTorchNet("net_parallel", "l5_torchMerge");
}

TEST_P(Test_Torch_layers, net_residual)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_RELEASE == 2018050000
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && (target == DNN_TARGET_OPENCL ||
                                                    target == DNN_TARGET_OPENCL_FP16))
        applyTestTag(target == DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                     CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
    runTorchNet("net_residual", "", false, true);
}

class Test_Torch_nets : public DNNTestLayer {};

TEST_P(Test_Torch_nets, OpenFace_accuracy)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2018050000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    checkBackend();

    const string model = findDataFile("dnn/openface_nn4.small2.v1.t7", false);
    Net net = readNetFromTorch(model);

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    if (target == DNN_TARGET_CPU_FP16)
        net.enableWinograd(false);

    Mat sample = imread(findDataFile("cv/shared/lena.png"));
    Mat sampleF32(sample.size(), CV_32FC3);
    sample.convertTo(sampleF32, sampleF32.type());
    sampleF32 /= 255;
    resize(sampleF32, sampleF32, Size(96, 96), 0, 0, INTER_NEAREST);

    Mat inputBlob = blobFromImage(sampleF32, 1.0, Size(), Scalar(), /*swapRB*/true);

    net.setInput(inputBlob);
    Mat out = net.forward();

    // Reference output values are in range [-0.17212, 0.263492]
    // on Myriad problem layer: l4_Pooling - does not use pads_begin
    float l1 = 1e-5, lInf = 1e-3;
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
    {
        l1 = 2e-3;
        lInf = 5e-3;
    }
    else if (target == DNN_TARGET_CUDA_FP16)
    {
        l1 = 0.0004;
        lInf = 0.0012;
    }
    Mat outRef = readTorchBlob(_tf("net_openface_output.dat"), true);
    normAssert(out, outRef, "", l1, lInf);
}

static Mat getSegmMask(const Mat& scores)
{
    const int rows = scores.size[2];
    const int cols = scores.size[3];
    const int numClasses = scores.size[1];

    Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1, Scalar(0));
    for (int ch = 0; ch < numClasses; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = scores.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }
    return maxCl;
}

// Computer per-class intersection over union metric.
static void normAssertSegmentation(const Mat& ref, const Mat& test)
{
    CV_Assert_N(ref.dims == 4, test.dims == 4);
    const int numClasses = ref.size[1];
    CV_Assert(numClasses == test.size[1]);

    Mat refMask = getSegmMask(ref);
    Mat testMask = getSegmMask(test);
    EXPECT_EQ(countNonZero(refMask != testMask), 0);
}

TEST_P(Test_Torch_nets, ENet_accuracy)
{
    applyTestTag(target == DNN_TARGET_CPU ? "" : CV_TEST_TAG_MEMORY_512MB);
    checkBackend();
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    if (target == DNN_TARGET_CPU_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2020010000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#else
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target != DNN_TARGET_CPU)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
        throw SkipTestException("");
    }
#endif
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2023000000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU)
    {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        throw SkipTestException("");
    }

    Net net;
    {
        const string model = findDataFile("dnn/Enet-model-best.net", false);
        net = readNetFromTorch(model, true);
        ASSERT_TRUE(!net.empty());
    }

    net.enableWinograd(false);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat sample = imread(_tf("street.png", false));
    Mat inputBlob = blobFromImage(sample, 1./255, Size(), Scalar(), /*swapRB*/true);

    net.setInput(inputBlob, "");
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("torch_enet_prob.npy", false));
    // Due to numerical instability in Pooling-Unpooling layers (indexes jittering)
    // thresholds for ENet must be changed. Accuracy of results was checked on
    // Cityscapes dataset and difference in mIOU with Torch is 10E-4%
    normAssert(ref, out, "", 0.0005, /*target == DNN_TARGET_CPU ? 0.453 : */0.552);
    normAssertSegmentation(ref, out);

    const int N = 3;
    for (int i = 0; i < N; i++)
    {
        net.setInput(inputBlob, "");
        Mat out = net.forward();
        normAssert(ref, out, "", 0.0005, /*target == DNN_TARGET_CPU ? 0.453 : */0.552);
        normAssertSegmentation(ref, out);
    }
}

// Check accuracy of style transfer models from https://github.com/jcjohnson/fast-neural-style
// th fast_neural_style.lua \
//   -input_image ~/opencv_extra/testdata/dnn/googlenet_1.png \
//   -output_image lena.png \
//   -median_filter 0 \
//   -image_size 0 \
//   -model models/eccv16/starry_night.t7
// th fast_neural_style.lua \
//   -input_image ~/opencv_extra/testdata/dnn/googlenet_1.png \
//   -output_image lena.png \
//   -median_filter 0 \
//   -image_size 0 \
//   -model models/instance_norm/feathers.t7
TEST_P(Test_Torch_nets, FastNeuralStyle_accuracy)
{
#if defined INF_ENGINE_RELEASE
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD
            && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
#endif

    checkBackend();

#if defined(INF_ENGINE_RELEASE)
#if INF_ENGINE_RELEASE <= 2018050000
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_OPENCL)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif
#endif

    std::string models[] = {"dnn/fast_neural_style_eccv16_starry_night.t7",
                            "dnn/fast_neural_style_instance_norm_feathers.t7"};
    std::string targets[] = {"dnn/lena_starry_night.png", "dnn/lena_feathers.png"};

    for (int i = 0; i < 2; ++i)
    {
        const string model = findDataFile(models[i], false);
        Net net = readNetFromTorch(model);

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        Mat img = imread(findDataFile("dnn/googlenet_1.png"));
        Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(103.939, 116.779, 123.68), false);

        if (target == DNN_TARGET_CPU_FP16)
            net.enableWinograd(false);

        net.setInput(inputBlob);
        Mat out = net.forward();

        // Deprocessing.
        getPlane(out, 0, 0) += 103.939;
        getPlane(out, 0, 1) += 116.779;
        getPlane(out, 0, 2) += 123.68;
        out = cv::min(cv::max(0, out), 255);

        Mat ref = imread(findDataFile(targets[i]));
        Mat refBlob = blobFromImage(ref, 1.0, Size(), Scalar(), false);

        if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
        {
            double normL1 = cvtest::norm(refBlob, out, cv::NORM_L1) / refBlob.total();
            if (target == DNN_TARGET_MYRIAD)
                EXPECT_LE(normL1, 4.0f);
            else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
                EXPECT_LE(normL1, 1.0f);
            else
                EXPECT_LE(normL1, 0.6f);
        }
        else if(target == DNN_TARGET_CUDA_FP16)
        {
            normAssert(out, refBlob, "", 0.6, 26);
        }
        else if (target == DNN_TARGET_CPU_FP16)
        {
            normAssert(out, refBlob, "", 0.7, 25);
        }
        else
            normAssert(out, refBlob, "", 0.5, 1.16);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Torch_nets, dnnBackendsAndTargets());

// Test a custom layer
// https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialUpSamplingNearest
class SpatialUpSamplingNearestLayer CV_FINAL : public Layer
{
public:
    SpatialUpSamplingNearestLayer(const LayerParams &params) : Layer(params)
    {
        scale = params.get<int>("scale_factor");
    }

    static Ptr<Layer> create(LayerParams& params)
    {
        return Ptr<Layer>(new SpatialUpSamplingNearestLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE
    {
        std::vector<int> outShape(4);
        outShape[0] = inputs[0][0];  // batch size
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = scale * inputs[0][2];
        outShape[3] = scale * inputs[0][3];
        outputs.assign(1, outShape);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat& inp = inputs[0];
        Mat& out = outputs[0];
        const int outHeight = out.size[2];
        const int outWidth = out.size[3];
        for (size_t n = 0; n < inp.size[0]; ++n)
        {
            for (size_t ch = 0; ch < inp.size[1]; ++ch)
            {
                resize(getPlane(inp, n, ch), getPlane(out, n, ch),
                       Size(outWidth, outHeight), 0, 0, INTER_NEAREST);
            }
        }
    }

private:
    int scale;
};

TEST_P(Test_Torch_layers, upsampling_nearest)
{
#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021030000)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_MYRIAD)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);  // TODO
#endif

    // Test a custom layer.
    CV_DNN_REGISTER_LAYER_CLASS(SpatialUpSamplingNearest, SpatialUpSamplingNearestLayer);
    try
    {
        runTorchNet("net_spatial_upsampling_nearest", "", false, true);
    }
    catch (...)
    {
        LayerFactory::unregisterLayer("SpatialUpSamplingNearest");
        throw;
    }
    LayerFactory::unregisterLayer("SpatialUpSamplingNearest");

    // Test an implemented layer.
    runTorchNet("net_spatial_upsampling_nearest", "", false, true);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Torch_layers, dnnBackendsAndTargets());

}
