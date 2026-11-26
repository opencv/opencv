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
#include <cstdio>
#include <fstream>

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

TEST(Test_Torch, padding_rejects_invalid_dim_and_min_pad)
{
    auto appendInt32 = [](std::string& out, int32_t v) {
        out.append(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto appendFloat64 = [](std::string& out, double v) {
        out.append(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    auto appendString = [&](std::string& out, const std::string& s) {
        appendInt32(out, static_cast<int32_t>(s.size()));
        out.append(s);
    };
    auto makePaddingT7 = [&](int32_t dimVal, int32_t padVal, int32_t nInputDim) {
        std::string buf;
        appendInt32(buf, 4);  // TYPE_TORCH
        appendInt32(buf, 1);  // object index
        appendString(buf, "V 1");
        appendString(buf, "nn.Padding");
        appendInt32(buf, 3);  // TYPE_TABLE
        appendInt32(buf, 2);  // table index
        appendInt32(buf, 4);  // 4 pairs
        appendInt32(buf, 2);  // TYPE_STRING
        appendString(buf, "dim");
        appendInt32(buf, 1);  // TYPE_NUMBER
        appendFloat64(buf, static_cast<double>(dimVal));
        appendInt32(buf, 2);  // TYPE_STRING
        appendString(buf, "pad");
        appendInt32(buf, 1);  // TYPE_NUMBER
        appendFloat64(buf, static_cast<double>(padVal));
        appendInt32(buf, 2);  // TYPE_STRING
        appendString(buf, "nInputDim");
        appendInt32(buf, 1);  // TYPE_NUMBER
        appendFloat64(buf, static_cast<double>(nInputDim));
        appendInt32(buf, 2);  // TYPE_STRING
        appendString(buf, "value");
        appendInt32(buf, 1);  // TYPE_NUMBER
        appendFloat64(buf, 0.0);
        return buf;
    };
    auto loadPaddingBytes = [](const std::string& bytes) {
        const std::string path = cv::tempfile(".t7");
        FILE* fp = fopen(path.c_str(), "wb");
        CV_Assert(fp != NULL);
        const size_t written = fwrite(bytes.data(), 1, bytes.size(), fp);
        fclose(fp);
        CV_Assert(written == bytes.size());
        try
        {
            Net net = readNetFromTorch(path);
            std::remove(path.c_str());
            return net;
        }
        catch (...)
        {
            std::remove(path.c_str());
            throw;
        }
    };

    Net validNet;
    ASSERT_NO_THROW(validNet = loadPaddingBytes(makePaddingT7(2, 1, 2)));
    validNet.setInput(Mat(1, 3, CV_32F, Scalar(2.0f)));
    Mat out = validNet.forward();
    ASSERT_EQ(out.dims, 2);
    EXPECT_EQ(out.rows, 1);
    EXPECT_EQ(out.cols, 4);
    EXPECT_FLOAT_EQ(out.at<float>(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(out.at<float>(0, 3), 0.0f);

    EXPECT_THROW(loadPaddingBytes(makePaddingT7(0, 1, 2)), cv::Exception);
    EXPECT_THROW(loadPaddingBytes(makePaddingT7(-1, -1, 2)), cv::Exception);
    EXPECT_THROW(loadPaddingBytes(makePaddingT7(INT_MIN, 1, 2)), cv::Exception);
    EXPECT_THROW(loadPaddingBytes(makePaddingT7(CV_MAX_DIM + 1, 1, 2)), cv::Exception);
    EXPECT_THROW(loadPaddingBytes(makePaddingT7(2, INT_MIN, 2)), cv::Exception);
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
    EXPECT_LE(countNonZero(refMask != testMask), 2);
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

namespace torch_fixture {

// Torch7 binary serialization: little-endian int32 for object tags and indices, int64 for
// sizes, strides and offsets.
static void putInt(std::string& out, int32_t v)
{
    for (int b = 0; b < 4; b++)
        out.push_back((char)(((uint32_t)v >> (8 * b)) & 0xff));
}

static void putLong(std::string& out, int64_t v)
{
    for (int b = 0; b < 8; b++)
        out.push_back((char)(((uint64_t)v >> (8 * b)) & 0xff));
}

static void putString(std::string& out, const std::string& text)
{
    putInt(out, (int32_t)text.size());
    out += text;
}

enum { TYPE_STRING = 2, TYPE_TABLE = 3, TYPE_TORCH = 4 };

static void putObject(std::string& out, int index, const std::string& className)
{
    putInt(out, TYPE_TORCH);
    putInt(out, index);
    putString(out, "V 1");
    putString(out, className);
}

static void putTensorHeader(std::string& out, const std::vector<int64_t>& sizes,
                            const std::vector<int64_t>& strides, int64_t offset1)
{
    putInt(out, (int32_t)sizes.size());
    for (size_t i = 0; i < sizes.size(); i++)
        putLong(out, sizes[i]);
    for (size_t i = 0; i < strides.size(); i++)
        putLong(out, strides[i]);
    putLong(out, offset1);
}

// Element i of every storage holds the value sampleValue(i) (or i for a plain float tensor).
static float sampleValue(int i) { return 0.25f * (float)((i % 7) + 1); }

static void putFloatBits(std::string& out, float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    putInt(out, (int32_t)bits);
}

static void putDoubleBits(std::string& out, double v)
{
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    putLong(out, (int64_t)bits);
}

// A torch.FloatTensor whose FloatStorage holds 0, 1, ..., storageElems - 1.
static std::string floatTensor(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                               int64_t offset1, int storageElems)
{
    std::string out;
    putObject(out, 1, "torch.FloatTensor");
    putTensorHeader(out, sizes, strides, offset1);
    putObject(out, 2, "torch.FloatStorage");
    putLong(out, storageElems);
    for (int i = 0; i < storageElems; i++)
        putFloatBits(out, (float)i);
    return out;
}

// A torch.FloatTensor whose ndims field is written directly and followed by nothing else.
static std::string tensorWithRawNdims(int32_t ndims)
{
    std::string out;
    putObject(out, 1, "torch.FloatTensor");
    putInt(out, ndims);
    return out;
}

enum StorageKind { BYTE_STORAGE, FLOAT_STORAGE, DOUBLE_STORAGE, LONG_STORAGE };

static std::string tensorClass(StorageKind k)
{
    static const char* names[] = {"torch.ByteTensor", "torch.FloatTensor", "torch.DoubleTensor", "torch.LongTensor"};
    return names[k];
}

static std::string storageClass(StorageKind k)
{
    static const char* names[] = {"torch.ByteStorage", "torch.FloatStorage", "torch.DoubleStorage", "torch.LongStorage"};
    return names[k];
}

// nn.Linear whose bias (offset 1) and weight (`weightOffset1`) tensors share one storage. The
// bias tensor owns the storage (object 4); the weight tensor refers back to object
// `weightStorageIndex`, which is the storage unless the fixture deliberately points elsewhere.
static std::string linearSharedStorage(StorageKind biasKind, StorageKind weightKind, StorageKind storageKind,
                                       int count, int storageElems, int64_t weightOffset1,
                                       bool constantValues, int weightStorageIndex = 4)
{
    std::string out;
    putObject(out, 1, "nn.Linear");
    putInt(out, TYPE_TABLE);
    putInt(out, 2);
    putInt(out, 2);

    putInt(out, TYPE_STRING);
    putString(out, "bias");
    putObject(out, 3, tensorClass(biasKind));
    putTensorHeader(out, {count}, {1}, 1);
    putObject(out, 4, storageClass(storageKind));
    putLong(out, storageElems);
    for (int i = 0; i < storageElems; i++)
    {
        const int v = constantValues ? 1 : (i % 7) + 1;
        switch (storageKind)
        {
        case BYTE_STORAGE: out.push_back((char)v); break;
        case FLOAT_STORAGE: putFloatBits(out, sampleValue(i)); break;
        case DOUBLE_STORAGE: putDoubleBits(out, (double)sampleValue(i)); break;
        case LONG_STORAGE: putLong(out, v); break;
        }
    }

    putInt(out, TYPE_STRING);
    putString(out, "weight");
    putObject(out, 5, tensorClass(weightKind));
    putTensorHeader(out, {count, 1}, {1, 1}, weightOffset1);
    putInt(out, TYPE_TORCH);
    putInt(out, weightStorageIndex);
    return out;
}

// Writes the serialized bytes to a temporary file for the duration of one call.
struct TempFile
{
    std::string path;
    explicit TempFile(const std::string& bytes) : path(cv::tempfile(".t7"))
    {
        std::ofstream f(path.c_str(), std::ios::binary);
        f.write(bytes.data(), (std::streamsize)bytes.size());
        f.close();
    }
    ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace torch_fixture

static void expectTorchBlobRejected(const std::string& bytes, const char* reason)
{
    SCOPED_TRACE(reason);
    torch_fixture::TempFile file(bytes);
    try
    {
        (void)readTorchBlob(file.path, true);
        FAIL() << "Torch blob was accepted";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
    }
}

static void expectTorchModelRejected(const std::string& bytes, const char* reason)
{
    SCOPED_TRACE(reason);
    torch_fixture::TempFile file(bytes);
    try
    {
        (void)readNetFromTorch(file.path);
        FAIL() << "Torch model with an inconsistent tensor was accepted";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
    }
}

// Reads a float tensor view and compares each logical element with the storage element it names.
static void expectTorchView(const std::string& bytes, const std::vector<int>& sizes,
                            const std::vector<int>& steps, int offset0)
{
    torch_fixture::TempFile file(bytes);
    Mat blob;
    ASSERT_NO_THROW(blob = readTorchBlob(file.path, true));
    ASSERT_EQ(blob.dims, (int)sizes.size());
    size_t total = 1;
    for (size_t i = 0; i < sizes.size(); i++)
    {
        ASSERT_EQ(blob.size[(int)i], sizes[i]);
        total *= (size_t)sizes[i];
    }
    ASSERT_EQ(blob.total(), total);
    ASSERT_TRUE(blob.isContinuous());

    std::vector<int> index(sizes.size(), 0);
    for (size_t linear = 0; linear < total; linear++)
    {
        int at = offset0;
        for (size_t d = 0; d < sizes.size(); d++)
            at += index[d] * steps[d];
        EXPECT_EQ(blob.ptr<float>()[linear], (float)at) << "element " << linear;
        for (int d = (int)sizes.size() - 1; d >= 0; d--)
        {
            if (++index[d] < sizes[d])
                break;
            index[d] = 0;
        }
    }
}

TEST(Test_Torch, readTorchBlob_tensor_views_match_logical_elements)
{
    using namespace torch_fixture;
    expectTorchView(floatTensor({1, 16}, {16, 1}, 1, 16), {1, 16}, {16, 1}, 0);
    expectTorchView(floatTensor({2, 3, 4}, {12, 4, 1}, 1, 24), {2, 3, 4}, {12, 4, 1}, 0);
    expectTorchView(floatTensor({2, 3, 4}, {12, 4, 1}, 6, 29), {2, 3, 4}, {12, 4, 1}, 5);
    // sizes [2,3], strides [4,1] over an 8-element storage: element (i, j) is storage[4*i + j].
    expectTorchView(floatTensor({2, 3}, {4, 1}, 1, 8), {2, 3}, {4, 1}, 0);
    // a dimension of size 1 is never stepped over, so its stride is free
    expectTorchView(floatTensor({1, 3, 1, 4}, {100, 4, 7, 1}, 3, 14), {1, 3, 1, 4}, {100, 4, 7, 1}, 2);
    expectTorchView(floatTensor({3, 2}, {1, 3}, 1, 6), {3, 2}, {1, 3}, 0);
    expectTorchView(floatTensor({2, 3, 4}, {1, 2, 6}, 2, 25), {2, 3, 4}, {1, 2, 6}, 1);
    expectTorchView(floatTensor({2, 3}, {5, 1}, 4, 11), {2, 3}, {5, 1}, 3);
    expectTorchView(floatTensor({4, 1, 3}, {1, 12, 4}, 1, 12), {4, 1, 3}, {1, 12, 4}, 0);
    // sizes [2,2], strides [1,1] over a 3-element storage: both rows share storage[1]
    expectTorchView(floatTensor({2, 2}, {1, 1}, 1, 3), {2, 2}, {1, 1}, 0);
}

TEST(Test_Torch, readTorchBlob_empty_tensor_offset_bound)
{
    using namespace torch_fixture;
    // [0,2] at 1-based offset 5 over 4 floats: the offset equals the storage size, which is allowed
    {
        TempFile file(floatTensor({0, 2}, {2, 1}, 5, 4));
        Mat blob;
        ASSERT_NO_THROW(blob = readTorchBlob(file.path, true));
        EXPECT_EQ(blob.total(), (size_t)0);
    }
    expectTorchBlobRejected(floatTensor({0, 2}, {2, 1}, 6, 4), "tensor storage offset out of bounds");
}

TEST(Test_Torch, readTorchBlob_rejects_invalid_tensor_metadata)
{
    using namespace torch_fixture;
    expectTorchBlobRejected(tensorWithRawNdims(-1), "invalid tensor dimensionality");
    expectTorchBlobRejected(tensorWithRawNdims(33), "invalid tensor dimensionality");
    expectTorchBlobRejected(floatTensor({-1, 2}, {2, 1}, 1, 4), "invalid tensor dimension size");
    expectTorchBlobRejected(floatTensor({(int64_t)1 << 31, 1}, {1, 1}, 1, 4), "invalid tensor dimension size");
    expectTorchBlobRejected(floatTensor({4, 4}, {0, 1}, 1, 4), "non-positive tensor stride");
    expectTorchBlobRejected(floatTensor({2, 2}, {-1, 1}, 3, 4), "non-positive tensor stride");
    expectTorchBlobRejected(floatTensor({16, 1}, {(int64_t)3 << 60, 1}, 1, 8), "strided span overflow");
    expectTorchBlobRejected(floatTensor({1, 64}, {1, 1}, 1, 16), "exceed backing storage size");
    expectTorchBlobRejected(floatTensor({1, 8}, {7, 1}, 10, 16), "exceed backing storage size");
}

TEST(Test_Torch, readTorchBlob_validates_storage_offset_before_use)
{
    using namespace torch_fixture;
    const int validOffsets[] = {1, 5, 9, 13};
    for (size_t k = 0; k < sizeof(validOffsets) / sizeof(validOffsets[0]); k++)
    {
        SCOPED_TRACE(validOffsets[k]);
        TempFile file(floatTensor({4}, {1}, validOffsets[k], 16));
        Mat blob;
        ASSERT_NO_THROW(blob = readTorchBlob(file.path, true));
        ASSERT_EQ(blob.total(), (size_t)4);
        for (int i = 0; i < 4; i++)
            EXPECT_EQ(blob.ptr<float>()[i], (float)(validOffsets[k] - 1 + i));
    }

    const int64_t nonPositive[] = {0, -1, std::numeric_limits<int64_t>::min(),
                                   std::numeric_limits<int64_t>::min() + 1};
    for (size_t k = 0; k < sizeof(nonPositive) / sizeof(nonPositive[0]); k++)
        expectTorchBlobRejected(floatTensor({4}, {1}, nonPositive[k], 16), "storage offset must be positive");

    // 17 is one element past the storage; the next values have the low 32 bits of a valid offset
    const int64_t beyondStorage[] = {17, 0x100000001LL, 0x100000005LL, 0x7FFFFFFFLL + 2,
                                     std::numeric_limits<int64_t>::max()};
    for (size_t k = 0; k < sizeof(beyondStorage) / sizeof(beyondStorage[0]); k++)
        expectTorchBlobRejected(floatTensor({4}, {1}, beyondStorage[k], 16), "exceed backing storage size");
}

TEST(Test_Torch, tensor_reuses_storage_of_the_same_type)
{
    // Each nn.Linear has bias [32] at 1-based offset 1 and weight [32,1] at 1-based offset 33 sharing
    // a single 64-element storage.
    using namespace torch_fixture;
    const int count = 32, total = 64;
    const StorageKind kinds[] = {FLOAT_STORAGE, DOUBLE_STORAGE, LONG_STORAGE};
    for (size_t k = 0; k < 3; k++)
    {
        SCOPED_TRACE((int)kinds[k]);
        TempFile file(linearSharedStorage(kinds[k], kinds[k], kinds[k], count, total, count + 1, false));
        Net net;
        ASSERT_NO_THROW(net = readNetFromTorch(file.path));
        net.setInput(Mat(1, 1, CV_32F, Scalar(2.0f)));
        Mat out;
        ASSERT_NO_THROW(out = net.forward());
        ASSERT_EQ(out.total(), (size_t)count);
        for (int i = 0; i < count; i++)
        {
            const float scale = kinds[k] == LONG_STORAGE ? 4.0f : 1.0f;  // integers 1..7 versus 0.25 * (1..7)
            const float biasVal = scale * 0.25f * (float)((i % 7) + 1);
            const float weightVal = scale * 0.25f * (float)(((count + i) % 7) + 1);
            EXPECT_EQ(out.at<float>(i), 2.0f * weightVal + biasVal);
        }
    }
}

TEST(Test_Torch, tensor_type_must_match_reused_storage)
{
    using namespace torch_fixture;
    const char* mismatch = "tensor type does not match the type of its storage";
    const int n = 64;
    expectTorchModelRejected(linearSharedStorage(BYTE_STORAGE, FLOAT_STORAGE, BYTE_STORAGE, n, n, 1, true), mismatch);
    expectTorchModelRejected(linearSharedStorage(LONG_STORAGE, FLOAT_STORAGE, LONG_STORAGE, n, n, 1, true), mismatch);
    expectTorchModelRejected(linearSharedStorage(FLOAT_STORAGE, LONG_STORAGE, FLOAT_STORAGE, n, n, 1, false), mismatch);
    expectTorchModelRejected(linearSharedStorage(DOUBLE_STORAGE, LONG_STORAGE, DOUBLE_STORAGE, n, n, 1, false), mismatch);
    expectTorchModelRejected(linearSharedStorage(LONG_STORAGE, DOUBLE_STORAGE, LONG_STORAGE, n, n, 1, true), mismatch);
}

// The first tensor reads a FloatStorage (4-byte elements); a second DoubleTensor then reuses that storage. Reading
// it with 8-byte elements would copy twice the storage size, so the already-read path must compare the types too.
TEST(Test_Torch, double_tensor_must_not_reuse_float_storage)
{
    using namespace torch_fixture;
    const char* mismatch = "tensor type does not match the type of its storage";
    const int n = 64;
    expectTorchModelRejected(linearSharedStorage(FLOAT_STORAGE, DOUBLE_STORAGE, FLOAT_STORAGE, n, n, 1, false), mismatch);
    expectTorchModelRejected(linearSharedStorage(DOUBLE_STORAGE, FLOAT_STORAGE, DOUBLE_STORAGE, n, n, 1, false), mismatch);
}

TEST(Test_Torch, tensor_storage_index_must_refer_to_a_storage)
{
    using namespace torch_fixture;
    const int n = 64;
    // the weight tensor names object 3, which is the bias tensor
    expectTorchModelRejected(linearSharedStorage(FLOAT_STORAGE, FLOAT_STORAGE, FLOAT_STORAGE, n, n, 1, false, 3),
                             "is not a storage");
}

}
