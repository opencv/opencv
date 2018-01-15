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
#include <opencv2/ts/ocl_test.hpp>

namespace cvtest
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

template<typename TStr>
static std::string _tf(TStr filename, bool inTorchDir = true)
{
    String path = "dnn/";
    if (inTorchDir)
        path += "torch/";
    path += filename;
    return findDataFile(path, false);
}

TEST(Torch_Importer, simple_read)
{
    Net net;
    ASSERT_NO_THROW(net = readNetFromTorch(_tf("net_simple_net.txt"), false));
    ASSERT_FALSE(net.empty());
}

static void runTorchNet(String prefix, int targetId = DNN_TARGET_CPU, String outLayerName = "",
                        bool check2ndBlob = false, bool isBinary = false)
{
    String suffix = (isBinary) ? ".dat" : ".txt";

    Net net = readNetFromTorch(_tf(prefix + "_net" + suffix), isBinary);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(targetId);

    Mat inp, outRef;
    ASSERT_NO_THROW( inp = readTorchBlob(_tf(prefix + "_input" + suffix), isBinary) );
    ASSERT_NO_THROW( outRef = readTorchBlob(_tf(prefix + "_output" + suffix), isBinary) );

    if (outLayerName.empty())
        outLayerName = net.getLayerNames().back();

    net.setInput(inp, "0");
    std::vector<Mat> outBlobs;
    net.forward(outBlobs, outLayerName);
    normAssert(outRef, outBlobs[0]);

    if (check2ndBlob)
    {
        Mat out2 = outBlobs[1];
        Mat ref2 = readTorchBlob(_tf(prefix + "_output_2" + suffix), isBinary);
        normAssert(out2, ref2);
    }
}

TEST(Torch_Importer, run_convolution)
{
    runTorchNet("net_conv");
}

OCL_TEST(Torch_Importer, run_convolution)
{
    runTorchNet("net_conv", DNN_TARGET_OPENCL);
}

TEST(Torch_Importer, run_pool_max)
{
    runTorchNet("net_pool_max", DNN_TARGET_CPU, "", true);
}

OCL_TEST(Torch_Importer, run_pool_max)
{
    runTorchNet("net_pool_max", DNN_TARGET_OPENCL, "", true);
}

TEST(Torch_Importer, run_pool_ave)
{
    runTorchNet("net_pool_ave");
}

OCL_TEST(Torch_Importer, run_pool_ave)
{
    runTorchNet("net_pool_ave", DNN_TARGET_OPENCL);
}

TEST(Torch_Importer, run_reshape)
{
    runTorchNet("net_reshape");
    runTorchNet("net_reshape_batch");
    runTorchNet("net_reshape_single_sample");
    runTorchNet("net_reshape_channels", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, run_linear)
{
    runTorchNet("net_linear_2d");
}

TEST(Torch_Importer, run_paralel)
{
    runTorchNet("net_parallel", DNN_TARGET_CPU, "l5_torchMerge");
}

TEST(Torch_Importer, run_concat)
{
    runTorchNet("net_concat", DNN_TARGET_CPU, "l5_torchMerge");
    runTorchNet("net_depth_concat", DNN_TARGET_CPU, "", false, true);
}

OCL_TEST(Torch_Importer, run_concat)
{
    runTorchNet("net_concat", DNN_TARGET_OPENCL, "l5_torchMerge");
    runTorchNet("net_depth_concat", DNN_TARGET_OPENCL, "", false, true);
}

TEST(Torch_Importer, run_deconv)
{
    runTorchNet("net_deconv");
}

TEST(Torch_Importer, run_batch_norm)
{
    runTorchNet("net_batch_norm", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, net_prelu)
{
    runTorchNet("net_prelu");
}

TEST(Torch_Importer, net_cadd_table)
{
    runTorchNet("net_cadd_table");
}

TEST(Torch_Importer, net_softmax)
{
    runTorchNet("net_softmax");
    runTorchNet("net_softmax_spatial");
}

OCL_TEST(Torch_Importer, net_softmax)
{
    runTorchNet("net_softmax", DNN_TARGET_OPENCL);
    runTorchNet("net_softmax_spatial", DNN_TARGET_OPENCL);
}

TEST(Torch_Importer, net_logsoftmax)
{
    runTorchNet("net_logsoftmax");
    runTorchNet("net_logsoftmax_spatial");
}

OCL_TEST(Torch_Importer, net_logsoftmax)
{
    runTorchNet("net_logsoftmax", DNN_TARGET_OPENCL);
    runTorchNet("net_logsoftmax_spatial", DNN_TARGET_OPENCL);
}

TEST(Torch_Importer, net_lp_pooling)
{
    runTorchNet("net_lp_pooling_square", DNN_TARGET_CPU, "", false, true);
    runTorchNet("net_lp_pooling_power", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, net_conv_gemm_lrn)
{
    runTorchNet("net_conv_gemm_lrn", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, net_inception_block)
{
    runTorchNet("net_inception_block", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, net_normalize)
{
    runTorchNet("net_normalize", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, net_padding)
{
    runTorchNet("net_padding", DNN_TARGET_CPU, "", false, true);
    runTorchNet("net_spatial_zero_padding", DNN_TARGET_CPU, "", false, true);
    runTorchNet("net_spatial_reflection_padding", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, net_non_spatial)
{
    runTorchNet("net_non_spatial", DNN_TARGET_CPU, "", false, true);
}

TEST(Torch_Importer, ENet_accuracy)
{
    Net net;
    {
        const string model = findDataFile("dnn/Enet-model-best.net", false);
        net = readNetFromTorch(model, true);
        ASSERT_FALSE(net.empty());
    }

    Mat sample = imread(_tf("street.png", false));
    Mat inputBlob = blobFromImage(sample, 1./255);

    net.setInput(inputBlob, "");
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("torch_enet_prob.npy", false));
    // Due to numerical instability in Pooling-Unpooling layers (indexes jittering)
    // thresholds for ENet must be changed. Accuracy of resuults was checked on
    // Cityscapes dataset and difference in mIOU with Torch is 10E-4%
    normAssert(ref, out, "", 0.00044, 0.44);

    const int N = 3;
    for (int i = 0; i < N; i++)
    {
        net.setInput(inputBlob, "");
        Mat out = net.forward();
        normAssert(ref, out, "", 0.00044, 0.44);
    }
}

TEST(Torch_Importer, OpenFace_accuracy)
{
    const string model = findDataFile("dnn/openface_nn4.small2.v1.t7", false);
    Net net = readNetFromTorch(model);

    Mat sample = imread(findDataFile("cv/shared/lena.png", false));
    Mat sampleF32(sample.size(), CV_32FC3);
    sample.convertTo(sampleF32, sampleF32.type());
    sampleF32 /= 255;
    resize(sampleF32, sampleF32, Size(96, 96), 0, 0, INTER_NEAREST);

    Mat inputBlob = blobFromImage(sampleF32);

    net.setInput(inputBlob);
    Mat out = net.forward();

    Mat outRef = readTorchBlob(_tf("net_openface_output.dat"), true);
    normAssert(out, outRef);
}

OCL_TEST(Torch_Importer, OpenFace_accuracy)
{
    const string model = findDataFile("dnn/openface_nn4.small2.v1.t7", false);
    Net net = readNetFromTorch(model);

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    Mat sample = imread(findDataFile("cv/shared/lena.png", false));
    Mat sampleF32(sample.size(), CV_32FC3);
    sample.convertTo(sampleF32, sampleF32.type());
    sampleF32 /= 255;
    resize(sampleF32, sampleF32, Size(96, 96), 0, 0, INTER_NEAREST);

    Mat inputBlob = blobFromImage(sampleF32);

    net.setInput(inputBlob);
    Mat out = net.forward();

    Mat outRef = readTorchBlob(_tf("net_openface_output.dat"), true);
    normAssert(out, outRef);
}

OCL_TEST(Torch_Importer, ENet_accuracy)
{
    Net net;
    {
        const string model = findDataFile("dnn/Enet-model-best.net", false);
        net = readNetFromTorch(model, true);
        ASSERT_TRUE(!net.empty());
    }

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    Mat sample = imread(_tf("street.png", false));
    Mat inputBlob = blobFromImage(sample, 1./255);

    net.setInput(inputBlob, "");
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("torch_enet_prob.npy", false));
    // Due to numerical instability in Pooling-Unpooling layers (indexes jittering)
    // thresholds for ENet must be changed. Accuracy of resuults was checked on
    // Cityscapes dataset and difference in mIOU with Torch is 10E-4%
    normAssert(ref, out, "", 0.00044, 0.44);

    const int N = 3;
    for (int i = 0; i < N; i++)
    {
        net.setInput(inputBlob, "");
        Mat out = net.forward();
        normAssert(ref, out, "", 0.00044, 0.44);
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
TEST(Torch_Importer, FastNeuralStyle_accuracy)
{
    std::string models[] = {"dnn/fast_neural_style_eccv16_starry_night.t7",
                            "dnn/fast_neural_style_instance_norm_feathers.t7"};
    std::string targets[] = {"dnn/lena_starry_night.png", "dnn/lena_feathers.png"};

    for (int i = 0; i < 2; ++i)
    {
        const string model = findDataFile(models[i], false);
        Net net = readNetFromTorch(model);

        Mat img = imread(findDataFile("dnn/googlenet_1.png", false));
        Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(103.939, 116.779, 123.68), false);

        net.setInput(inputBlob);
        Mat out = net.forward();

        // Deprocessing.
        getPlane(out, 0, 0) += 103.939;
        getPlane(out, 0, 1) += 116.779;
        getPlane(out, 0, 2) += 123.68;
        out = cv::min(cv::max(0, out), 255);

        Mat ref = imread(findDataFile(targets[i]));
        Mat refBlob = blobFromImage(ref, 1.0, Size(), Scalar(), false);

        normAssert(out, refBlob, "", 0.5, 1.1);
    }
}

OCL_TEST(Torch_Importer, FastNeuralStyle_accuracy)
{
    std::string models[] = {"dnn/fast_neural_style_eccv16_starry_night.t7",
                            "dnn/fast_neural_style_instance_norm_feathers.t7"};
    std::string targets[] = {"dnn/lena_starry_night.png", "dnn/lena_feathers.png"};

    for (int i = 0; i < 2; ++i)
    {
        const string model = findDataFile(models[i], false);
        Net net = readNetFromTorch(model);

        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_OPENCL);

        Mat img = imread(findDataFile("dnn/googlenet_1.png", false));
        Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(103.939, 116.779, 123.68), false);

        net.setInput(inputBlob);
        Mat out = net.forward();

        // Deprocessing.
        getPlane(out, 0, 0) += 103.939;
        getPlane(out, 0, 1) += 116.779;
        getPlane(out, 0, 2) += 123.68;
        out = cv::min(cv::max(0, out), 255);

        Mat ref = imread(findDataFile(targets[i]));
        Mat refBlob = blobFromImage(ref, 1.0, Size(), Scalar(), false);

        normAssert(out, refBlob, "", 0.5, 1.1);
    }
}

}
