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

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    Mat inp, outRef;
    ASSERT_NO_THROW( inp = readTorchBlob(_tf(prefix + "_input" + suffix), isBinary) );
    ASSERT_NO_THROW( outRef = readTorchBlob(_tf(prefix + "_output" + suffix), isBinary) );

    if (outLayerName.empty())
        outLayerName = net.getLayerNames().back();

    net.setInput(inp);
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

typedef testing::TestWithParam<DNNTarget> Test_Torch_layers;

TEST_P(Test_Torch_layers, run_convolution)
{
    runTorchNet("net_conv", GetParam(), "", false, true);
}

TEST_P(Test_Torch_layers, run_pool_max)
{
    runTorchNet("net_pool_max", GetParam(), "", true);
}

TEST_P(Test_Torch_layers, run_pool_ave)
{
    runTorchNet("net_pool_ave", GetParam());
}

TEST_P(Test_Torch_layers, run_reshape)
{
    int targetId = GetParam();
    runTorchNet("net_reshape", targetId);
    runTorchNet("net_reshape_batch", targetId);
    runTorchNet("net_reshape_single_sample", targetId);
    runTorchNet("net_reshape_channels", targetId, "", false, true);
}

TEST_P(Test_Torch_layers, run_linear)
{
    runTorchNet("net_linear_2d", GetParam());
}

TEST_P(Test_Torch_layers, run_concat)
{
    int targetId = GetParam();
    runTorchNet("net_concat", targetId, "l5_torchMerge");
    runTorchNet("net_depth_concat", targetId, "", false, true);
}

TEST_P(Test_Torch_layers, run_deconv)
{
    runTorchNet("net_deconv", GetParam());
}

TEST_P(Test_Torch_layers, run_batch_norm)
{
    runTorchNet("net_batch_norm", GetParam(), "", false, true);
}

TEST_P(Test_Torch_layers, net_prelu)
{
    runTorchNet("net_prelu", GetParam());
}

TEST_P(Test_Torch_layers, net_cadd_table)
{
    runTorchNet("net_cadd_table", GetParam());
}

TEST_P(Test_Torch_layers, net_softmax)
{
    int targetId = GetParam();
    runTorchNet("net_softmax", targetId);
    runTorchNet("net_softmax_spatial", targetId);
}

TEST_P(Test_Torch_layers, net_logsoftmax)
{
    runTorchNet("net_logsoftmax");
    runTorchNet("net_logsoftmax_spatial");
}

TEST_P(Test_Torch_layers, net_lp_pooling)
{
    int targetId = GetParam();
    runTorchNet("net_lp_pooling_square", targetId, "", false, true);
    runTorchNet("net_lp_pooling_power", targetId, "", false, true);
}

TEST_P(Test_Torch_layers, net_conv_gemm_lrn)
{
    runTorchNet("net_conv_gemm_lrn", GetParam(), "", false, true);
}

TEST_P(Test_Torch_layers, net_inception_block)
{
    runTorchNet("net_inception_block", GetParam(), "", false, true);
}

TEST_P(Test_Torch_layers, net_normalize)
{
    runTorchNet("net_normalize", GetParam(), "", false, true);
}

TEST_P(Test_Torch_layers, net_padding)
{
    int targetId = GetParam();
    runTorchNet("net_padding", targetId, "", false, true);
    runTorchNet("net_spatial_zero_padding", targetId, "", false, true);
    runTorchNet("net_spatial_reflection_padding", targetId, "", false, true);
}

TEST_P(Test_Torch_layers, net_non_spatial)
{
    runTorchNet("net_non_spatial", GetParam(), "", false, true);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_Torch_layers, availableDnnTargets());

typedef testing::TestWithParam<DNNTarget> Test_Torch_nets;

TEST_P(Test_Torch_nets, OpenFace_accuracy)
{
    const string model = findDataFile("dnn/openface_nn4.small2.v1.t7", false);
    Net net = readNetFromTorch(model);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(GetParam());

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

TEST_P(Test_Torch_nets, ENet_accuracy)
{
    Net net;
    {
        const string model = findDataFile("dnn/Enet-model-best.net", false);
        net = readNetFromTorch(model, true);
        ASSERT_TRUE(!net.empty());
    }

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(GetParam());

    Mat sample = imread(_tf("street.png", false));
    Mat inputBlob = blobFromImage(sample, 1./255);

    net.setInput(inputBlob, "");
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("torch_enet_prob.npy", false));
    // Due to numerical instability in Pooling-Unpooling layers (indexes jittering)
    // thresholds for ENet must be changed. Accuracy of results was checked on
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
TEST_P(Test_Torch_nets, FastNeuralStyle_accuracy)
{
    std::string models[] = {"dnn/fast_neural_style_eccv16_starry_night.t7",
                            "dnn/fast_neural_style_instance_norm_feathers.t7"};
    std::string targets[] = {"dnn/lena_starry_night.png", "dnn/lena_feathers.png"};

    for (int i = 0; i < 2; ++i)
    {
        const string model = findDataFile(models[i], false);
        Net net = readNetFromTorch(model);

        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(GetParam());

        Mat img = imread(findDataFile("dnn/googlenet_1.png", false));
        Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(103.939, 116.779, 123.68), false);

        net.setInput(inputBlob);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
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

INSTANTIATE_TEST_CASE_P(/**/, Test_Torch_nets, availableDnnTargets());

// TODO: fix OpenCL and add to the rest of tests
TEST(Torch_Importer, run_paralel)
{
    runTorchNet("net_parallel", DNN_TARGET_CPU, "l5_torchMerge");
}

TEST(Torch_Importer, DISABLED_run_paralel)
{
    runTorchNet("net_parallel", DNN_TARGET_OPENCL, "l5_torchMerge");
}

TEST(Torch_Importer, net_residual)
{
    runTorchNet("net_residual", DNN_TARGET_CPU, "", false, true);
}

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

    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals) CV_OVERRIDE
    {
        Mat& inp = *inputs[0];
        Mat& out = outputs[0];
        const int outHeight = out.size[2];
        const int outWidth = out.size[3];
        for (size_t n = 0; n < inputs[0]->size[0]; ++n)
        {
            for (size_t ch = 0; ch < inputs[0]->size[1]; ++ch)
            {
                resize(getPlane(inp, n, ch), getPlane(out, n, ch),
                       Size(outWidth, outHeight), 0, 0, INTER_NEAREST);
            }
        }
    }

    virtual void forward(InputArrayOfArrays, OutputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE {}

private:
    int scale;
};

TEST(Torch_Importer, upsampling_nearest)
{
    CV_DNN_REGISTER_LAYER_CLASS(SpatialUpSamplingNearest, SpatialUpSamplingNearestLayer);
    runTorchNet("net_spatial_upsampling_nearest", DNN_TARGET_CPU, "", false, true);
    LayerFactory::unregisterLayer("SpatialUpSamplingNearest");
}

}
