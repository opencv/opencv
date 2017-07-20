// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// This tests doesn't require any external data. They just compare outputs of
// layers using different computation backends. Input and parameters are random.

#include "test_precomp.hpp"

namespace cvtest
{

#ifdef HAVE_HALIDE
using namespace cv;
using namespace cv::dnn;
using namespace testing;

static void test(LayerParams& params, Mat& input)
{
    randu(input, -1.0f, 1.0f);

    Net net;
    int lid = net.addLayer(params.name, params.type, params);
    net.connect(0, 0, lid, 0);

    net.setInput(input);
    Mat outputDefault = net.forward(params.name).clone();

    net.setPreferableBackend(DNN_BACKEND_HALIDE);
    Mat outputHalide = net.forward(params.name).clone();
    normAssert(outputDefault, outputHalide);
}

////////////////////////////////////////////////////////////////////////////////
// Convolution
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<Vec3i, Size, Size, Size, Size, Size, bool> > Convolution;
TEST_P(Convolution, Accuracy)
{
    int inChannels = get<0>(GetParam())[0];
    int outChannels = get<0>(GetParam())[1];
    int group = get<0>(GetParam())[2];
    Size inSize = get<1>(GetParam());
    Size kernel = get<2>(GetParam());
    Size stride = get<3>(GetParam());
    Size pad = get<4>(GetParam());
    Size dilation = get<5>(GetParam());
    bool hasBias = get<6>(GetParam());

    Mat weights({outChannels, inChannels / group, kernel.height, kernel.width}, CV_32F);
    randu(weights, -1.0f, 1.0f);

    LayerParams lp;
    lp.set("kernel_w", kernel.width);
    lp.set("kernel_h", kernel.height);
    lp.set("pad_w", pad.width);
    lp.set("pad_h", pad.height);
    lp.set("stride_w", stride.width);
    lp.set("stride_h", stride.height);
    lp.set("dilation_w", dilation.width);
    lp.set("dilation_h", dilation.height);
    lp.set("num_output", outChannels);
    lp.set("group", group);
    lp.set("bias_term", hasBias);
    lp.type = "Convolution";
    lp.name = "testLayer";
    lp.blobs.push_back(weights);
    if (hasBias)
    {
        Mat bias({outChannels}, CV_32F);
        randu(bias, -1.0f, 1.0f);
        lp.blobs.push_back(bias);
    }
    Mat input({1, inChannels, inSize.height, inSize.width}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, Convolution, Combine(
/*in channels, out channels, group*/
             Values(Vec3i(6, 4, 1), Vec3i(6, 9, 1),
                    Vec3i(6, 4, 2), Vec3i(6, 9, 3)),
/*in size*/  Values(Size(5, 6)),
/*kernel*/   Values(Size(3, 1), Size(1, 3)),
/*stride*/   Values(Size(1, 1), Size(2, 2)),
/*pad*/      Values(Size(1, 0), Size(0, 1)),
/*dilation*/ Values(Size(1, 1), Size(2, 2)),
/*has bias*/ Bool()
));

////////////////////////////////////////////////////////////////////////////////
// Deconvolution
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<Vec3i, Size, Size, Size, Size, Vec4i, bool> > Deconvolution;
TEST_P(Deconvolution, Accuracy)
{
    int inChannels = get<0>(GetParam())[0];
    int outChannels = get<0>(GetParam())[1];
    int group = get<0>(GetParam())[2];
    Size inSize = get<1>(GetParam());
    Size kernel = get<2>(GetParam());
    Size pad = get<3>(GetParam());
    Size dilation = get<4>(GetParam());
    Size stride = Size(get<5>(GetParam())[0], get<5>(GetParam())[1]);
    Size adjPad = Size(get<5>(GetParam())[2], get<5>(GetParam())[3]);
    bool hasBias = get<6>(GetParam());

    Mat weights({outChannels, inChannels / group, kernel.height, kernel.width}, CV_32F);
    randu(weights, -1.0f, 1.0f);

    LayerParams lp;
    lp.set("kernel_w", kernel.width);
    lp.set("kernel_h", kernel.height);
    lp.set("pad_w", pad.width);
    lp.set("pad_h", pad.height);
    lp.set("stride_w", stride.width);
    lp.set("stride_h", stride.height);
    lp.set("dilation_w", dilation.width);
    lp.set("dilation_h", dilation.height);
    lp.set("adj_w", adjPad.width);
    lp.set("adj_h", adjPad.height);
    lp.set("num_output", outChannels);
    lp.set("group", group);
    lp.set("bias_term", hasBias);
    lp.type = "Deconvolution";
    lp.name = "testLayer";
    lp.blobs.push_back(weights);
    if (hasBias)
    {
        Mat bias({outChannels}, CV_32F);
        randu(bias, -1.0f, 1.0f);
        lp.blobs.push_back(bias);
    }
    Mat input({1, inChannels, inSize.height, inSize.width}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, Deconvolution, Combine(
/*in channels, out channels, group*/
             Values(Vec3i(6, 4, 1), Vec3i(6, 9, 1)),
/*in size*/  Values(Size(5, 6)),
/*kernel*/   Values(Size(3, 1), Size(1, 3)),
/*pad*/      Values(Size(1, 0), Size(0, 1)),
/*dilation*/ Values(Size(1, 1), Size(2, 2)),
/*stride, adj. pad*/ Values(Vec4i(1,1, 0,0), Vec4i(2,2, 1,0), Vec4i(1,2, 0,1)),
/*has bias*/ Bool()
));

////////////////////////////////////////////////////////////////////////////////
// LRN
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<Vec3i, int, Vec3f, bool, std::string> > LRN;
TEST_P(LRN, Accuracy)
{
    int inChannels = get<0>(GetParam())[0];
    Size inSize = Size(get<0>(GetParam())[1], get<0>(GetParam())[2]);
    int localSize = get<1>(GetParam());
    float alpha = get<2>(GetParam())[0];
    float beta = get<2>(GetParam())[1];
    float bias = get<2>(GetParam())[2];
    bool normBySize = get<3>(GetParam());
    std::string nrmType = get<4>(GetParam());

    LayerParams lp;
    lp.set("norm_region", nrmType);
    lp.set("local_size", localSize);
    lp.set("alpha", alpha);
    lp.set("beta", beta);
    lp.set("bias", bias);
    lp.set("norm_by_size", normBySize);
    lp.type = "LRN";
    lp.name = "testLayer";

    Mat input({1, inChannels, inSize.height, inSize.width}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, LRN, Combine(
/*input ch,w,h*/ Values(Vec3i(6, 5, 8), Vec3i(7, 11, 6)),
/*local size*/   Values(3, 5),
                 Values(Vec3f(0.9f, 1.0f, 1.1f), Vec3f(0.9f, 1.1f, 1.0f),
/*alpha, beta,*/        Vec3f(1.0f, 0.9f, 1.1f), Vec3f(1.0f, 1.1f, 0.9f),
/*bias */               Vec3f(1.1f, 0.9f, 1.0f), Vec3f(1.1f, 1.0f, 0.9f)),
/*norm_by_size*/ Bool(),
/*norm_type*/    Values("ACROSS_CHANNELS", "WITHIN_CHANNEL")
));

////////////////////////////////////////////////////////////////////////////////
// Average pooling
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int, Size, Size, Size> > AvePooling;
TEST_P(AvePooling, Accuracy)
{
    int inChannels = get<0>(GetParam());
    Size outSize = get<1>(GetParam());;  // Input size will be computed from parameters.
    Size kernel = get<2>(GetParam());
    Size stride = get<3>(GetParam());

    const int inWidth = (outSize.width - 1) * stride.width + kernel.width;
    const int inHeight = (outSize.height - 1) * stride.height + kernel.height;

    LayerParams lp;
    lp.set("pool", "ave");
    lp.set("kernel_w", kernel.width);
    lp.set("kernel_h", kernel.height);
    lp.set("stride_w", stride.width);
    lp.set("stride_h", stride.height);
    lp.type = "Pooling";
    lp.name = "testLayer";

    Mat input({1, inChannels, inHeight, inWidth}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, AvePooling, Combine(
/*in channels*/ Values(3, 4),
/*out size*/    Values(Size(1, 1), Size(2, 2), Size(3, 2), Size(4, 7)),
/*kernel*/      Values(Size(1, 1), Size(2, 2), Size(3, 3), Size(3, 2)),
/*stride*/      Values(Size(1, 1), Size(2, 2), Size(3, 2))
));

////////////////////////////////////////////////////////////////////////////////
// Maximum pooling
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int, Size, Size, Size, Size> > MaxPooling;
TEST_P(MaxPooling, Accuracy)
{
    int inChannels = get<0>(GetParam());
    Size inSize = get<1>(GetParam());
    Size kernel = get<2>(GetParam());
    Size stride = get<3>(GetParam());
    Size pad = get<4>(GetParam());

    LayerParams lp;
    lp.set("pool", "max");
    lp.set("kernel_w", kernel.width);
    lp.set("kernel_h", kernel.height);
    lp.set("stride_w", stride.width);
    lp.set("stride_h", stride.height);
    lp.set("pad_w", pad.width);
    lp.set("pad_h", pad.height);
    lp.type = "Pooling";
    lp.name = "testLayer";

    Mat input({1, inChannels, inSize.height, inSize.width}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, MaxPooling, Combine(
/*in channels*/ Values(3, 4),
/*in size*/     Values(Size(5, 5), Size(7, 6)),
/*kernel*/      Values(Size(2, 2), Size(3, 3), Size(3, 2)),
/*stride*/      Values(Size(1, 1), Size(2, 2), Size(3, 2)),
/*pad*/         Values(Size(0, 0), Size(1, 1), Size(0, 1))
));

////////////////////////////////////////////////////////////////////////////////
// Fully-connected
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int, Size, int, bool> > FullyConnected;
TEST_P(FullyConnected, Accuracy)
{
    int inChannels = get<0>(GetParam());
    Size inSize = get<1>(GetParam());
    int outChannels = get<2>(GetParam());
    bool hasBias = get<3>(GetParam());

    Mat weights(outChannels, inChannels * inSize.height * inSize.width, CV_32F);
    randu(weights, -1.0f, 1.0f);

    Mat bias(1, outChannels, CV_32F);
    randu(bias, -1.0f, 1.0f);

    LayerParams lp;
    lp.set("num_output", outChannels);
    lp.set("bias_term", hasBias);
    lp.blobs.push_back(weights);
    lp.blobs.push_back(bias);
    lp.type = "InnerProduct";
    lp.name = "testLayer";

    Mat input({1, inChannels, inSize.height, inSize.width}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, FullyConnected, Combine(
/*in channels*/  Values(3, 4),
/*in size*/      Values(Size(5, 4), Size(4, 5), Size(1, 1)),
/*out channels*/ Values(3, 4),
/*has bias*/     Bool()
));

////////////////////////////////////////////////////////////////////////////////
// SoftMax
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int> > SoftMax;
TEST_P(SoftMax, Accuracy)
{
    int inChannels = get<0>(GetParam());
    LayerParams lp;
    lp.type = "SoftMax";
    lp.name = "testLayer";

    Mat input({1, inChannels, 1, 1}, CV_32F);
    test(lp, input);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, SoftMax, Values(3, 4, 5, 1024));

//////////////////////////////////////////////////////////////////////////////
// Max pooling - unpooling
//////////////////////////////////////////////////////////////////////////////
TEST(MaxPoolUnpool_Halide, Accuracy)
{
    LayerParams pool;
    pool.set("pool", "max");
    pool.set("kernel_w", 2);
    pool.set("kernel_h", 2);
    pool.set("stride_w", 2);
    pool.set("stride_h", 2);
    pool.set("pad_w", 0);
    pool.set("pad_h", 0);
    pool.type = "Pooling";
    pool.name = "testPool";

    LayerParams unpool;
    unpool.set("pool_k_w", 2);
    unpool.set("pool_k_h", 2);
    unpool.set("pool_stride_w", 2);
    unpool.set("pool_stride_h", 2);
    unpool.set("pool_pad_w", 0);
    unpool.set("pool_pad_h", 0);
    unpool.type = "MaxUnpool";
    unpool.name = "testUnpool";

    Net net;
    int poolId = net.addLayer(pool.name, pool.type, pool);
    net.connect(0, 0, poolId, 0);

    int unpoolId = net.addLayer(unpool.name, unpool.type, unpool);
    net.connect(poolId, 0, unpoolId, 0);
    net.connect(poolId, 1, unpoolId, 1);

    Mat input({1, 1, 4, 4}, CV_32F);
    randu(input, -1.0f, 1.0f);
    net.setInput(input);
    Mat outputDefault = net.forward("testUnpool").clone();

    net.setPreferableBackend(DNN_BACKEND_HALIDE);
    net.setInput(input);
    Mat outputHalide = net.forward("testUnpool").clone();
    normAssert(outputDefault, outputHalide);
}

////////////////////////////////////////////////////////////////////////////////
// AvePooling + in-place layers
////////////////////////////////////////////////////////////////////////////////
static const int kNumChannels = 3;

void testInPlaceActivation(LayerParams& lp)
{
    EXPECT_FALSE(lp.name.empty());

    LayerParams pool;
    pool.set("pool", "ave");
    pool.set("kernel_w", 2);
    pool.set("kernel_h", 2);
    pool.set("stride_w", 2);
    pool.set("stride_h", 2);
    pool.type = "Pooling";

    Net net;
    int poolId = net.addLayer(pool.name, pool.type, pool);
    net.connect(0, 0, poolId, 0);
    net.addLayerToPrev(lp.name, lp.type, lp);

    Mat input({1, kNumChannels, 10, 10}, CV_32F);
    randu(input, -1.0f, 1.0f);
    net.setInput(input);
    Mat outputDefault = net.forward(lp.name).clone();

    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_HALIDE);
    Mat outputHalide = net.forward(lp.name).clone();
    normAssert(outputDefault, outputHalide);
}

typedef TestWithParam<tuple<bool, bool, float> > BatchNorm;
TEST_P(BatchNorm, Accuracy)
{
    bool hasWeights = get<0>(GetParam());
    bool hasBias = get<1>(GetParam());
    float epsilon = get<2>(GetParam());

    LayerParams lp;
    lp.set("has_weight", hasWeights);
    lp.set("has_bias", hasBias);
    lp.set("eps", epsilon);
    lp.type = "BatchNorm";
    lp.name = "testLayer";

    lp.blobs.reserve(4);
    for (int i = 0; i < 3; ++i)
        lp.blobs.push_back(Mat({kNumChannels}, CV_32F));
    if (hasBias || hasWeights)
        lp.blobs.push_back(Mat({kNumChannels}, CV_32F));

    for (Mat& m : lp.blobs)
        randu(m, 0.0f, 1.0f);

    testInPlaceActivation(lp);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, BatchNorm, Combine(
/*has weights*/ Bool(),
/*has bias*/    Bool(),
/*epsilon*/     Values(1e-3f, 1e-5f)
));

typedef TestWithParam<tuple<float> > ReLU;
TEST_P(ReLU, Accuracy)
{
    float negativeSlope = get<0>(GetParam());

    LayerParams lp;
    lp.set("negative_slope", negativeSlope);
    lp.type = "ReLU";
    lp.name = "testLayer";
    testInPlaceActivation(lp);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, ReLU, Values(
/*negative slope*/ 2.0f, 0.3f, -0.1f, 0.0f
));

typedef TestWithParam<tuple<std::string> > NoParamActivation;
TEST_P(NoParamActivation, Accuracy)
{
    LayerParams lp;
    lp.type = get<0>(GetParam());
    lp.name = "testLayer";
    testInPlaceActivation(lp);
}
INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, NoParamActivation, Values(
/*type*/ "TanH", "Sigmoid", "AbsVal", "BNLL"
));

typedef TestWithParam<tuple<Vec3f> > Power;
TEST_P(Power, Accuracy)
{
    float power = get<0>(GetParam())[0];
    float scale = get<0>(GetParam())[1];
    float shift = get<0>(GetParam())[2];

    LayerParams lp;
    lp.set("power", power);
    lp.set("scale", scale);
    lp.set("shift", shift);
    lp.type = "Power";
    lp.name = "testLayer";
    testInPlaceActivation(lp);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, Power,
/*power, scale, shift*/ Values(Vec3f(0.9f, 1.0f, 1.1f), Vec3f(0.9f, 1.1f, 1.0f),
                               Vec3f(1.0f, 0.9f, 1.1f), Vec3f(1.0f, 1.1f, 0.9f),
                               Vec3f(1.1f, 0.9f, 1.0f), Vec3f(1.1f, 1.0f, 0.9f))
);

TEST(ChannelsPReLU, Accuracy)
{
    LayerParams lp;
    lp.type = "ChannelsPReLU";
    lp.name = "testLayer";
    lp.blobs.push_back(Mat({kNumChannels}, CV_32F));
    randu(lp.blobs[0], -1.0f, 1.0f);

    testInPlaceActivation(lp);
}

typedef TestWithParam<tuple<bool> > Scale;
TEST_P(Scale, Accuracy)
{
    bool hasBias = get<0>(GetParam());

    LayerParams lp;
    lp.set("bias_term", hasBias);
    lp.type = "Scale";
    lp.name = "testLayer";
    lp.blobs.push_back(Mat({kNumChannels}, CV_32F));
    randu(lp.blobs[0], -1.0f, 1.0f);
    if (hasBias)
    {
        lp.blobs.push_back(Mat({kNumChannels}, CV_32F));
        randu(lp.blobs[1], -1.0f, 1.0f);
    }
    testInPlaceActivation(lp);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, Scale, Values(true, false));

////////////////////////////////////////////////////////////////////////////////
// Concat layer
////////////////////////////////////////////////////////////////////////////////
//
// input --- conv --- concat --- output
//      `--- conv ----^ ^ ^
//      `---- ... ------' '
//      `-----------------'
typedef TestWithParam<tuple<Vec3i, Vec3i> > Concat;
TEST_P(Concat, Accuracy)
{
    Vec3i inSize = get<0>(GetParam());
    Vec3i numChannels = get<1>(GetParam());

    Net net;

    std::vector<int> convLayerIds;
    convLayerIds.reserve(numChannels.channels);
    for (int i = 0, n = numChannels.channels; i < n; ++i)
    {
        if (!numChannels[i])
            break;

        Mat weights({numChannels[i], inSize[0], 1, 1}, CV_32F);
        randu(weights, -1.0f, 1.0f);

        LayerParams convParam;
        convParam.set("kernel_w", 1);
        convParam.set("kernel_h", 1);
        convParam.set("num_output", numChannels[i]);
        convParam.set("bias_term", false);
        convParam.type = "Convolution";
        std::ostringstream ss;
        ss << "convLayer" << i;
        convParam.name = ss.str();
        convParam.blobs.push_back(weights);

        int layerId = net.addLayer(convParam.name, convParam.type, convParam);
        convLayerIds.push_back(layerId);
        net.connect(0, 0, layerId, 0);
    }

    LayerParams concatParam;
    concatParam.type = "Concat";
    concatParam.name = "testLayer";
    int concatId = net.addLayer(concatParam.name, concatParam.type, concatParam);
    net.connect(0, 0, concatId, 0);
    for (int i = 0; i < convLayerIds.size(); ++i)
    {
        net.connect(convLayerIds[i], 0, concatId, i + 1);
    }

    Mat input({1, inSize[0], inSize[1], inSize[2]}, CV_32F);
    randu(input, -1.0f, 1.0f);

    net.setInput(input);
    Mat outputDefault = net.forward(concatParam.name).clone();

    net.setPreferableBackend(DNN_BACKEND_HALIDE);
    Mat outputHalide = net.forward(concatParam.name).clone();
    normAssert(outputDefault, outputHalide);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, Concat, Combine(
/*input size*/ Values(Vec3i(1, 4, 5), Vec3i(2, 8, 6)),
/*channels*/   Values(Vec3i(2, 0, 0), Vec3i(3, 4, 0), Vec3i(1, 6, 2))
));

////////////////////////////////////////////////////////////////////////////////
// Element-wise layers
////////////////////////////////////////////////////////////////////////////////
//
// input --- conv --- eltwise --- output
//      `--- conv ----^ ^ ^
//      `---- ... ------' '
//      `-----------------'
typedef TestWithParam<tuple<Vec3i, std::string, int, bool> > Eltwise;
TEST_P(Eltwise, Accuracy)
{
    Vec3i inSize = get<0>(GetParam());
    std::string op = get<1>(GetParam());
    int numConv = get<2>(GetParam());
    bool weighted = get<3>(GetParam());

    Net net;

    std::vector<int> convLayerIds(numConv);
    for (int i = 0; i < numConv; ++i)
    {
        Mat weights({inSize[0], inSize[0], 1, 1}, CV_32F);
        randu(weights, -1.0f, 1.0f);

        LayerParams convParam;
        convParam.set("kernel_w", 1);
        convParam.set("kernel_h", 1);
        convParam.set("num_output", inSize[0]);
        convParam.set("bias_term", false);
        convParam.type = "Convolution";
        std::ostringstream ss;
        ss << "convLayer" << i;
        convParam.name = ss.str();
        convParam.blobs.push_back(weights);

        convLayerIds[i] = net.addLayer(convParam.name, convParam.type, convParam);
        net.connect(0, 0, convLayerIds[i], 0);
    }

    LayerParams eltwiseParam;
    eltwiseParam.set("operation", op);
    if (op == "sum" && weighted)
    {
        RNG rng = cv::theRNG();
        std::vector<float> coeff(1 + numConv);
        for (int i = 0; i < coeff.size(); ++i)
        {
            coeff[i] = rng.uniform(-2.0f, 2.0f);
        }
        eltwiseParam.set("coeff", DictValue::arrayReal<float*>(&coeff[0], coeff.size()));
    }
    eltwiseParam.type = "Eltwise";
    eltwiseParam.name = "testLayer";
    int eltwiseId = net.addLayer(eltwiseParam.name, eltwiseParam.type, eltwiseParam);
    net.connect(0, 0, eltwiseId, 0);
    for (int i = 0; i < numConv; ++i)
    {
        net.connect(convLayerIds[i], 0, eltwiseId, i + 1);
    }

    Mat input({1, inSize[0], inSize[1], inSize[2]}, CV_32F);
    randu(input, -1.0f, 1.0f);

    net.setInput(input);
    Mat outputDefault = net.forward(eltwiseParam.name).clone();

    net.setPreferableBackend(DNN_BACKEND_HALIDE);
    Mat outputHalide = net.forward(eltwiseParam.name).clone();
    normAssert(outputDefault, outputHalide);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_Halide, Eltwise, Combine(
/*input size*/ Values(Vec3i(1, 4, 5), Vec3i(2, 8, 6)),
/*operation*/  Values("prod", "sum", "max"),
/*num convs*/  Values(1, 2, 3),
/*weighted(for sum only)*/ Bool()
));
#endif  // HAVE_HALIDE

}  // namespace cvtest
