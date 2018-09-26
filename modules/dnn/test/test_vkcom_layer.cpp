// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// This tests doesn't require any external data. They just compare outputs of
// layers using different computation backends. Input and parameters are random.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::dnn;
using namespace testing;

static void test(Mat& input, Net& net, Backend backendId, Target targetId, bool skipCheck = false)
{
    DNNTestLayer::checkBackend(backendId, targetId);
    randu(input, -1.0f, 1.0f);

    net.setInput(input);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    Mat outputDefault = net.forward().clone();

    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    Mat outputHalide = net.forward().clone();

    if (skipCheck)
        return;

    double l1, lInf;
    DNNTestLayer::getDefaultThresholds(backendId, targetId, &l1, &lInf);
    normAssert(outputDefault, outputHalide, "", l1, lInf);
}

static void test(LayerParams& params, Mat& input, Backend backendId, Target targetId, bool skipCheck = false)
{
    Net net;
    net.addLayerToPrev(params.name, params.type, params);
    test(input, net, backendId, targetId, skipCheck);
}

static inline testing::internal::ParamGenerator<tuple<Backend, Target> > dnnBackendsAndTargetsWithVkCom()
{
    return dnnBackendsAndTargets(false, false, false, true);
}

////////////////////////////////////////////////////////////////////////////////
// Convolution
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<Vec3i, Size, Size, Size, Size, Size, bool, tuple<Backend, Target> > > Convolution;
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
    Backend backendId = get<0>(get<7>(GetParam()));
    Target targetId = get<1>(get<7>(GetParam()));

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_RELEASE < 2018030000
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE && targetId == DNN_TARGET_MYRIAD)
        throw SkipTestException("Test is enabled starts from OpenVINO 2018R3");
#endif

    bool skipCheck = false;
    if (cvtest::skipUnstableTests && backendId == DNN_BACKEND_OPENCV &&
        (targetId == DNN_TARGET_OPENCL || targetId == DNN_TARGET_OPENCL_FP16) &&
        (
            (kernel == Size(3, 1) && stride == Size(1, 1) && pad == Size(0, 1)) ||
            (stride.area() > 1 && !(pad.width == 0 && pad.height == 0))
        )
    )
        skipCheck = true;

    int sz[] = {outChannels, inChannels / group, kernel.height, kernel.width};
    Mat weights(4, &sz[0], CV_32F);
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
        Mat bias(1, outChannels, CV_32F);
        randu(bias, -1.0f, 1.0f);
        lp.blobs.push_back(bias);
    }
    int inpSz[] = {1, inChannels, inSize.height, inSize.width};
    Mat input(4, &inpSz[0], CV_32F);
    test(lp, input, backendId, targetId, skipCheck);
    if (skipCheck)
        throw SkipTestException("Skip checks in unstable test");
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, Convolution, Combine(
/*in channels, out channels, group*/
             Values(Vec3i(6, 4, 1), Vec3i(6, 9, 1),
                    Vec3i(6, 4, 2), Vec3i(6, 9, 3)),
/*in size*/  Values(Size(5, 6)),
/*kernel*/   Values(Size(3, 1), Size(1, 3)),
/*stride*/   Values(Size(1, 1), Size(2, 2)),
/*pad*/      Values(Size(1, 0), Size(0, 1)),
/*dilation*/ Values(Size(1, 1), Size(2, 2)),
/*has bias*/ Bool(),
             dnnBackendsAndTargetsWithVkCom()
));

////////////////////////////////////////////////////////////////////////////////
// LRN
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<Vec3i, int, Vec3f, bool, std::string, tuple<Backend, Target> > > LRN;
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
    Backend backendId = get<0>(get<5>(GetParam()));
    Target targetId = get<1>(get<5>(GetParam()));
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
        throw SkipTestException("");

    LayerParams lp;
    lp.set("norm_region", nrmType);
    lp.set("local_size", localSize);
    lp.set("alpha", alpha);
    lp.set("beta", beta);
    lp.set("bias", bias);
    lp.set("norm_by_size", normBySize);
    lp.type = "LRN";
    lp.name = "testLayer";

    int sz[] = {1, inChannels, inSize.height, inSize.width};
    Mat input(4, &sz[0], CV_32F);
    test(lp, input, backendId, targetId);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, LRN, Combine(
/*input ch,w,h*/ Values(Vec3i(6, 5, 8), Vec3i(7, 11, 6)),
/*local size*/   Values(3, 5),
                 Values(Vec3f(0.9f, 1.0f, 1.1f), Vec3f(0.9f, 1.1f, 1.0f),
/*alpha, beta,*/        Vec3f(1.0f, 0.9f, 1.1f), Vec3f(1.0f, 1.1f, 0.9f),
/*bias */               Vec3f(1.1f, 0.9f, 1.0f), Vec3f(1.1f, 1.0f, 0.9f)),
/*norm_by_size*/ Bool(),
/*norm_type*/    Values("ACROSS_CHANNELS", "WITHIN_CHANNEL"),
                 dnnBackendsAndTargetsWithVkCom()
));

////////////////////////////////////////////////////////////////////////////////
// Average pooling
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int, Size, Size, Size, tuple<Backend, Target> > > AvePooling;
TEST_P(AvePooling, Accuracy)
{
    int inChannels = get<0>(GetParam());
    Size outSize = get<1>(GetParam());;  // Input size will be computed from parameters.
    Size kernel = get<2>(GetParam());
    Size stride = get<3>(GetParam());
    Backend backendId = get<0>(get<4>(GetParam()));
    Target targetId = get<1>(get<4>(GetParam()));
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE && targetId == DNN_TARGET_MYRIAD &&
        stride == Size(3, 2) && kernel == Size(3, 3) && outSize != Size(1, 1))
        throw SkipTestException("");

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

    int sz[] = {1, inChannels, inHeight, inWidth};
    Mat input(4, &sz[0], CV_32F);
    test(lp, input, backendId, targetId);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, AvePooling, Combine(
/*in channels*/ Values(3, 4),
/*out size*/    Values(Size(1, 1), Size(2, 2), Size(3, 2), Size(4, 7)),
/*kernel*/      Values(Size(1, 1), Size(2, 2), Size(3, 3), Size(3, 2)),
/*stride*/      Values(Size(1, 1), Size(2, 2), Size(3, 2)),
                dnnBackendsAndTargetsWithVkCom()
));

////////////////////////////////////////////////////////////////////////////////
// Maximum pooling
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int, Size, Size, Size, Size, tuple<Backend, Target> > > MaxPooling;
TEST_P(MaxPooling, Accuracy)
{
    int inChannels = get<0>(GetParam());
    Size inSize = get<1>(GetParam());
    Size kernel = get<2>(GetParam());
    Size stride = get<3>(GetParam());
    Size pad = get<4>(GetParam());
    Backend backendId = get<0>(get<5>(GetParam()));
    Target targetId = get<1>(get<5>(GetParam()));

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

    int sz[] = {1, inChannels, inSize.height, inSize.width};
    Mat input(4, &sz[0], CV_32F);
    test(lp, input, backendId, targetId);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, MaxPooling, Combine(
/*in channels*/ Values(3, 4),
/*in size*/     Values(Size(5, 5), Size(7, 6)),
/*kernel*/      Values(Size(2, 2), Size(3, 3), Size(3, 2)),
/*stride*/      Values(Size(1, 1), Size(2, 2), Size(3, 2)),
/*pad*/         Values(Size(0, 0), Size(1, 1), Size(0, 1)),
                dnnBackendsAndTargetsWithVkCom()
));

////////////////////////////////////////////////////////////////////////////////
// SoftMax
////////////////////////////////////////////////////////////////////////////////
typedef TestWithParam<tuple<int,  tuple<Backend, Target> > > SoftMax;
TEST_P(SoftMax, Accuracy)
{
    int inChannels = get<0>(GetParam());
    Backend backendId = get<0>(get<1>(GetParam()));
    Target targetId = get<1>(get<1>(GetParam()));
    LayerParams lp;
    lp.type = "SoftMax";
    lp.name = "testLayer";

    int sz[] = {1, inChannels, 1, 1};
    Mat input(4, &sz[0], CV_32F);
    test(lp, input, backendId, targetId);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, SoftMax, Combine(
    Values(3, 4, 5, 1024),
    dnnBackendsAndTargetsWithVkCom()
));

////////////////////////////////////////////////////////////////////////////////
// AvePooling + in-place layers
////////////////////////////////////////////////////////////////////////////////
static const int kNumChannels = 3;

void testInPlaceActivation(LayerParams& lp, Backend backendId, Target targetId)
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

    int sz[] = {1, kNumChannels, 10, 10};
    Mat input(4, &sz[0], CV_32F);
    test(input, net, backendId, targetId);
}

typedef TestWithParam<tuple<float, tuple<Backend, Target> > > ReLU;
TEST_P(ReLU, Accuracy)
{
    float negativeSlope = get<0>(GetParam());
    Backend backendId = get<0>(get<1>(GetParam()));
    Target targetId = get<1>(get<1>(GetParam()));

    LayerParams lp;
    lp.set("negative_slope", negativeSlope);
    lp.type = "ReLU";
    lp.name = "testLayer";
    testInPlaceActivation(lp, backendId, targetId);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, ReLU, Combine(
/*negative slope*/ Values(2.0f, 0.3f, -0.1f, 0.0f),
                   dnnBackendsAndTargetsWithVkCom()
));

////////////////////////////////////////////////////////////////////////////////
// Concat layer
////////////////////////////////////////////////////////////////////////////////
//
// input --- conv --- concat --- output
//      `--- conv ----^ ^ ^
//      `---- ... ------' '
//      `-----------------'
typedef TestWithParam<tuple<Vec3i, Vec3i, tuple<Backend, Target> > > Concat;
TEST_P(Concat, Accuracy)
{
    Vec3i inSize = get<0>(GetParam());
    Vec3i numChannels = get<1>(GetParam());
    Backend backendId = get<0>(get<2>(GetParam()));
    Target targetId = get<1>(get<2>(GetParam()));

    Net net;

    std::vector<int> convLayerIds;
    convLayerIds.reserve(numChannels.channels);
    for (int i = 0, n = numChannels.channels; i < n; ++i)
    {
        if (!numChannels[i])
            break;

        int sz[] = {numChannels[i], inSize[0], 1, 1};
        Mat weights(4, &sz[0], CV_32F);
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

    int sz[] = {1, inSize[0], inSize[1], inSize[2]};
    Mat input(4, &sz[0], CV_32F);
    test(input, net, backendId, targetId);
}

INSTANTIATE_TEST_CASE_P(Layer_Test_VkCom, Concat, Combine(
/*input size*/ Values(Vec3i(1, 4, 5), Vec3i(2, 8, 6)),
/*channels*/   Values(Vec3i(2, 0, 0), Vec3i(3, 4, 0), Vec3i(1, 6, 2)),
               dnnBackendsAndTargetsWithVkCom()
));

}} // namespace
