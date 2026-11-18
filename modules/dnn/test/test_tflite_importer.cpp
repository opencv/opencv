// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
Test for TFLite models loading
*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS
#include <opencv2/dnn/utils/debug_utils.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#ifdef OPENCV_TEST_DNN_TFLITE

#include "schema_generated.h"

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::dnn;

class Test_TFLite : public DNNTestLayer {
public:
    void testModel(Net& net, const std::string& modelName, const Mat& input, double l1 = 0, double lInf = 0);
    void testModel(const std::string& modelName, const Mat& input, double l1 = 0, double lInf = 0);
    void testModel(const std::string& modelName, const Size& inpSize, double l1 = 0, double lInf = 0);
    void testLayer(const std::string& modelName, double l1 = 0, double lInf = 0);
};

void testInputShapes(const Net& net, const std::vector<Mat>& inps) {
    std::vector<MatShape> inLayerShapes;
    std::vector<MatShape> outLayerShapes;
    net.getLayerShapes(MatShape(), 0, inLayerShapes, outLayerShapes);
    ASSERT_EQ(inLayerShapes.size(), inps.size());

    for (int i = 0; i < inps.size(); ++i) {
        ASSERT_EQ(inLayerShapes[i], shape(inps[i]));
    }
}

void Test_TFLite::testModel(Net& net, const std::string& modelName, const Mat& input, double l1, double lInf)
{
    l1 = l1 ? l1 : default_l1;
    lInf = lInf ? lInf : default_lInf;

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    testInputShapes(net, {input});
    net.setInput(input);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    std::vector<Mat> outs;
    net.forward(outs, outNames);

    ASSERT_EQ(outs.size(), outNames.size());
    for (int i = 0; i < outNames.size(); ++i) {
        std::replace(outNames[i].begin(), outNames[i].end(), ':', '_');
        Mat ref = blobFromNPY(findDataFile(format("dnn/tflite/%s_out_%s.npy", modelName.c_str(), outNames[i].c_str())));
        // A workaround solution for the following cases due to inconsistent shape definitions.
        // The details please see: https://github.com/opencv/opencv/pull/25297#issuecomment-2039081369
        if (modelName == "face_landmark" || modelName == "selfie_segmentation") {
            ref = ref.reshape(1, 1);
            outs[i] = outs[i].reshape(1, 1);
        }
        normAssert(ref, outs[i], outNames[i].c_str(), l1, lInf);
    }
}

void Test_TFLite::testModel(const std::string& modelName, const Mat& input, double l1, double lInf)
{
    Net net = readNet(findDataFile("dnn/tflite/" + modelName + ".tflite", false));
    testModel(net, modelName, input, l1, lInf);
}

void Test_TFLite::testModel(const std::string& modelName, const Size& inpSize, double l1, double lInf)
{
    Mat input = imread(findDataFile("cv/shared/lena.png"));
    input = blobFromImage(input, 1.0 / 255, inpSize, 0, true);
    testModel(modelName, input, l1, lInf);
}

void Test_TFLite::testLayer(const std::string& modelName, double l1, double lInf)
{
    Mat inp = blobFromNPY(findDataFile("dnn/tflite/" + modelName + "_inp.npy"));
    Net net = readNet(findDataFile("dnn/tflite/" + modelName + ".tflite"));
    testModel(net, modelName, inp, l1, lInf);
}

// https://google.github.io/mediapipe/solutions/face_mesh
TEST_P(Test_TFLite, face_landmark)
{
    if (backend == DNN_BACKEND_CUDA && target == DNN_TARGET_CUDA_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
    double l1 = 0.066, lInf = 0.21;
    if (target == DNN_TARGET_CPU_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD ||
        (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL))
    {
        l1 = 0.15;
        lInf = 0.82;
    }
    testModel("face_landmark", Size(192, 192), l1, lInf);
}

// https://google.github.io/mediapipe/solutions/face_detection
TEST_P(Test_TFLite, face_detection_short_range)
{
    double l1 = 0, lInf = 2e-4;
    if (target == DNN_TARGET_CPU_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD ||
        (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL))
    {
        l1 = 0.04;
        lInf = 0.8;
    }
    testModel("face_detection_short_range", Size(128, 128), l1, lInf);
}

// https://google.github.io/mediapipe/solutions/selfie_segmentation
TEST_P(Test_TFLite, selfie_segmentation)
{
    double l1 = 0.002, lInf = 0.24;
    if (target == DNN_TARGET_CPU_FP16 || target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD ||
        (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL))
    {
        l1 = 0.01;
        lInf = 0.48;
    }
    testModel("selfie_segmentation", Size(256, 256), l1, lInf);
}

TEST_P(Test_TFLite, max_unpooling)
{
    if (backend == DNN_BACKEND_CUDA)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA);

#if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2022010000)
        if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
#endif

    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target != DNN_TARGET_CPU) {
        if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_OPENCL)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        if (target == DNN_TARGET_MYRIAD)      applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
    }

    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);

    // Due Max Unpoling is a numerically unstable operation and small difference between frameworks
    // might lead to positional difference of maximal elements in the tensor, this test checks
    // behavior of Max Unpooling layer only.
    Net net = readNet(findDataFile("dnn/tflite/hair_segmentation.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input = imread(findDataFile("cv/shared/lena.png"));
    cvtColor(input, input, COLOR_BGR2RGBA);
    input = input.mul(Scalar(1, 1, 1, 0));
    input = blobFromImage(input, 1.0 / 255);
    testInputShapes(net, {input});
    net.setInput(input);

    std::vector<std::vector<Mat> > outs;
    net.forward(outs, {"p_re_lu_1", "max_pooling_with_argmax2d", "conv2d_86", "max_unpooling2d_2"});

    ASSERT_EQ(outs.size(), 4);
    ASSERT_EQ(outs[0].size(), 1);
    ASSERT_EQ(outs[1].size(), 2);
    ASSERT_EQ(outs[2].size(), 1);
    ASSERT_EQ(outs[3].size(), 1);
    Mat poolInp = outs[0][0];
    Mat poolOut = outs[1][0];
    Mat poolIds = outs[1][1];
    Mat unpoolInp = outs[2][0];
    Mat unpoolOut = outs[3][0];

    ASSERT_EQ(poolInp.size, unpoolOut.size);
    ASSERT_EQ(poolOut.size, poolIds.size);
    ASSERT_EQ(poolOut.size, unpoolInp.size);

    ASSERT_EQ(countNonZero(poolInp), poolInp.total());

    for (int c = 0; c < 32; ++c) {
        float *poolInpData = poolInp.ptr<float>(0, c);
        float *poolOutData = poolOut.ptr<float>(0, c);
        float *poolIdsData = poolIds.ptr<float>(0, c);
        float *unpoolInpData = unpoolInp.ptr<float>(0, c);
        float *unpoolOutData = unpoolOut.ptr<float>(0, c);
        for (int y = 0; y < 64; ++y) {
            for (int x = 0; x < 64; ++x) {
                int maxIdx = (y * 128 + x) * 2;
                std::vector<int> indices{maxIdx + 1, maxIdx + 128, maxIdx + 129};
                std::string errMsg = format("Channel %d, y: %d, x: %d", c, y, x);
                for (int idx : indices) {
                    if (poolInpData[idx] > poolInpData[maxIdx]) {
                        EXPECT_EQ(unpoolOutData[maxIdx], 0.0f) << errMsg;
                        maxIdx = idx;
                    }
                }
                EXPECT_EQ(poolInpData[maxIdx], poolOutData[y * 64 + x]) << errMsg;
                if (backend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
                    EXPECT_EQ(poolIdsData[y * 64 + x], (float)maxIdx) << errMsg;
                }
                EXPECT_EQ(unpoolOutData[maxIdx], unpoolInpData[y * 64 + x]) << errMsg;
            }
        }
    }
}

TEST_P(Test_TFLite, EfficientDet_int8) {
    if (target != DNN_TARGET_CPU || (backend != DNN_BACKEND_OPENCV &&
        backend != DNN_BACKEND_TIMVX && backend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)) {
        throw SkipTestException("Only OpenCV, TimVX and OpenVINO targets support INT8 on CPU");
    }
    Net net = readNet(findDataFile("dnn/tflite/coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat img = imread(findDataFile("dnn/dog416.png"));
    Mat blob = blobFromImage(img, 1.0, Size(320, 320));

    net.setInput(blob);
    Mat out = net.forward();
    Mat_<float> ref({3, 7}, {
        0, 7, 0.62890625, 0.6014542579650879, 0.13300055265426636, 0.8977657556533813, 0.292389452457428,
        0, 17, 0.56640625, 0.15983937680721283, 0.35905322432518005, 0.5155506730079651, 0.9409466981887817,
        0, 1, 0.5, 0.14357104897499084, 0.2240825891494751, 0.7183101177215576, 0.9140362739562988
    });
    normAssertDetections(ref, out, "", 0.5, 0.05, 0.1);
}

TEST_P(Test_TFLite, replicate_by_pack) {
    double l1 = 0, lInf = 0;
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL)
    {
        l1 = 4e-4;
        lInf = 2e-3;
    }
    testLayer("replicate_by_pack", l1, lInf);
}

TEST_P(Test_TFLite, split) {
    testLayer("split");
}

TEST_P(Test_TFLite, fully_connected) {
    if (backend == DNN_BACKEND_VKCOM)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN);
    testLayer("fully_connected");
}

TEST_P(Test_TFLite, permute) {
    testLayer("permutation_3d");
    // Temporarily disabled as TFLiteConverter produces a incorrect graph in this case
    //testLayer("permutation_4d_0123");
    testLayer("permutation_4d_0132");
    testLayer("permutation_4d_0213");
    testLayer("permutation_4d_0231");
}

TEST_P(Test_TFLite, global_average_pooling_2d) {
    testLayer("global_average_pooling_2d");
}

TEST_P(Test_TFLite, global_max_pooling_2d) {
    testLayer("global_max_pooling_2d");
}

TEST_P(Test_TFLite, leakyRelu) {
    testLayer("leakyRelu");
}

TEST_P(Test_TFLite, StridedSlice) {
    testLayer("strided_slice");
}

TEST_P(Test_TFLite, face_blendshapes)
{
    Mat inp = blobFromNPY(findDataFile("dnn/tflite/face_blendshapes_inp.npy"));
    testModel("face_blendshapes", inp);
}

TEST_P(Test_TFLite, maximum)
{
    Net net = readNetFromTFLite(findDataFile("dnn/tflite/maximum.tflite"));

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input_x = blobFromNPY(findDataFile("dnn/tflite/maximum_input_x.npy"));
    Mat input_y = blobFromNPY(findDataFile("dnn/tflite/maximum_input_y.npy"));

    net.setInput(input_x, "x");
    net.setInput(input_y, "y");

    Mat out = net.forward();
    Mat ref = blobFromNPY(findDataFile("dnn/tflite/maximum_output.npy"));

    double l1 = 1e-5;
    double lInf = 1e-4;

    if (target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 1e-3;
        lInf = 1e-3;
    }

    normAssert(ref, out, "", l1, lInf);
}

TEST_P(Test_TFLite, minimum)
{
    Net net = readNetFromTFLite(findDataFile("dnn/tflite/minimum.tflite"));

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input_x = blobFromNPY(findDataFile("dnn/tflite/minimum_input_x.npy"));
    Mat input_y = blobFromNPY(findDataFile("dnn/tflite/minimum_input_y.npy"));

    net.setInput(input_x, "x");
    net.setInput(input_y, "y");

    Mat out = net.forward();
    Mat ref = blobFromNPY(findDataFile("dnn/tflite/minimum_output.npy"));

    double l1 = 1e-5;
    double lInf = 1e-4;

    if (target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_OPENCL_FP16)
    {
        l1 = 1e-3;
        lInf = 1e-3;
    }

    normAssert(ref, out, "", l1, lInf);
}

static std::string tfliteBoundsModel(const char* name)
{
    return findDataFile(std::string("dnn/tflite/") + name + ".tflite");
}

static void expectTFLiteRejected(const char* model, const char* reason)
{
    try
    {
        (void)readNetFromTFLite(tfliteBoundsModel(model));
        FAIL() << "Malformed TFLite tensor was accepted";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_NE(std::string(e.what()).find(reason), std::string::npos);
    }
}

TEST(Test_TFLite_Importer, ControlPadding)
{
    EXPECT_NO_THROW(readNetFromTFLite(tfliteBoundsModel("tfliteControl")));
}

TEST(Test_TFLite_Importer, ShortInt32Buffer)
{
    expectTFLiteRejected("tfliteShortBuffer", "tensor requires");
}

TEST(Test_TFLite_Importer, ShortInt32ElementWidth)
{
    expectTFLiteRejected("tfliteShortElementWidth", "tensor requires 32 bytes but its buffer has 16 bytes");
}

TEST(Test_TFLite_Importer, NegativeShape)
{
    expectTFLiteRejected("tfliteNegativeShape", "negative dimension");
}

TEST(Test_TFLite_Importer, OverflowShape)
{
    expectTFLiteRejected("tfliteOverflowShape", "overflows size_t");
}

TEST(Test_TFLite_Importer, BufferIndexEqualCount)
{
    expectTFLiteRejected("tfliteInvalidBufferIndex", "tensor buffer index is out of range");
}

TEST(Test_TFLite_Importer, BufferIndexUintMax)
{
    expectTFLiteRejected("tfliteHugeBufferIndex", "tensor buffer index is out of range");
}

TEST(Test_TFLite_Importer, Int8Control)
{
    EXPECT_NO_THROW(readNetFromTFLite(tfliteBoundsModel("tfliteInt8Control")));
}

TEST(Test_TFLite_Importer, ShortInt8Buffer)
{
    expectTFLiteRejected("tfliteShortInt8Buffer", "tensor requires 8 bytes but its buffer has 4 bytes");
}

TEST(Test_TFLite_Importer, ScalarControl)
{
    EXPECT_NO_THROW(readNetFromTFLite(tfliteBoundsModel("tfliteScalarControl")));
}

TEST(Test_TFLite_Importer, ZeroElementControl)
{
    EXPECT_NO_THROW(readNetFromTFLite(tfliteBoundsModel("tfliteZeroElementControl")));
}

TEST(Test_TFLite_Importer, EmptyDataControl)
{
    EXPECT_NO_THROW(readNetFromTFLite(tfliteBoundsModel("tfliteEmptyDataControl")));
}

TEST(Test_TFLite_Importer, ZeroFirstOverflow)
{
    expectTFLiteRejected("tfliteZeroFirstOverflow", "overflows size_t");
}

TEST(Test_TFLite_Importer, ZeroLastOverflow)
{
    expectTFLiteRejected("tfliteZeroLastOverflow", "overflows size_t");
}

TEST(Test_TFLite_Importer, SparseTensor)
{
    expectTFLiteRejected("tfliteSparseTensor", "sparse tensors are not supported");
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TFLite, dnnBackendsAndTargets());

// The models below are assembled in memory with the generated TFLite schema API, so these tests
// need no external model files.
namespace tflite_fixture {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using namespace opencv_tflite;

static std::vector<uint8_t> int32Bytes(const std::vector<int32_t>& values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(int32_t));
    if (!bytes.empty())
        std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

static std::vector<uint8_t> firstBytes(const std::vector<uint8_t>& bytes, size_t count)
{
    return std::vector<uint8_t>(bytes.begin(), bytes.begin() + count);
}

// TfLitePoolParams fields up to and including `activation`, as MediaPipe serializes them.
static std::vector<uint8_t> poolParams(int strideW = 2, int strideH = 2, int filterW = 2, int filterH = 2)
{
    return int32Bytes({1 /*padding SAME*/, strideW, strideH, filterW, filterH, 0 /*activation*/});
}

static std::vector<uint8_t> transposeConvParams(int strideW = 2, int strideH = 2)
{
    return int32Bytes({1 /*padding SAME*/, strideW, strideH});
}

struct Case
{
    // operator code
    int8_t deprecatedCode = BuiltinOperator_MAX_POOL_2D;
    const char* customCode = nullptr;
    // operator
    uint32_t opcodeIndex = 0;
    std::vector<int32_t> inputs{0};
    std::vector<int32_t> outputs{1};
    int poolOptions = 1;  // 0: none, 1: Pool2DOptions(2,2,2,2), 2: Conv2DOptions in the Pool2DOptions slot,
                          // 3: FullyConnectedOptions
    int strideW = 2, strideH = 2, filterW = 2, filterH = 2;
    bool hasCustomOptions = false;
    std::vector<uint8_t> customOptions;
    // tensors; every tensor but an optional unnamed one gets a name
    std::vector<std::vector<int32_t> > shapes{{1, 4, 4, 1}, {1, 2, 2, 1}};
    int unnamedTensor = -1;
    std::vector<std::vector<uint8_t> > buffers{std::vector<uint8_t>()};
    std::vector<uint32_t> tensorBuffers{0, 0};
    std::vector<int32_t> graphInputs{0}, graphOutputs{1};
    // optional second operator, a custom operator with operator code 1
    const char* secondCustomCode = nullptr;
    std::vector<uint8_t> secondCustomOptions;
    std::vector<int32_t> secondInputs, secondOutputs;
};

static Case maxPool()
{
    return Case();
}

static Case customOp(const char* code, std::vector<uint8_t> options, bool present = true)
{
    Case c;
    c.deprecatedCode = BuiltinOperator_CUSTOM;
    c.customCode = code;
    c.poolOptions = 0;
    c.hasCustomOptions = present;
    c.customOptions = options;
    return c;
}

static Case argmaxPool(std::vector<uint8_t> options, bool present = true)
{
    Case c = customOp("MaxPoolingWithArgmax2D", options, present);
    c.shapes = {{1, 4, 4, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}};
    c.tensorBuffers = {0, 0, 0};
    c.outputs = {1, 2};
    c.graphOutputs = {1, 2};
    return c;
}

static Case unpooling(std::vector<uint8_t> options, bool present = true)
{
    Case c = customOp("MaxUnpooling2D", options, present);
    c.shapes = {{1, 2, 2, 1}, {1, 2, 2, 1}, {1, 4, 4, 1}};
    c.tensorBuffers = {0, 0, 0};
    c.inputs = {0, 1};
    c.outputs = {2};
    c.graphInputs = {0, 1};
    c.graphOutputs = {2};
    return c;
}

static Case transposeConv(std::vector<uint8_t> options, bool present = true)
{
    Case c = customOp("Convolution2DTransposeBias", options, present);
    c.shapes = {{1, 2, 2, 1}, {1, 2, 2, 1}, {1}, {1, 4, 4, 1}};
    c.tensorBuffers = {0, 1, 2, 0};
    c.buffers = {std::vector<uint8_t>(),
                 std::vector<uint8_t>(4 * sizeof(float), 0),
                 std::vector<uint8_t>(sizeof(float), 0)};
    c.inputs = {0, 1, 2};
    c.outputs = {3};
    return c;
}

// MaxPoolingWithArgmax2D feeding both of its outputs to MaxUnpooling2D.
static Case argmaxPoolThenUnpooling()
{
    Case c = argmaxPool(poolParams());
    c.shapes = {{1, 4, 4, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}, {1, 4, 4, 1}};
    c.tensorBuffers = {0, 0, 0, 0};
    c.graphOutputs = {3};
    c.secondCustomCode = "MaxUnpooling2D";
    c.secondCustomOptions = poolParams();
    c.secondInputs = {1, 2};
    c.secondOutputs = {3};
    return c;
}

// A fully connected operator whose optional bias input is omitted (-1).
static Case fullyConnectedWithoutBias()
{
    Case c;
    c.deprecatedCode = BuiltinOperator_FULLY_CONNECTED;
    c.poolOptions = 3;
    c.shapes = {{1, 4}, {2, 4}, {1, 2}};
    c.tensorBuffers = {0, 1, 0};
    c.buffers = {std::vector<uint8_t>(), std::vector<uint8_t>(8 * sizeof(float), 0)};
    c.inputs = {0, 1, -1};
    c.outputs = {2};
    c.graphOutputs = {2};
    return c;
}

static Case dequantize()
{
    Case c;
    c.deprecatedCode = BuiltinOperator_DEQUANTIZE;
    c.poolOptions = 0;
    return c;
}

static std::vector<char> build(const Case& c)
{
    FlatBufferBuilder fbb;

    std::vector<Offset<OperatorCode> > codes;
    codes.push_back(CreateOperatorCodeDirect(fbb, c.deprecatedCode, c.customCode, 1,
                                             (BuiltinOperator)c.deprecatedCode));
    if (c.secondCustomCode)
        codes.push_back(CreateOperatorCodeDirect(fbb, BuiltinOperator_CUSTOM, c.secondCustomCode, 1,
                                                 BuiltinOperator_CUSTOM));

    std::vector<Offset<Tensor> > tensors;
    for (size_t i = 0; i < c.shapes.size(); ++i)
    {
        const std::string name = i == 0 ? "input" : cv::format("tensor%d", (int)i);
        tensors.push_back(CreateTensorDirect(fbb, &c.shapes[i], TensorType_FLOAT32, c.tensorBuffers[i],
                                             (int)i == c.unnamedTensor ? nullptr : name.c_str(),
                                             0, false, 0, nullptr, true));
    }

    Offset<void> builtin;
    BuiltinOptions builtinType = BuiltinOptions_NONE;
    if (c.poolOptions == 1)
    {
        builtinType = BuiltinOptions_Pool2DOptions;
        builtin = CreatePool2DOptions(fbb, Padding_SAME, c.strideW, c.strideH, c.filterW, c.filterH,
                                      ActivationFunctionType_NONE).Union();
    }
    else if (c.poolOptions == 2)
    {
        builtinType = BuiltinOptions_Conv2DOptions;
        builtin = CreateConv2DOptions(fbb, Padding_SAME, 1, 1, ActivationFunctionType_NONE).Union();
    }
    else if (c.poolOptions == 3)
    {
        builtinType = BuiltinOptions_FullyConnectedOptions;
        builtin = CreateFullyConnectedOptions(fbb, ActivationFunctionType_NONE).Union();
    }

    std::vector<Offset<Operator> > operators;
    operators.push_back(CreateOperatorDirect(fbb, c.opcodeIndex, &c.inputs, &c.outputs, builtinType, builtin,
                                             c.hasCustomOptions ? &c.customOptions : nullptr));
    if (c.secondCustomCode)
        operators.push_back(CreateOperatorDirect(fbb, 1, &c.secondInputs, &c.secondOutputs, BuiltinOptions_NONE, 0,
                                                 &c.secondCustomOptions));

    std::vector<Offset<SubGraph> > subgraphs;
    subgraphs.push_back(CreateSubGraphDirect(fbb, &tensors, &c.graphInputs, &c.graphOutputs, &operators, "main"));

    std::vector<Offset<Buffer> > buffers;
    for (size_t i = 0; i < c.buffers.size(); ++i)
        buffers.push_back(c.buffers[i].empty() ? CreateBuffer(fbb) : CreateBufferDirect(fbb, &c.buffers[i]));

    FinishModelBuffer(fbb, CreateModelDirect(fbb, 3, &codes, &subgraphs, "fixture", &buffers));
    return std::vector<char>(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
}

}  // namespace tflite_fixture

static void expectTFLiteAccepted(const tflite_fixture::Case& c)
{
    const std::vector<char> model = tflite_fixture::build(c);
    EXPECT_NO_THROW(readNetFromTFLite(model.data(), model.size()));
}

static void expectTFLiteRejected(const tflite_fixture::Case& c, const char* reason)
{
    SCOPED_TRACE(reason);
    const std::vector<char> model = tflite_fixture::build(c);
    try
    {
        (void)readNetFromTFLite(model.data(), model.size());
        FAIL() << "Invalid TFLite model was accepted";
    }
    catch (const cv::Exception& e)
    {
        EXPECT_NE(std::string(e.what()).find(reason), std::string::npos) << e.what();
    }
}

TEST(Test_TFLite_Importer, OperatorOutputsAndOpcode)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(maxPool());

    Case noOutputs = maxPool();
    noOutputs.outputs = {};
    expectTFLiteRejected(noOutputs, "operator has no outputs");

    Case negativeOutput = maxPool();
    negativeOutput.outputs = {-1};
    expectTFLiteRejected(negativeOutput, "output tensor index must be non-negative");

    Case outputOutOfRange = maxPool();
    outputOutOfRange.outputs = {2};
    expectTFLiteRejected(outputOutOfRange, "output tensor index out of range");

    // A flatbuffer vector element is addressed by offset and never resolves to a null pointer,
    // so the null-tensor and null-operator-code assertions have no malformed-model witness.
    Case unnamedOutput = maxPool();
    unnamedOutput.unnamedTensor = 1;
    expectTFLiteRejected(unnamedOutput, "outputName");

    Case opcodeOutOfRange = maxPool();
    opcodeOutOfRange.opcodeIndex = 3;
    expectTFLiteRejected(opcodeOutOfRange, "opcode_index out of range");

    Case customWithoutCode = customOp(nullptr, poolParams());
    expectTFLiteRejected(customWithoutCode, "customCode");
}

TEST(Test_TFLite_Importer, OperatorTensorIndexes)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(argmaxPoolThenUnpooling());
    expectTFLiteAccepted(fullyConnectedWithoutBias());

    // The second output of the first operator is out of range and the second operator consumes it.
    Case laterOutputOutOfRange = argmaxPoolThenUnpooling();
    laterOutputOutOfRange.outputs = {1, 7};
    laterOutputOutOfRange.secondInputs = {1, 7};
    expectTFLiteRejected(laterOutputOutOfRange, "output tensor index out of range");

    Case laterOutputNegative = argmaxPool(poolParams());
    laterOutputNegative.outputs = {1, -2};
    expectTFLiteRejected(laterOutputNegative, "output tensor index must be non-negative");

    Case inputOutOfRange = maxPool();
    inputOutOfRange.inputs = {2};
    expectTFLiteRejected(inputOutOfRange, "input tensor index out of range");

    Case inputNegative = maxPool();
    inputNegative.inputs = {-2};
    expectTFLiteRejected(inputNegative, "input tensor index must be non-negative or -1");

    Case optionalInputOutOfRange = fullyConnectedWithoutBias();
    optionalInputOutOfRange.inputs = {0, 1, 3};
    expectTFLiteRejected(optionalInputOutOfRange, "input tensor index out of range");
}

TEST(Test_TFLite_Importer, SubgraphInputs)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(maxPool());

    Case inputOutOfRange = maxPool();
    inputOutOfRange.graphInputs = {2};
    expectTFLiteRejected(inputOutOfRange, "subgraph input tensor index out of range");

    Case inputNegative = maxPool();
    inputNegative.graphInputs = {-1};
    expectTFLiteRejected(inputNegative, "subgraph input tensor index must be non-negative");

    Case unnamedInput = maxPool();
    unnamedInput.unnamedTensor = 0;
    expectTFLiteRejected(unnamedInput, "inputName");
}

TEST(Test_TFLite_Importer, DequantizeInputs)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(dequantize());

    Case noInputs = dequantize();
    noInputs.inputs = {};
    expectTFLiteRejected(noInputs, "DEQUANTIZE operator requires an input");

    Case omittedInput = dequantize();
    omittedInput.inputs = {-1};
    expectTFLiteRejected(omittedInput, "DEQUANTIZE input must not be omitted");
}

TEST(Test_TFLite_Importer, PoolingOptions)
{
    using namespace tflite_fixture;

    Case noOptions = maxPool();
    noOptions.poolOptions = 0;
    expectTFLiteRejected(noOptions, "options");

    Case wrongOptions = maxPool();
    wrongOptions.poolOptions = 2;
    expectTFLiteRejected(wrongOptions, "options");

    Case zeroStrideW = maxPool();
    zeroStrideW.strideW = 0;
    expectTFLiteRejected(zeroStrideW, "stride_w must be positive");

    Case zeroStrideH = maxPool();
    zeroStrideH.strideH = 0;
    expectTFLiteRejected(zeroStrideH, "stride_h must be positive");

    Case zeroFilterW = maxPool();
    zeroFilterW.filterW = 0;
    expectTFLiteRejected(zeroFilterW, "filter_w must be positive");

    Case zeroFilterH = maxPool();
    zeroFilterH.filterH = 0;
    expectTFLiteRejected(zeroFilterH, "filter_h must be positive");
}

TEST(Test_TFLite_Importer, MaxPoolingWithArgmaxOptions)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(argmaxPool(poolParams()));
    expectTFLiteRejected(argmaxPool(std::vector<uint8_t>(), false), "custom_options() != nullptr");
    expectTFLiteRejected(argmaxPool(firstBytes(poolParams(), 4)),
                         "custom_options too small");
    expectTFLiteRejected(argmaxPool(poolParams(0, 2, 2, 2)), "stride_w must be positive");
    expectTFLiteRejected(argmaxPool(poolParams(2, 0, 2, 2)), "stride_h must be positive");
    expectTFLiteRejected(argmaxPool(poolParams(2, 2, 0, 2)), "filter_w must be positive");
    expectTFLiteRejected(argmaxPool(poolParams(2, 2, 2, 0)), "filter_h must be positive");
}

TEST(Test_TFLite_Importer, MaxUnpoolingOptions)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(unpooling(poolParams()));
    expectTFLiteRejected(unpooling(std::vector<uint8_t>(), false), "custom_options() != nullptr");
    expectTFLiteRejected(unpooling(firstBytes(poolParams(), 4)),
                         "custom_options too small");
    expectTFLiteRejected(unpooling(poolParams(0, 2, 2, 2)), "stride_w must be positive");
    expectTFLiteRejected(unpooling(poolParams(2, 0, 2, 2)), "stride_h must be positive");
    expectTFLiteRejected(unpooling(poolParams(2, 2, 0, 2)), "filter_w must be positive");
    expectTFLiteRejected(unpooling(poolParams(2, 2, 2, 0)), "filter_h must be positive");
}

TEST(Test_TFLite_Importer, TransposeConvolutionOptions)
{
    using namespace tflite_fixture;
    expectTFLiteAccepted(transposeConv(transposeConvParams()));
    expectTFLiteRejected(transposeConv(std::vector<uint8_t>(), false), "custom_options() != nullptr");
    expectTFLiteRejected(transposeConv(firstBytes(transposeConvParams(), 4)),
                         "custom_options too small");
    expectTFLiteRejected(transposeConv(transposeConvParams(0, 2)), "stride_w must be positive");
    expectTFLiteRejected(transposeConv(transposeConvParams(2, 0)), "stride_h must be positive");
}

}}  // namespace

#endif  // OPENCV_TEST_DNN_TFLITE
