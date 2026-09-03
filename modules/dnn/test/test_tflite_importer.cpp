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
    net.getLayerShapes(MatShape(), CV_32F, 0, inLayerShapes, outLayerShapes);
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
        int64_t *poolIdsData = poolIds.ptr<int64_t>(0, c);
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
                    EXPECT_EQ(poolIdsData[y * 64 + x], (int64_t)maxIdx) << errMsg;
                }
                EXPECT_EQ(unpoolOutData[maxIdx], unpoolInpData[y * 64 + x]) << errMsg;
            }
        }
    }
}

TEST_P(Test_TFLite, EfficientDet_int8) {
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH); // TODO: fix this test for OpenVINO

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

// shrink_axis_mask: single and multiple shrunk axes.
TEST_P(Test_TFLite, StridedSliceShrink) {
    testLayer("strided_slice_shrink_1");
    testLayer("strided_slice_shrink_2");
}

TEST_P(Test_TFLite, Slice) {
    testLayer("slice");
}

TEST_P(Test_TFLite, Sign) {
    testLayer("sign");
}

TEST_P(Test_TFLite, BatchMatMul) {
    testLayer("batch_matmul", 1e-5, 1e-4);
}

TEST_P(Test_TFLite, Select) {
    testLayer("select");
}

TEST_P(Test_TFLite, TopK) {
    testLayer("top_k");
}

TEST_P(Test_TFLite, Less) {
    testLayer("less");
}

TEST_P(Test_TFLite, NotEqual) {
    testLayer("not_equal");
}

TEST_P(Test_TFLite, LogicalAnd) {
    testLayer("logical_and");
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

// A multi-output model must keep all declared outputs, not just the last operator's.
TEST_P(Test_TFLite, multi_output_names)
{
    Net net = readNetFromTFLite(findDataFile("dnn/tflite/face_detection_short_range.tflite", false));

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    std::sort(outNames.begin(), outNames.end());
    ASSERT_EQ(outNames, (std::vector<String>{"classificators", "regressors"}));
}

// end2end head [1,300,C] rows -> Nx7 [batch,cls,conf,x1,y1,x2,y2] for normAssertDetections
static Mat decodeYoloEnd2End(const Mat& out, float confThr = 0.25f)
{
    int n = out.size[out.dims - 2];
    Mat d = out.reshape(1, n);
    std::vector<float> rows;
    for (int i = 0; i < n; ++i)
    {
        const float* r = d.ptr<float>(i);
        if (r[4] < confThr)
            continue;
        float v[7] = {0.f, r[5], r[4], r[0], r[1], r[2], r[3]};
        rows.insert(rows.end(), v, v + 7);
    }
    return Mat((int)(rows.size() / 7), 7, CV_32F, rows.data()).clone();
}

// classic head [1, 4+numClasses, numAnchors] -> Nx7 [batch,cls,conf,x1,y1,x2,y2] after NMS
static Mat decodeYoloClassic(const Mat& out, float confThr = 0.25f, float nmsThr = 0.45f)
{
    Mat t;
    cv::transpose(out.reshape(1, out.size[1]), t);   // [numAnchors, 4+numClasses]
    int nc = t.cols - 4;
    std::vector<Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> ids;
    for (int i = 0; i < t.rows; ++i)
    {
        const float* r = t.ptr<float>(i);
        Point maxLoc;
        double conf;
        minMaxLoc(Mat(1, nc, CV_32F, (void*)(r + 4)), 0, &conf, 0, &maxLoc);
        if (conf < confThr)
            continue;
        boxes.push_back(Rect2d(r[0] - r[2] / 2, r[1] - r[3] / 2, r[2], r[3]));
        scores.push_back((float)conf);
        ids.push_back(maxLoc.x);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThr, nmsThr, keep);
    std::vector<float> rows;
    for (int j : keep)
    {
        const Rect2d& b = boxes[j];
        float v[7] = {0.f, (float)ids[j], scores[j], (float)b.x, (float)b.y,
                      (float)(b.x + b.width), (float)(b.y + b.height)};
        rows.insert(rows.end(), v, v + 7);
    }
    return Mat((int)(rows.size() / 7), 7, CV_32F, rows.data()).clone();
}

TEST_P(Test_TFLite, yolov8n)
{
    Net net = readNet(findDataFile("dnn/tflite/yolov8n.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat input = blobFromImage(imread(findDataFile("dnn/dog416.png")), 1.0 / 255, Size(640, 640), Scalar(), true, false);
    testInputShapes(net, {input});
    net.setInput(input);
    Mat out = net.forward();
    // reference detections from the real TFLite runtime
    float refData[] = {
        0, 16, 0.829000f, 0.171249f, 0.385993f, 0.403052f, 0.938615f,
        0, 1,  0.806363f, 0.161169f, 0.236250f, 0.738703f, 0.730268f,
        0, 7,  0.576536f, 0.607749f, 0.130276f, 0.900613f, 0.297468f,
    };
    normAssertDetections(Mat(3, 7, CV_32F, refData), decodeYoloClassic(out), "", 0.25, 0.01, 0.02);
}

TEST_P(Test_TFLite, yolov5nu)
{
    Net net = readNet(findDataFile("dnn/tflite/yolov5nu.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat input = blobFromImage(imread(findDataFile("dnn/dog416.png")), 1.0 / 255, Size(640, 640), Scalar(), true, false);
    testInputShapes(net, {input});
    net.setInput(input);
    Mat out = net.forward();
    float refData[] = {
        0, 16, 0.840335f, 0.171856f, 0.382436f, 0.406565f, 0.921805f,
        0, 7,  0.647125f, 0.607092f, 0.128508f, 0.902305f, 0.299463f,
        0, 1,  0.494057f, 0.160287f, 0.242356f, 0.740009f, 0.736629f,
    };
    normAssertDetections(Mat(3, 7, CV_32F, refData), decodeYoloClassic(out), "", 0.25, 0.01, 0.02);
}

TEST_P(Test_TFLite, yolo26n)
{
    Net net = readNet(findDataFile("dnn/tflite/yolo26n.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat input = blobFromImage(imread(findDataFile("dnn/dog416.png")), 1.0 / 255, Size(640, 640), Scalar(), true, false);
    testInputShapes(net, {input});
    net.setInput(input);
    Mat out = net.forward();
    Mat ref = blobFromNPY(findDataFile("dnn/tflite/yolo26n_out_serving_default_output_0_output.npy"));
    normAssertDetections(decodeYoloEnd2End(ref), decodeYoloEnd2End(out), "", 0.5, 0.01, 0.02);
}

TEST_P(Test_TFLite, yolo26n_seg)
{
    Net net = readNet(findDataFile("dnn/tflite/yolo26n-seg.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat input = blobFromImage(imread(findDataFile("dnn/street.png")), 1.0 / 255, Size(640, 640), Scalar(), true, false);
    testInputShapes(net, {input});
    net.setInput(input);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    net.forward(outs, outNames);
    for (size_t i = 0; i < outNames.size(); ++i)
    {
        // validate only the detection head; the mask-proto tensor is intermediate
        if (outs[i].size[outs[i].dims - 2] != 300)
            continue;
        Mat ref = blobFromNPY(findDataFile(format("dnn/tflite/yolo26n-seg_out_%s.npy", outNames[i].c_str())));
        normAssertDetections(decodeYoloEnd2End(ref), decodeYoloEnd2End(outs[i]), "", 0.5, 0.01, 0.02);
    }
}

TEST_P(Test_TFLite, resnet18)
{
    Net net = readNet(findDataFile("dnn/tflite/resnet18.tflite", false));
    Mat input = blobFromImage(imread(findDataFile("dnn/space_shuttle.jpg")), 1.0 / 255, Size(224, 224), Scalar(), true, false);
    testModel(net, "resnet18", input, 0.05, 0.2);
}

TEST_P(Test_TFLite, mobilenet_v2)
{
    Net net = readNet(findDataFile("dnn/tflite/mobilenet_v2.tflite", false));
    Mat input = blobFromImage(imread(findDataFile("dnn/space_shuttle.jpg")), 1.0 / 255, Size(224, 224), Scalar(), true, false);
    testModel(net, "mobilenet_v2", input, 0.05, 0.15);
}

TEST_P(Test_TFLite, squeezenet1_1)
{
    Net net = readNet(findDataFile("dnn/tflite/squeezenet1_1.tflite", false));
    Mat input = blobFromImage(imread(findDataFile("dnn/space_shuttle.jpg")), 1.0 / 255, Size(224, 224), Scalar(), true, false);
    testModel(net, "squeezenet1_1", input, 0.05, 0.15);
}

TEST_P(Test_TFLite, yunet)
{
    Net net = readNet(findDataFile("dnn/tflite/yunet_float32.tflite", false));
    Mat input = blobFromImage(imread(findDataFile("cv/shared/lena.png")), 1.0 / 255, Size(160, 120), Scalar(), true, false);
    testModel(net, "yunet_float32", input, 1e-4, 1e-2);
}

TEST_P(Test_TFLite, hand_landmark)
{
    Net net = readNet(findDataFile("dnn/tflite/hand_landmark_lite.tflite", false));
    Mat hand = imread(findDataFile("dnn/pose.png"))(Rect(0, 138, 140, 140));
    Mat input = blobFromImage(hand, 1.0 / 255, Size(224, 224), Scalar(), true, false);
    testModel(net, "hand_landmark_lite", input, 1e-4, 1e-2);
}

TEST_P(Test_TFLite, pose_landmark)
{
    Net net = readNet(findDataFile("dnn/tflite/pose_landmark_lite.tflite", false));
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    Mat input = blobFromImage(imread(findDataFile("dnn/pose.png")), 1.0 / 255, Size(256, 256), Scalar(), true, false);
    testInputShapes(net, {input});
    net.setInput(input);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    std::vector<Mat> outs;
    net.forward(outs, outNames);
    // The reference stores 4D outputs as NHWC; compare the 2D outputs only.
    for (size_t i = 0; i < outNames.size(); ++i)
    {
        if (outs[i].dims > 2)
            continue;
        std::replace(outNames[i].begin(), outNames[i].end(), ':', '_');
        Mat ref = blobFromNPY(findDataFile("dnn/tflite/pose_landmark_lite_out_" + outNames[i] + ".npy"));
        normAssert(ref, outs[i], outNames[i].c_str(), 0.15, 1.5);
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TFLite, dnnBackendsAndTargets());

}}  // namespace

#endif  // OPENCV_TEST_DNN_TFLITE
