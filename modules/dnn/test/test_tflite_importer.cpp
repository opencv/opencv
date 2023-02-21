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

#ifdef OPENCV_TEST_DNN_TFLITE

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::dnn;

void testModel(const std::string& modelName, const Mat& input, double l1 = 1e-5, double lInf = 1e-4)
{
    Net net = readNet(findDataFile("dnn/tflite/" + modelName + ".tflite", false));
    net.setInput(input);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    std::vector<Mat> outs;
    net.forward(outs, outNames);

    ASSERT_EQ(outs.size(), outNames.size());
    for (int i = 0; i < outNames.size(); ++i) {
        Mat ref = blobFromNPY(findDataFile(format("dnn/tflite/%s_out_%s.npy", modelName.c_str(), outNames[i].c_str())));
        normAssert(ref.reshape(1, 1), outs[i].reshape(1, 1), outNames[i].c_str(), l1, lInf);
    }
}

void testModel(const std::string& modelName, const Size& inpSize, double l1 = 1e-5, double lInf = 1e-4)
{
    Mat input = imread(findDataFile("cv/shared/lena.png"));
    input = blobFromImage(input, 1.0 / 255, inpSize, 0, true);
    testModel(modelName, input, l1, lInf);
}

// https://google.github.io/mediapipe/solutions/face_mesh
TEST(Test_TFLite, face_landmark)
{
    testModel("face_landmark", Size(192, 192), 2e-5, 2e-4);
}

// https://google.github.io/mediapipe/solutions/face_detection
TEST(Test_TFLite, face_detection_short_range)
{
    testModel("face_detection_short_range", Size(128, 128));
}

// https://google.github.io/mediapipe/solutions/selfie_segmentation
TEST(Test_TFLite, selfie_segmentation)
{
    testModel("selfie_segmentation", Size(256, 256));
}

TEST(Test_TFLite, max_unpooling)
{
    // Due Max Unpoling is a numerically unstable operation and small difference between frameworks
    // might lead to positional difference of maximal elements in the tensor, this test checks
    // behavior of Max Unpooling layer only.
    Net net = readNet(findDataFile("dnn/tflite/hair_segmentation.tflite", false));

    Mat input = imread(findDataFile("cv/shared/lena.png"));
    cvtColor(input, input, COLOR_BGR2RGBA);
    input = input.mul(Scalar(1, 1, 1, 0));
    input = blobFromImage(input, 1.0 / 255);
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
                EXPECT_EQ(poolIdsData[y * 64 + x], (float)maxIdx) << errMsg;
                EXPECT_EQ(unpoolOutData[maxIdx], unpoolInpData[y * 64 + x]) << errMsg;
            }
        }
    }
}

}}  // namespace

#endif  // OPENCV_TEST_DNN_TFLITE
