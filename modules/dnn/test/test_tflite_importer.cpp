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

namespace opencv_test
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/tflite/") + filename;
}

void testModel(const std::string& modelName, const Mat& input, double norm = 1e-5) {
#ifndef HAVE_FLATBUFFERS
    throw SkipTestException("FlatBuffers required for TFLite importer");
#endif

    Net net = readNet(_tf(modelName + ".tflite"));
    net.setInput(input);

    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    std::vector<Mat> outs;
    net.forward(outs, outNames);

    CV_CheckEQ(outs.size(), outNames.size(), "");
    for (int i = 0; i < outNames.size(); ++i) {
        Mat ref = blobFromNPY(_tf(format("%s_out_%s.npy", modelName.c_str(), outNames[i].c_str())));
        normAssert(ref.reshape(1, 1), outs[i].reshape(1, 1), outNames[i].c_str(), norm);
    }
}

void testModel(const std::string& modelName, const Size& inpSize, double norm = 1e-5) {
    Mat input = imread(getOpenCVExtraDir() + "/cv/shared/lena.png");
    input = blobFromImage(input, 1.0 / 255, inpSize, 0, true);
    testModel(modelName, input, norm);
}

// https://google.github.io/mediapipe/solutions/face_mesh
TEST(Test_TFLite, face_landmark)
{
    testModel("face_landmark", Size(192, 192), 2e-5);
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
    Net net = readNet(_tf("hair_segmentation.tflite"));

    Mat input = imread(getOpenCVExtraDir() + "/cv/shared/lena.png");
    cvtColor(input, input, COLOR_BGR2RGBA);
    input = input.mul(Scalar(1, 1, 1, 0));
    input = blobFromImage(input, 1.0 / 255);
    net.setInput(input);

    std::vector<std::vector<Mat> > outs;
    net.forward(outs, {"p_re_lu_1", "max_pooling_with_argmax2d", "conv2d_86", "max_unpooling2d_2"});
    CV_CheckEQ(outs.size(), (size_t)4, "");
    CV_CheckEQ(outs[0].size(), (size_t)1, "");
    CV_CheckEQ(outs[1].size(), (size_t)2, "");
    CV_CheckEQ(outs[2].size(), (size_t)1, "");
    CV_CheckEQ(outs[3].size(), (size_t)1, "");
    Mat poolInp = outs[0][0];
    Mat poolOut = outs[1][0];
    Mat poolIds = outs[1][1];
    Mat unpoolInp = outs[2][0];
    Mat unpoolOut = outs[3][0];

    CV_CheckEQ(poolInp.size, unpoolOut.size, "");
    CV_CheckEQ(poolOut.size, poolIds.size, "");
    CV_CheckEQ(poolOut.size, unpoolInp.size, "");

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
                for (int idx : indices) {
                    if (poolInpData[idx] > poolInpData[maxIdx]) {
                        CV_CheckEQ(unpoolOutData[maxIdx], 0.0f, "");
                        maxIdx = idx;
                    }
                }
                CV_CheckEQ(poolInpData[maxIdx], poolOutData[y * 64 + x], "");
                CV_CheckEQ(poolIdsData[y * 64 + x], (float)maxIdx, "");
                CV_CheckEQ(unpoolOutData[maxIdx], unpoolInpData[y * 64 + x], "");
            }
        }
    }
}

}
