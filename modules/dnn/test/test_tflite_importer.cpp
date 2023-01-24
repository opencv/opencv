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

void testModel(const std::string& modelName, const Size& inpSize, double norm = 1e-5) {
#ifndef HAVE_FLATBUFFERS
    throw SkipTestException("FlatBuffers required for TFLite importer");
#endif

    Net net = readNet(_tf(modelName + ".tflite"));

    Mat input = imread(getOpenCVExtraDir() + "/cv/shared/lena.png");
    input = blobFromImage(input, 1.0 / 255, inpSize, 0, true);
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

}
