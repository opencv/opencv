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

TEST(Test_TFLite, face_landmark)
{
    Net net = readNet(_tf("face_landmark.tflite"));

    Mat input = imread(getOpenCVExtraDir() + "/cv/shared/lena.png");
    input = blobFromImage(input, 1.0 / 255, Size(192, 192), 0, true);
    net.setInput(input);
    Mat out = net.forward();

    Mat ref = blobFromNPY(_tf("face_landmark_out.npy"));

    normAssert(ref.reshape(1, 1), out.reshape(1, 1), "", 2e-5);
}

} 