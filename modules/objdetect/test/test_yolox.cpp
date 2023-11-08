// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


TEST(Objdetect_yolox_detection, regression)
{
    float confThreshold = 0.5;
    float nmsThreshold = 0.5;
    float objThreshold = 0.5;
    dnn::Backend backendId = dnn::DNN_BACKEND_OPENCV;
    dnn::Target targetId = dnn::DNN_TARGET_CPU;

    // Initialize detector
    std::string modelPath = findDataFile("dnn/yolox_s.onnx", false);
    Ptr<ObjectDetectorYX> detector = ObjectDetectorYX::create(modelPath, confThreshold, nmsThreshold, objThreshold, backendId, targetId);
}


}} // namespace
