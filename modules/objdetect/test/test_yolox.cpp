// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


TEST(Objdetect_yolox_detection, regression)
{
    // Initialize detector
    std::string detect_model = findDataFile("dnn/onnx/models/object_detection_yolox_2022nov.onnx", false);
}


}} // namespace
