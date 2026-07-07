// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include <opencv2/ptcloud.hpp>
#include <opencv2/features.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>

#include "../dnn/common.hpp"

using namespace cv;
using namespace std;

const string about =
    "Monocular visual odometry using ALIKED + LightGlue.\n\n"
    "Runs feature detection (ALIKED) and matching (LightGlue) from ONNX models over a\n"
    "directory of images, estimates the camera trajectory and writes it to --output.\n"
    "To run:\n"
    "\t ./example_slam_visual_odometry --aliked=aliked.onnx --lightglue=lightglue.onnx --images=./seq\n"
    "Sample command (run on the GPU):\n"
    "\t ./example_slam_visual_odometry --aliked=aliked.onnx --lightglue=lightglue.onnx --images=./seq --target=cuda\n";

const string param_keys =
    "{ help h           |        | Print help message }"
    "{ aliked           | <none> | Path to ALIKED ONNX model }"
    "{ lightglue        | <none> | Path to LightGlue ONNX model }"
    "{ images           | <none> | Path to directory with input images }"
    "{ output           | vo_out | Output directory for trajectory and map }"
    "{ fx               | 718.856  | Camera focal length X }"
    "{ fy               | 718.856  | Camera focal length Y }"
    "{ cx               | 607.1928 | Camera principal point X }"
    "{ cy               | 185.2157 | Camera principal point Y }"
    "{ min-parallax     | 1.5    | Minimum initialisation parallax in degrees }"
    "{ min-points       | 50     | Minimum initialisation map points }";

const string backend_keys = format(
    "{ backend          | default | Choose one of computation backends: "
                              "default: automatically (by default), "
                              "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                              "opencv: OpenCV implementation, "
                              "vkcom: VKCOM, "
                              "cuda: CUDA, "
                              "webnn: WebNN }");

const string target_keys = format(
    "{ target           | cpu | Choose one of target computation devices: "
                              "cpu: CPU target (by default), "
                              "opencl: OpenCL, "
                              "opencl_fp16: OpenCL fp16 (half-float precision), "
                              "vpu: VPU, "
                              "vulkan: Vulkan, "
                              "cuda: CUDA, "
                              "cuda_fp16: CUDA fp16 (half-float preprocess) }");

string keys = param_keys + backend_keys + target_keys;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const String alikedPath = parser.get<String>("aliked");
    const String lightgluePath = parser.get<String>("lightglue");
    const String imagesDir = parser.get<String>("images");
    const int backendId = getBackendID(parser.get<String>("backend"));
    const int targetId = getTargetID(parser.get<String>("target"));

    if (!parser.check() || alikedPath == "<none>" || lightgluePath == "<none>" || imagesDir == "<none>")
    {
        parser.printErrors();
        parser.printMessage();
        return 1;
    }

    const String outputDir = parser.get<String>("output");

    const Matx33d K(parser.get<double>("fx"), 0., parser.get<double>("cx"),
                    0., parser.get<double>("fy"), parser.get<double>("cy"),
                    0., 0., 1.);

    ALIKED::Params detParams;
    detParams.inputSize = Size(640, 640);
    detParams.engine    = dnn::ENGINE_NEW;
    detParams.backend   = backendId;
    detParams.target    = targetId;
    auto detector = ALIKED::create(alikedPath, detParams);

    auto matcher = LightGlueMatcher::create(lightgluePath, 0.0f,
                                            backendId, targetId);

    slam::OdometryParams voParams;
    voParams.minInitParallaxDeg = parser.get<double>("min-parallax");
    voParams.minInitPoints      = parser.get<int>("min-points");

    auto vo = slam::VisualOdometry::create(
        detector, matcher,
        imagesDir, outputDir,
        Mat(K), Mat(), voParams);

    const int64  t0      = getTickCount();
    const bool   ok      = vo->run();
    const double elapsed = (getTickCount() - t0) / getTickFrequency();

    std::cout << "run="      << (ok ? "ok" : "FAILED")
              << "  frames=" << vo->getTrajectory().size()
              << "  elapsed=" << elapsed << "s\n"
              << "output -> " << outputDir << "\n";
    return ok ? 0 : 1;
}
