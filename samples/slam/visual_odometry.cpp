// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include <opencv2/ptcloud.hpp>
#include <opencv2/features.hpp>
#include <opencv2/core.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

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
    "{ dist             |        | Lens distortion coeffs, comma-separated k1,k2,p1,p2[,k3,...] (default: none) }";

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

    // Optional lens distortion (k1,k2,p1,p2[,k3,...]); empty means no distortion.
    Mat distCoeffs;
    {
        std::stringstream ss(parser.get<String>("dist"));
        std::vector<double> coeffs;
        String tok;
        while (std::getline(ss, tok, ','))
        {
            const size_t b = tok.find_first_not_of(" \t");
            const size_t e = tok.find_last_not_of(" \t");
            if (b == String::npos) continue;
            coeffs.push_back(std::stod(tok.substr(b, e - b + 1)));
        }
        if (!coeffs.empty())
            distCoeffs = Mat(coeffs, true).reshape(1, 1);
    }

    ALIKED::Params detParams;
    detParams.inputSize = Size(640, 640);
    detParams.engine    = dnn::ENGINE_NEW;
    detParams.backend   = backendId;
    detParams.target    = targetId;
    auto detector = ALIKED::create(alikedPath, detParams);

    auto matcher = LightGlueMatcher::create(lightgluePath, 0.0f,
                                            backendId, targetId);

    slam::OdometryParams voParams;

    voParams.minInitParallaxDeg = 1.5;
    voParams.minInitPoints      = 50;
    voParams.pnpReprojThresh    = 4.0;
    voParams.kfMaxFrames        = 30;
    voParams.localMapTopK       = 10;

    auto vo = slam::VisualOdometry::create(
        detector, matcher, imagesDir, outputDir, Mat(K), distCoeffs, voParams);

    std::cout << "images folder : " << imagesDir << "\n"
              << "output folder : " << outputDir << "\n\n"
              << "Running the batch pipeline (tracking + local/global BA + loop closure).\n"
              << "Per-frame progress and loop-closure / global-BA diagnostics are written to "
              << (outputDir.empty() ? String("vo.log") : (outputDir + "/vo.log")) << "\n\n";

    // run() drives the full pipeline: it processes every image in --images, runs local BA
    // and loop detection/closure at each keyframe, applies global BA once at the end, and
    // writes the *corrected* outputs. We deliberately do NOT export vo.getTrajectory() here:
    // that is the raw, per-frame log appended at emission time and is never rewritten by loop
    // closure or BA, so plotting it would hide every correction. Instead, use the files run()
    // writes below (keyframe_images.txt / images.txt), whose poses ride on the corrected
    // keyframe graph.
    const int64 t0 = getTickCount();
    const bool ok = vo->run();
    const double elapsed = (getTickCount() - t0) / getTickFrequency();

    std::cout << "\n"
              << "==================== Visual Odometry Result ====================\n"
              << "status        : " << (ok ? "OK" : "FAILED") << "\n"
              << "camera poses  : " << vo->getTrajectory().size() << "\n"
              << "keyframes     : " << vo->getMap().numKeyframes() << "\n"
              << "map points    : " << vo->getMap().numMapPoints() << "\n"
              << "elapsed time  : " << std::fixed << std::setprecision(2) << elapsed << " s\n"
              << "output dir    : " << outputDir << "\n";
    if (ok && !outputDir.empty())
    {
        std::cout << "corrected     : " << outputDir << "/keyframe_images.txt (keyframe centres)\n"
                  << "                " << outputDir << "/images.txt          (per-frame centres)\n"
                  << "map points    : " << outputDir << "/point3d.txt\n";
    }
    std::cout << "====================================================================\n";
    return ok ? 0 : 1;
}
