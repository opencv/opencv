// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include <opencv2/slam.hpp>
#include <opencv2/features.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;

static const char* keys =
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

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Monocular visual odometry using ALIKED + LightGlue\n"
                 "  Example: visual_odometry --aliked=aliked.onnx --lightglue=lg.onnx --images=./seq\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const String alikedPath    = parser.get<String>("aliked");
    const String lightgluePath = parser.get<String>("lightglue");
    const String imagesDir     = parser.get<String>("images");

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
    auto detector = ALIKED::create(alikedPath, detParams);

    auto matcher = LightGlueMatcher::create(lightgluePath, 0.0f,
                                            dnn::DNN_BACKEND_DEFAULT,
                                            dnn::DNN_TARGET_CPU);

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
