// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/slam.hpp>
#include <opencv2/features.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;

static const char* ALIKED_MODEL    = "/media/user/path/to/models/aliked-n16rot-top1k-640.onnx";
static const char* LIGHTGLUE_MODEL = "/media/user/path/to/models/aliked_lightglue.onnx";
static const char* IMAGES_DIR      = "/media/user/path/to/dataset";
static const char* OUTPUT_DIR      = "vo_out";

// KITTI-00: fx, fy, cx, cy
static const Matx33d K(718.856, 0., 607.1928,
                        0., 718.856, 185.2157,
                        0., 0.,      1.);

// k1, k2, p1, p2, k3
static const std::vector<double> DIST = { -0.2811, 0.0723, -0.0003, 0.0001, 0.0 };

static Ptr<Feature2D> makeDetector()
{
    ALIKED::Params p;
    p.inputSize = Size(640, 640);
    p.engine    = dnn::ENGINE_NEW;
    return ALIKED::create(ALIKED_MODEL, p);
}

static Ptr<DescriptorMatcher> makeMatcher()
{
    return LightGlueMatcher::create(LIGHTGLUE_MODEL, 0.0f,
                                    dnn::DNN_BACKEND_DEFAULT,
                                    dnn::DNN_TARGET_CPU);
}

int main()
{
    slam::OdometryParams params;
    params.minInitParallaxDeg = 1.5;
    params.minInitPoints      = 50;

    auto vo = slam::VisualOdometry::create(
        makeDetector(), makeMatcher(),
        IMAGES_DIR, OUTPUT_DIR,
        Mat(K), Mat(DIST), params);

    const int64 t0      = getTickCount();
    const bool  ok      = vo->run();
    const double elapsed = (getTickCount() - t0) / getTickFrequency();

    std::cout << "run=" << (ok ? "ok" : "FAILED")
              << "  frames=" << vo->getTrajectory().size()
              << "  elapsed=" << elapsed << "s\n"
              << "output -> " << OUTPUT_DIR << "\n";
    return ok ? 0 : 1;
}
