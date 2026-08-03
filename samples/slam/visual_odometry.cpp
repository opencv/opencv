// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include <opencv2/ptcloud.hpp>
#include <opencv2/features.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../dnn/common.hpp"

using namespace cv;
using namespace std;

const string about =
    "Modular monocular visual odometry: feature tracking + PnP, no bundle adjustment\n"
    "or loop closure.\n\n"
    "Feeds a directory of images through a pluggable feature detector + matcher pair,\n"
    "estimates the camera trajectory, and writes the result to --output in COLMAP text\n"
    "format. This sample wires up one ONNX detector/matcher pair by default; the\n"
    "pipeline (cv::slam::VisualOdometry) accepts any cv::Feature2D + cv::DescriptorMatcher.\n"
    "To run:\n"
    "\t ./example_slam_visual_odometry --aliked=aliked.onnx --lightglue=lightglue.onnx --images=./seq\n"
    "Sample command (run on the GPU):\n"
    "\t ./example_slam_visual_odometry --aliked=aliked.onnx --lightglue=lightglue.onnx --images=./seq --target=cuda\n";

const string param_keys =
    "{ help h           |        | Print help message }"
    "{ aliked           | <none> | Path to detector ONNX model }"
    "{ lightglue        | <none> | Path to matcher ONNX model }"
    "{ images           | <none> | Path to directory with input images }"
    "{ output           | vo_out | Output directory for trajectory and map }"
    "{ fx               | 718.856  | Camera focal length X }"
    "{ fy               | 718.856  | Camera focal length Y }"
    "{ cx               | 607.1928 | Camera principal point X }"
    "{ cy               | 185.2157 | Camera principal point Y }"
    "{ dist             |        | Lens distortion coeffs, comma-separated k1,k2,p1,p2[,k3,...] (default: none) }"
    "{ progress         | true   | Print per-frame progress logs to the console as they happen }";

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

static vector<String> collectImages(const String& imagesDir)
{
    vector<String> allFiles;
    try
    {
        glob(imagesDir, allFiles, false);
    }
    catch (const Exception& e)
    {
        cerr << "glob failed on " << imagesDir << ": " << e.what() << endl;
        return {};
    }

    vector<String> imageFiles;
    imageFiles.reserve(allFiles.size());
    for (const String& file : allFiles)
        if (haveImageReader(file))
            imageFiles.push_back(file);

    sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

static Mat parseDistCoeffs(const String& text)
{
    stringstream stream(text);
    vector<double> coeffs;
    String token;
    while (getline(stream, token, ','))
    {
        const size_t begin = token.find_first_not_of(" \t");
        const size_t end   = token.find_last_not_of(" \t");
        if (begin == String::npos) continue;
        coeffs.push_back(stod(token.substr(begin, end - begin + 1)));
    }
    return coeffs.empty() ? Mat() : Mat(coeffs, true).reshape(1, 1);
}

static bool writeColmapFiles(const Ptr<slam::VisualOdometry>& vo,
                             const Matx33d& K, const Mat& distCoeffs, Size imageSize,
                             const vector<String>& poseImageNames,
                             const String& outputFolder)
{
    if (!utils::fs::createDirectories(outputFolder))
    {
        cerr << "cannot create output directory " << outputFolder << endl;
        return false;
    }

    // camera.txt
    {
        ofstream file(utils::fs::join(outputFolder, "camera.txt").c_str());
        if (!file.is_open()) { cerr << "cannot write camera.txt" << endl; return false; }
        file << "# Camera list with one line of data per camera:\n"
             << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx, fy, cx, cy, dist...]\n";
        file << setprecision(9);
        file << "1 " << (distCoeffs.empty() ? "PINHOLE" : "FULL_OPENCV") << " "
             << imageSize.width << " " << imageSize.height << " "
             << K(0, 0) << " " << K(1, 1) << " " << K(0, 2) << " " << K(1, 2);
        for (int i = 0; i < (int)distCoeffs.total(); ++i)
            file << " " << distCoeffs.at<double>(i);
        file << "\n";
    }

    // images.txt
    {
        const vector<Matx44d>& trajectory = vo->getTrajectory();
        ofstream file(utils::fs::join(outputFolder, "images.txt").c_str());
        if (!file.is_open()) { cerr << "cannot write images.txt" << endl; return false; }
        file << "# Image list with one line of data per image:\n"
             << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
             << "# Number of images: " << trajectory.size() << "\n";
        file << setprecision(9);
        for (size_t i = 0; i < trajectory.size(); ++i)
        {
            const Matx44d& poseCw = trajectory[i];
            const Quatd q = Quatd::createFromRotMat(poseCw.get_minor<3, 3>(0, 0));
            // COLMAP expects a name relative to the image directory, not a full path.
            const String name = i < poseImageNames.size()
                              ? poseImageNames[i].substr(poseImageNames[i].find_last_of("/\\") + 1)
                              : format("pose_%zu", i);
            file << i << " " << q.w << " " << q.x << " " << q.y << " " << q.z << " "
                 << poseCw(0, 3) << " " << poseCw(1, 3) << " " << poseCw(2, 3)
                 << " 1 " << name << "\n";
        }
    }

    // point3d.txt
    {
        ofstream file(utils::fs::join(outputFolder, "point3d.txt").c_str());
        if (!file.is_open()) { cerr << "cannot write point3d.txt" << endl; return false; }
        file << "# 3D point list with one line of data per point:\n"
             << "#   POINT3D_ID, X, Y, Z, TRACK_LENGTH\n";
        file << setprecision(9);
        for (const slam::MapPoint* point : vo->getMap().mapPoints())
        {
            if (!point || point->bad) continue;
            file << point->id << " "
                 << point->pos.x << " " << point->pos.y << " " << point->pos.z << " "
                 << point->observations.size() << "\n";
        }
    }

    return true;
}

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

    const bool showProgress = parser.get<bool>("progress");
    if (showProgress)
        utils::logging::setLogLevel(utils::logging::LOG_LEVEL_INFO);

    const String outputDir = parser.get<String>("output");

    const Matx33d K(parser.get<double>("fx"), 0., parser.get<double>("cx"),
                    0., parser.get<double>("fy"), parser.get<double>("cy"),
                    0., 0., 1.);
    const Mat distCoeffs = parseDistCoeffs(parser.get<String>("dist"));

    const vector<String> imageFiles = collectImages(imagesDir);
    if (imageFiles.empty())
    {
        cerr << "no images found in " << imagesDir << endl;
        return 1;
    }

    ALIKED::Params detParams;
    detParams.inputSize = Size(640, 640);
    detParams.engine    = dnn::ENGINE_OPENCV;
    detParams.backend   = backendId;
    detParams.target    = targetId;
    auto detector = ALIKED::create(alikedPath, detParams);

    auto matcher = LightGlueMatcher::create(lightgluePath, 0.0f, backendId, targetId);

    // tracking parameters
    slam::OdometryParams voParams;
    voParams.minInitParallaxDeg = 1.5;
    voParams.minInitPoints      = 50;
    voParams.pnpReprojThresh    = 4.0;
    voParams.kfMaxFrames        = 30;
    voParams.localMapTopK       = 10;
    voParams.poseOptEnable      = false;
    voParams.localBaEnable      = false;
    voParams.globalBaEnable     = false;
    voParams.loopEnable         = false;
    voParams.loopCloseEnable    = false;

    auto vo = slam::VisualOdometry::create(detector, matcher, Mat(K), distCoeffs, voParams);

    cout << "images folder : " << imagesDir << "\n"
         << "output folder : " << outputDir << "\n"
         << "images found  : " << imageFiles.size() << "\n\n"
         << "Running feature tracking (PnP), no BA or loop closure.\n"
         << (showProgress
                ? "Per-frame progress is printed below (OpenCV logs to stderr). Pass --progress=false to silence it.\n\n"
                : "Per-frame progress is disabled. Re-run with --progress=true (the default) to see it.\n\n");

    vector<String> poseImageNames;
    String refImageName;
    Size imageSize;
    size_t previousPoseCount = 0;

    const int64 ticksBefore = getTickCount();
    for (size_t i = 0; i < imageFiles.size(); ++i)
    {
        const Mat image = imread(imageFiles[i]);
        if (image.empty())
        {
            cerr << "[frame " << i << "] imread failed: " << imageFiles[i] << endl;
            continue;
        }
        imageSize = image.size();

        const slam::OdometryState stateBefore = vo->getState();
        vo->processFrame(image);
        const slam::OdometryState stateAfter = vo->getState();

        if (stateBefore == slam::NOT_INITIALIZED ||
            (stateBefore == slam::TRACKING && stateAfter == slam::INITIALIZING))
            refImageName = imageFiles[i];

        const size_t poseCount = vo->getTrajectory().size();
        if (poseCount == previousPoseCount + 1)
            poseImageNames.push_back(imageFiles[i]);
        else if (poseCount == previousPoseCount + 2)
        {
            poseImageNames.push_back(refImageName);
            poseImageNames.push_back(imageFiles[i]);
        }
        previousPoseCount = poseCount;
    }
    const double elapsed = (getTickCount() - ticksBefore) / getTickFrequency();

    const bool ok = !vo->getTrajectory().empty();
    bool exported = false;
    if (ok && !outputDir.empty())
    {
        try
        {
            exported = writeColmapFiles(vo, K, distCoeffs, imageSize, poseImageNames, outputDir);
        }
        catch (const Exception& e)
        {
            cerr << "writing output failed: " << e.what() << endl;
        }
    }

    cout << "\n"
         << "================ Visual Odometry Result ================\n"
         << "status        : " << (ok ? "OK" : "FAILED") << "\n"
         << "camera poses  : " << vo->getTrajectory().size() << "\n"
         << "keyframes     : " << vo->getNumKeyframes() << "\n"
         << "map points    : " << vo->getNumMapPoints() << "\n"
         << "elapsed time  : " << fixed << setprecision(2) << elapsed << " s\n";
    if (exported)
        cout << "output        : " << outputDir << "/{camera,images,point3d}.txt\n";
    cout << "==========================================================\n";

    return ok ? 0 : 1;
}
