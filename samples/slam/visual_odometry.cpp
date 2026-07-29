// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.

#include <opencv2/ptcloud.hpp>
#include <opencv2/features.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../dnn/common.hpp"

using namespace cv;
using namespace std;

namespace {

// path helper: strips the directory part off a path
String basenameOf(const String& path)
{
    const size_t slash = path.find_last_of("/\\");
    return (slash == String::npos) ? path : path.substr(slash + 1);
}

// reader: lists readable image files under a directory, sorted by filename
std::vector<String> listImageFiles(const String& imagesDir)
{
    std::vector<String> allFiles;
    glob(imagesDir, allFiles, false);
    std::vector<String> imgFiles;
    imgFiles.reserve(allFiles.size());
    for (const auto& f : allFiles)
        if (haveImageReader(f)) imgFiles.push_back(f);
    std::sort(imgFiles.begin(), imgFiles.end());
    return imgFiles;
}

// bookkeeping: maps each pose in vo->getTrajectory() back to its source image filename.
//
// processFrame() doesn't report which frame a pose came from, so this infers it: most frames
// add one pose, but the frame where two-view initialization completes adds two at once (the
// reference frame's pose, remembered when initialization started, plus the current frame's).
class PoseFilenameTracker
{
public:
    void update(slam::OdometryState before, slam::OdometryState after,
                size_t trajectorySize, const String& filename)
    {
        if (before == slam::NOT_INITIALIZED ||
            (before == slam::TRACKING && after == slam::INITIALIZING))
            refFilename_ = filename;

        const size_t added = trajectorySize - prevTrajectorySize_;
        if (added == 1)
            filenames_.push_back(filename);
        else if (added == 2)
        {
            filenames_.push_back(refFilename_);
            filenames_.push_back(filename);
        }
        prevTrajectorySize_ = trajectorySize;
    }

    const std::vector<String>& filenames() const { return filenames_; }

private:
    std::vector<String> filenames_;
    size_t prevTrajectorySize_ = 0;
    String refFilename_;
};

// writer: dumps camera.txt, point3d.txt and images.txt (COLMAP text format) for @p vo into
// @p outputFolder; @p poseFilenames[i] is the source image for trajectory pose i
void writeColmapFiles(const slam::VisualOdometry& vo, const Mat& K,
                      const std::vector<String>& poseFilenames,
                      const String& outputFolder)
{
    cv::utils::fs::createDirectories(outputFolder);

    {
        std::ofstream f(cv::utils::fs::join(outputFolder, "camera.txt").c_str());
        int width = 0, height = 0;
        if (!vo.getMap().keyframes().empty())
        {
            const slam::KeyFrame* kf = *vo.getMap().keyframes().begin();
            width  = kf->imageSize.width;
            height = kf->imageSize.height;
        }
        f.setf(std::ios::fixed); f.precision(4);
        f << "fx " << K.at<double>(0, 0) << "\n"
          << "fy " << K.at<double>(1, 1) << "\n"
          << "cx " << K.at<double>(0, 2) << "\n"
          << "cy " << K.at<double>(1, 2) << "\n"
          << "width "  << width  << "\n"
          << "height " << height << "\n";
    }

    {
        std::ofstream f(cv::utils::fs::join(outputFolder, "point3d.txt").c_str());
        f << "# Map points in world coordinates.\n# Columns: id X Y Z n_observations\n";
        f.setf(std::ios::scientific); f.precision(9);
        for (slam::MapPoint* mp : vo.getMap().mapPoints())
        {
            if (!mp || mp->bad) continue;
            f << mp->id << " "
              << mp->pos.x << " " << mp->pos.y << " " << mp->pos.z << " "
              << mp->observations.size() << "\n";
        }
    }

    {
        std::ofstream f(cv::utils::fs::join(outputFolder, "images.txt").c_str());
        const auto& traj = vo.getTrajectory();
        f << "# Image list with two lines of data per image:\n"
          << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
          << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
          << "# Number of images: " << traj.size() << ", mean observations per image: 0.0\n";
        f.setf(std::ios::fixed); f.precision(6);

        for (size_t i = 0; i < traj.size(); ++i)
        {
            const Matx44d& T = traj[i];
            Matx33d R;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) R(r, c) = T(r, c);

            const Quatd q = Quatd::createFromRotMat(R);
            const String name = (i < poseFilenames.size())
                ? basenameOf(poseFilenames[i])
                : (String("pose_") + std::to_string(i));

            f << i << " " << q.w << " " << q.x << " " << q.y << " " << q.z << " "
              << T(0,3) << " " << T(1,3) << " " << T(2,3) << " " << 1 << " " << name << "\n";
        }
    }
}

} // namespace

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
    detParams.engine    = dnn::ENGINE_OPENCV;
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
        detector, matcher, Mat(K), distCoeffs, voParams);

    const std::vector<String> imgFiles = listImageFiles(imagesDir);

    if (imgFiles.empty())
    {
        std::cerr << "no images found in " << imagesDir << "\n";
        return 1;
    }

    std::cout << "images folder : " << imagesDir << "\n"
              << "output folder : " << outputDir << "\n"
              << "total frames  : " << imgFiles.size() << "\n\n";

    PoseFilenameTracker poseTracker;
    int nEmitted = 0;

    const int nDigits = static_cast<int>(std::to_string(imgFiles.size()).size());
    const int64 t0 = getTickCount();
    for (size_t i = 0; i < imgFiles.size(); ++i)
    {
        const int64 tFrameStart = getTickCount();

        Mat img = imread(imgFiles[i]);
        if (img.empty())
        {
            std::cerr << "Frame " << std::setw(nDigits) << (i + 1) << "/" << imgFiles.size()
                      << " failed to read: " << imgFiles[i] << "\n";
            continue;
        }

        const slam::OdometryState before = vo->getState();
        const bool emitted = vo->processFrame(img);
        const slam::OdometryState after = vo->getState();
        if (emitted) ++nEmitted;

        poseTracker.update(before, after, vo->getTrajectory().size(), imgFiles[i]);

        const double frameElapsed = (getTickCount() - tFrameStart) / getTickFrequency();
        const double fps = frameElapsed > 0. ? 1. / frameElapsed : 0.;

        std::cout << "Frame " << std::setw(nDigits) << (i + 1) << "/" << imgFiles.size()
                  << " processed | fps: " << std::fixed << std::setprecision(1) << std::setw(5) << fps
                  << " | emitted: " << (emitted ? "yes" : "no ")
                  << " | keyframes: " << vo->getMap().numKeyframes()
                  << " | map points: " << vo->getMap().numMapPoints() << "\n";
    }
    const double elapsed = (getTickCount() - t0) / getTickFrequency();
    const double avgFps = elapsed > 0. ? imgFiles.size() / elapsed : 0.;
    const bool ok = nEmitted > 0;

    if (ok && !outputDir.empty())
        writeColmapFiles(*vo, Mat(K), poseTracker.filenames(), outputDir);

    std::cout << "\n"
              << "==================== Visual Odometry Result ====================\n"
              << "status        : " << (ok ? "OK" : "FAILED") << "\n"
              << "total frames  : " << imgFiles.size() << "\n"
              << "camera poses  : " << vo->getTrajectory().size() << "\n"
              << "keyframes     : " << vo->getMap().numKeyframes() << "\n"
              << "map points    : " << vo->getMap().numMapPoints() << "\n"
              << "elapsed time  : " << std::fixed << std::setprecision(2) << elapsed << " s\n"
              << "average fps   : " << std::fixed << std::setprecision(2) << avgFps << "\n"
              << "output dir    : " << outputDir << "\n"
              << "====================================================================\n";
    return ok ? 0 : 1;
}
