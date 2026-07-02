// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "vo_impl.hpp"

#include <fstream>
#include <sstream>

namespace cv {
namespace slam {

namespace {

const char* stateName(OdometryState s)
{
    switch (s)
    {
    case NOT_INITIALIZED: return "NOT_INITIALIZED";
    case INITIALIZING:    return "INITIALIZING";
    case TRACKING:        return "TRACKING";
    }
    return "NOT_INITIALIZED";
}

String joinPath(const String& dir, const String& name)
{
    if (dir.empty()) return name;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

} // anonymous namespace

// Factory

VisualOdometry::VisualOdometry() = default;
VisualOdometry::~VisualOdometry() = default;

Ptr<VisualOdometry> VisualOdometry::create(
    const Ptr<Feature2D>& detector,
    const Ptr<DescriptorMatcher>& matcher,
    const String& imagesFolder,
    const String& outputFolder,
    InputArray cameraMatrix,
    InputArray distCoeffs,
    const OdometryParams& params)
{
    CV_Assert(detector && "VisualOdometry::create: detector must not be null");
    CV_Assert(matcher  && "VisualOdometry::create: matcher must not be null");

    Mat K = cameraMatrix.getMat();
    CV_Assert(!K.empty() && K.rows == 3 && K.cols == 3);
    Mat dist = distCoeffs.empty() ? Mat() : distCoeffs.getMat();

    return makePtr<VisualOdometryImpl>(
        detector, matcher, imagesFolder, outputFolder, K, dist, params);
}

// Constructor

VisualOdometryImpl::VisualOdometryImpl(
    const Ptr<Feature2D>& detector,
    const Ptr<DescriptorMatcher>& matcher,
    const String& imagesFolder,
    const String& outputFolder,
    const Mat& cameraMatrix,
    const Mat& distCoeffs,
    const OdometryParams& params)
    : detector(detector), matcher(matcher), params(params),
      imagesFolder(imagesFolder), outputFolder(outputFolder)
{
    cameraMatrix.convertTo(K, CV_64F);
    if (!distCoeffs.empty())
        distCoeffs.convertTo(dist, CV_64F);
}

// reset / processFrame

void VisualOdometryImpl::reset()
{
    state = NOT_INITIALIZED;
    lastPoseCw = Matx44d::eye();
    refFrame = Frame();
    lastKf = nullptr;
    framesSinceKf = 0;
    lastKfInliers = 0;
    velocity = Matx44d::eye();
    hasVelocity = false;
    prevFrame = Frame();
    hasPrevFrame = false;
    lastEvent.clear();
    poseFilenames.clear();
    map.clear();
}

bool VisualOdometryImpl::processFrame(InputArray image)
{
    CV_INSTRUMENT_REGION();

    if (image.empty()) return false;
    lastEvent.clear();

    Frame cur;
    extractFeatures(image, cur);
    if (cur.keypoints.empty() || cur.descriptors.empty()) return false;

    cur.mapPoints.assign(cur.keypoints.size(), nullptr);
    cur.outliers.assign(cur.keypoints.size(), false);
    cur.buildGrid();

    switch (state)
    {
    case NOT_INITIALIZED:
        refFrame = cur;
        state = INITIALIZING;
        return false;

    case INITIALIZING:
        return bootstrap(cur);

    case TRACKING:
        return track(cur);
    }
    return false;
}

// Feature extraction

void VisualOdometryImpl::extractFeatures(InputArray image, Frame& out) const
{
    Mat img = image.getMat();
    out.imageSize = img.size();
    out.keypoints.clear();

    // Detect and compute on the original image (color/grey is up to the detector).
    detector->detectAndCompute(img, noArray(), out.keypoints, out.descriptors);

    // Store a greyscale copy for the optical-flow fallback.
    if (img.channels() > 1)
        cvtColor(img, out.image, COLOR_BGR2GRAY);
    else
        out.image = img.clone();

    // Pre-compute undistorted pixel coordinates used by every stage.
    if (!out.keypoints.empty())
    {
        std::vector<Point2f> raw;
        raw.reserve(out.keypoints.size());
        for (const auto& kp : out.keypoints)
            raw.push_back(kp.pt);

        if (!dist.empty())
            undistortPoints(raw, out.undistKpts, K, dist, noArray(), K);
        else
            out.undistKpts = raw;
    }
}

// Frame matching helper

void VisualOdometryImpl::matchFrames(
    const std::vector<KeyPoint>& qKp, const Mat& qDesc, Size qSz,
    const std::vector<KeyPoint>& tKp, const Mat& tDesc, Size tSz,
    std::vector<DMatch>& matches) const
{
    matches.clear();
    if (qDesc.empty() || tDesc.empty()) return;
    if (qKp.empty()   || tKp.empty())   return;

    LightGlueMatcher* lg = dynamic_cast<LightGlueMatcher*>(matcher.get());
    if (lg)
    {
        Mat qk((int)qKp.size(), 2, CV_32F);
        for (size_t i = 0; i < qKp.size(); ++i)
        { qk.at<float>((int)i,0) = qKp[i].pt.x; qk.at<float>((int)i,1) = qKp[i].pt.y; }

        Mat tk((int)tKp.size(), 2, CV_32F);
        for (size_t i = 0; i < tKp.size(); ++i)
        { tk.at<float>((int)i,0) = tKp[i].pt.x; tk.at<float>((int)i,1) = tKp[i].pt.y; }

        lg->setPairInfo(qk, tk, qSz, tSz);
    }

    matcher->match(qDesc, tDesc, matches);
}

// Batch run()

bool VisualOdometryImpl::run()
{
    CV_INSTRUMENT_REGION();

    if (imagesFolder.empty())
    {
        CV_LOG_ERROR(NULL, "VisualOdometry::run: imagesFolder is empty");
        return false;
    }

    std::vector<String> allFiles;
    try { cv::glob(imagesFolder, allFiles, false); }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, "VisualOdometry::run: glob failed: " << e.what());
        return false;
    }

    std::vector<String> imgFiles;
    imgFiles.reserve(allFiles.size());
    for (const auto& f : allFiles)
        if (cv::haveImageReader(f)) imgFiles.push_back(f);
    std::sort(imgFiles.begin(), imgFiles.end());

    if (imgFiles.empty())
    {
        CV_LOG_WARNING(NULL, "VisualOdometry::run: no images in " << imagesFolder);
        return false;
    }

    if (!outputFolder.empty())
        cv::utils::fs::createDirectories(outputFolder);
    auto logln = [](const String&) {};

    logln("[INFO] optimizer = reprojection inlier check");
    logln(String("[INFO] images_folder = ") + imagesFolder);
    logln(String("[INFO] output_folder = ") + outputFolder);
    {
        std::ostringstream ss;
        ss << "[INFO] found " << imgFiles.size() << " image(s)";
        logln(ss.str());
    }

    reset();

    int nEmitted = 0;
    size_t prevTrajLen = 0;
    String refFilename;

    for (size_t i = 0; i < imgFiles.size(); ++i)
    {
        Mat img = imread(imgFiles[i]);
        if (img.empty())
        {
            std::ostringstream ss;
            ss << "[FRAME " << i << "] file=" << imgFiles[i] << " ERROR: imread failed";
            logln(ss.str()); continue;
        }

        OdometryState before = state;
        bool emitted = processFrame(img);
        OdometryState after = state;
        if (emitted) ++nEmitted;

        // Track which input image maps to each trajectory pose.
        if (before == NOT_INITIALIZED ||
            (before == TRACKING && after == INITIALIZING))
            refFilename = imgFiles[i];

        const size_t added = map.trajectory().size() - prevTrajLen;
        if (added == 1)
            poseFilenames.push_back(imgFiles[i]);
        else if (added == 2)
        {
            poseFilenames.push_back(refFilename);
            poseFilenames.push_back(imgFiles[i]);
        }
        prevTrajLen = map.trajectory().size();

        std::ostringstream ss;
        ss << "[FRAME " << i << "] file=" << imgFiles[i]
           << " state=" << stateName(before);
        if (before != after) ss << "->" << stateName(after);
        ss << " emitted=" << (emitted ? "yes" : "no")
           << " keyframes=" << map.numKeyframes()
           << " map_points=" << map.numMapPoints();
        if (!lastEvent.empty()) ss << " [" << lastEvent << "]";
        if (emitted)
        {
            Point3d C = detail::cameraCenterWorld(lastPoseCw);
            ss << " C=(" << C.x << "," << C.y << "," << C.z << ")";
        }
        logln(ss.str());
    }

    if (!outputFolder.empty())
    {
        writeCameraIntrinsics(joinPath(outputFolder, "camera.txt"));
        writeMapPoints       (joinPath(outputFolder, "point3d.txt"));
        writeImagesTxt       (joinPath(outputFolder, "images.txt"));
    }

    return nEmitted > 0;
}

// IO helpers

void VisualOdometryImpl::writeCameraIntrinsics(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeCameraIntrinsics: cannot open " << path); return; }

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    int width = 0, height = 0;
    if (!map.keyframes().empty())
    {
        const KeyFrame* kf = *map.keyframes().begin();
        width  = kf->imageSize.width;
        height = kf->imageSize.height;
    }

    f.setf(std::ios::fixed); f.precision(4);
    f << "fx " << fx << "\n"
      << "fy " << fy << "\n"
      << "cx " << cx << "\n"
      << "cy " << cy << "\n"
      << "width "  << width  << "\n"
      << "height " << height << "\n";
}

void VisualOdometryImpl::writeMapPoints(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeMapPoints: cannot open " << path); return; }
    f << "# Map points in world coordinates.\n# Columns: id X Y Z n_observations\n";
    f.setf(std::ios::scientific); f.precision(9);
    for (MapPoint* mp : map.mapPoints())
    {
        if (!mp || mp->bad) continue;
        f << mp->id << " "
          << mp->pos.x << " " << mp->pos.y << " " << mp->pos.z << " "
          << mp->observations.size() << "\n";
    }
}

// Shepperd's method: numerically-stable R → unit quaternion (qw, qx, qy, qz).
static void rotMatToQuat(const Matx33d& R,
                         double& qw, double& qx, double& qy, double& qz)
{
    const double tr = R(0,0) + R(1,1) + R(2,2);
    if (tr > 0.0)
    {
        double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s;
        qx = (R(2,1) - R(1,2)) / s;
        qy = (R(0,2) - R(2,0)) / s;
        qz = (R(1,0) - R(0,1)) / s;
    }
    else if (R(0,0) > R(1,1) && R(0,0) > R(2,2))
    {
        double s = std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2)) * 2.0;
        qw = (R(2,1) - R(1,2)) / s; qx = 0.25 * s;
        qy = (R(0,1) + R(1,0)) / s; qz = (R(0,2) + R(2,0)) / s;
    }
    else if (R(1,1) > R(2,2))
    {
        double s = std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2)) * 2.0;
        qw = (R(0,2) - R(2,0)) / s; qx = (R(0,1) + R(1,0)) / s;
        qy = 0.25 * s;              qz = (R(1,2) + R(2,1)) / s;
    }
    else
    {
        double s = std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1)) * 2.0;
        qw = (R(1,0) - R(0,1)) / s; qx = (R(0,2) + R(2,0)) / s;
        qy = (R(1,2) + R(2,1)) / s; qz = 0.25 * s;
    }
}

static String basenameOf(const String& path)
{
    const size_t slash = path.find_last_of("/\\");
    return (slash == String::npos) ? path : path.substr(slash + 1);
}

void VisualOdometryImpl::writeImagesTxt(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeImagesTxt: cannot open " << path); return; }

    const auto& traj = map.trajectory();
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
            for (int c = 0; c < 3; ++c) R(r,c) = T(r,c);
        double qw, qx, qy, qz;
        rotMatToQuat(R, qw, qx, qy, qz);

        const String name = (i < poseFilenames.size())
            ? basenameOf(poseFilenames[i])
            : (String("pose_") + std::to_string(i));

        f << i << " " << qw << " " << qx << " " << qy << " " << qz << " "
          << T(0,3) << " " << T(1,3) << " " << T(2,3) << " " << 1 << " " << name << "\n";
    }
}

}} // namespace cv::slam
