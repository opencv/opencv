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
    frameRecords.clear();
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

    std::ofstream log;
    if (!outputFolder.empty())
    {
        cv::utils::fs::createDirectories(outputFolder);
        log.open(joinPath(outputFolder, "vo.log").c_str());
    }
    auto logln = [&](const String& s) { if (log.is_open()) log << s << "\n"; };

    {
        std::ostringstream ss;
        ss << "[INFO] optimizer: pose_ba=" << (params.poseOptEnable ? "g2o" : "reproj")
           << " local_ba=" << (params.localBaEnable ? "on" : "off")
           << " global_ba=" << (params.globalBaEnable ? "on" : "off")
           << " loop=" << (params.loopEnable ? "on" : "off");
        #ifndef HAVE_G2O
        ss << " [WARNING: g2o unavailable — BA and loop-closure are no-ops]";
        #endif
        logln(ss.str());
    }
    logln(String("[INFO] images_folder = ") + imagesFolder);
    logln(String("[INFO] output_folder = ") + outputFolder);
    {
        std::ostringstream ss;
        ss << "[INFO] found " << imgFiles.size() << " image(s)";
        logln(ss.str());
    }

    reset();

    int nEmitted = 0;

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
    
    // Global BA: full map refinement after all frames are processed.
    {
        Optimizer::GlobalBAStats gbaStats;
        Optimizer::GlobalBundleAdjustment(map, K, params.globalBaIters,
                                          params.globalBaMinObs, params.globalBaEnable,
                                          nullptr, &gbaStats);
        if (gbaStats.ran)
        {
            std::ostringstream gs;
            gs << "[INFO] global BA: chi2 " << gbaStats.chi2Before
               << " -> " << gbaStats.chi2After
               << " (" << gbaStats.posesUpdated << " poses updated, "
               << gbaStats.culled << " observations culled)";
            logln(gs.str());
            CV_LOG_INFO(NULL, "slam global BA complete: chi2 "
                              << gbaStats.chi2Before << " -> " << gbaStats.chi2After
                              << " (" << gbaStats.posesUpdated << " poses updated, "
                              << gbaStats.culled << " observations culled)");
        }
    }

    if (!outputFolder.empty())
    {
        writeImages        (joinPath(outputFolder, "images.txt"));
        writeKeyframeImages(joinPath(outputFolder, "keyframe_images.txt"));
        writePoint3D       (joinPath(outputFolder, "point3d.txt"));
        writeCamera        (joinPath(outputFolder, "camera.txt"));

        std::ostringstream ss;
        ss << "[INFO] run complete: frames=" << imgFiles.size()
           << " emitted=" << nEmitted
           << " keyframes=" << map.numKeyframes()
           << " map_points=" << map.numMapPoints();
        logln(ss.str());
        logln("[INFO] wrote images.txt, keyframe_images.txt, point3d.txt, camera.txt");
    }

    return nEmitted > 0;
}

// IO helpers

// all frames, corrected — each frame rides along with its reference keyframe's corrected pose
void VisualOdometryImpl::writeImages(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeImages: cannot open " << path); return; }
    f << "# Corrected per-frame camera centre in world coordinates.\n# Columns: Cx Cy Cz\n";
    f.setf(std::ios::scientific); f.precision(9);
    for (const auto& rec : frameRecords)
    {
        if (!rec.refKf) continue;
        Matx44d corrected = rec.relPose * rec.refKf->poseCw;
        Point3d C = detail::cameraCenterWorld(corrected);
        f << C.x << " " << C.y << " " << C.z << "\n";
    }
}

// keyframes only, corrected — final poses after local BA, loop closure, global BA
void VisualOdometryImpl::writeKeyframeImages(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeKeyframeImages: cannot open " << path); return; }
    f << "# Corrected keyframe camera centres in world coordinates.\n# Columns: kf_id Cx Cy Cz\n";
    f.setf(std::ios::scientific); f.precision(9);
    std::vector<KeyFrame*> kfs(map.keyframes().begin(), map.keyframes().end());
    std::sort(kfs.begin(), kfs.end(),
              [](const KeyFrame* a, const KeyFrame* b){ return a->id < b->id; });
    for (const KeyFrame* kf : kfs)
    {
        if (!kf || kf->bad) continue;
        Point3d C = detail::cameraCenterWorld(kf->poseCw);
        f << kf->id << " " << C.x << " " << C.y << " " << C.z << "\n";
    }
}

// all 3D map points, corrected
void VisualOdometryImpl::writePoint3D(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writePoint3D: cannot open " << path); return; }
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

// camera intrinsics as-is
void VisualOdometryImpl::writeCamera(const String& path) const
{
    std::ofstream f(path.c_str());
    if (!f.is_open()) { CV_LOG_WARNING(NULL, "writeCamera: cannot open " << path); return; }
    f << "# Camera intrinsics\n";
    f << "# fx fy cx cy\n";
    f.setf(std::ios::scientific); f.precision(9);
    f << K.at<double>(0,0) << " " << K.at<double>(1,1) << " "
      << K.at<double>(0,2) << " " << K.at<double>(1,2) << "\n";
    if (!dist.empty())
    {
        f << "# distortion coefficients\n";
        Mat d = dist.reshape(1, 1);
        for (int i = 0; i < d.cols; ++i)
            f << d.at<double>(0, i) << (i + 1 < d.cols ? " " : "\n");
    }
}

}} // namespace cv::slam
