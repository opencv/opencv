// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "vo_impl.hpp"

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

} // anonymous namespace

// Factory

VisualOdometry::VisualOdometry() = default;
VisualOdometry::~VisualOdometry() = default;

Ptr<VisualOdometry> VisualOdometry::create(
    const Ptr<Feature2D>& detector,
    const Ptr<DescriptorMatcher>& matcher,
    InputArray cameraMatrix,
    InputArray distCoeffs,
    const OdometryParams& params)
{
    CV_Assert(detector && "VisualOdometry::create: detector must not be null");
    CV_Assert(matcher  && "VisualOdometry::create: matcher must not be null");

    Mat K = cameraMatrix.getMat();
    CV_Assert(!K.empty() && K.rows == 3 && K.cols == 3);
    Mat dist = distCoeffs.empty() ? Mat() : distCoeffs.getMat();

    return makePtr<VisualOdometryImpl>(detector, matcher, K, dist, params);
}

// Constructor

VisualOdometryImpl::VisualOdometryImpl(
    const Ptr<Feature2D>& _detector,
    const Ptr<DescriptorMatcher>& _matcher,
    const Mat& _cameraMatrix,
    const Mat& _distCoeffs,
    const OdometryParams& _params)
    : detector(_detector), matcher(_matcher), params(_params)
{
    _cameraMatrix.convertTo(K, CV_64F);
    if (!_distCoeffs.empty())
        _distCoeffs.convertTo(dist, CV_64F);

    CV_LOG_INFO(NULL, "slam: optimizer pose_ba=" << (params.poseOptEnable ? "g2o" : "reproj")
                      << " local_ba=" << (params.localBaEnable ? "on" : "off")
                      << " global_ba=" << (params.globalBaEnable ? "on" : "off")
                      << " loop=" << (params.loopEnable ? "on" : "off"));
#ifndef HAVE_G2O
    CV_LOG_WARNING(NULL, "slam: built without g2o — bundle adjustment and loop closure are no-ops");
#endif
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

    Frame currentFrame;
    extractFeatures(image, currentFrame);
    if (currentFrame.keypoints.empty() || currentFrame.descriptors.empty()) return false;

    currentFrame.mapPoints.assign(currentFrame.keypoints.size(), nullptr);
    currentFrame.outliers.assign(currentFrame.keypoints.size(), false);
    currentFrame.buildGrid();

    const OdometryState stateBefore = state;
    bool emitted = false;

    switch (state)
    {
    case NOT_INITIALIZED:
        refFrame = currentFrame;
        state = INITIALIZING;
        break;

    case INITIALIZING:
        emitted = bootstrap(currentFrame);
        break;

    case TRACKING:
        emitted = track(currentFrame);
        break;
    }

    // Per-frame progress, at INFO so it survives release builds (CV_LOG_DEBUG is compiled
    // out when CV_LOG_STRIP_LEVEL defaults to DEBUG). The logger is gated at WARNING by
    // default, so this stays silent unless the caller asks for it via OPENCV_LOG_LEVEL=INFO.
    CV_LOG_INFO(NULL, "slam: state=" << stateName(stateBefore)
                      << (stateBefore != state ? String(" -> ") + stateName(state) : String())
                      << " emitted=" << (emitted ? "yes" : "no")
                      << " keyframes=" << map.numKeyframes()
                      << " map_points=" << map.numMapPoints()
                      << (lastEvent.empty() ? String() : " [" + lastEvent + "]"));

    return emitted;
}

// End-of-sequence refinement

bool VisualOdometryImpl::finalizeMap()
{
    CV_INSTRUMENT_REGION();

    Optimizer::GlobalBAStats stats;
    Optimizer::GlobalBundleAdjustment(map, K, params.globalBaIters,
                                      params.globalBaMinObs, params.globalBaEnable,
                                      nullptr, &stats);
    if (!stats.ran)
    {
        CV_LOG_INFO(NULL, "slam: global BA skipped");
        return false;
    }

    CV_LOG_INFO(NULL, "slam: global BA chi2 " << stats.chi2Before << " -> " << stats.chi2After
                      << " (" << stats.posesUpdated << " poses updated, "
                      << stats.culled << " observations culled)");
    return true;
}

// Per-frame poses re-expressed on the corrected keyframe graph

std::vector<Matx44d> VisualOdometryImpl::getCorrectedTrajectory() const
{
    // frameRecords is appended in lockstep with map.trajectory(), so the result stays
    // index-aligned with getTrajectory(). A frame whose reference keyframe was culled
    // cannot be corrected and falls back to its raw pose.
    const std::vector<Matx44d>& raw = map.trajectory();
    CV_Assert(frameRecords.size() == raw.size());

    std::vector<Matx44d> corrected;
    corrected.reserve(frameRecords.size());
    for (size_t i = 0; i < frameRecords.size(); ++i)
    {
        const FrameRecord& record = frameRecords[i];
        if (record.refKf && !record.refKf->bad)
            corrected.push_back(record.relPose * record.refKf->poseCw);
        else
            corrected.push_back(raw[i]);
    }
    return corrected;
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

    // No-op for matchers that do not use pair context (e.g. BFMatcher); consumed by
    // keypoint-aware matchers such as LightGlue.
    matcher->setImagePairInfo(qKp, tKp, qSz, tSz);

    matcher->match(qDesc, tDesc, matches);
}

}} // namespace cv::slam
