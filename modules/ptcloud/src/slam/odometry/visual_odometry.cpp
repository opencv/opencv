// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "vo_impl.hpp"

namespace cv {
namespace slam {

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
    const Ptr<Feature2D>& detector_,
    const Ptr<DescriptorMatcher>& matcher_,
    const Mat& cameraMatrix,
    const Mat& distCoeffs,
    const OdometryParams& params_)
    : detector(detector_), matcher(matcher_), params(params_)
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

    switch (state)
    {
    case NOT_INITIALIZED:
        refFrame = currentFrame;
        state = INITIALIZING;
        return false;

    case INITIALIZING:
        return bootstrap(currentFrame);

    case TRACKING:
        return track(currentFrame);
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

    matcher->setImagePairInfo(qKp, tKp, qSz, tSz);
    matcher->match(qDesc, tDesc, matches);
}

}} // namespace cv::slam
