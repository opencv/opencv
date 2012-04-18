/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/videostab/ring_buffer.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

// does isotropic normalization
static Mat normalizePoints(int npoints, Point2f *points)
{
    float cx = 0.f, cy = 0.f;
    for (int i = 0; i < npoints; ++i)
    {
        cx += points[i].x;
        cy += points[i].y;
    }
    cx /= npoints;
    cy /= npoints;

    float d = 0.f;
    for (int i = 0; i < npoints; ++i)
    {
        points[i].x -= cx;
        points[i].y -= cy;
        d += sqrt(sqr(points[i].x) + sqr(points[i].y));
    }
    d /= npoints;

    float s = sqrt(2.f) / d;
    for (int i = 0; i < npoints; ++i)
    {
        points[i].x *= s;
        points[i].y *= s;
    }

    Mat_<float> T = Mat::eye(3, 3, CV_32F);
    T(0,0) = T(1,1) = s;
    T(0,2) = -cx*s;
    T(1,2) = -cy*s;
    return T;
}


static Mat estimateGlobMotionLeastSquaresTranslation(
        int npoints, Point2f *points0, Point2f *points1, float *rmse)
{
    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    for (int i = 0; i < npoints; ++i)
    {
        M(0,2) += points1[i].x - points0[i].x;
        M(1,2) += points1[i].y - points0[i].y;
    }
    M(0,2) /= npoints;
    M(1,2) /= npoints;

    if (rmse)
    {
        *rmse = 0;
        for (int i = 0; i < npoints; ++i)
            *rmse += sqr(points1[i].x - points0[i].x - M(0,2)) +
                     sqr(points1[i].y - points0[i].y - M(1,2));
        *rmse = sqrt(*rmse / npoints);
    }

    return M;
}


static Mat estimateGlobMotionLeastSquaresTranslationAndScale(
        int npoints, Point2f *points0, Point2f *points1, float *rmse)
{
    Mat_<float> T0 = normalizePoints(npoints, points0);
    Mat_<float> T1 = normalizePoints(npoints, points1);

    Mat_<float> A(2*npoints, 3), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = 1; a0[2] = 0;
        a1[0] = p0.y; a1[1] = 0; a1[2] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    M(0,0) = M(1,1) = sol(0,0);
    M(0,2) = sol(1,0);
    M(1,2) = sol(2,0);

    return T1.inv() * M * T0;
}


static Mat estimateGlobMotionLeastSquaresSimilarity(
        int npoints, Point2f *points0, Point2f *points1, float *rmse)
{
    Mat_<float> T0 = normalizePoints(npoints, points0);
    Mat_<float> T1 = normalizePoints(npoints, points1);

    Mat_<float> A(2*npoints, 4), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = p0.y; a0[2] = 1; a0[3] = 0;
        a1[0] = p0.y; a1[1] = -p0.x; a1[2] = 0; a1[3] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    M(0,0) = M(1,1) = sol(0,0);
    M(0,1) = sol(1,0);
    M(1,0) = -sol(1,0);
    M(0,2) = sol(2,0);
    M(1,2) = sol(3,0);

    return T1.inv() * M * T0;
}


static Mat estimateGlobMotionLeastSquaresAffine(
        int npoints, Point2f *points0, Point2f *points1, float *rmse)
{
    Mat_<float> T0 = normalizePoints(npoints, points0);
    Mat_<float> T1 = normalizePoints(npoints, points1);

    Mat_<float> A(2*npoints, 6), b(2*npoints, 1);
    float *a0, *a1;
    Point2f p0, p1;

    for (int i = 0; i < npoints; ++i)
    {
        a0 = A[2*i];
        a1 = A[2*i+1];
        p0 = points0[i];
        p1 = points1[i];
        a0[0] = p0.x; a0[1] = p0.y; a0[2] = 1; a0[3] = a0[4] = a0[5] = 0;
        a1[0] = a1[1] = a1[2] = 0; a1[3] = p0.x; a1[4] = p0.y; a1[5] = 1;
        b(2*i,0) = p1.x;
        b(2*i+1,0) = p1.y;
    }

    Mat_<float> sol;
    solve(A, b, sol, DECOMP_SVD);

    if (rmse)
        *rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

    Mat_<float> M = Mat::eye(3, 3, CV_32F);
    for (int i = 0, k = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j, ++k)
            M(i,j) = sol(k,0);

    return T1.inv() * M * T0;
}


Mat estimateGlobalMotionLeastSquares(
        int npoints, Point2f *points0, Point2f *points1, int model, float *rmse)
{
    CV_Assert(model <= MM_AFFINE);

    typedef Mat (*Impl)(int, Point2f*, Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresSimilarity,
                            estimateGlobMotionLeastSquaresAffine };

    return impls[model](npoints, points0, points1, rmse);
}


Mat estimateGlobalMotionRobust(
        int npoints, const Point2f *points0, const Point2f *points1, int model,
        const RansacParams &params, float *rmse, int *ninliers)
{
    CV_Assert(model <= MM_AFFINE);

    const int niters = static_cast<int>(ceil(log(1 - params.prob) /
                                             log(1 - pow(1 - params.eps, params.size))));

    // current hypothesis
    vector<int> indices(params.size);
    vector<Point2f> subset0(params.size);
    vector<Point2f> subset1(params.size);

    // best hypothesis
    vector<Point2f> subset0best(params.size);
    vector<Point2f> subset1best(params.size);
    Mat_<float> bestM;
    int ninliersMax = -1;

    RNG rng(0);
    Point2f p0, p1;
    float x, y;

    for (int iter = 0; iter < niters; ++iter)
    {
        for (int i = 0; i < params.size; ++i)
        {
            bool ok = false;
            while (!ok)
            {
                ok = true;
                indices[i] = static_cast<unsigned>(rng) % npoints;
                for (int j = 0; j < i; ++j)
                    if (indices[i] == indices[j])
                        { ok = false; break; }
            }
        }
        for (int i = 0; i < params.size; ++i)
        {
            subset0[i] = points0[indices[i]];
            subset1[i] = points1[indices[i]];
        }

        Mat_<float> M = estimateGlobalMotionLeastSquares(
                params.size, &subset0[0], &subset1[0], model, 0);

        int ninliers = 0;
        for (int i = 0; i < npoints; ++i)
        {
            p0 = points0[i]; p1 = points1[i];
            x = M(0,0)*p0.x + M(0,1)*p0.y + M(0,2);
            y = M(1,0)*p0.x + M(1,1)*p0.y + M(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
                ninliers++;
        }
        if (ninliers >= ninliersMax)
        {
            bestM = M;
            ninliersMax = ninliers;
            subset0best.swap(subset0);
            subset1best.swap(subset1);
        }
    }

    if (ninliersMax < params.size)
        // compute RMSE
        bestM = estimateGlobalMotionLeastSquares(
                params.size, &subset0best[0], &subset1best[0], model, rmse);
    else
    {
        subset0.resize(ninliersMax);
        subset1.resize(ninliersMax);
        for (int i = 0, j = 0; i < npoints; ++i)
        {
            p0 = points0[i]; p1 = points1[i];
            x = bestM(0,0)*p0.x + bestM(0,1)*p0.y + bestM(0,2);
            y = bestM(1,0)*p0.x + bestM(1,1)*p0.y + bestM(1,2);
            if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
            {
                subset0[j] = p0;
                subset1[j] = p1;
                j++;
            }
        }
        bestM = estimateGlobalMotionLeastSquares(
                ninliersMax, &subset0[0], &subset1[0], model, rmse);
    }

    if (ninliers)
        *ninliers = ninliersMax;

    return bestM;
}


FromFileMotionReader::FromFileMotionReader(const string &path)
{
    file_.open(path.c_str());
    CV_Assert(file_.is_open());
}


Mat FromFileMotionReader::estimate(const Mat &/*frame0*/, const Mat &/*frame1*/, bool *ok)
{
    Mat_<float> M(3, 3);
    bool ok_;
    file_ >> M(0,0) >> M(0,1) >> M(0,2)
          >> M(1,0) >> M(1,1) >> M(1,2)
          >> M(2,0) >> M(2,1) >> M(2,2) >> ok_;
    if (ok) *ok = ok_;
    return M;
}


ToFileMotionWriter::ToFileMotionWriter(const string &path, Ptr<GlobalMotionEstimatorBase> estimator)
{
    file_.open(path.c_str());
    CV_Assert(file_.is_open());
    estimator_ = estimator;
}


Mat ToFileMotionWriter::estimate(const Mat &frame0, const Mat &frame1, bool *ok)
{
    bool ok_;
    Mat_<float> M = estimator_->estimate(frame0, frame1, &ok_);
    file_ << M(0,0) << " " << M(0,1) << " " << M(0,2) << " "
          << M(1,0) << " " << M(1,1) << " " << M(1,2) << " "
          << M(2,0) << " " << M(2,1) << " " << M(2,2) << " " << ok_ << endl;
    if (ok) *ok = ok_;
    return M;
}


PyrLkRobustMotionEstimator::PyrLkRobustMotionEstimator(MotionModel model)
{
    setDetector(new GoodFeaturesToTrackDetector());
    setOptFlowEstimator(new SparsePyrLkOptFlowEstimator());
    setMotionModel(model);
    setRansacParams(RansacParams::default2dMotion(model));
    setMinInlierRatio(0.1f);
    setGridSize(Size(0,0));
}


Mat PyrLkRobustMotionEstimator::estimate(const Mat &frame0, const Mat &frame1, bool *ok)
{
    // find keypoints

    detector_->detect(frame0, keypointsPrev_);

    // add extra keypoints

    if (gridSize_.width > 0 && gridSize_.height > 0)
    {
        float dx = static_cast<float>(frame0.cols) / (gridSize_.width + 1);
        float dy = static_cast<float>(frame0.rows) / (gridSize_.height + 1);
        for (int x = 0; x < gridSize_.width; ++x)
            for (int y = 0; y < gridSize_.height; ++y)
                keypointsPrev_.push_back(KeyPoint((x+1)*dx, (y+1)*dy, 0.f));
    }

    // extract points from keypoints

    pointsPrev_.resize(keypointsPrev_.size());
    for (size_t i = 0; i < keypointsPrev_.size(); ++i)
        pointsPrev_[i] = keypointsPrev_[i].pt;

    // find correspondences

    optFlowEstimator_->run(frame0, frame1, pointsPrev_, points_, status_, noArray());

    // leave good correspondences only

    pointsPrevGood_.clear(); pointsPrevGood_.reserve(points_.size());
    pointsGood_.clear(); pointsGood_.reserve(points_.size());

    for (size_t i = 0; i < points_.size(); ++i)
    {
        if (status_[i])
        {
            pointsPrevGood_.push_back(pointsPrev_[i]);
            pointsGood_.push_back(points_[i]);
        }
    }

    size_t npoints = pointsGood_.size();

    // find motion

    int ninliers = 0;
    Mat_<float> M;

    if (motionModel_ != MM_HOMOGRAPHY)
        M = estimateGlobalMotionRobust(
                npoints, &pointsPrevGood_[0], &pointsGood_[0], motionModel_,
                ransacParams_, 0, &ninliers);
    else
    {
        vector<uchar> mask;
        M = findHomography(pointsPrevGood_, pointsGood_, mask, CV_RANSAC, ransacParams_.thresh);
        for (size_t i  = 0; i < npoints; ++i)
            if (mask[i]) ninliers++;
    }

    // check if we're confident enough in estimated motion

    if (ok) *ok = true;
    if (static_cast<float>(ninliers) / npoints < minInlierRatio_)
    {
        M = Mat::eye(3, 3, CV_32F);
        if (ok) *ok = false;
    }

    return M;
}


#if HAVE_OPENCV_GPU
PyrLkRobustMotionEstimatorGpu::PyrLkRobustMotionEstimatorGpu(MotionModel model)
{
    CV_Assert(gpu::getCudaEnabledDeviceCount() > 0);
    setMotionModel(model);
    setRansacParams(RansacParams::default2dMotion(model));
    setMinInlierRatio(0.1f);
}


Mat PyrLkRobustMotionEstimatorGpu::estimate(const Mat &frame0, const Mat &frame1, bool *ok)
{
    frame0_.upload(frame0);
    frame1_.upload(frame1);
    return estimate(frame0_, frame1_, ok);
}


Mat PyrLkRobustMotionEstimatorGpu::estimate(const gpu::GpuMat &frame0, const gpu::GpuMat &frame1, bool *ok)
{
    // convert frame to gray if it's color

    gpu::GpuMat grayFrame0;
    if (frame0.channels() == 1)
        grayFrame0 = frame0;
    else
    {
        gpu::cvtColor(frame0_, grayFrame0_, CV_BGR2GRAY);
        grayFrame0 = grayFrame0_;
    }

    // find keypoints

    detector_(grayFrame0, pointsPrev_);

    // find correspondences

    optFlowEstimator_.run(frame0, frame1, pointsPrev_, points_, status_);    

    // leave good correspondences only

    gpu::compactPoints(pointsPrev_, points_, status_);

    pointsPrev_.download(hostPointsPrev_);
    points_.download(hostPoints_);

    int npoints = hostPointsPrev_.cols;

    // find motion

    int ninliers = 0;
    Mat_<float> M;

    if (motionModel_ != MM_HOMOGRAPHY)
        M = estimateGlobalMotionRobust(
                npoints, hostPointsPrev_.ptr<Point2f>(0), hostPoints_.ptr<Point2f>(), motionModel_,
                ransacParams_, 0, &ninliers);
    else
    {
        vector<uchar> mask;
        M = findHomography(hostPointsPrev_, hostPoints_, mask, CV_RANSAC, ransacParams_.thresh);
        for (int i  = 0; i < npoints; ++i)
            if (mask[i]) ninliers++;
    }

    // check if we're confident enough in estimated motion

    if (ok) *ok = true;
    if (static_cast<float>(ninliers) / npoints < minInlierRatio_)
    {
        M = Mat::eye(3, 3, CV_32F);
        if (ok) *ok = false;
    }

    return M;
}
#endif // #if HAVE_OPENCV_GPU


Mat getMotion(int from, int to, const vector<Mat> &motions)
{
    Mat M = Mat::eye(3, 3, CV_32F);
    if (to > from)
    {
        for (int i = from; i < to; ++i)
            M = at(i, motions) * M;
    }
    else if (from > to)
    {
        for (int i = to; i < from; ++i)
            M = at(i, motions) * M;
        M = M.inv();
    }
    return M;
}

} // namespace videostab
} // namespace cv
