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
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videostab/global_motion.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

static Mat estimateGlobMotionLeastSquaresTranslation(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
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
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
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
    return M;
}


static Mat estimateGlobMotionLeastSquaresAffine(
        int npoints, const Point2f *points0, const Point2f *points1, float *rmse)
{
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

    return M;
}


Mat estimateGlobalMotionLeastSquares(
        const vector<Point2f> &points0, const vector<Point2f> &points1, int model, float *rmse)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresAffine };

    int npoints = static_cast<int>(points0.size());
    return impls[model](npoints, &points0[0], &points1[0], rmse);
}


Mat estimateGlobalMotionRobust(
        const vector<Point2f> &points0, const vector<Point2f> &points1, int model,
        const RansacParams &params, float *rmse, int *ninliers)
{
    CV_Assert(points0.size() == points1.size());

    typedef Mat (*Impl)(int, const Point2f*, const Point2f*, float*);
    static Impl impls[] = { estimateGlobMotionLeastSquaresTranslation,
                            estimateGlobMotionLeastSquaresTranslationAndScale,
                            estimateGlobMotionLeastSquaresAffine };

    const int npoints = static_cast<int>(points0.size());
    const int niters = static_cast<int>(ceil(log(1 - params.prob) /
                                             log(1 - pow(1 - params.eps, params.size))));

    RNG rng(0);
    vector<int> indices(params.size);
    vector<Point2f> subset0(params.size), subset1(params.size);
    vector<Point2f> subset0best(params.size), subset1best(params.size);
    Mat_<float> bestM;
    int ninliersMax = -1;
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

        Mat_<float> M = impls[model](params.size, &subset0[0], &subset1[0], 0);

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
        // compute rmse
        bestM = impls[model](params.size, &subset0best[0], &subset1best[0], rmse);
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
        bestM = impls[model](ninliersMax, &subset0[0], &subset1[0], rmse);
    }

    if (ninliers)
        *ninliers = ninliersMax;

    return bestM;
}


PyrLkRobustMotionEstimator::PyrLkRobustMotionEstimator()
    : ransacParams_(RansacParams::affine2dMotionStd())
{
    setDetector(new GoodFeaturesToTrackDetector());
    setOptFlowEstimator(new SparsePyrLkOptFlowEstimator());
    setMotionModel(AFFINE);
    setMaxRmse(0.5f);
    setMinInlierRatio(0.1f);
}


Mat PyrLkRobustMotionEstimator::estimate(const Mat &frame0, const Mat &frame1)
{
    detector_->detect(frame0, keypointsPrev_);

    pointsPrev_.resize(keypointsPrev_.size());
    for (size_t i = 0; i < keypointsPrev_.size(); ++i)
        pointsPrev_[i] = keypointsPrev_[i].pt;

    optFlowEstimator_->run(frame0, frame1, pointsPrev_, points_, status_, noArray());

    size_t npoints = points_.size();
    pointsPrevGood_.clear();
    pointsPrevGood_.reserve(npoints);
    pointsGood_.clear();
    pointsGood_.reserve(npoints);
    for (size_t i = 0; i < npoints; ++i)
    {
        if (status_[i])
        {
            pointsPrevGood_.push_back(pointsPrev_[i]);
            pointsGood_.push_back(points_[i]);
        }
    }

    float rmse;
    int ninliers;
    Mat M = estimateGlobalMotionRobust(
            pointsPrevGood_, pointsGood_, motionModel_, ransacParams_, &rmse, &ninliers);

    if (rmse > maxRmse_ || static_cast<float>(ninliers) / pointsGood_.size() < minInlierRatio_)
        M = Mat::eye(3, 3, CV_32F);

    return M;
}


Mat getMotion(int from, int to, const Mat *motions, int size)
{
    Mat M = Mat::eye(3, 3, CV_32F);
    if (to > from)
    {
        for (int i = from; i < to; ++i)
            M = at(i, motions, size) * M;
    }
    else if (from > to)
    {
        for (int i = to; i < from; ++i)
            M = at(i, motions, size) * M;
        M = M.inv();
    }
    return M;
}


Mat getMotion(int from, int to, const vector<Mat> &motions)
{
    return getMotion(from, to, &motions[0], motions.size());
}

} // namespace videostab
} // namespace cv
