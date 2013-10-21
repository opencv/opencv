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
#include "opencv2/videostab/motion_stabilizing.hpp"
#include "opencv2/videostab/global_motion.hpp"
#include "opencv2/videostab/ring_buffer.hpp"
#include "clp.hpp"

namespace cv
{
namespace videostab
{

void MotionStabilizationPipeline::stabilize(
        int size, const std::vector<Mat> &motions, std::pair<int,int> range, Mat *stabilizationMotions)
{
    std::vector<Mat> updatedMotions(motions.size());
    for (size_t i = 0; i < motions.size(); ++i)
        updatedMotions[i] = motions[i].clone();

    std::vector<Mat> stabilizationMotions_(size);

    for (int i = 0; i < size; ++i)
        stabilizationMotions[i] = Mat::eye(3, 3, CV_32F);

    for (size_t i = 0; i < stabilizers_.size(); ++i)
    {
        stabilizers_[i]->stabilize(size, updatedMotions, range, &stabilizationMotions_[0]);

        for (int k = 0; k < size; ++k)
            stabilizationMotions[k] = stabilizationMotions_[k] * stabilizationMotions[k];

        for (int j = 0; j + 1 < size; ++j)
        {
            Mat S0 = stabilizationMotions[j];
            Mat S1 = stabilizationMotions[j+1];
            at(j, updatedMotions) = S1 * at(j, updatedMotions) * S0.inv();
        }
    }
}


void MotionFilterBase::stabilize(
        int size, const std::vector<Mat> &motions, std::pair<int,int> range, Mat *stabilizationMotions)
{
    for (int i = 0; i < size; ++i)
        stabilizationMotions[i] = stabilize(i, motions, range);
}


void GaussianMotionFilter::setParams(int _radius, float _stdev)
{
    radius_ = _radius;
    stdev_ = _stdev > 0.f ? _stdev : std::sqrt(static_cast<float>(_radius));

    float sum = 0;
    weight_.resize(2*radius_ + 1);
    for (int i = -radius_; i <= radius_; ++i)
        sum += weight_[radius_ + i] = std::exp(-i*i/(stdev_*stdev_));
    for (int i = -radius_; i <= radius_; ++i)
        weight_[radius_ + i] /= sum;
}


Mat GaussianMotionFilter::stabilize(int idx, const std::vector<Mat> &motions, std::pair<int,int> range)
{
    const Mat &cur = at(idx, motions);
    Mat res = Mat::zeros(cur.size(), cur.type());
    float sum = 0.f;
    int iMin = std::max(idx - radius_, range.first);
    int iMax = std::min(idx + radius_, range.second);
    for (int i = iMin; i <= iMax; ++i)
    {
        res += weight_[radius_ + i - idx] * getMotion(idx, i, motions);
        sum += weight_[radius_ + i - idx];
    }
    return sum > 0.f ? res / sum : Mat::eye(cur.size(), cur.type());
}


LpMotionStabilizer::LpMotionStabilizer(MotionModel model)
{
    setMotionModel(model);
    setFrameSize(Size(0,0));
    setTrimRatio(0.1f);
    setWeight1(1);
    setWeight2(10);
    setWeight3(100);
    setWeight4(100);
}


#ifndef HAVE_CLP

void LpMotionStabilizer::stabilize(int, const std::vector<Mat>&, std::pair<int,int>, Mat*)
{
    CV_Error(Error::StsError, "The library is built without Clp support");
}

#else

void LpMotionStabilizer::stabilize(
        int size, const std::vector<Mat> &motions, std::pair<int,int> /*range*/, Mat *stabilizationMotions)
{
    CV_Assert(model_ <= MM_AFFINE);

    int N = size;
    const std::vector<Mat> &M = motions;
    Mat *S = stabilizationMotions;

    double w = frameSize_.width, h = frameSize_.height;
    double tw = w * trimRatio_, th = h * trimRatio_;

    int ncols = 4*N + 6*(N-1) + 6*(N-2) + 6*(N-3);
    int nrows = 8*N + 2*6*(N-1) + 2*6*(N-2) + 2*6*(N-3);

    rows_.clear();
    cols_.clear();
    elems_.clear();

    obj_.assign(ncols, 0);
    collb_.assign(ncols, -INF);
    colub_.assign(ncols, INF);
    int c = 4*N;

    // for each slack variable e[t] (error bound)
    for (int t = 0; t < N-1; ++t, c += 6)
    {
        // e[t](0,0)
        obj_[c] = w4_*w1_;
        collb_[c] = 0;

        // e[t](0,1)
        obj_[c+1] = w4_*w1_;
        collb_[c+1] = 0;

        // e[t](0,2)
        obj_[c+2] = w1_;
        collb_[c+2] = 0;

        // e[t](1,0)
        obj_[c+3] = w4_*w1_;
        collb_[c+3] = 0;

        // e[t](1,1)
        obj_[c+4] = w4_*w1_;
        collb_[c+4] = 0;

        // e[t](1,2)
        obj_[c+5] = w1_;
        collb_[c+5] = 0;
    }
    for (int t = 0; t < N-2; ++t, c += 6)
    {
        // e[t](0,0)
        obj_[c] = w4_*w2_;
        collb_[c] = 0;

        // e[t](0,1)
        obj_[c+1] = w4_*w2_;
        collb_[c+1] = 0;

        // e[t](0,2)
        obj_[c+2] = w2_;
        collb_[c+2] = 0;

        // e[t](1,0)
        obj_[c+3] = w4_*w2_;
        collb_[c+3] = 0;

        // e[t](1,1)
        obj_[c+4] = w4_*w2_;
        collb_[c+4] = 0;

        // e[t](1,2)
        obj_[c+5] = w2_;
        collb_[c+5] = 0;
    }
    for (int t = 0; t < N-3; ++t, c += 6)
    {
        // e[t](0,0)
        obj_[c] = w4_*w3_;
        collb_[c] = 0;

        // e[t](0,1)
        obj_[c+1] = w4_*w3_;
        collb_[c+1] = 0;

        // e[t](0,2)
        obj_[c+2] = w3_;
        collb_[c+2] = 0;

        // e[t](1,0)
        obj_[c+3] = w4_*w3_;
        collb_[c+3] = 0;

        // e[t](1,1)
        obj_[c+4] = w4_*w3_;
        collb_[c+4] = 0;

        // e[t](1,2)
        obj_[c+5] = w3_;
        collb_[c+5] = 0;
    }

    elems_.clear();
    rowlb_.assign(nrows, -INF);
    rowub_.assign(nrows, INF);

    int r = 0;

    // frame corners
    const Point2d pt[] = {Point2d(0,0), Point2d(w,0), Point2d(w,h), Point2d(0,h)};

    // for each frame
    for (int t = 0; t < N; ++t)
    {
        c = 4*t;

        // for each frame corner
        for (int i = 0; i < 4; ++i, r += 2)
        {
            set(r, c, pt[i].x); set(r, c+1, pt[i].y); set(r, c+2, 1);
            set(r+1, c, pt[i].y); set(r+1, c+1, -pt[i].x); set(r+1, c+3, 1);
            rowlb_[r] = pt[i].x-tw;
            rowub_[r] = pt[i].x+tw;
            rowlb_[r+1] = pt[i].y-th;
            rowub_[r+1] = pt[i].y+th;
        }
    }

    // for each S[t+1]M[t] - S[t] - e[t] <= 0 condition
    for (int t = 0; t < N-1; ++t, r += 6)
    {
        Mat_<float> M0 = at(t,M);

        c = 4*t;
        set(r, c, -1);
        set(r+1, c+1, -1);
        set(r+2, c+2, -1);
        set(r+3, c+1, 1);
        set(r+4, c, -1);
        set(r+5, c+3, -1);

        c = 4*(t+1);
        set(r, c, M0(0,0)); set(r, c+1, M0(1,0));
        set(r+1, c, M0(0,1)); set(r+1, c+1, M0(1,1));
        set(r+2, c, M0(0,2)); set(r+2, c+1, M0(1,2)); set(r+2, c+2, 1);
        set(r+3, c, M0(1,0)); set(r+3, c+1, -M0(0,0));
        set(r+4, c, M0(1,1)); set(r+4, c+1, -M0(0,1));
        set(r+5, c, M0(1,2)); set(r+5, c+1, -M0(0,2)); set(r+5, c+3, 1);

        c = 4*N + 6*t;
        for (int i = 0; i < 6; ++i)
            set(r+i, c+i, -1);

        rowub_[r] = 0;
        rowub_[r+1] = 0;
        rowub_[r+2] = 0;
        rowub_[r+3] = 0;
        rowub_[r+4] = 0;
        rowub_[r+5] = 0;
    }

    // for each 0 <= S[t+1]M[t] - S[t] + e[t] condition
    for (int t = 0; t < N-1; ++t, r += 6)
    {
        Mat_<float> M0 = at(t,M);

        c = 4*t;
        set(r, c, -1);
        set(r+1, c+1, -1);
        set(r+2, c+2, -1);
        set(r+3, c+1, 1);
        set(r+4, c, -1);
        set(r+5, c+3, -1);

        c = 4*(t+1);
        set(r, c, M0(0,0)); set(r, c+1, M0(1,0));
        set(r+1, c, M0(0,1)); set(r+1, c+1, M0(1,1));
        set(r+2, c, M0(0,2)); set(r+2, c+1, M0(1,2)); set(r+2, c+2, 1);
        set(r+3, c, M0(1,0)); set(r+3, c+1, -M0(0,0));
        set(r+4, c, M0(1,1)); set(r+4, c+1, -M0(0,1));
        set(r+5, c, M0(1,2)); set(r+5, c+1, -M0(0,2)); set(r+5, c+3, 1);

        c = 4*N + 6*t;
        for (int i = 0; i < 6; ++i)
            set(r+i, c+i, 1);

        rowlb_[r] = 0;
        rowlb_[r+1] = 0;
        rowlb_[r+2] = 0;
        rowlb_[r+3] = 0;
        rowlb_[r+4] = 0;
        rowlb_[r+5] = 0;
    }

    // for each S[t+2]M[t+1] - S[t+1]*(I+M[t]) + S[t] - e[t] <= 0 condition
    for (int t = 0; t < N-2; ++t, r += 6)
    {
        Mat_<float> M0 = at(t,M), M1 = at(t+1,M);

        c = 4*t;
        set(r, c, 1);
        set(r+1, c+1, 1);
        set(r+2, c+2, 1);
        set(r+3, c+1, -1);
        set(r+4, c, 1);
        set(r+5, c+3, 1);

        c = 4*(t+1);
        set(r, c, -M0(0,0)-1); set(r, c+1, -M0(1,0));
        set(r+1, c, -M0(0,1)); set(r+1, c+1, -M0(1,1)-1);
        set(r+2, c, -M0(0,2)); set(r+2, c+1, -M0(1,2)); set(r+2, c+2, -2);
        set(r+3, c, -M0(1,0)); set(r+3, c+1, M0(0,0)+1);
        set(r+4, c, -M0(1,1)-1); set(r+4, c+1, M0(0,1));
        set(r+5, c, -M0(1,2)); set(r+5, c+1, M0(0,2)); set(r+5, c+3, -2);

        c = 4*(t+2);
        set(r, c, M1(0,0)); set(r, c+1, M1(1,0));
        set(r+1, c, M1(0,1)); set(r+1, c+1, M1(1,1));
        set(r+2, c, M1(0,2)); set(r+2, c+1, M1(1,2)); set(r+2, c+2, 1);
        set(r+3, c, M1(1,0)); set(r+3, c+1, -M1(0,0));
        set(r+4, c, M1(1,1)); set(r+4, c+1, -M1(0,1));
        set(r+5, c, M1(1,2)); set(r+5, c+1, -M1(0,2)); set(r+5, c+3, 1);

        c = 4*N + 6*(N-1) + 6*t;
        for (int i = 0; i < 6; ++i)
            set(r+i, c+i, -1);

        rowub_[r] = 0;
        rowub_[r+1] = 0;
        rowub_[r+2] = 0;
        rowub_[r+3] = 0;
        rowub_[r+4] = 0;
        rowub_[r+5] = 0;
    }

    // for each 0 <= S[t+2]M[t+1]] - S[t+1]*(I+M[t]) + S[t] + e[t] condition
    for (int t = 0; t < N-2; ++t, r += 6)
    {
        Mat_<float> M0 = at(t,M), M1 = at(t+1,M);

        c = 4*t;
        set(r, c, 1);
        set(r+1, c+1, 1);
        set(r+2, c+2, 1);
        set(r+3, c+1, -1);
        set(r+4, c, 1);
        set(r+5, c+3, 1);

        c = 4*(t+1);
        set(r, c, -M0(0,0)-1); set(r, c+1, -M0(1,0));
        set(r+1, c, -M0(0,1)); set(r+1, c+1, -M0(1,1)-1);
        set(r+2, c, -M0(0,2)); set(r+2, c+1, -M0(1,2)); set(r+2, c+2, -2);
        set(r+3, c, -M0(1,0)); set(r+3, c+1, M0(0,0)+1);
        set(r+4, c, -M0(1,1)-1); set(r+4, c+1, M0(0,1));
        set(r+5, c, -M0(1,2)); set(r+5, c+1, M0(0,2)); set(r+5, c+3, -2);

        c = 4*(t+2);
        set(r, c, M1(0,0)); set(r, c+1, M1(1,0));
        set(r+1, c, M1(0,1)); set(r+1, c+1, M1(1,1));
        set(r+2, c, M1(0,2)); set(r+2, c+1, M1(1,2)); set(r+2, c+2, 1);
        set(r+3, c, M1(1,0)); set(r+3, c+1, -M1(0,0));
        set(r+4, c, M1(1,1)); set(r+4, c+1, -M1(0,1));
        set(r+5, c, M1(1,2)); set(r+5, c+1, -M1(0,2)); set(r+5, c+3, 1);

        c = 4*N + 6*(N-1) + 6*t;
        for (int i = 0; i < 6; ++i)
            set(r+i, c+i, 1);

        rowlb_[r] = 0;
        rowlb_[r+1] = 0;
        rowlb_[r+2] = 0;
        rowlb_[r+3] = 0;
        rowlb_[r+4] = 0;
        rowlb_[r+5] = 0;
    }

    // for each S[t+3]M[t+2] - S[t+2]*(I+2M[t+1]) + S[t+1]*(2*I+M[t]) - S[t] - e[t] <= 0 condition
    for (int t = 0; t < N-3; ++t, r += 6)
    {
        Mat_<float> M0 = at(t,M), M1 = at(t+1,M), M2 = at(t+2,M);

        c = 4*t;
        set(r, c, -1);
        set(r+1, c+1, -1);
        set(r+2, c+2, -1);
        set(r+3, c+1, 1);
        set(r+4, c, -1);
        set(r+5, c+3, -1);

        c = 4*(t+1);
        set(r, c, M0(0,0)+2); set(r, c+1, M0(1,0));
        set(r+1, c, M0(0,1)); set(r+1, c+1, M0(1,1)+2);
        set(r+2, c, M0(0,2)); set(r+2, c+1, M0(1,2)); set(r+2, c+2, 3);
        set(r+3, c, M0(1,0)); set(r+3, c+1, -M0(0,0)-2);
        set(r+4, c, M0(1,1)+2); set(r+4, c+1, -M0(0,1));
        set(r+5, c, M0(1,2)); set(r+5, c+1, -M0(0,2)); set(r+5, c+3, 3);

        c = 4*(t+2);
        set(r, c, -2*M1(0,0)-1); set(r, c+1, -2*M1(1,0));
        set(r+1, c, -2*M1(0,1)); set(r+1, c+1, -2*M1(1,1)-1);
        set(r+2, c, -2*M1(0,2)); set(r+2, c+1, -2*M1(1,2)); set(r+2, c+2, -3);
        set(r+3, c, -2*M1(1,0)); set(r+3, c+1, 2*M1(0,0)+1);
        set(r+4, c, -2*M1(1,1)-1); set(r+4, c+1, 2*M1(0,1));
        set(r+5, c, -2*M1(1,2)); set(r+5, c+1, 2*M1(0,2)); set(r+5, c+3, -3);

        c = 4*(t+3);
        set(r, c, M2(0,0)); set(r, c+1, M2(1,0));
        set(r+1, c, M2(0,1)); set(r+1, c+1, M2(1,1));
        set(r+2, c, M2(0,2)); set(r+2, c+1, M2(1,2)); set(r+2, c+2, 1);
        set(r+3, c, M2(1,0)); set(r+3, c+1, -M2(0,0));
        set(r+4, c, M2(1,1)); set(r+4, c+1, -M2(0,1));
        set(r+5, c, M2(1,2)); set(r+5, c+1, -M2(0,2)); set(r+5, c+3, 1);

        c = 4*N + 6*(N-1) + 6*(N-2) + 6*t;
        for (int i = 0; i < 6; ++i)
            set(r+i, c+i, -1);

        rowub_[r] = 0;
        rowub_[r+1] = 0;
        rowub_[r+2] = 0;
        rowub_[r+3] = 0;
        rowub_[r+4] = 0;
        rowub_[r+5] = 0;
    }

    // for each 0 <= S[t+3]M[t+2] - S[t+2]*(I+2M[t+1]) + S[t+1]*(2*I+M[t]) + e[t] condition
    for (int t = 0; t < N-3; ++t, r += 6)
    {
        Mat_<float> M0 = at(t,M), M1 = at(t+1,M), M2 = at(t+2,M);

        c = 4*t;
        set(r, c, -1);
        set(r+1, c+1, -1);
        set(r+2, c+2, -1);
        set(r+3, c+1, 1);
        set(r+4, c, -1);
        set(r+5, c+3, -1);

        c = 4*(t+1);
        set(r, c, M0(0,0)+2); set(r, c+1, M0(1,0));
        set(r+1, c, M0(0,1)); set(r+1, c+1, M0(1,1)+2);
        set(r+2, c, M0(0,2)); set(r+2, c+1, M0(1,2)); set(r+2, c+2, 3);
        set(r+3, c, M0(1,0)); set(r+3, c+1, -M0(0,0)-2);
        set(r+4, c, M0(1,1)+2); set(r+4, c+1, -M0(0,1));
        set(r+5, c, M0(1,2)); set(r+5, c+1, -M0(0,2)); set(r+5, c+3, 3);

        c = 4*(t+2);
        set(r, c, -2*M1(0,0)-1); set(r, c+1, -2*M1(1,0));
        set(r+1, c, -2*M1(0,1)); set(r+1, c+1, -2*M1(1,1)-1);
        set(r+2, c, -2*M1(0,2)); set(r+2, c+1, -2*M1(1,2)); set(r+2, c+2, -3);
        set(r+3, c, -2*M1(1,0)); set(r+3, c+1, 2*M1(0,0)+1);
        set(r+4, c, -2*M1(1,1)-1); set(r+4, c+1, 2*M1(0,1));
        set(r+5, c, -2*M1(1,2)); set(r+5, c+1, 2*M1(0,2)); set(r+5, c+3, -3);

        c = 4*(t+3);
        set(r, c, M2(0,0)); set(r, c+1, M2(1,0));
        set(r+1, c, M2(0,1)); set(r+1, c+1, M2(1,1));
        set(r+2, c, M2(0,2)); set(r+2, c+1, M2(1,2)); set(r+2, c+2, 1);
        set(r+3, c, M2(1,0)); set(r+3, c+1, -M2(0,0));
        set(r+4, c, M2(1,1)); set(r+4, c+1, -M2(0,1));
        set(r+5, c, M2(1,2)); set(r+5, c+1, -M2(0,2)); set(r+5, c+3, 1);

        c = 4*N + 6*(N-1) + 6*(N-2) + 6*t;
        for (int i = 0; i < 6; ++i)
            set(r+i, c+i, 1);

        rowlb_[r] = 0;
        rowlb_[r+1] = 0;
        rowlb_[r+2] = 0;
        rowlb_[r+3] = 0;
        rowlb_[r+4] = 0;
        rowlb_[r+5] = 0;
    }

    // solve

    CoinPackedMatrix A(true, &rows_[0], &cols_[0], &elems_[0], elems_.size());
    A.setDimensions(nrows, ncols);

    ClpSimplex model(false);
    model.loadProblem(A, &collb_[0], &colub_[0], &obj_[0], &rowlb_[0], &rowub_[0]);

    ClpDualRowSteepest dualSteep(1);
    model.setDualRowPivotAlgorithm(dualSteep);

    ClpPrimalColumnSteepest primalSteep(1);
    model.setPrimalColumnPivotAlgorithm(primalSteep);

    model.scaling(1);

    ClpPresolve presolveInfo;
    Ptr<ClpSimplex> presolvedModel(presolveInfo.presolvedModel(model));

    if (presolvedModel)
    {
        presolvedModel->dual();
        presolveInfo.postsolve(true);
        model.checkSolution();
        model.primal(1);
    }
    else
    {
        model.dual();
        model.checkSolution();
        model.primal(1);
    }

    // save results

    const double *sol = model.getColSolution();
    c = 0;

    for (int t = 0; t < N; ++t, c += 4)
    {
        Mat_<float> S0 = Mat::eye(3, 3, CV_32F);
        S0(1,1) = S0(0,0) = sol[c];
        S0(0,1) = sol[c+1];
        S0(1,0) = -sol[c+1];
        S0(0,2) = sol[c+2];
        S0(1,2) = sol[c+3];
        S[t] = S0;
    }
}
#endif // #ifndef HAVE_CLP


static inline int areaSign(Point2f a, Point2f b, Point2f c)
{
    double area = (b-a).cross(c-a);
    if (area < -1e-5) return -1;
    if (area > 1e-5) return 1;
    return 0;
}


static inline bool segmentsIntersect(Point2f a, Point2f b, Point2f c, Point2f d)
{
    return areaSign(a,b,c) * areaSign(a,b,d) < 0 &&
           areaSign(c,d,a) * areaSign(c,d,b) < 0;
}


// Checks if rect a (with sides parallel to axis) is inside rect b (arbitrary).
// Rects must be passed in the [(0,0), (w,0), (w,h), (0,h)] order.
static inline bool isRectInside(const Point2f a[4], const Point2f b[4])
{
    for (int i = 0; i < 4; ++i)
        if (b[i].x > a[0].x && b[i].x < a[2].x && b[i].y > a[0].y && b[i].y < a[2].y)
            return false;
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
        if (segmentsIntersect(a[i], a[(i+1)%4], b[j], b[(j+1)%4]))
            return false;
    return true;
}


static inline bool isGoodMotion(const float M[], float w, float h, float dx, float dy)
{
    Point2f pt[4] = {Point2f(0,0), Point2f(w,0), Point2f(w,h), Point2f(0,h)};
    Point2f Mpt[4];
    float z;

    for (int i = 0; i < 4; ++i)
    {
        Mpt[i].x = M[0]*pt[i].x + M[1]*pt[i].y + M[2];
        Mpt[i].y = M[3]*pt[i].x + M[4]*pt[i].y + M[5];
        z = M[6]*pt[i].x + M[7]*pt[i].y + M[8];
        Mpt[i].x /= z;
        Mpt[i].y /= z;
    }

    pt[0] = Point2f(dx, dy);
    pt[1] = Point2f(w - dx, dy);
    pt[2] = Point2f(w - dx, h - dy);
    pt[3] = Point2f(dx, h - dy);

    return isRectInside(pt, Mpt);
}


static inline void relaxMotion(const float M[], float t, float res[])
{
    res[0] = M[0]*(1.f-t) + t;
    res[1] = M[1]*(1.f-t);
    res[2] = M[2]*(1.f-t);
    res[3] = M[3]*(1.f-t);
    res[4] = M[4]*(1.f-t) + t;
    res[5] = M[5]*(1.f-t);
    res[6] = M[6]*(1.f-t);
    res[7] = M[7]*(1.f-t);
    res[8] = M[8]*(1.f-t) + t;
}


Mat ensureInclusionConstraint(const Mat &M, Size size, float trimRatio)
{
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    const float w = static_cast<float>(size.width);
    const float h = static_cast<float>(size.height);
    const float dx = floor(w * trimRatio);
    const float dy = floor(h * trimRatio);
    const float srcM[] =
            {M.at<float>(0,0), M.at<float>(0,1), M.at<float>(0,2),
             M.at<float>(1,0), M.at<float>(1,1), M.at<float>(1,2),
             M.at<float>(2,0), M.at<float>(2,1), M.at<float>(2,2)};

    float curM[9];
    float t = 0;
    relaxMotion(srcM, t, curM);
    if (isGoodMotion(curM, w, h, dx, dy))
        return M;

    float l = 0, r = 1;
    while (r - l > 1e-3f)
    {
        t = (l + r) * 0.5f;
        relaxMotion(srcM, t, curM);
        if (isGoodMotion(curM, w, h, dx, dy))
            r = t;
        else
            l = t;
    }

    return (1 - r) * M + r * Mat::eye(3, 3, CV_32F);
}


// TODO can be estimated for O(1) time
float estimateOptimalTrimRatio(const Mat &M, Size size)
{
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    const float w = static_cast<float>(size.width);
    const float h = static_cast<float>(size.height);
    Mat_<float> M_(M);

    Point2f pt[4] = {Point2f(0,0), Point2f(w,0), Point2f(w,h), Point2f(0,h)};
    Point2f Mpt[4];
    float z;

    for (int i = 0; i < 4; ++i)
    {
        Mpt[i].x = M_(0,0)*pt[i].x + M_(0,1)*pt[i].y + M_(0,2);
        Mpt[i].y = M_(1,0)*pt[i].x + M_(1,1)*pt[i].y + M_(1,2);
        z = M_(2,0)*pt[i].x + M_(2,1)*pt[i].y + M_(2,2);
        Mpt[i].x /= z;
        Mpt[i].y /= z;
    }

    float l = 0, r = 0.5f;
    while (r - l > 1e-3f)
    {
        float t = (l + r) * 0.5f;
        float dx = floor(w * t);
        float dy = floor(h * t);
        pt[0] = Point2f(dx, dy);
        pt[1] = Point2f(w - dx, dy);
        pt[2] = Point2f(w - dx, h - dy);
        pt[3] = Point2f(dx, h - dy);
        if (isRectInside(pt, Mpt))
            r = t;
        else
            l = t;
    }

    return r;
}

} // namespace videostab
} // namespace cv
