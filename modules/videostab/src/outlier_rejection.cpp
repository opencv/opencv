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
#include "opencv2/core/core.hpp"
#include "opencv2/videostab/outlier_rejection.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

void NullOutlierRejector::process(
        Size /*frameSize*/, InputArray points0, InputArray points1, OutputArray mask)
{
    CV_Assert(points0.type() == points1.type());
    CV_Assert(points0.getMat().checkVector(2) == points1.getMat().checkVector(2));

    int npoints = points0.getMat().checkVector(2);
    mask.create(1, npoints, CV_8U);
    Mat mask_ = mask.getMat();
    mask_.setTo(1);
}

TranslationBasedLocalOutlierRejector::TranslationBasedLocalOutlierRejector()
{
    setCellSize(Size(50, 50));
    setRansacParams(RansacParams::default2dMotion(MM_TRANSLATION));
}


void TranslationBasedLocalOutlierRejector::process(
        Size frameSize, InputArray points0, InputArray points1, OutputArray mask)
{
    CV_Assert(points0.type() == points1.type());
    CV_Assert(points0.getMat().checkVector(2) == points1.getMat().checkVector(2));

    int npoints = points0.getMat().checkVector(2);

    const Point2f* points0_ = points0.getMat().ptr<Point2f>();
    const Point2f* points1_ = points1.getMat().ptr<Point2f>();

    mask.create(1, npoints, CV_8U);
    uchar* mask_ = mask.getMat().ptr<uchar>();

    Size ncells((frameSize.width + cellSize_.width - 1) / cellSize_.width,
                (frameSize.height + cellSize_.height - 1) / cellSize_.height);

    int cx, cy;

    // fill grid cells

    grid_.assign(ncells.area(), Cell());

    for (int i = 0; i < npoints; ++i)
    {
        cx = std::min(cvRound(points0_[i].x / cellSize_.width), ncells.width - 1);
        cy = std::min(cvRound(points0_[i].y / cellSize_.height), ncells.height - 1);
        grid_[cy * ncells.width + cx].push_back(i);
    }

    // process each cell

    RNG rng(0);
    int niters = ransacParams_.niters();
    int ninliers, ninliersMax;
    vector<int> inliers;
    float dx, dy, dxBest, dyBest;
    float x1, y1;
    int idx;

    for (size_t ci = 0; ci < grid_.size(); ++ci)
    {
        // estimate translation model at the current cell using RANSAC

        const Cell &cell = grid_[ci];
        ninliersMax = 0;
        dxBest = dyBest = 0.f;

        // find the best hypothesis

        if (!cell.empty())
        {
            for (int iter = 0; iter < niters; ++iter)
            {
                idx = cell[static_cast<unsigned>(rng) % cell.size()];
                dx = points1_[idx].x - points0_[idx].x;
                dy = points1_[idx].y - points0_[idx].y;

                ninliers = 0;
                for (size_t i = 0; i < cell.size(); ++i)
                {
                    x1 = points0_[cell[i]].x + dx;
                    y1 = points0_[cell[i]].y + dy;
                    if (sqr(x1 - points1_[cell[i]].x) + sqr(y1 - points1_[cell[i]].y) <
                        sqr(ransacParams_.thresh))
                    {
                        ninliers++;
                    }
                }

                if (ninliers > ninliersMax)
                {
                    ninliersMax = ninliers;
                    dxBest = dx;
                    dyBest = dy;
                }
            }
        }

        // get the best hypothesis inliers

        ninliers = 0;
        inliers.resize(ninliersMax);
        for (size_t i = 0; i < cell.size(); ++i)
        {
            x1 = points0_[cell[i]].x + dxBest;
            y1 = points0_[cell[i]].y + dyBest;
            if (sqr(x1 - points1_[cell[i]].x) + sqr(y1 - points1_[cell[i]].y) <
                sqr(ransacParams_.thresh))
            {
                inliers[ninliers++] = cell[i];
            }
        }

        // refine the best hypothesis

        dxBest = dyBest = 0.f;
        for (size_t i = 0; i < inliers.size(); ++i)
        {
            dxBest += points1_[inliers[i]].x - points0_[inliers[i]].x;
            dyBest += points1_[inliers[i]].y - points0_[inliers[i]].y;
        }
        if (!inliers.empty())
        {
            dxBest /= inliers.size();
            dyBest /= inliers.size();
        }

        // set mask elements for refined model inliers

        for (size_t i = 0; i < cell.size(); ++i)
        {
            x1 = points0_[cell[i]].x + dxBest;
            y1 = points0_[cell[i]].y + dyBest;
            if (sqr(x1 - points1_[cell[i]].x) + sqr(y1 - points1_[cell[i]].y) <
                sqr(ransacParams_.thresh))
            {
                mask_[cell[i]] = 1;
            }
            else
            {
                mask_[cell[i]] = 0;
            }
        }
    }
}

} // namespace videostab
} // namespace cv
