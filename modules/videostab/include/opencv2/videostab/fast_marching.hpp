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

#ifndef __OPENCV_VIDEOSTAB_FAST_MARCHING_HPP__
#define __OPENCV_VIDEOSTAB_FAST_MARCHING_HPP__

#include <cmath>
#include <queue>
#include <algorithm>
#include "opencv2/core/core.hpp"

namespace cv
{
namespace videostab
{

// See http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf
class CV_EXPORTS FastMarchingMethod
{
public:
    FastMarchingMethod() : inf_(1e6f) {}

    template <typename Inpaint>
    Inpaint run(const Mat &mask, Inpaint inpaint);

    Mat distanceMap() const { return dist_; }

private:
    enum { INSIDE = 0, BAND = 1, KNOWN = 255 };

    struct DXY
    {
        float dist;
        int x, y;

        DXY() : dist(0), x(0), y(0) {}
        DXY(float _dist, int _x, int _y) : dist(_dist), x(_x), y(_y) {}
        bool operator <(const DXY &dxy) const { return dist < dxy.dist; }
    };

    float solve(int x1, int y1, int x2, int y2) const;
    int& indexOf(const DXY &dxy) { return index_(dxy.y, dxy.x); }

    void heapUp(int idx);
    void heapDown(int idx);
    void heapAdd(const DXY &dxy);
    void heapRemoveMin();

    float inf_;

    cv::Mat_<uchar> flag_; // flag map
    cv::Mat_<float> dist_; // distance map

    cv::Mat_<int> index_; // index of point in the narrow band
    std::vector<DXY> narrowBand_; // narrow band heap
    int size_; // narrow band size
};

} // namespace videostab
} // namespace cv

#include "fast_marching_inl.hpp"

#endif
