/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#ifndef __OPENCV_STITCHING_UTIL_INL_HPP__
#define __OPENCV_STITCHING_UTIL_INL_HPP__

#include <queue>
#include "util.hpp" // Make your IDE see declarations

template <typename B>
B Graph::forEach(B body) const
{
    for (int i = 0; i < numVertices(); ++i)
    {
        std::list<GraphEdge>::const_iterator edge = edges_[i].begin();
        for (; edge != edges_[i].end(); ++edge)
            body(*edge);
    }
    return body;
}


template <typename B>
B Graph::walkBreadthFirst(int from, B body) const
{
    std::vector<bool> was(numVertices(), false);
    std::queue<int> vertices;

    was[from] = true;
    vertices.push(from);

    while (!vertices.empty())
    {
        int vertex = vertices.front();
        vertices.pop();

        std::list<GraphEdge>::const_iterator edge = edges_[vertex].begin();
        for (; edge != edges_[vertex].end(); ++edge)
        {
            if (!was[edge->to])
            {
                body(*edge);
                was[edge->to] = true;
                vertices.push(edge->to);
            }
        }
    }

    return body;
}


//////////////////////////////////////////////////////////////////////////////
// Some auxiliary math functions

static inline
float normL2(const cv::Point3f& a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}


static inline
float normL2(const cv::Point3f& a, const cv::Point3f& b)
{
    return normL2(a - b);
}


static inline
double normL2sq(const cv::Mat &r)
{
    return r.dot(r);
}


static inline int sqr(int x) { return x * x; }
static inline float sqr(float x) { return x * x; }
static inline double sqr(double x) { return x * x; }

#endif // __OPENCV_STITCHING_UTIL_INL_HPP__
