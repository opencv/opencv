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
#include "precomp.hpp"

using namespace std;
using namespace cv;

void cv::DisjointSets::createOneElemSets(int n)
{
    rank_.assign(n, 0);
    size.assign(n, 1);
    parent.resize(n);
    for (int i = 0; i < n; ++i)
        parent[i] = i;
}


int cv::DisjointSets::findSetByElem(int elem)
{
    int set = elem;
    while (set != parent[set])
        set = parent[set];
    int next;
    while (elem != parent[elem]) 
    {
        next = parent[elem];
        parent[elem] = set;
        elem = next;
    }
    return set;
}


int cv::DisjointSets::mergeSets(int set1, int set2)
{
    if (rank_[set1] < rank_[set2]) 
    {
        parent[set1] = set2;
        size[set2] += size[set1];
        return set2;
    }
    if (rank_[set2] < rank_[set1]) 
    {
        parent[set2] = set1;
        size[set1] += size[set2];
        return set1;
    }
    parent[set1] = set2;
    rank_[set2]++;
    size[set2] += size[set1];
    return set2;
}


void cv::Graph::addEdge(int from, int to, float weight)
{
    edges_[from].push_back(GraphEdge(from, to, weight));
}


bool cv::overlapRoi(Point tl1, Point tl2, Size sz1, Size sz2, Rect &roi)
{
    int x_tl = max(tl1.x, tl2.x);
    int y_tl = max(tl1.y, tl2.y);
    int x_br = min(tl1.x + sz1.width, tl2.x + sz2.width);
    int y_br = min(tl1.y + sz1.height, tl2.y + sz2.height);
    if (x_tl < x_br && y_tl < y_br)
    {
        roi = Rect(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
        return true;
    }
    return false;
}


Rect cv::resultRoi(const vector<Point> &corners, const vector<Mat> &images)
{
    vector<Size> sizes(images.size());
    for (size_t i = 0; i < images.size(); ++i)
        sizes[i] = images[i].size();
    return resultRoi(corners, sizes);
}


Rect cv::resultRoi(const vector<Point> &corners, const vector<Size> &sizes)
{
    CV_Assert(sizes.size() == corners.size());
    Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
    Point br(numeric_limits<int>::min(), numeric_limits<int>::min());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        tl.x = min(tl.x, corners[i].x);
        tl.y = min(tl.y, corners[i].y);
        br.x = max(br.x, corners[i].x + sizes[i].width);
        br.y = max(br.y, corners[i].y + sizes[i].height);
    }
    return Rect(tl, br);
}


Point cv::resultTl(const vector<Point> &corners)
{
    Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        tl.x = min(tl.x, corners[i].x);
        tl.y = min(tl.y, corners[i].y);
    }
    return tl;
}


void cv::selectRandomSubset(int count, int size, vector<int> &subset)
{
    subset.clear();
    for (int i = 0; i < size; ++i)
    {
        if (randu<int>() % (size - i) < count)
        {
            subset.push_back(i);
            count--;
        }
    }
}
