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
#include "util.hpp"

using namespace std;
using namespace cv;

void DjSets::create(int n) {
    rank_.assign(n, 0);
    size.assign(n, 1);
    parent.resize(n);
    for (int i = 0; i < n; ++i)
        parent[i] = i;
}


int DjSets::find(int elem) {
    int set = elem;
    while (set != parent[set])
        set = parent[set];
    int next;
    while (elem != parent[elem]) {
        next = parent[elem];
        parent[elem] = set;
        elem = next;
    }
    return set;
}


int DjSets::merge(int set1, int set2) {
    if (rank_[set1] < rank_[set2]) {
        parent[set1] = set2;
        size[set2] += size[set1];
        return set2;
    }
    if (rank_[set2] < rank_[set1]) {
        parent[set2] = set1;
        size[set1] += size[set2];
        return set1;
    }
    parent[set1] = set2;
    rank_[set2]++;
    size[set2] += size[set1];
    return set2;
}


void Graph::addEdge(int from, int to, float weight)
{
    edges_[from].push_back(GraphEdge(from, to, weight));
}
