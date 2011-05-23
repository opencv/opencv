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
#ifndef __OPENCV_STITCHING_UTIL_HPP__
#define __OPENCV_STITCHING_UTIL_HPP__

#include <list>
#include "precomp.hpp"

#define ENABLE_LOG 1

#if ENABLE_LOG
  #include <iostream>
  #define LOG(msg) std::cout << msg;
#else
  #define LOG(msg)
#endif

#define LOGLN(msg) LOG(msg << std::endl)


class DjSets
{
public:
    DjSets(int n = 0) { create(n); }

    void create(int n);
    int find(int elem);
    int merge(int set1, int set2);

    std::vector<int> parent;
    std::vector<int> size;

private:
    std::vector<int> rank_;
};


struct GraphEdge
{
    GraphEdge(int from, int to, float weight) 
        : from(from), to(to), weight(weight) {}
    bool operator <(const GraphEdge& other) const { return weight < other.weight; }
    bool operator >(const GraphEdge& other) const { return weight > other.weight; }

    int from, to;
    float weight;
};


class Graph
{
public:
    Graph(int num_vertices = 0) { create(num_vertices); }

    void create(int num_vertices) { edges_.assign(num_vertices, std::list<GraphEdge>()); }

    int numVertices() const { return static_cast<int>(edges_.size()); }

    void addEdge(int from, int to, float weight);

    template <typename B>
    B forEach(B body) const;

    template <typename B> 
    B walkBreadthFirst(int from, B body) const;
    
private:
    std::vector< std::list<GraphEdge> > edges_;
};

#include "util_inl.hpp"

#endif // __OPENCV_STITCHING_UTIL_HPP__
