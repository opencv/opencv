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

#include <time.h>
#include <vector>
#include "precomp.hpp"

#if !defined(HAVE_CUDA)

namespace cv
{
namespace gpu
{

void meanShiftSegmentation(const GpuMat&, Mat&, int, int, int, TermCriteria) { throw_nogpu(); }

} // namespace gpu
} // namespace cv

#else

//#define _MSSEGMENTATION_DBG

#ifdef _MSSEGMENTATION_DBG
#include <iostream>
#define LOG(s) std::cout << (s) << std::endl
#define LOG2(s1, s2) std::cout << (s1) << (s2) << std::endl
#define DBG(code) code
#else
#define LOG(s1)
#define LOG2(s1, s2)
#define DBG(code)
#endif

#define PIX(y, x) ((y) * ncols + (x))

using namespace std;

// Auxiliray stuff
namespace
{

//
// Declarations
//

class DjSets
{
public:
    DjSets(int n);
    ~DjSets();
    int find(int elem) const;
    int merge(int set1, int set2);

    int* parent;
    int* rank;
    int* size;
private:
    DjSets(const DjSets&) {}
    DjSets operator =(const DjSets&) {}
};


template <typename T>
struct GraphEdge
{
    GraphEdge() {}
    GraphEdge(int to, int next, const T& val) : to(to), next(next), val(val) {}
    int to;
    int next;
    T val;
};


template <typename T>
class Graph
{
public:
    typedef GraphEdge<T> Edge;

    Graph(int numv, int nume_max);
    ~Graph();

    void addEdge(int from, int to, const T& val=T());

    int* start;
    Edge* edges;

    int numv;
    int nume_max;
    int nume;
private:
    Graph(const Graph&) {}
    Graph operator =(const Graph&) {}
};


struct SegmLinkVal
{
    SegmLinkVal() {}
    SegmLinkVal(int dr, int dsp) : dr(dr), dsp(dsp) {}
    bool operator <(const SegmLinkVal& other) const
    {
        return dr + dsp < other.dr + other.dsp;
    }
    int dr;
    int dsp;
};


struct SegmLink
{
    SegmLink() {}
    SegmLink(int from, int to, const SegmLinkVal& val) 
        : from(from), to(to), val(val) {}
    int from;
    int to;
    SegmLinkVal val;
};


struct SegmLinkCmp
{
    bool operator ()(const SegmLink& lhs, const SegmLink& rhs) const
    {
        return lhs.val < rhs.val;
    }
};

//
// Implementation
//

DjSets::DjSets(int n)
{
    parent = new int[n];
    rank = new int[n];
    size = new int[n];   
    for (int i = 0; i < n; ++i)
    {
        parent[i] = i;
        rank[i] = 0;
        size[i] = 1;
    }
}


DjSets::~DjSets()
{
    delete[] parent;
    delete[] rank;
    delete[] size;
}


inline int DjSets::find(int elem) const
{
    int set = elem;
    while (set != parent[set])
        set = parent[set];
    while (elem != parent[elem])
    {
        int next = parent[elem];
        parent[elem] = set;
        elem = next;
    }
    return set;
}


inline int DjSets::merge(int set1, int set2)
{
    if (rank[set1] < rank[set2])
    {
        parent[set1] = set2;
        size[set2] += size[set1];
        return set2;
    }
    if (rank[set2] < rank[set1])
    {
        parent[set2] = set1;
        size[set1] += size[set2];
        return set1;
    }
    parent[set1] = set2;
    rank[set2]++;
    size[set2] += size[set1];
    return set2;
}


template <typename T>
Graph<T>::Graph(int numv, int nume_max)
{
    this->numv = numv;
    this->nume_max = nume_max;
    start = new int[numv];
    for (int i = 0; i < numv; ++i)
        start[i] = -1;
    edges = new Edge[nume_max];
    nume = 0;
}


template <typename T>
Graph<T>::~Graph()
{
    delete[] start;
    delete[] edges;
}


template <typename T>
inline void Graph<T>::addEdge(int from, int to, const T& val)
{
    Edge* edge = edges + nume;
    new (edge) SegmLink(to, start[from], val);
    start[from] = nume;
    nume++;
}


inline int sqr(int x)
{
    return x * x;
}

} // anonymous namespace


namespace cv
{
namespace gpu
{

void meanShiftSegmentation(const GpuMat& src, Mat& dst, int sp, int sr, int minsize, TermCriteria criteria)
{
    CV_Assert(src.type() == CV_8UC4);
    const int nrows = src.rows;
    const int ncols = src.cols;
    const int hr = sr;
    const int hsp = sp;

    DBG(clock_t start = clock());

    // Perform mean shift procedure and obtain region and spatial maps
    GpuMat h_rmap, h_spmap;
    meanShiftProc(src, h_rmap, h_spmap, sp, sr, criteria);
    Mat rmap = h_rmap;
    Mat spmap = h_spmap;

    LOG2("meanshift:", clock() - start);
    DBG(start = clock());

    Graph<SegmLinkVal> g(nrows * ncols, 4 * (nrows - 1) * (ncols - 1)
                                        + (nrows - 1) + (ncols - 1));

    LOG2("ragalloc:", clock() - start);
    DBG(start = clock());

    // Make region adjacent graph from image
    // TODO: SSE?
    Vec4b r1;
    Vec4b r2[4];
    Point_<short> sp1;
    Point_<short> sp2[4];
    int dr[4];
    int dsp[4];
    for (int y = 0; y < nrows - 1; ++y)
    {
        Vec4b* ry = rmap.ptr<Vec4b>(y);
        Vec4b* ryp = rmap.ptr<Vec4b>(y + 1);
        Point_<short>* spy = spmap.ptr<Point_<short> >(y);
        Point_<short>* spyp = spmap.ptr<Point_<short> >(y + 1);
        for (int x = 0; x < ncols - 1; ++x)
        {
            r1 = ry[x];
            sp1 = spy[x];

            r2[0] = ry[x + 1];
            r2[1] = ryp[x];
            r2[2] = ryp[x + 1];
            r2[3] = ryp[x];

            sp2[0] = spy[x + 1];
            sp2[1] = spyp[x];
            sp2[2] = spyp[x + 1];
            sp2[3] = spyp[x];

            dr[0] = sqr(r1[0] - r2[0][0]) + sqr(r1[1] - r2[0][1]) + sqr(r1[2] - r2[0][2]);
            dr[1] = sqr(r1[0] - r2[1][0]) + sqr(r1[1] - r2[1][1]) + sqr(r1[2] - r2[1][2]);
            dr[2] = sqr(r1[0] - r2[2][0]) + sqr(r1[1] - r2[2][1]) + sqr(r1[2] - r2[2][2]);
            dsp[0] = sqr(sp1.x - sp2[0].x) + sqr(sp1.y - sp2[0].y);
            dsp[1] = sqr(sp1.x - sp2[1].x) + sqr(sp1.y - sp2[1].y);
            dsp[2] = sqr(sp1.x - sp2[2].x) + sqr(sp1.y - sp2[2].y); 

            r1 = ry[x + 1];
            sp1 = spy[x + 1];

            dr[3] = sqr(r1[0] - r2[3][0]) + sqr(r1[1] - r2[3][1]) + sqr(r1[2] - r2[3][2]);
            dsp[3] = sqr(sp1.x - sp2[3].x) + sqr(sp1.y - sp2[3].y); 

            g.addEdge(PIX(y, x), PIX(y, x + 1), SegmLinkVal(dr[0], dsp[0]));
            g.addEdge(PIX(y, x), PIX(y + 1, x), SegmLinkVal(dr[1], dsp[1]));
            g.addEdge(PIX(y, x), PIX(y + 1, x + 1), SegmLinkVal(dr[2], dsp[2]));
            g.addEdge(PIX(y, x + 1), PIX(y, x + 1), SegmLinkVal(dr[3], dsp[3]));
        }
    }
    for (int y = 0; y < nrows - 1; ++y)
    {
        r1 = rmap.at<Vec4b>(y, ncols - 1);
        r2[0] = rmap.at<Vec4b>(y + 1, ncols - 1);
        sp1 = spmap.at<Point_<short> >(y, ncols - 1);
        sp2[0] = spmap.at<Point_<short> >(y + 1, ncols - 1);
        dr[0] = sqr(r1[0] - r2[0][0]) + sqr(r1[1] - r2[0][1]) + sqr(r1[2] - r2[0][2]);
        dsp[0] = sqr(sp1.x - sp2[0].x) + sqr(sp1.y - sp2[0].y);   
        g.addEdge(PIX(y, ncols - 1), PIX(y + 1, ncols - 1), SegmLinkVal(dr[0], dsp[0]));
    }
    for (int x = 0; x < ncols - 1; ++x)
    {
        r1 = rmap.at<Vec4b>(nrows - 1, x);
        r2[0] = rmap.at<Vec4b>(nrows - 1, x + 1);
        sp1 = spmap.at<Point_<short> >(nrows - 1, x);
        sp2[0] = spmap.at<Point_<short> >(nrows - 1, x + 1);
        dr[0] = sqr(r1[0] - r2[0][0]) + sqr(r1[1] - r2[0][1]) + sqr(r1[2] - r2[0][2]);
        dsp[0] = sqr(sp1.x - sp2[0].x) + sqr(sp1.y - sp2[0].y);   
        g.addEdge(PIX(nrows - 1, x), PIX(nrows - 1, x + 1), SegmLinkVal(dr[0], dsp[0]));
    }

    LOG2("raginit:", clock() - start);
    DBG(start = clock());

    DjSets comps(g.numv);

    LOG2("djsetinit:", clock() - start);
    DBG(start = clock());

    // Find adjacent components
    for (int v = 0; v < g.numv; ++v)
    {
        for (int e_it = g.start[v]; e_it != -1; e_it = g.edges[e_it].next)
        {
            int comp1 = comps.find(v);
            int comp2 = comps.find(g.edges[e_it].to);
            if (comp1 != comp2 && g.edges[e_it].val.dr < hr 
                               && g.edges[e_it].val.dsp < hsp)
                comps.merge(comp1, comp2);
        }
    }

    LOG2("findadjacent:", clock() - start);
    DBG(start = clock());

    vector<SegmLink> edges;
    edges.reserve(g.numv);

    LOG2("initedges:", clock() - start);
    DBG(start = clock());

    for (int v = 0; v < g.numv; ++v)
    {
        int comp1 = comps.find(v);
        for (int e_it = g.start[v]; e_it != -1; e_it = g.edges[e_it].next)
        {
            int comp2 = comps.find(g.edges[e_it].to);
            if (comp1 != comp2)
                edges.push_back(SegmLink(comp1, comp2, g.edges[e_it].val));
        }
    }

    LOG2("prepareforsort:", clock() - start);
    DBG(start = clock());

    // Sort all graph's edges connecting differnet components (in asceding order)
    sort(edges.begin(), edges.end(), SegmLinkCmp());

    LOG2("sortedges:", clock() - start);
    DBG(start = clock());

    // Exclude small components (starting from the nearest couple)
    vector<SegmLink>::iterator e_it = edges.begin();
    for (; e_it != edges.end(); ++e_it)
    {
        int comp1 = comps.find(e_it->from);
        int comp2 = comps.find(e_it->to);
        if (comp1 != comp2 && (comps.size[comp1] < minsize || comps.size[comp2] < minsize))
            comps.merge(comp1, comp2);
    }

    LOG2("excludesmall:", clock() - start);
    DBG(start = clock());

    // Compute sum of the pixel's colors which are in the same segment
    Mat h_src = src;
    vector<Vec4i> sumcols(nrows * ncols, Vec4i(0, 0, 0, 0));
    for (int y = 0; y < nrows; ++y)
    {
        Vec4b* h_srcy = h_src.ptr<Vec4b>(y);
        for (int x = 0; x < ncols; ++x)
        {
            int parent = comps.find(PIX(y, x));
            Vec4b col = h_srcy[x];
            Vec4i& sumcol = sumcols[parent];
            sumcol[0] += col[0];
            sumcol[1] += col[1];
            sumcol[2] += col[2];
        }
    }

    LOG2("computesum:", clock() - start);
    DBG(start = clock());

    // Create final image, color of each segment is the average color of its pixels
    dst.create(src.size(), src.type());

    for (int y = 0; y < nrows; ++y)
    {
        Vec4b* dsty = dst.ptr<Vec4b>(y);
        for (int x = 0; x < ncols; ++x)
        {
            int parent = comps.find(PIX(y, x));
            const Vec4i& sumcol = sumcols[parent];
            Vec4b& dstcol = dsty[x];
            dstcol[0] = static_cast<uchar>(sumcol[0] / comps.size[parent]);
            dstcol[1] = static_cast<uchar>(sumcol[1] / comps.size[parent]);
            dstcol[2] = static_cast<uchar>(sumcol[2] / comps.size[parent]);
        }
    }

    LOG2("createfinal:", clock() - start);
}

} // namespace gpu
} // namespace cv

#endif // #if !defined (HAVE_CUDA)