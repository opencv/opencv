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
#include "precomp.hpp"

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::meanShiftSegmentation(InputArray, OutputArray, int, int, int, TermCriteria, Stream&) { throw_no_cuda(); }

#else

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
    int find(int elem);
    int merge(int set1, int set2);

    std::vector<int> parent;
    std::vector<int> rank;
    std::vector<int> size;
private:
    DjSets(const DjSets&);
    void operator =(const DjSets&);
};


template <typename T>
struct GraphEdge
{
    GraphEdge() {}
    GraphEdge(int to_, int next_, const T& val_) : to(to_), next(next_), val(val_) {}
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

    void addEdge(int from, int to, const T& val=T());

    std::vector<int> start;
    std::vector<Edge> edges;

    int numv;
    int nume_max;
    int nume;
private:
    Graph(const Graph&);
    void operator =(const Graph&);
};


struct SegmLinkVal
{
    SegmLinkVal() {}
    SegmLinkVal(int dr_, int dsp_) : dr(dr_), dsp(dsp_) {}
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
    SegmLink(int from_, int to_, const SegmLinkVal& val_)
        : from(from_), to(to_), val(val_) {}
    bool operator <(const SegmLink& other) const
    {
        return val < other.val;
    }
    int from;
    int to;
    SegmLinkVal val;
};

//
// Implementation
//

DjSets::DjSets(int n) : parent(n), rank(n, 0), size(n, 1)
{
    for (int i = 0; i < n; ++i)
        parent[i] = i;
}


inline int DjSets::find(int elem)
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
Graph<T>::Graph(int numv_, int nume_max_) : start(numv_, -1), edges(nume_max_)
{
    this->numv = numv_;
    this->nume_max = nume_max_;
    nume = 0;
}


template <typename T>
inline void Graph<T>::addEdge(int from, int to, const T& val)
{
    edges[nume] = Edge(to, start[from], val);
    start[from] = nume;
    nume++;
}


inline int pix(int y, int x, int ncols)
{
    return y * ncols + x;
}


inline int sqr(int x)
{
    return x * x;
}


inline int dist2(const cv::Vec4b& lhs, const cv::Vec4b& rhs)
{
    return sqr(lhs[0] - rhs[0]) + sqr(lhs[1] - rhs[1]) + sqr(lhs[2] - rhs[2]);
}


inline int dist2(const cv::Vec2s& lhs, const cv::Vec2s& rhs)
{
    return sqr(lhs[0] - rhs[0]) + sqr(lhs[1] - rhs[1]);
}

} // anonymous namespace


void cv::cuda::meanShiftSegmentation(InputArray _src, OutputArray _dst, int sp, int sr, int minsize, TermCriteria criteria, Stream& stream)
{
    GpuMat src = _src.getGpuMat();

    CV_Assert( src.type() == CV_8UC4 );

    const int nrows = src.rows;
    const int ncols = src.cols;
    const int hr = sr;
    const int hsp = sp;

    // Perform mean shift procedure and obtain region and spatial maps
    GpuMat d_rmap, d_spmap;
    cuda::meanShiftProc(src, d_rmap, d_spmap, sp, sr, criteria, stream);

    stream.waitForCompletion();

    Mat rmap(d_rmap);
    Mat spmap(d_spmap);

    Graph<SegmLinkVal> g(nrows * ncols, 4 * (nrows - 1) * (ncols - 1)
                                        + (nrows - 1) + (ncols - 1));

    // Make region adjacent graph from image
    Vec4b r1;
    Vec4b r2[4];
    Vec2s sp1;
    Vec2s sp2[4];
    int dr[4];
    int dsp[4];
    for (int y = 0; y < nrows - 1; ++y)
    {
        Vec4b* ry = rmap.ptr<Vec4b>(y);
        Vec4b* ryp = rmap.ptr<Vec4b>(y + 1);
        Vec2s* spy = spmap.ptr<Vec2s>(y);
        Vec2s* spyp = spmap.ptr<Vec2s>(y + 1);
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

            dr[0] = dist2(r1, r2[0]);
            dr[1] = dist2(r1, r2[1]);
            dr[2] = dist2(r1, r2[2]);
            dsp[0] = dist2(sp1, sp2[0]);
            dsp[1] = dist2(sp1, sp2[1]);
            dsp[2] = dist2(sp1, sp2[2]);

            r1 = ry[x + 1];
            sp1 = spy[x + 1];

            dr[3] = dist2(r1, r2[3]);
            dsp[3] = dist2(sp1, sp2[3]);

            g.addEdge(pix(y, x, ncols), pix(y, x + 1, ncols), SegmLinkVal(dr[0], dsp[0]));
            g.addEdge(pix(y, x, ncols), pix(y + 1, x, ncols), SegmLinkVal(dr[1], dsp[1]));
            g.addEdge(pix(y, x, ncols), pix(y + 1, x + 1, ncols), SegmLinkVal(dr[2], dsp[2]));
            g.addEdge(pix(y, x + 1, ncols), pix(y + 1, x, ncols), SegmLinkVal(dr[3], dsp[3]));
        }
    }
    for (int y = 0; y < nrows - 1; ++y)
    {
        r1 = rmap.at<Vec4b>(y, ncols - 1);
        r2[0] = rmap.at<Vec4b>(y + 1, ncols - 1);
        sp1 = spmap.at<Vec2s>(y, ncols - 1);
        sp2[0] = spmap.at<Vec2s>(y + 1, ncols - 1);
        dr[0] = dist2(r1, r2[0]);
        dsp[0] = dist2(sp1, sp2[0]);
        g.addEdge(pix(y, ncols - 1, ncols), pix(y + 1, ncols - 1, ncols), SegmLinkVal(dr[0], dsp[0]));
    }
    for (int x = 0; x < ncols - 1; ++x)
    {
        r1 = rmap.at<Vec4b>(nrows - 1, x);
        r2[0] = rmap.at<Vec4b>(nrows - 1, x + 1);
        sp1 = spmap.at<Vec2s>(nrows - 1, x);
        sp2[0] = spmap.at<Vec2s>(nrows - 1, x + 1);
        dr[0] = dist2(r1, r2[0]);
        dsp[0] = dist2(sp1, sp2[0]);
        g.addEdge(pix(nrows - 1, x, ncols), pix(nrows - 1, x + 1, ncols), SegmLinkVal(dr[0], dsp[0]));
    }

    DjSets comps(g.numv);

    // Find adjacent components
    for (int v = 0; v < g.numv; ++v)
    {
        for (int e_it = g.start[v]; e_it != -1; e_it = g.edges[e_it].next)
        {
            int c1 = comps.find(v);
            int c2 = comps.find(g.edges[e_it].to);
            if (c1 != c2 && g.edges[e_it].val.dr < hr && g.edges[e_it].val.dsp < hsp)
                comps.merge(c1, c2);
        }
    }

    std::vector<SegmLink> edges;
    edges.reserve(g.numv);

    // Prepare edges connecting different components
    for (int v = 0; v < g.numv; ++v)
    {
        int c1 = comps.find(v);
        for (int e_it = g.start[v]; e_it != -1; e_it = g.edges[e_it].next)
        {
            int c2 = comps.find(g.edges[e_it].to);
            if (c1 != c2)
                edges.push_back(SegmLink(c1, c2, g.edges[e_it].val));
        }
    }

    // Sort all graph's edges connecting different components (in ascending order)
    std::sort(edges.begin(), edges.end());

    // Exclude small components (starting from the nearest couple)
    for (size_t i = 0; i < edges.size(); ++i)
    {
        int c1 = comps.find(edges[i].from);
        int c2 = comps.find(edges[i].to);
        if (c1 != c2 && (comps.size[c1] < minsize || comps.size[c2] < minsize))
            comps.merge(c1, c2);
    }

    // Compute sum of the pixel's colors which are in the same segment
    Mat h_src(src);
    std::vector<Vec4i> sumcols(nrows * ncols, Vec4i(0, 0, 0, 0));
    for (int y = 0; y < nrows; ++y)
    {
        Vec4b* h_srcy = h_src.ptr<Vec4b>(y);
        for (int x = 0; x < ncols; ++x)
        {
            int parent = comps.find(pix(y, x, ncols));
            Vec4b col = h_srcy[x];
            Vec4i& sumcol = sumcols[parent];
            sumcol[0] += col[0];
            sumcol[1] += col[1];
            sumcol[2] += col[2];
        }
    }

    // Create final image, color of each segment is the average color of its pixels
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();

    for (int y = 0; y < nrows; ++y)
    {
        Vec4b* dsty = dst.ptr<Vec4b>(y);
        for (int x = 0; x < ncols; ++x)
        {
            int parent = comps.find(pix(y, x, ncols));
            const Vec4i& sumcol = sumcols[parent];
            Vec4b& dstcol = dsty[x];
            dstcol[0] = static_cast<uchar>(sumcol[0] / comps.size[parent]);
            dstcol[1] = static_cast<uchar>(sumcol[1] / comps.size[parent]);
            dstcol[2] = static_cast<uchar>(sumcol[2] / comps.size[parent]);
            dstcol[3] = 255;
        }
    }
}

#endif // #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
