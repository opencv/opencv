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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
#include "kdtree.hpp"

namespace cv
{
namespace ml
{
// This is reimplementation of kd-trees from cvkdtree*.* by Xavier Delacour, cleaned-up and
// adopted to work with the new OpenCV data structures.

// The algorithm is taken from:
// J.S. Beis and D.G. Lowe. Shape indexing using approximate nearest-neighbor search
// in highdimensional spaces. In Proc. IEEE Conf. Comp. Vision Patt. Recog.,
// pages 1000--1006, 1997. http://citeseer.ist.psu.edu/beis97shape.html

const int MAX_TREE_DEPTH = 32;

KDTree::KDTree()
{
    maxDepth = -1;
    normType = NORM_L2;
}

KDTree::KDTree(InputArray _points, bool _copyData)
{
    maxDepth = -1;
    normType = NORM_L2;
    build(_points, _copyData);
}

KDTree::KDTree(InputArray _points, InputArray _labels, bool _copyData)
{
    maxDepth = -1;
    normType = NORM_L2;
    build(_points, _labels, _copyData);
}

struct SubTree
{
    SubTree() : first(0), last(0), nodeIdx(0), depth(0) {}
    SubTree(int _first, int _last, int _nodeIdx, int _depth)
        : first(_first), last(_last), nodeIdx(_nodeIdx), depth(_depth) {}
    int first;
    int last;
    int nodeIdx;
    int depth;
};


static float
medianPartition( size_t* ofs, int a, int b, const float* vals )
{
    int k, a0 = a, b0 = b;
    int middle = (a + b)/2;
    while( b > a )
    {
        int i0 = a, i1 = (a+b)/2, i2 = b;
        float v0 = vals[ofs[i0]], v1 = vals[ofs[i1]], v2 = vals[ofs[i2]];
        int ip = v0 < v1 ? (v1 < v2 ? i1 : v0 < v2 ? i2 : i0) :
                 v0 < v2 ? (v1 == v0 ? i2 : i0): (v1 < v2 ? i2 : i1);
        float pivot = vals[ofs[ip]];
        std::swap(ofs[ip], ofs[i2]);

        for( i1 = i0, i0--; i1 <= i2; i1++ )
            if( vals[ofs[i1]] <= pivot )
            {
                i0++;
                std::swap(ofs[i0], ofs[i1]);
            }
        if( i0 == middle )
            break;
        if( i0 > middle )
            b = i0 - (b == i0);
        else
            a = i0;
    }

    float pivot = vals[ofs[middle]];
    for( k = a0; k < middle; k++ )
    {
        CV_Assert(vals[ofs[k]] <= pivot);
    }
    for( k = b0; k > middle; k-- )
    {
        CV_Assert(vals[ofs[k]] >= pivot);
    }

    return vals[ofs[middle]];
}

static void
computeSums( const Mat& points, const size_t* ofs, int a, int b, double* sums )
{
    int i, j, dims = points.cols;
    const float* data = points.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        sums[j*2] = sums[j*2+1] = 0;
    for( i = a; i <= b; i++ )
    {
        const float* row = data + ofs[i];
        for( j = 0; j < dims; j++ )
        {
            double t = row[j], s = sums[j*2] + t, s2 = sums[j*2+1] + t*t;
            sums[j*2] = s; sums[j*2+1] = s2;
        }
    }
}


void KDTree::build(InputArray _points, bool _copyData)
{
    build(_points, noArray(), _copyData);
}


void KDTree::build(InputArray __points, InputArray __labels, bool _copyData)
{
    Mat _points = __points.getMat(), _labels = __labels.getMat();
    CV_Assert(_points.type() == CV_32F && !_points.empty());
    std::vector<KDTree::Node>().swap(nodes);

    if( !_copyData )
        points = _points;
    else
    {
        points.release();
        points.create(_points.size(), _points.type());
    }

    int i, j, n = _points.rows, ptdims = _points.cols, top = 0;
    const float* data = _points.ptr<float>(0);
    float* dstdata = points.ptr<float>(0);
    size_t step = _points.step1();
    size_t dstep = points.step1();
    int ptpos = 0;
    labels.resize(n);
    const int* _labels_data = 0;

    if( !_labels.empty() )
    {
        int nlabels = _labels.checkVector(1, CV_32S, true);
        CV_Assert(nlabels == n);
        _labels_data = _labels.ptr<int>();
    }

    Mat sumstack(MAX_TREE_DEPTH*2, ptdims*2, CV_64F);
    SubTree stack[MAX_TREE_DEPTH*2];

    std::vector<size_t> _ptofs(n);
    size_t* ptofs = &_ptofs[0];

    for( i = 0; i < n; i++ )
        ptofs[i] = i*step;

    nodes.push_back(Node());
    computeSums(points, ptofs, 0, n-1, sumstack.ptr<double>(top));
    stack[top++] = SubTree(0, n-1, 0, 0);
    int _maxDepth = 0;

    while( --top >= 0 )
    {
        int first = stack[top].first, last = stack[top].last;
        int depth = stack[top].depth, nidx = stack[top].nodeIdx;
        int count = last - first + 1, dim = -1;
        const double* sums = sumstack.ptr<double>(top);
        double invCount = 1./count, maxVar = -1.;

        if( count == 1 )
        {
            int idx0 = (int)(ptofs[first]/step);
            int idx = _copyData ? ptpos++ : idx0;
            nodes[nidx].idx = ~idx;
            if( _copyData )
            {
                const float* src = data + ptofs[first];
                float* dst = dstdata + idx*dstep;
                for( j = 0; j < ptdims; j++ )
                    dst[j] = src[j];
            }
            labels[idx] = _labels_data ? _labels_data[idx0] : idx0;
            _maxDepth = std::max(_maxDepth, depth);
            continue;
        }

        // find the dimensionality with the biggest variance
        for( j = 0; j < ptdims; j++ )
        {
            double m = sums[j*2]*invCount;
            double varj = sums[j*2+1]*invCount - m*m;
            if( maxVar < varj )
            {
                maxVar = varj;
                dim = j;
            }
        }

        int left = (int)nodes.size(), right = left + 1;
        nodes.push_back(Node());
        nodes.push_back(Node());
        nodes[nidx].idx = dim;
        nodes[nidx].left = left;
        nodes[nidx].right = right;
        nodes[nidx].boundary = medianPartition(ptofs, first, last, data + dim);

        int middle = (first + last)/2;
        double *lsums = (double*)sums, *rsums = lsums + ptdims*2;
        computeSums(points, ptofs, middle+1, last, rsums);
        for( j = 0; j < ptdims*2; j++ )
            lsums[j] = sums[j] - rsums[j];
        stack[top++] = SubTree(first, middle, left, depth+1);
        stack[top++] = SubTree(middle+1, last, right, depth+1);
    }
    maxDepth = _maxDepth;
}


struct PQueueElem
{
    PQueueElem() : dist(0), idx(0) {}
    PQueueElem(float _dist, int _idx) : dist(_dist), idx(_idx) {}
    float dist;
    int idx;
};


int KDTree::findNearest(InputArray _vec, int K, int emax,
                        OutputArray _neighborsIdx, OutputArray _neighbors,
                        OutputArray _dist, OutputArray _labels) const

{
    Mat vecmat = _vec.getMat();
    CV_Assert( vecmat.isContinuous() && vecmat.type() == CV_32F && vecmat.total() == (size_t)points.cols );
    const float* vec = vecmat.ptr<float>();
    K = std::min(K, points.rows);
    int ptdims = points.cols;

    CV_Assert(K > 0 && (normType == NORM_L2 || normType == NORM_L1));

    AutoBuffer<uchar> _buf((K+1)*(sizeof(float) + sizeof(int)));
    int* idx = (int*)_buf.data();
    float* dist = (float*)(idx + K + 1);
    int i, j, ncount = 0, e = 0;

    int qsize = 0, maxqsize = 1 << 10;
    AutoBuffer<uchar> _pqueue(maxqsize*sizeof(PQueueElem));
    PQueueElem* pqueue = (PQueueElem*)_pqueue.data();
    emax = std::max(emax, 1);

    for( e = 0; e < emax; )
    {
        float d, alt_d = 0.f;
        int nidx;

        if( e == 0 )
            nidx = 0;
        else
        {
            // take the next node from the priority queue
            if( qsize == 0 )
                break;
            nidx = pqueue[0].idx;
            alt_d = pqueue[0].dist;
            if( --qsize > 0 )
            {
                std::swap(pqueue[0], pqueue[qsize]);
                d = pqueue[0].dist;
                for( i = 0;;)
                {
                    int left = i*2 + 1, right = i*2 + 2;
                    if( left >= qsize )
                        break;
                    if( right < qsize && pqueue[right].dist < pqueue[left].dist )
                        left = right;
                    if( pqueue[left].dist >= d )
                        break;
                    std::swap(pqueue[i], pqueue[left]);
                    i = left;
                }
            }

            if( ncount == K && alt_d > dist[ncount-1] )
                continue;
        }

        for(;;)
        {
            if( nidx < 0 )
                break;
            const Node& n = nodes[nidx];

            if( n.idx < 0 )
            {
                i = ~n.idx;
                const float* row = points.ptr<float>(i);
                if( normType == NORM_L2 )
                    for( j = 0, d = 0.f; j < ptdims; j++ )
                    {
                        float t = vec[j] - row[j];
                        d += t*t;
                    }
                else
                    for( j = 0, d = 0.f; j < ptdims; j++ )
                        d += std::abs(vec[j] - row[j]);

                dist[ncount] = d;
                idx[ncount] = i;
                for( i = ncount-1; i >= 0; i-- )
                {
                    if( dist[i] <= d )
                        break;
                    std::swap(dist[i], dist[i+1]);
                    std::swap(idx[i], idx[i+1]);
                }
                ncount += ncount < K;
                e++;
                break;
            }

            int alt;
            if( vec[n.idx] <= n.boundary )
            {
                nidx = n.left;
                alt = n.right;
            }
            else
            {
                nidx = n.right;
                alt = n.left;
            }

            d = vec[n.idx] - n.boundary;
            if( normType == NORM_L2 )
                d = d*d + alt_d;
            else
                d = std::abs(d) + alt_d;
            // subtree prunning
            if( ncount == K && d > dist[ncount-1] )
                continue;
            // add alternative subtree to the priority queue
            pqueue[qsize] = PQueueElem(d, alt);
            for( i = qsize; i > 0; )
            {
                int parent = (i-1)/2;
                if( parent < 0 || pqueue[parent].dist <= d )
                    break;
                std::swap(pqueue[i], pqueue[parent]);
                i = parent;
            }
            qsize += qsize+1 < maxqsize;
        }
    }

    K = std::min(K, ncount);
    if( _neighborsIdx.needed() )
    {
        _neighborsIdx.create(K, 1, CV_32S, -1, true);
        Mat nidx = _neighborsIdx.getMat();
        Mat(nidx.size(), CV_32S, &idx[0]).copyTo(nidx);
    }
    if( _dist.needed() )
        sqrt(Mat(K, 1, CV_32F, dist), _dist);

    if( _neighbors.needed() || _labels.needed() )
        getPoints(Mat(K, 1, CV_32S, idx), _neighbors, _labels);
    return K;
}


void KDTree::findOrthoRange(InputArray _lowerBound,
                            InputArray _upperBound,
                            OutputArray _neighborsIdx,
                            OutputArray _neighbors,
                            OutputArray _labels ) const
{
    int ptdims = points.cols;
    Mat lowerBound = _lowerBound.getMat(), upperBound = _upperBound.getMat();
    CV_Assert( lowerBound.size == upperBound.size &&
               lowerBound.isContinuous() &&
               upperBound.isContinuous() &&
               lowerBound.type() == upperBound.type() &&
               lowerBound.type() == CV_32F &&
               lowerBound.total() == (size_t)ptdims );
    const float* L = lowerBound.ptr<float>();
    const float* R = upperBound.ptr<float>();

    std::vector<int> idx;
    AutoBuffer<int> _stack(MAX_TREE_DEPTH*2 + 1);
    int* stack = _stack.data();
    int top = 0;

    stack[top++] = 0;

    while( --top >= 0 )
    {
        int nidx = stack[top];
        if( nidx < 0 )
            break;
        const Node& n = nodes[nidx];
        if( n.idx < 0 )
        {
            int j, i = ~n.idx;
            const float* row = points.ptr<float>(i);
            for( j = 0; j < ptdims; j++ )
                if( row[j] < L[j] || row[j] >= R[j] )
                    break;
            if( j == ptdims )
                idx.push_back(i);
            continue;
        }
        if( L[n.idx] <= n.boundary )
            stack[top++] = n.left;
        if( R[n.idx] > n.boundary )
            stack[top++] = n.right;
    }

    if( _neighborsIdx.needed() )
    {
        _neighborsIdx.create((int)idx.size(), 1, CV_32S, -1, true);
        Mat nidx = _neighborsIdx.getMat();
        Mat(nidx.size(), CV_32S, &idx[0]).copyTo(nidx);
    }
    getPoints( idx, _neighbors, _labels );
}


void KDTree::getPoints(InputArray _idx, OutputArray _pts, OutputArray _labels) const
{
    Mat idxmat = _idx.getMat(), pts, labelsmat;
    CV_Assert( idxmat.isContinuous() && idxmat.type() == CV_32S &&
               (idxmat.cols == 1 || idxmat.rows == 1) );
    const int* idx = idxmat.ptr<int>();
    int* dstlabels = 0;

    int ptdims = points.cols;
    int i, nidx = (int)idxmat.total();
    if( nidx == 0 )
    {
        _pts.release();
        _labels.release();
        return;
    }

    if( _pts.needed() )
    {
        _pts.create( nidx, ptdims, points.type());
        pts = _pts.getMat();
    }

    if(_labels.needed())
    {
        _labels.create(nidx, 1, CV_32S, -1, true);
        labelsmat = _labels.getMat();
        CV_Assert( labelsmat.isContinuous() );
        dstlabels = labelsmat.ptr<int>();
    }
    const int* srclabels = !labels.empty() ? &labels[0] : 0;

    for( i = 0; i < nidx; i++ )
    {
        int k = idx[i];
        CV_Assert( (unsigned)k < (unsigned)points.rows );
        const float* src = points.ptr<float>(k);
        if( !pts.empty() )
            std::copy(src, src + ptdims, pts.ptr<float>(i));
        if( dstlabels )
            dstlabels[i] = srclabels ? srclabels[k] : k;
    }
}


const float* KDTree::getPoint(int ptidx, int* label) const
{
    CV_Assert( (unsigned)ptidx < (unsigned)points.rows);
    if(label)
        *label = labels[ptidx];
    return points.ptr<float>(ptidx);
}


int KDTree::dims() const
{
    return !points.empty() ? points.cols : 0;
}

}
}
