//M*//////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#undef INFINITY
#define INFINITY 10000
#define OCCLUSION_PENALTY 10000
#define OCCLUSION_PENALTY2 1000
#define DENOMINATOR 16
#undef OCCLUDED
#define OCCLUDED CV_STEREO_GC_OCCLUDED
#define CUTOFF 1000
#define IS_BLOCKED(d1, d2) ((d1) > (d2))

typedef struct GCVtx
{
    GCVtx *next;
    int parent;
    int first;
    int ts;
    int dist;
    short weight;
    uchar t;
}
GCVtx;

typedef struct GCEdge
{
    GCVtx* dst;
    int next;
    int weight;
}
GCEdge;

typedef struct CvStereoGCState2
{
    int Ithreshold, interactionRadius;
    int lambda, lambda1, lambda2, K;
    int dataCostFuncTab[CUTOFF+1];
    int smoothnessR[CUTOFF*2+1];
    int smoothnessGrayDiff[512];
    GCVtx** orphans;
    int maxOrphans;
}
CvStereoGCState2;

// truncTab[x+255] = MAX(x-255,0)
static uchar icvTruncTab[512];
// cutoffSqrTab[x] = MIN(x*x, CUTOFF)
static int icvCutoffSqrTab[256];

static void icvInitStereoConstTabs()
{
    static volatile int initialized = 0;
    if( !initialized )
    {
        int i;
        for( i = 0; i < 512; i++ )
            icvTruncTab[i] = (uchar)MIN(MAX(i-255,0),255);
        for( i = 0; i < 256; i++ )
            icvCutoffSqrTab[i] = MIN(i*i, CUTOFF);
        initialized = 1;
    }
}

static void icvInitStereoTabs( CvStereoGCState2* state2 )
{
    int i, K = state2->K;

    for( i = 0; i <= CUTOFF; i++ )
        state2->dataCostFuncTab[i] = MIN(i*DENOMINATOR - K, 0);

    for( i = 0; i < CUTOFF*2 + 1; i++ )
        state2->smoothnessR[i] = MIN(abs(i-CUTOFF), state2->interactionRadius);

    for( i = 0; i < 512; i++ )
    {
        int diff = abs(i - 255);
        state2->smoothnessGrayDiff[i] = diff < state2->Ithreshold ? state2->lambda1 : state2->lambda2;
    }
}


static int icvGCResizeOrphansBuf( GCVtx**& orphans, int norphans )
{
    int i, newNOrphans = MAX(norphans*3/2, 256);
    GCVtx** newOrphans = (GCVtx**)cvAlloc( newNOrphans*sizeof(orphans[0]) );
    for( i = 0; i < norphans; i++ )
        newOrphans[i] = orphans[i];
    cvFree( &orphans );
    orphans = newOrphans;
    return newNOrphans;
}

static int64 icvGCMaxFlow( GCVtx* vtx, int nvtx, GCEdge* edges, GCVtx**& _orphans, int& _maxOrphans )
{
    const int TERMINAL = -1, ORPHAN = -2;
    GCVtx stub, *nilNode = &stub, *first = nilNode, *last = nilNode;
    int i, k;
    int curr_ts = 0;
    int64 flow = 0;
    int norphans = 0, maxOrphans = _maxOrphans;
    GCVtx** orphans = _orphans;
    stub.next = nilNode;
    
    // initialize the active queue and the graph vertices
    for( i = 0; i < nvtx; i++ )
    {
        GCVtx* v = vtx + i;
        v->ts = 0;
        if( v->weight != 0 )
        {
            last = last->next = v;
            v->dist = 1;
            v->parent = TERMINAL;
            v->t = v->weight < 0;
        }
        else
            v->parent = 0;
    }

    first = first->next;
    last->next = nilNode;
    nilNode->next = 0;

    // run the search-path -> augment-graph -> restore-trees loop
    for(;;)
    {
        GCVtx* v, *u;
        int e0 = -1, ei = 0, ej = 0, min_weight, weight;
        uchar vt;
        
        // grow S & T search trees, find an edge connecting them
        while( first != nilNode )
        {
            v = first;
            if( v->parent )
            {
                vt = v->t;
                for( ei = v->first; ei != 0; ei = edges[ei].next )
                {
                    if( edges[ei^vt].weight == 0 )
                        continue;
                    u = edges[ei].dst;
                    if( !u->parent )
                    {
                        u->t = vt;
                        u->parent = ei ^ 1;
                        u->ts = v->ts;
                        u->dist = v->dist + 1;
                        if( !u->next )
                        {
                            u->next = nilNode;
                            last = last->next = u;
                        }
                        continue;
                    }

                    if( u->t != vt )
                    {
                        e0 = ei ^ vt;
                        break;
                    }

                    if( u->dist > v->dist+1 && u->ts <= v->ts )
                    {
                        // reassign the parent
                        u->parent = ei ^ 1;
                        u->ts = v->ts;
                        u->dist = v->dist + 1;
                    }
                }
                if( e0 > 0 )
                    break;
            }
            // exclude the vertex from the active list
            first = first->next;
            v->next = 0;
        }

        if( e0 <= 0 )
            break;

        // find the minimum edge weight along the path
        min_weight = edges[e0].weight;
        assert( min_weight > 0 );
        // k = 1: source tree, k = 0: destination tree
        for( k = 1; k >= 0; k-- )
        {
            for( v = edges[e0^k].dst;; v = edges[ei].dst )
            {
                if( (ei = v->parent) < 0 )
                    break;
                weight = edges[ei^k].weight;
                min_weight = MIN(min_weight, weight);
                assert( min_weight > 0 );
            }
            weight = abs(v->weight);
            min_weight = MIN(min_weight, weight);
            assert( min_weight > 0 );
        }

        // modify weights of the edges along the path and collect orphans
        edges[e0].weight -= min_weight;
        edges[e0^1].weight += min_weight;
        flow += min_weight;

        // k = 1: source tree, k = 0: destination tree
        for( k = 1; k >= 0; k-- )
        {
            for( v = edges[e0^k].dst;; v = edges[ei].dst )
            {
                if( (ei = v->parent) < 0 )
                    break;
                edges[ei^(k^1)].weight += min_weight;
                if( (edges[ei^k].weight -= min_weight) == 0 )
                {
                    if( norphans >= maxOrphans )
                        maxOrphans = icvGCResizeOrphansBuf( orphans, norphans );
                    orphans[norphans++] = v;
                    v->parent = ORPHAN;
                }
            }
            
            v->weight = (short)(v->weight + min_weight*(1-k*2));
            if( v->weight == 0 )
            {
                if( norphans >= maxOrphans )
                    maxOrphans = icvGCResizeOrphansBuf( orphans, norphans );
                orphans[norphans++] = v;
                v->parent = ORPHAN;
            }
        }

        // restore the search trees by finding new parents for the orphans
        curr_ts++;
        while( norphans > 0 )
        {
            GCVtx* v = orphans[--norphans];
            int d, min_dist = INT_MAX;
            e0 = 0;
            vt = v->t;

            for( ei = v->first; ei != 0; ei = edges[ei].next )
            {
                if( edges[ei^(vt^1)].weight == 0 )
                    continue;
                u = edges[ei].dst;
                if( u->t != vt || u->parent == 0 )
                    continue;
                // compute the distance to the tree root
                for( d = 0;; )
                {
                    if( u->ts == curr_ts )
                    {
                        d += u->dist;
                        break;
                    }
                    ej = u->parent;
                    d++;
                    if( ej < 0 )
                    {
                        if( ej == ORPHAN )
                            d = INT_MAX-1;
                        else
                        {
                            u->ts = curr_ts;
                            u->dist = 1;
                        }
                        break;
                    }
                    u = edges[ej].dst;
                }

                // update the distance
                if( ++d < INT_MAX )
                {
                    if( d < min_dist )
                    {
                        min_dist = d;
                        e0 = ei;
                    }
                    for( u = edges[ei].dst; u->ts != curr_ts; u = edges[u->parent].dst )
                    {
                        u->ts = curr_ts;
                        u->dist = --d;
                    }
                }
            }

            if( (v->parent = e0) > 0 )
            {
                v->ts = curr_ts;
                v->dist = min_dist;
                continue;
            }

            /* no parent is found */
            v->ts = 0;
            for( ei = v->first; ei != 0; ei = edges[ei].next )
            {
                u = edges[ei].dst;
                ej = u->parent;
                if( u->t != vt || !ej )
                    continue;
                if( edges[ei^(vt^1)].weight && !u->next )
                {
                    u->next = nilNode;
                    last = last->next = u;
                }
                if( ej > 0 && edges[ej].dst == v )
                {
                    if( norphans >= maxOrphans )
                        maxOrphans = icvGCResizeOrphansBuf( orphans, norphans );
                    orphans[norphans++] = u;
                    u->parent = ORPHAN;
                }
            }
        }
    }

    _orphans = orphans;
    _maxOrphans = maxOrphans;

    return flow;
}


CvStereoGCState* cvCreateStereoGCState( int numberOfDisparities, int maxIters )
{
    CvStereoGCState* state = 0;

    state = (CvStereoGCState*)cvAlloc( sizeof(*state) );
    memset( state, 0, sizeof(*state) );
    state->minDisparity = 0;
    state->numberOfDisparities = numberOfDisparities;
    state->maxIters = maxIters <= 0 ? 3 : maxIters;
    state->Ithreshold = 5;
    state->interactionRadius = 1;
    state->K = state->lambda = state->lambda1 = state->lambda2 = -1.f;
    state->occlusionCost = OCCLUSION_PENALTY;

    return state;
}

void cvReleaseStereoGCState( CvStereoGCState** _state )
{
    CvStereoGCState* state;
    
    if( !_state && !*_state )
        return;

    state = *_state;
    cvReleaseMat( &state->left );
    cvReleaseMat( &state->right );
    cvReleaseMat( &state->ptrLeft );
    cvReleaseMat( &state->ptrRight );
    cvReleaseMat( &state->vtxBuf );
    cvReleaseMat( &state->edgeBuf );
    cvFree( _state );
}

// ||I(x) - J(x')|| =
// min(CUTOFF,
//   min(
//     max(
//       max(minJ(x') - I(x), 0),
//       max(I(x) - maxJ(x'), 0)),
//     max(
//       max(minI(x) - J(x'), 0),
//       max(J(x') - maxI(x), 0)))**2) ==
// min(CUTOFF,
//   min(
//       max(minJ(x') - I(x), 0) +
//       max(I(x) - maxJ(x'), 0),
//
//       max(minI(x) - J(x'), 0) +
//       max(J(x') - maxI(x), 0)))**2)
// where (I, minI, maxI) and
//       (J, minJ, maxJ) are stored as interleaved 3-channel images.
// minI, maxI are computed from I,
// minJ, maxJ are computed from J - see icvInitGraySubPix.
static inline int icvDataCostFuncGraySubpix( const uchar* a, const uchar* b )
{
    int va = a[0], vb = b[0];
    int da = icvTruncTab[b[1] - va + 255] + icvTruncTab[va - b[2] + 255];
    int db = icvTruncTab[a[1] - vb + 255] + icvTruncTab[vb - a[2] + 255];
    return icvCutoffSqrTab[MIN(da,db)];
}

static inline int icvSmoothnessCostFunc( int da, int db, int maxR, const int* stabR, int scale )
{
    return da == db ? 0 : (da == OCCLUDED || db == OCCLUDED ? maxR : stabR[da - db])*scale;
}

static void icvInitGraySubpix( const CvMat* left, const CvMat* right,
                               CvMat* left3, CvMat* right3 )
{
    int k, x, y, rows = left->rows, cols = left->cols;
    
    for( k = 0; k < 2; k++ )
    {
        const CvMat* src = k == 0 ? left : right;
        CvMat* dst = k == 0 ? left3 : right3;
        int sstep = src->step;

        for( y = 0; y < rows; y++ )
        {
            const uchar* sptr = src->data.ptr + sstep*y;
            const uchar* sptr_prev = y > 0 ? sptr - sstep : sptr;
            const uchar* sptr_next = y < rows-1 ? sptr + sstep : sptr;
            uchar* dptr = dst->data.ptr + dst->step*y;
            int v_prev = sptr[0];
            
            for( x = 0; x < cols; x++, dptr += 3 )
            {
                int v = sptr[x], v1, minv = v, maxv = v;
                
                v1 = (v + v_prev)/2;
                minv = MIN(minv, v1); maxv = MAX(maxv, v1);
                v1 = (v + sptr_prev[x])/2;
                minv = MIN(minv, v1); maxv = MAX(maxv, v1);
                v1 = (v + sptr_next[x])/2;
                minv = MIN(minv, v1); maxv = MAX(maxv, v1);
                if( x < cols-1 )
                {
                    v1 = (v + sptr[x+1])/2;
                    minv = MIN(minv, v1); maxv = MAX(maxv, v1);
                }
                v_prev = v;
                dptr[0] = (uchar)v;
                dptr[1] = (uchar)minv;
                dptr[2] = (uchar)maxv;
            }
        }
    }
}

// Optimal K is computed as avg_x(k-th-smallest_d(||I(x)-J(x+d)||)),
// where k = number_of_disparities*0.25.
static float
icvComputeK( CvStereoGCState* state )
{
    int x, y, x1, d, i, j, rows = state->left->rows, cols = state->left->cols, n = 0;
    int mind = state->minDisparity, nd = state->numberOfDisparities, maxd = mind + nd;
    int k = MIN(MAX((nd + 2)/4, 3), nd);
    int *arr = (int*)cvStackAlloc(k*sizeof(arr[0])), delta, t, sum = 0;

    for( y = 0; y < rows; y++ )
    {
        const uchar* lptr = state->left->data.ptr + state->left->step*y;
        const uchar* rptr = state->right->data.ptr + state->right->step*y;
        
        for( x = 0; x < cols; x++ )
        {
            for( d = maxd-1, i = 0; d >= mind; d-- )
            {
                x1 = x - d;
                if( (unsigned)x1 >= (unsigned)cols )
                    continue;
                delta = icvDataCostFuncGraySubpix( lptr + x*3, rptr + x1*3 );
                if( i < k )
                    arr[i++] = delta;
                else
                    for( i = 0; i < k; i++ )
                        if( delta < arr[i] )
                            CV_SWAP( arr[i], delta, t );
            }
            delta = arr[0];
            for( j = 1; j < i; j++ )
                delta = MAX(delta, arr[j]);
            sum += delta;
            n++;
        }
    }

    return (float)sum/n;
}

static int64 icvComputeEnergy( const CvStereoGCState* state, const CvStereoGCState2* state2,
                               bool allOccluded )
{
    int x, y, rows = state->left->rows, cols = state->left->cols;
    int64 E = 0;
    const int* dtab = state2->dataCostFuncTab;
    int maxR = state2->interactionRadius;
    const int* stabR = state2->smoothnessR + CUTOFF;
    const int* stabI = state2->smoothnessGrayDiff + 255;
    const uchar* left = state->left->data.ptr;
    const uchar* right = state->right->data.ptr;
    short* dleft = state->dispLeft->data.s;
    short* dright = state->dispRight->data.s;
    int step = state->left->step;
    int dstep = (int)(state->dispLeft->step/sizeof(short));

    assert( state->left->step == state->right->step &&
        state->dispLeft->step == state->dispRight->step );

    if( allOccluded )
        return (int64)OCCLUSION_PENALTY*rows*cols*2;

    for( y = 0; y < rows; y++, left += step, right += step, dleft += dstep, dright += dstep )
    {
        for( x = 0; x < cols; x++ )
        {
            int d = dleft[x], x1, d1;
            if( d == OCCLUDED )
                E += OCCLUSION_PENALTY;
            else
            {
                x1 = x + d;
                if( (unsigned)x1 >= (unsigned)cols )
                    continue;
                d1 = dright[x1];
                if( d == -d1 )
                    E += dtab[icvDataCostFuncGraySubpix( left + x*3, right + x1*3 )];
            }

            if( x < cols-1 )
            {
                d1 = dleft[x+1];
                E += icvSmoothnessCostFunc(d, d1, maxR, stabR, stabI[left[x*3] - left[x*3+3]] );
            }
            if( y < rows-1 )
            {
                d1 = dleft[x+dstep];
                E += icvSmoothnessCostFunc(d, d1, maxR, stabR, stabI[left[x*3] - left[x*3+step]] );
            }

            d = dright[x];
            if( d == OCCLUDED )
                E += OCCLUSION_PENALTY;

            if( x < cols-1 )
            {
                d1 = dright[x+1];
                E += icvSmoothnessCostFunc(d, d1, maxR, stabR, stabI[right[x*3] - right[x*3+3]] );
            }
            if( y < rows-1 )
            {
                d1 = dright[x+dstep];
                E += icvSmoothnessCostFunc(d, d1, maxR, stabR, stabI[right[x*3] - right[x*3+step]] );
            }
            assert( E >= 0 );
        }
    }

    return E;
}

static inline void icvAddEdge( GCVtx *x, GCVtx* y, GCEdge* edgeBuf, int nedges, int w, int rw )
{
    GCEdge *xy = edgeBuf + nedges, *yx = xy + 1;

    assert( x != 0 && y != 0 );
    xy->dst = y;
    xy->next = x->first;
    xy->weight = (short)w;
    x->first = nedges;

    yx->dst = x;
    yx->next = y->first;
    yx->weight = (short)rw;
    y->first = nedges+1;
}

static inline int icvAddTWeights( GCVtx* vtx, int sourceWeight, int sinkWeight )
{
    int w = vtx->weight;
    if( w > 0 )
        sourceWeight += w;
    else
        sinkWeight -= w;
    vtx->weight = (short)(sourceWeight - sinkWeight);
    return MIN(sourceWeight, sinkWeight);
}

static inline int icvAddTerm( GCVtx* x, GCVtx* y, int A, int B, int C, int D,
                              GCEdge* edgeBuf, int& nedges )
{
    int dE = 0, w;

    assert(B - A + C - D >= 0);
    if( B < A )
    {
        dE += icvAddTWeights(x, D, B);
        dE += icvAddTWeights(y, 0, A - B);
        if( (w = B - A + C - D) != 0 )
        {
            icvAddEdge( x, y, edgeBuf, nedges, 0, w );
            nedges += 2;
        }
    }
    else if( C < D )
    {
        dE += icvAddTWeights(x, D, A + D - C);
        dE += icvAddTWeights(y, 0, C - D);
        if( (w = B - A + C - D) != 0 )
        {
            icvAddEdge( x, y, edgeBuf, nedges, w, 0 );
            nedges += 2;
        }
    }
    else
    {
        dE += icvAddTWeights(x, D, A);
        if( B != A || C != D )
        {
            icvAddEdge( x, y, edgeBuf, nedges, B - A, C - D );
            nedges += 2;
        }
    }
    return dE;
}

static int64 icvAlphaExpand( int64 Eprev, int alpha, CvStereoGCState* state, CvStereoGCState2* state2 )
{
    GCVtx *var, *var1;
    int64 E = 0;
    int delta, E00=0, E0a=0, Ea0=0, Eaa=0;
    int k, a, d, d1, x, y, x1, y1, rows = state->left->rows, cols = state->left->cols;
    int nvtx = 0, nedges = 2;
    GCVtx* vbuf = (GCVtx*)state->vtxBuf->data.ptr;
    GCEdge* ebuf = (GCEdge*)state->edgeBuf->data.ptr;
    int maxR = state2->interactionRadius;
    const int* dtab = state2->dataCostFuncTab;
    const int* stabR = state2->smoothnessR + CUTOFF;
    const int* stabI = state2->smoothnessGrayDiff + 255;
    const uchar* left0 = state->left->data.ptr;
    const uchar* right0 = state->right->data.ptr;
    short* dleft0 = state->dispLeft->data.s;
    short* dright0 = state->dispRight->data.s;
    GCVtx** pleft0 = (GCVtx**)state->ptrLeft->data.ptr;
    GCVtx** pright0 = (GCVtx**)state->ptrRight->data.ptr;
    int step = state->left->step;
    int dstep = (int)(state->dispLeft->step/sizeof(short));
    int pstep = (int)(state->ptrLeft->step/sizeof(GCVtx*));
    int aa[] = { alpha, -alpha };

    //double t = (double)cvGetTickCount();

    assert( state->left->step == state->right->step &&
            state->dispLeft->step == state->dispRight->step &&
            state->ptrLeft->step == state->ptrRight->step );
    for( k = 0; k < 2; k++ )
    {
        ebuf[k].dst = 0;
        ebuf[k].next = 0;
        ebuf[k].weight = 0;
    }

    for( y = 0; y < rows; y++ )
    {
        const uchar* left = left0 + step*y;
        const uchar* right = right0 + step*y;
        const short* dleft = dleft0 + dstep*y;
        const short* dright = dright0 + dstep*y;
        GCVtx** pleft = pleft0 + pstep*y;
        GCVtx** pright = pright0 + pstep*y;
        const uchar* lr[] = { left, right };
        const short* dlr[] = { dleft, dright };
        GCVtx** plr[] = { pleft, pright }; 

        for( k = 0; k < 2; k++ )
        {
            a = aa[k];
            for( y1 = y+(y>0); y1 <= y+(y<rows-1); y1++ )
            {
                const short* disp = (k == 0 ? dleft0 : dright0) + y1*dstep;
                GCVtx** ptr = (k == 0 ? pleft0 : pright0) + y1*pstep;
                for( x = 0; x < cols; x++ )
                {
                    GCVtx* v = ptr[x] = &vbuf[nvtx++];
                    v->first = 0;
                    v->weight = disp[x] == (short)(OCCLUDED ? -OCCLUSION_PENALTY2 : 0);
                }
            }
        }

        for( x = 0; x < cols; x++ )
        {
            d = dleft[x];
            x1 = x + d;
            var = pleft[x];

            // (left + x, right + x + d)
            if( d != alpha && d != OCCLUDED && (unsigned)x1 < (unsigned)cols )
            {
                var1 = pright[x1];
                d1 = dright[x1];
                if( d == -d1 )
                {
                    assert( var1 != 0 );
                    delta = IS_BLOCKED(alpha, d) ? INFINITY : 0;
                    //add inter edge
                    E += icvAddTerm( var, var1,
                        dtab[icvDataCostFuncGraySubpix( left + x*3, right + x1*3 )],
                        delta, delta, 0, ebuf, nedges );
                }
                else if( IS_BLOCKED(alpha, d) )
                    E += icvAddTerm( var, var1, 0, INFINITY, 0, 0, ebuf, nedges );
            }

            // (left + x, right + x + alpha)
            x1 = x + alpha;
            if( (unsigned)x1 < (unsigned)cols )
            {
                var1 = pright[x1];
                d1 = dright[x1];

                E0a = IS_BLOCKED(d, alpha) ? INFINITY : 0;
                Ea0 = IS_BLOCKED(-d1, alpha) ? INFINITY : 0;
                Eaa = dtab[icvDataCostFuncGraySubpix( left + x*3, right + x1*3 )];
                E += icvAddTerm( var, var1, 0, E0a, Ea0, Eaa, ebuf, nedges );
            }

            // smoothness
            for( k = 0; k < 2; k++ )
            {
                GCVtx** p = plr[k];
                const short* disp = dlr[k];
                const uchar* img = lr[k] + x*3;
                int scale;
                var = p[x];
                d = disp[x];
                a = aa[k];

                if( x < cols - 1 )
                {
                    var1 = p[x+1];
                    d1 = disp[x+1];
                    scale = stabI[img[0] - img[3]];
                    E0a = icvSmoothnessCostFunc( d, a, maxR, stabR, scale );
                    Ea0 = icvSmoothnessCostFunc( a, d1, maxR, stabR, scale );
                    E00 = icvSmoothnessCostFunc( d, d1, maxR, stabR, scale );
                    E += icvAddTerm( var, var1, E00, E0a, Ea0, 0, ebuf, nedges );
                }

                if( y < rows - 1 )
                {
                    var1 = p[x+pstep];
                    d1 = disp[x+dstep];
                    scale = stabI[img[0] - img[step]];
                    E0a = icvSmoothnessCostFunc( d, a, maxR, stabR, scale );
                    Ea0 = icvSmoothnessCostFunc( a, d1, maxR, stabR, scale );
                    E00 = icvSmoothnessCostFunc( d, d1, maxR, stabR, scale );
                    E += icvAddTerm( var, var1, E00, E0a, Ea0, 0, ebuf, nedges );
                }
            }

            // visibility term
            if( d != OCCLUDED && IS_BLOCKED(alpha, -d))
            {
                x1 = x + d;
                if( (unsigned)x1 < (unsigned)cols )
                {
                    if( d != -dleft[x1] )
                    {
                        var1 = pleft[x1];
                        E += icvAddTerm( var, var1, 0, INFINITY, 0, 0, ebuf, nedges );
                    }
                }
            }
        }
    }

    //t = (double)cvGetTickCount() - t;
    ebuf[0].weight = ebuf[1].weight = 0;
    E += icvGCMaxFlow( vbuf, nvtx, ebuf, state2->orphans, state2->maxOrphans );

    if( E < Eprev )
    {
        for( y = 0; y < rows; y++ )
        {
            short* dleft = dleft0 + dstep*y;
            short* dright = dright0 + dstep*y;
            GCVtx** pleft = pleft0 + pstep*y;
            GCVtx** pright = pright0 + pstep*y;
            for( x = 0; x < cols; x++ )
            {
                GCVtx* var = pleft[x];
                if( var && var->parent && var->t )
                    dleft[x] = (short)alpha; 

                var = pright[x];
                if( var && var->parent && var->t )
                    dright[x] = (short)-alpha;
            }
        }
    }

    return MIN(E, Eprev);
}


CV_IMPL void cvFindStereoCorrespondenceGC( const CvArr* _left, const CvArr* _right,
    CvArr* _dispLeft, CvArr* _dispRight, CvStereoGCState* state, int useDisparityGuess )
{
    CvStereoGCState2 state2;
    state2.orphans = 0;
    state2.maxOrphans = 0;

    CvMat lstub, *left = cvGetMat( _left, &lstub );
    CvMat rstub, *right = cvGetMat( _right, &rstub );
    CvMat dlstub, *dispLeft = cvGetMat( _dispLeft, &dlstub );
    CvMat drstub, *dispRight = cvGetMat( _dispRight, &drstub );
    CvSize size;
    int iter, i, nZeroExpansions = 0;
    CvRNG rng = cvRNG(-1);
    int* disp;
    CvMat _disp;
    int64 E;

    CV_Assert( state != 0 );
    CV_Assert( CV_ARE_SIZES_EQ(left, right) && CV_ARE_TYPES_EQ(left, right) &&
               CV_MAT_TYPE(left->type) == CV_8UC1 );
    CV_Assert( !dispLeft ||
        (CV_ARE_SIZES_EQ(dispLeft, left) && CV_MAT_CN(dispLeft->type) == 1) );
    CV_Assert( !dispRight ||
        (CV_ARE_SIZES_EQ(dispRight, left) && CV_MAT_CN(dispRight->type) == 1) );

    size = cvGetSize(left);
    if( !state->left || state->left->width != size.width || state->left->height != size.height )
    {
        int pcn = (int)(sizeof(GCVtx*)/sizeof(int));
        int vcn = (int)(sizeof(GCVtx)/sizeof(int));
        int ecn = (int)(sizeof(GCEdge)/sizeof(int));
        cvReleaseMat( &state->left );
        cvReleaseMat( &state->right );
        cvReleaseMat( &state->ptrLeft );
        cvReleaseMat( &state->ptrRight );
        cvReleaseMat( &state->dispLeft );
        cvReleaseMat( &state->dispRight );

        state->left = cvCreateMat( size.height, size.width, CV_8UC3 );
        state->right = cvCreateMat( size.height, size.width, CV_8UC3 );
        state->dispLeft = cvCreateMat( size.height, size.width, CV_16SC1 );
        state->dispRight = cvCreateMat( size.height, size.width, CV_16SC1 );
        state->ptrLeft = cvCreateMat( size.height, size.width, CV_32SC(pcn) );
        state->ptrRight = cvCreateMat( size.height, size.width, CV_32SC(pcn) );
        state->vtxBuf = cvCreateMat( 1, size.height*size.width*2, CV_32SC(vcn) );
        state->edgeBuf = cvCreateMat( 1, size.height*size.width*12 + 16, CV_32SC(ecn) );
    }

    if( !useDisparityGuess )
    {
        cvSet( state->dispLeft, cvScalarAll(OCCLUDED));
        cvSet( state->dispRight, cvScalarAll(OCCLUDED));
    }
    else
    {
        CV_Assert( dispLeft && dispRight );
        cvConvert( dispLeft, state->dispLeft );
        cvConvert( dispRight, state->dispRight );
    }

    state2.Ithreshold = state->Ithreshold;
    state2.interactionRadius = state->interactionRadius;
    state2.lambda = cvRound(state->lambda*DENOMINATOR);
    state2.lambda1 = cvRound(state->lambda1*DENOMINATOR);
    state2.lambda2 = cvRound(state->lambda2*DENOMINATOR);
    state2.K = cvRound(state->K*DENOMINATOR);

    icvInitStereoConstTabs();
    icvInitGraySubpix( left, right, state->left, state->right );
    disp = (int*)cvStackAlloc( state->numberOfDisparities*sizeof(disp[0]) );
    _disp = cvMat( 1, state->numberOfDisparities, CV_32S, disp );
    cvRange( &_disp, state->minDisparity, state->minDisparity + state->numberOfDisparities );
    cvRandShuffle( &_disp, &rng );

    if( state2.lambda < 0 && (state2.K < 0 || state2.lambda1 < 0 || state2.lambda2 < 0) )
    {
        float L = icvComputeK(state)*0.2f;
        state2.lambda = cvRound(L*DENOMINATOR);
    }

    if( state2.K < 0 )
        state2.K = state2.lambda*5;
    if( state2.lambda1 < 0 )
        state2.lambda1 = state2.lambda*3;
    if( state2.lambda2 < 0 )
        state2.lambda2 = state2.lambda;

    icvInitStereoTabs( &state2 );

    E = icvComputeEnergy( state, &state2, !useDisparityGuess );
    for( iter = 0; iter < state->maxIters; iter++ )
    {
        for( i = 0; i < state->numberOfDisparities; i++ )
        {
            int alpha = disp[i];
            int64 Enew = icvAlphaExpand( E, -alpha, state, &state2 );
            if( Enew < E )
            {
                nZeroExpansions = 0;
                E = Enew;
            }
            else if( ++nZeroExpansions >= state->numberOfDisparities )
                break;
        }
    }

    if( dispLeft )
        cvConvert( state->dispLeft, dispLeft );
    if( dispRight )
        cvConvert( state->dispRight, dispRight );

    cvFree( &state2.orphans );
}
