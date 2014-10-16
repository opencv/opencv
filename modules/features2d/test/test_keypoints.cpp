/*M///////////////////////////////////////////////////////////////////////////////////////
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

#include "test_precomp.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core_c.h"

using namespace std;
using namespace cv;

const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";

const int TABLE_SIZE = 400;

static const float chitab3[]=
{
    0.f,  0.0150057f,  0.0239478f,  0.0315227f,
    0.0383427f,  0.0446605f,  0.0506115f,  0.0562786f,
    0.0617174f,  0.0669672f,  0.0720573f,  0.0770099f,
    0.081843f,  0.0865705f,  0.0912043f,  0.0957541f,
    0.100228f,  0.104633f,  0.108976f,  0.113261f,
    0.117493f,  0.121676f,  0.125814f,  0.12991f,
    0.133967f,  0.137987f,  0.141974f,  0.145929f,
    0.149853f,  0.15375f,  0.15762f,  0.161466f,
    0.165287f,  0.169087f,  0.172866f,  0.176625f,
    0.180365f,  0.184088f,  0.187794f,  0.191483f,
    0.195158f,  0.198819f,  0.202466f,  0.2061f,
    0.209722f,  0.213332f,  0.216932f,  0.220521f,
    0.2241f,  0.22767f,  0.231231f,  0.234783f,
    0.238328f,  0.241865f,  0.245395f,  0.248918f,
    0.252435f,  0.255947f,  0.259452f,  0.262952f,
    0.266448f,  0.269939f,  0.273425f,  0.276908f,
    0.280386f,  0.283862f,  0.287334f,  0.290803f,
    0.29427f,  0.297734f,  0.301197f,  0.304657f,
    0.308115f,  0.311573f,  0.315028f,  0.318483f,
    0.321937f,  0.32539f,  0.328843f,  0.332296f,
    0.335749f,  0.339201f,  0.342654f,  0.346108f,
    0.349562f,  0.353017f,  0.356473f,  0.35993f,
    0.363389f,  0.366849f,  0.37031f,  0.373774f,
    0.377239f,  0.380706f,  0.384176f,  0.387648f,
    0.391123f,  0.3946f,  0.39808f,  0.401563f,
    0.405049f,  0.408539f,  0.412032f,  0.415528f,
    0.419028f,  0.422531f,  0.426039f,  0.429551f,
    0.433066f,  0.436586f,  0.440111f,  0.44364f,
    0.447173f,  0.450712f,  0.454255f,  0.457803f,
    0.461356f,  0.464915f,  0.468479f,  0.472049f,
    0.475624f,  0.479205f,  0.482792f,  0.486384f,
    0.489983f,  0.493588f,  0.4972f,  0.500818f,
    0.504442f,  0.508073f,  0.511711f,  0.515356f,
    0.519008f,  0.522667f,  0.526334f,  0.530008f,
    0.533689f,  0.537378f,  0.541075f,  0.54478f,
    0.548492f,  0.552213f,  0.555942f,  0.55968f,
    0.563425f,  0.56718f,  0.570943f,  0.574715f,
    0.578497f,  0.582287f,  0.586086f,  0.589895f,
    0.593713f,  0.597541f,  0.601379f,  0.605227f,
    0.609084f,  0.612952f,  0.61683f,  0.620718f,
    0.624617f,  0.628526f,  0.632447f,  0.636378f,
    0.64032f,  0.644274f,  0.648239f,  0.652215f,
    0.656203f,  0.660203f,  0.664215f,  0.668238f,
    0.672274f,  0.676323f,  0.680384f,  0.684457f,
    0.688543f,  0.692643f,  0.696755f,  0.700881f,
    0.70502f,  0.709172f,  0.713339f,  0.717519f,
    0.721714f,  0.725922f,  0.730145f,  0.734383f,
    0.738636f,  0.742903f,  0.747185f,  0.751483f,
    0.755796f,  0.760125f,  0.76447f,  0.768831f,
    0.773208f,  0.777601f,  0.782011f,  0.786438f,
    0.790882f,  0.795343f,  0.799821f,  0.804318f,
    0.808831f,  0.813363f,  0.817913f,  0.822482f,
    0.827069f,  0.831676f,  0.836301f,  0.840946f,
    0.84561f,  0.850295f,  0.854999f,  0.859724f,
    0.864469f,  0.869235f,  0.874022f,  0.878831f,
    0.883661f,  0.888513f,  0.893387f,  0.898284f,
    0.903204f,  0.908146f,  0.913112f,  0.918101f,
    0.923114f,  0.928152f,  0.933214f,  0.938301f,
    0.943413f,  0.94855f,  0.953713f,  0.958903f,
    0.964119f,  0.969361f,  0.974631f,  0.979929f,
    0.985254f,  0.990608f,  0.99599f,  1.0014f,
    1.00684f,  1.01231f,  1.01781f,  1.02335f,
    1.02891f,  1.0345f,  1.04013f,  1.04579f,
    1.05148f,  1.05721f,  1.06296f,  1.06876f,
    1.07459f,  1.08045f,  1.08635f,  1.09228f,
    1.09826f,  1.10427f,  1.11032f,  1.1164f,
    1.12253f,  1.1287f,  1.1349f,  1.14115f,
    1.14744f,  1.15377f,  1.16015f,  1.16656f,
    1.17303f,  1.17954f,  1.18609f,  1.19269f,
    1.19934f,  1.20603f,  1.21278f,  1.21958f,
    1.22642f,  1.23332f,  1.24027f,  1.24727f,
    1.25433f,  1.26144f,  1.26861f,  1.27584f,
    1.28312f,  1.29047f,  1.29787f,  1.30534f,
    1.31287f,  1.32046f,  1.32812f,  1.33585f,
    1.34364f,  1.3515f,  1.35943f,  1.36744f,
    1.37551f,  1.38367f,  1.39189f,  1.4002f,
    1.40859f,  1.41705f,  1.42561f,  1.43424f,
    1.44296f,  1.45177f,  1.46068f,  1.46967f,
    1.47876f,  1.48795f,  1.49723f,  1.50662f,
    1.51611f,  1.52571f,  1.53541f,  1.54523f,
    1.55517f,  1.56522f,  1.57539f,  1.58568f,
    1.59611f,  1.60666f,  1.61735f,  1.62817f,
    1.63914f,  1.65025f,  1.66152f,  1.67293f,
    1.68451f,  1.69625f,  1.70815f,  1.72023f,
    1.73249f,  1.74494f,  1.75757f,  1.77041f,
    1.78344f,  1.79669f,  1.81016f,  1.82385f,
    1.83777f,  1.85194f,  1.86635f,  1.88103f,
    1.89598f,  1.91121f,  1.92674f,  1.94257f,
    1.95871f,  1.97519f,  1.99201f,  2.0092f,
    2.02676f,  2.04471f,  2.06309f,  2.08189f,
    2.10115f,  2.12089f,  2.14114f,  2.16192f,
    2.18326f,  2.2052f,  2.22777f,  2.25101f,
    2.27496f,  2.29966f,  2.32518f,  2.35156f,
    2.37886f,  2.40717f,  2.43655f,  2.46709f,
    2.49889f,  2.53206f,  2.56673f,  2.60305f,
    2.64117f,  2.6813f,  2.72367f,  2.76854f,
    2.81623f,  2.86714f,  2.92173f,  2.98059f,
    3.04446f,  3.1143f,  3.19135f,  3.27731f,
    3.37455f,  3.48653f,  3.61862f,  3.77982f,
    3.98692f,  4.2776f,  4.77167f,  133.333f
};

struct MSCRNode;

struct TempMSCR
{
    MSCRNode* head;
    MSCRNode* tail;
    double m; // the margin used to prune area later
    int size;
};

struct MSCRNode
{
    MSCRNode* shortcut;
    // to make the finding of root less painful
    MSCRNode* prev;
    MSCRNode* next;
    // a point double-linked list
    TempMSCR* tmsr;
    // the temporary msr (set to NULL at every re-initialise)
    TempMSCR* gmsr;
    // the global msr (once set, never to NULL)
    int index;
    // the index of the node, at this point, it should be x at the first 16-bits, and y at the last 16-bits.
    int rank;
    int reinit;
    int size, sizei;
    double dt, di;
    double s;
};

struct MSCREdge
{
    double chi;
    MSCRNode* left;
    MSCRNode* right;
};

static double ChiSquaredDistance( uchar* x, uchar* y )
{
    return (double)((x[0]-y[0])*(x[0]-y[0]))/(double)(x[0]+y[0]+1e-10)+
    (double)((x[1]-y[1])*(x[1]-y[1]))/(double)(x[1]+y[1]+1e-10)+
    (double)((x[2]-y[2])*(x[2]-y[2]))/(double)(x[2]+y[2]+1e-10);
}

static void initMSCRNode( MSCRNode* node )
{
    node->gmsr = node->tmsr = NULL;
    node->reinit = 0xffff;
    node->rank = 0;
    node->sizei = node->size = 1;
    node->prev = node->next = node->shortcut = node;
}

// the preprocess to get the edge list with proper gaussian blur
static int preprocessMSER_8UC3( MSCRNode* node,
                               MSCREdge* edge,
                               double* total,
                               CvMat* src,
                               CvMat* mask,
                               CvMat* dx,
                               CvMat* dy,
                               int Ne,
                               int edgeBlurSize )
{
    int srccpt = src->step-src->cols*3;
    uchar* srcptr = src->data.ptr;
    uchar* lastptr = src->data.ptr+3;
    double* dxptr = dx->data.db;
    for ( int i = 0; i < src->rows; i++ )
    {
        for ( int j = 0; j < src->cols-1; j++ )
        {
            *dxptr = ChiSquaredDistance( srcptr, lastptr );
            dxptr++;
            srcptr += 3;
            lastptr += 3;
        }
        srcptr += srccpt+3;
        lastptr += srccpt+3;
    }
    srcptr = src->data.ptr;
    lastptr = src->data.ptr+src->step;
    double* dyptr = dy->data.db;
    for ( int i = 0; i < src->rows-1; i++ )
    {
        for ( int j = 0; j < src->cols; j++ )
        {
            *dyptr = ChiSquaredDistance( srcptr, lastptr );
            dyptr++;
            srcptr += 3;
            lastptr += 3;
        }
        srcptr += srccpt;
        lastptr += srccpt;
    }
    // get dx and dy and blur it
    if ( edgeBlurSize >= 1 )
    {
        Mat _dx(dx->rows, dx->cols, dx->type, dx->data.ptr, dx->step);
        Mat _dy(dy->rows, dy->cols, dy->type, dy->data.ptr, dy->step);
        GaussianBlur( _dx, _dx, Size(edgeBlurSize, edgeBlurSize), 0 );
        GaussianBlur( _dy, _dy, Size(edgeBlurSize, edgeBlurSize), 0 );
    }
    dxptr = dx->data.db;
    dyptr = dy->data.db;
    // assian dx, dy to proper edge list and initialize mscr node
    // the nasty code here intended to avoid extra loops
    if ( mask )
    {
        Ne = 0;
        int maskcpt = mask->step-mask->cols+1;
        uchar* maskptr = mask->data.ptr;
        MSCRNode* nodeptr = node;
        initMSCRNode( nodeptr );
        nodeptr->index = 0;
        *total += edge->chi = *dxptr;
        if ( maskptr[0] && maskptr[1] )
        {
            edge->left = nodeptr;
            edge->right = nodeptr+1;
            edge++;
            Ne++;
        }
        dxptr++;
        nodeptr++;
        maskptr++;
        for ( int i = 1; i < src->cols-1; i++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = i;
            if ( maskptr[0] && maskptr[1] )
            {
                *total += edge->chi = *dxptr;
                edge->left = nodeptr;
                edge->right = nodeptr+1;
                edge++;
                Ne++;
            }
            dxptr++;
            nodeptr++;
            maskptr++;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = src->cols-1;
        nodeptr++;
        maskptr += maskcpt;
        for ( int i = 1; i < src->rows-1; i++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = i<<16;
            if ( maskptr[0] )
            {
                if ( maskptr[-mask->step] )
                {
                    *total += edge->chi = *dyptr;
                    edge->left = nodeptr-src->cols;
                    edge->right = nodeptr;
                    edge++;
                    Ne++;
                }
                if ( maskptr[1] )
                {
                    *total += edge->chi = *dxptr;
                    edge->left = nodeptr;
                    edge->right = nodeptr+1;
                    edge++;
                    Ne++;
                }
            }
            dyptr++;
            dxptr++;
            nodeptr++;
            maskptr++;
            for ( int j = 1; j < src->cols-1; j++ )
            {
                initMSCRNode( nodeptr );
                nodeptr->index = (i<<16)|j;
                if ( maskptr[0] )
                {
                    if ( maskptr[-mask->step] )
                    {
                        *total += edge->chi = *dyptr;
                        edge->left = nodeptr-src->cols;
                        edge->right = nodeptr;
                        edge++;
                        Ne++;
                    }
                    if ( maskptr[1] )
                    {
                        *total += edge->chi = *dxptr;
                        edge->left = nodeptr;
                        edge->right = nodeptr+1;
                        edge++;
                        Ne++;
                    }
                }
                dyptr++;
                dxptr++;
                nodeptr++;
                maskptr++;
            }
            initMSCRNode( nodeptr );
            nodeptr->index = (i<<16)|(src->cols-1);
            if ( maskptr[0] && maskptr[-mask->step] )
            {
                *total += edge->chi = *dyptr;
                edge->left = nodeptr-src->cols;
                edge->right = nodeptr;
                edge++;
                Ne++;
            }
            dyptr++;
            nodeptr++;
            maskptr += maskcpt;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = (src->rows-1)<<16;
        if ( maskptr[0] )
        {
            if ( maskptr[1] )
            {
                *total += edge->chi = *dxptr;
                edge->left = nodeptr;
                edge->right = nodeptr+1;
                edge++;
                Ne++;
            }
            if ( maskptr[-mask->step] )
            {
                *total += edge->chi = *dyptr;
                edge->left = nodeptr-src->cols;
                edge->right = nodeptr;
                edge++;
                Ne++;
            }
        }
        dxptr++;
        dyptr++;
        nodeptr++;
        maskptr++;
        for ( int i = 1; i < src->cols-1; i++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = ((src->rows-1)<<16)|i;
            if ( maskptr[0] )
            {
                if ( maskptr[1] )
                {
                    *total += edge->chi = *dxptr;
                    edge->left = nodeptr;
                    edge->right = nodeptr+1;
                    edge++;
                    Ne++;
                }
                if ( maskptr[-mask->step] )
                {
                    *total += edge->chi = *dyptr;
                    edge->left = nodeptr-src->cols;
                    edge->right = nodeptr;
                    edge++;
                    Ne++;
                }
            }
            dxptr++;
            dyptr++;
            nodeptr++;
            maskptr++;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = ((src->rows-1)<<16)|(src->cols-1);
        if ( maskptr[0] && maskptr[-mask->step] )
        {
            *total += edge->chi = *dyptr;
            edge->left = nodeptr-src->cols;
            edge->right = nodeptr;
            Ne++;
        }
    } else {
        MSCRNode* nodeptr = node;
        initMSCRNode( nodeptr );
        nodeptr->index = 0;
        *total += edge->chi = *dxptr;
        dxptr++;
        edge->left = nodeptr;
        edge->right = nodeptr+1;
        edge++;
        nodeptr++;
        for ( int i = 1; i < src->cols-1; i++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = i;
            *total += edge->chi = *dxptr;
            dxptr++;
            edge->left = nodeptr;
            edge->right = nodeptr+1;
            edge++;
            nodeptr++;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = src->cols-1;
        nodeptr++;
        for ( int i = 1; i < src->rows-1; i++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = i<<16;
            *total += edge->chi = *dyptr;
            dyptr++;
            edge->left = nodeptr-src->cols;
            edge->right = nodeptr;
            edge++;
            *total += edge->chi = *dxptr;
            dxptr++;
            edge->left = nodeptr;
            edge->right = nodeptr+1;
            edge++;
            nodeptr++;
            for ( int j = 1; j < src->cols-1; j++ )
            {
                initMSCRNode( nodeptr );
                nodeptr->index = (i<<16)|j;
                *total += edge->chi = *dyptr;
                dyptr++;
                edge->left = nodeptr-src->cols;
                edge->right = nodeptr;
                edge++;
                *total += edge->chi = *dxptr;
                dxptr++;
                edge->left = nodeptr;
                edge->right = nodeptr+1;
                edge++;
                nodeptr++;
            }
            initMSCRNode( nodeptr );
            nodeptr->index = (i<<16)|(src->cols-1);
            *total += edge->chi = *dyptr;
            dyptr++;
            edge->left = nodeptr-src->cols;
            edge->right = nodeptr;
            edge++;
            nodeptr++;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = (src->rows-1)<<16;
        *total += edge->chi = *dxptr;
        dxptr++;
        edge->left = nodeptr;
        edge->right = nodeptr+1;
        edge++;
        *total += edge->chi = *dyptr;
        dyptr++;
        edge->left = nodeptr-src->cols;
        edge->right = nodeptr;
        edge++;
        nodeptr++;
        for ( int i = 1; i < src->cols-1; i++ )
        {
            initMSCRNode( nodeptr );
            nodeptr->index = ((src->rows-1)<<16)|i;
            *total += edge->chi = *dxptr;
            dxptr++;
            edge->left = nodeptr;
            edge->right = nodeptr+1;
            edge++;
            *total += edge->chi = *dyptr;
            dyptr++;
            edge->left = nodeptr-src->cols;
            edge->right = nodeptr;
            edge++;
            nodeptr++;
        }
        initMSCRNode( nodeptr );
        nodeptr->index = ((src->rows-1)<<16)|(src->cols-1);
        *total += edge->chi = *dyptr;
        edge->left = nodeptr-src->cols;
        edge->right = nodeptr;
    }
    return Ne;
}

class LessThanEdge
{
public:
    bool operator()(const MSCREdge& a, const MSCREdge& b) const { return a.chi < b.chi; }
};

// to find the root of one region
static MSCRNode* findMSCR( MSCRNode* x )
{
    MSCRNode* prev = x;
    MSCRNode* next;
    for ( ; ; )
    {
        next = x->shortcut;
        x->shortcut = prev;
        if ( next == x ) break;
        prev= x;
        x = next;
    }
    MSCRNode* root = x;
    for ( ; ; )
    {
        prev = x->shortcut;
        x->shortcut = root;
        if ( prev == x ) break;
        x = prev;
    }
    return root;
}

struct MSERParams
{
    MSERParams( int _delta=5, int _min_area=60, int _max_area=14400,
                      double _max_variation=0.25, double _min_diversity=.2,
                      int _max_evolution=200, double _area_threshold=1.01,
                      double _min_margin=0.003, int _edge_blur_size=5 )
    {
        delta = _delta;
        minArea = _min_area;
        maxArea = _max_area;
        maxVariation = _max_variation;
        minDiversity = _min_diversity;
        maxEvolution = _max_evolution;
        areaThreshold = _area_threshold;
        minMargin = _min_margin;
        edgeBlurSize = _edge_blur_size;
    }

    int delta;
    int minArea;
    int maxArea;
    double maxVariation;
    double minDiversity;
    int maxEvolution;
    double areaThreshold;
    double minMargin;
    int edgeBlurSize;
};

// the stable mscr should be:
// bigger than minArea and smaller than maxArea
// differ from its ancestor more than minDiversity
static bool MSCRStableCheck( MSCRNode* x, MSERParams params )
{
    if ( x->size <= params.minArea || x->size >= params.maxArea )
        return 0;
    if ( x->gmsr == NULL )
        return 1;
    double div = (double)(x->size-x->gmsr->size)/(double)x->size;
    return div > params.minDiversity;
}

static void
extractMSER_8UC3( CvMat* src,
                 CvMat* mask,
                 vector<vector<Point> >& msers,
                 MSERParams params )
{
    msers.clear();
    MSCRNode* map = (MSCRNode*)cvAlloc( src->cols*src->rows*sizeof(map[0]) );
    int Ne = src->cols*src->rows*2-src->cols-src->rows;
    MSCREdge* edge = (MSCREdge*)cvAlloc( Ne*sizeof(edge[0]) );
    TempMSCR* mscr = (TempMSCR*)cvAlloc( src->cols*src->rows*sizeof(mscr[0]) );
    double emean = 0;
    CvMat* dx = cvCreateMat( src->rows, src->cols-1, CV_64FC1 );
    CvMat* dy = cvCreateMat( src->rows-1, src->cols, CV_64FC1 );
    Ne = preprocessMSER_8UC3( map, edge, &emean, src, mask, dx, dy, Ne, params.edgeBlurSize );
    emean = emean / (double)Ne;
    std::sort(edge, edge + Ne, LessThanEdge());
    MSCREdge* edge_ub = edge+Ne;
    MSCREdge* edgeptr = edge;
    TempMSCR* mscrptr = mscr;
    // the evolution process
    for ( int i = 0; i < params.maxEvolution; i++ )
    {
        double k = (double)i/(double)params.maxEvolution*(TABLE_SIZE-1);
        int ti = cvFloor(k);
        double reminder = k-ti;
        double thres = emean*(chitab3[ti]*(1-reminder)+chitab3[ti+1]*reminder);
        // to process all the edges in the list that chi < thres
        while ( edgeptr < edge_ub && edgeptr->chi < thres )
        {
            MSCRNode* lr = findMSCR( edgeptr->left );
            MSCRNode* rr = findMSCR( edgeptr->right );
            // get the region root (who is responsible)
            if ( lr != rr )
            {
                // rank idea take from: N-tree Disjoint-Set Forests for Maximally Stable Extremal Regions
                if ( rr->rank > lr->rank )
                {
                    MSCRNode* tmp;
                    CV_SWAP( lr, rr, tmp );
                } else if ( lr->rank == rr->rank ) {
                    // at the same rank, we will compare the size
                    if ( lr->size > rr->size )
                    {
                        MSCRNode* tmp;
                        CV_SWAP( lr, rr, tmp );
                    }
                    lr->rank++;
                }
                rr->shortcut = lr;
                lr->size += rr->size;
                // join rr to the end of list lr (lr is a endless double-linked list)
                lr->prev->next = rr;
                lr->prev = rr->prev;
                rr->prev->next = lr;
                rr->prev = lr;
                // area threshold force to reinitialize
                if ( lr->size > (lr->size-rr->size)*params.areaThreshold )
                {
                    lr->sizei = lr->size;
                    lr->reinit = i;
                    if ( lr->tmsr != NULL )
                    {
                        lr->tmsr->m = lr->dt-lr->di;
                        lr->tmsr = NULL;
                    }
                    lr->di = edgeptr->chi;
                    lr->s = 1e10;
                }
                lr->dt = edgeptr->chi;
                if ( i > lr->reinit )
                {
                    double s = (double)(lr->size-lr->sizei)/(lr->dt-lr->di);
                    if ( s < lr->s )
                    {
                        // skip the first one and check stablity
                        if ( i > lr->reinit+1 && MSCRStableCheck( lr, params ) )
                        {
                            if ( lr->tmsr == NULL )
                            {
                                lr->gmsr = lr->tmsr = mscrptr;
                                mscrptr++;
                            }
                            lr->tmsr->size = lr->size;
                            lr->tmsr->head = lr;
                            lr->tmsr->tail = lr->prev;
                            lr->tmsr->m = 0;
                        }
                        lr->s = s;
                    }
                }
            }
            edgeptr++;
        }
        if ( edgeptr >= edge_ub )
            break;
    }
    for ( TempMSCR* ptr = mscr; ptr < mscrptr; ptr++ )
        // to prune area with margin less than minMargin
        if ( ptr->m > params.minMargin )
        {
            vector<Point> mser;
            MSCRNode* lpt = ptr->head;
            for ( int i = 0; i < ptr->size; i++ )
            {
                Point pt;
                pt.x = (lpt->index)&0xffff;
                pt.y = (lpt->index)>>16;
                lpt = lpt->next;
                mser.push_back(pt);
            }
            msers.push_back(mser);
        }
    cvReleaseMat( &dx );
    cvReleaseMat( &dy );
    cvFree( &mscr );
    cvFree( &edge );
    cvFree( &map );
}

/****************************************************************************************\
*                                     Test for KeyPoint                                  *
\****************************************************************************************/

class CV_FeatureDetectorKeypointsTest : public cvtest::BaseTest
{
public:
    CV_FeatureDetectorKeypointsTest(const Ptr<FeatureDetector>& _detector) :
        detector(_detector) {}

protected:
    virtual void run(int)
    {
        CV_Assert(detector);
        string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;

        // Read the test image.
        Mat image = imread(imgFilename);
        if(image.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        vector<KeyPoint> keypoints;
        vector<vector<Point> > msers;
        CvMat src = image;

        extractMSER_8UC3( &src, 0, msers, MSERParams());

        detector->detect(image, keypoints);

        if(keypoints.empty())
        {
            ts->printf(cvtest::TS::LOG, "Detector can't find keypoints in image.\n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        Rect r(0, 0, image.cols, image.rows);
        for(size_t i = 0; i < keypoints.size(); i++)
        {
            const KeyPoint& kp = keypoints[i];

            if(!r.contains(kp.pt))
            {
                ts->printf(cvtest::TS::LOG, "KeyPoint::pt is out of image (x=%f, y=%f).\n", kp.pt.x, kp.pt.y);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            if(kp.size <= 0.f)
            {
                ts->printf(cvtest::TS::LOG, "KeyPoint::size is not positive (%f).\n", kp.size);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            if((kp.angle < 0.f && kp.angle != -1.f) || kp.angle >= 360.f)
            {
                ts->printf(cvtest::TS::LOG, "KeyPoint::angle is out of range [0, 360). It's %f.\n", kp.angle);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }

    Ptr<FeatureDetector> detector;
};


// Registration of tests

TEST(Features2d_Detector_Keypoints_BRISK, validation)
{
    CV_FeatureDetectorKeypointsTest test(BRISK::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_FAST, validation)
{
    CV_FeatureDetectorKeypointsTest test(FastFeatureDetector::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_HARRIS, validation)
{
    CV_FeatureDetectorKeypointsTest test(GFTTDetector::create(1000, 0.01, 1, 3, true, 0.04));
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_GFTT, validation)
{
    CV_FeatureDetectorKeypointsTest test(GFTTDetector::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_MSER, validation)
{
    CV_FeatureDetectorKeypointsTest test(MSER::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_ORB, validation)
{
    CV_FeatureDetectorKeypointsTest test(ORB::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_KAZE, validation)
{
    CV_FeatureDetectorKeypointsTest test(KAZE::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_AKAZE, validation)
{
    CV_FeatureDetectorKeypointsTest test_kaze(AKAZE::create(AKAZE::DESCRIPTOR_KAZE));
    test_kaze.safe_run();

    CV_FeatureDetectorKeypointsTest test_mldb(AKAZE::create(AKAZE::DESCRIPTOR_MLDB));
    test_mldb.safe_run();
}
