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
#include "precomp.hpp"

namespace cv
{

int Subdiv2D::nextEdge(int edge) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    return qedges[edge >> 2].next[edge & 3];
}

int Subdiv2D::rotateEdge(int edge, int rotate) const
{
    return (edge & ~3) + ((edge + rotate) & 3);
}

int Subdiv2D::symEdge(int edge) const
{
    return edge ^ 2;
}

int Subdiv2D::getEdge(int edge, int nextEdgeType) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    edge = qedges[edge >> 2].next[(edge + nextEdgeType) & 3];
    return (edge & ~3) + ((edge + (nextEdgeType >> 4)) & 3);
}

int Subdiv2D::edgeOrg(int edge, CV_OUT Point2f* orgpt) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    int vidx = qedges[edge >> 2].pt[edge & 3];
    if( orgpt )
    {
        CV_DbgAssert((size_t)vidx < vtx.size());
        *orgpt = vtx[vidx].pt;
    }
    return vidx;
}

int Subdiv2D::edgeDst(int edge, CV_OUT Point2f* dstpt) const
{
    CV_DbgAssert((size_t)(edge >> 2) < qedges.size());
    int vidx = qedges[edge >> 2].pt[(edge + 2) & 3];
    if( dstpt )
    {
        CV_DbgAssert((size_t)vidx < vtx.size());
        *dstpt = vtx[vidx].pt;
    }
    return vidx;
}


Point2f Subdiv2D::getVertex(int vertex, CV_OUT int* firstEdge) const
{
    CV_DbgAssert((size_t)vertex < vtx.size());
    if( firstEdge )
        *firstEdge = vtx[vertex].firstEdge;
    return vtx[vertex].pt;
}


Subdiv2D::Subdiv2D(bool _flag)
{
    flag = _flag;
    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;
}

Subdiv2D::Subdiv2D(Rect rect, bool _flag)
{
    flag = _flag;

    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;

    initDelaunay(rect);
}


Subdiv2D::QuadEdge::QuadEdge()
{
    next[0] = next[1] = next[2] = next[3] = 0;
    pt[0] = pt[1] = pt[2] = pt[3] = 0;
}

Subdiv2D::QuadEdge::QuadEdge(int edgeidx)
{
    CV_DbgAssert((edgeidx & 3) == 0);
    next[0] = edgeidx;
    next[1] = edgeidx+3;
    next[2] = edgeidx+2;
    next[3] = edgeidx+1;

    pt[0] = pt[1] = pt[2] = pt[3] = 0;
}

bool Subdiv2D::QuadEdge::isfree() const
{
    return next[0] <= 0;
}

Subdiv2D::Vertex::Vertex()
{
    firstEdge = 0;
    type = PTTYPE_FREE;
}

Subdiv2D::Vertex::Vertex(Point2f _pt, bool _isvirtual, int _firstEdge)
{
    firstEdge = _firstEdge;
    type = _isvirtual ? PTTYPE_VORONOI : PTTYPE_DELAUNAY;
    pt = _pt;
}

Subdiv2D::Vertex::Vertex(Point2f _pt, int _type, int _firstEdge)
{
    firstEdge = _firstEdge;
    type = _type;
    pt = _pt;
}

bool Subdiv2D::Vertex::isvirtual() const
{
    return type == PTTYPE_VORONOI || type == PTTYPE_VORONOI_BOUNDING;
}

bool Subdiv2D::Vertex::isfree() const
{
    return type == PTTYPE_FREE;
}

bool Subdiv2D::Vertex::isbounding() const
{
    return type == PTTYPE_DELAUNAY_BOUNDING || type == PTTYPE_VORONOI_BOUNDING;
}

void Subdiv2D::splice( int edgeA, int edgeB )
{
    int& a_next = qedges[edgeA >> 2].next[edgeA & 3];
    int& b_next = qedges[edgeB >> 2].next[edgeB & 3];
    int a_rot = rotateEdge(a_next, 1);
    int b_rot = rotateEdge(b_next, 1);
    int& a_rot_next = qedges[a_rot >> 2].next[a_rot & 3];
    int& b_rot_next = qedges[b_rot >> 2].next[b_rot & 3];
    std::swap(a_next, b_next);
    std::swap(a_rot_next, b_rot_next);
}

void Subdiv2D::setEdgePoints(int edge, int orgPt, int dstPt)
{
    qedges[edge >> 2].pt[edge & 3] = orgPt;
    qedges[edge >> 2].pt[(edge + 2) & 3] = dstPt;
    vtx[orgPt].firstEdge = edge;
    vtx[dstPt].firstEdge = edge ^ 2;
}

int Subdiv2D::connectEdges( int edgeA, int edgeB )
{
    int edge = newEdge();

    splice(edge, getEdge(edgeA, NEXT_AROUND_LEFT));
    splice(symEdge(edge), edgeB);

    setEdgePoints(edge, edgeDst(edgeA), edgeOrg(edgeB));
    return edge;
}

void Subdiv2D::swapEdges( int edge )
{
    int sedge = symEdge(edge);
    int a = getEdge(edge, PREV_AROUND_ORG);
    int b = getEdge(sedge, PREV_AROUND_ORG);

    splice(edge, a);
    splice(sedge, b);

    setEdgePoints(edge, edgeDst(a), edgeDst(b));

    splice(edge, getEdge(a, NEXT_AROUND_LEFT));
    splice(sedge, getEdge(b, NEXT_AROUND_LEFT));
}

static int counterClockwise(Point2f a, Point2f b, Point2f c)
{
    double cp = (b - a).cross(c - a);
    return cp > FLT_EPSILON ? 1 : (cp < -FLT_EPSILON ? -1 : 0);
}

static bool parallel(Point2f u, Point2f v)
{
    return abs(u.cross(v)) < FLT_EPSILON;
}

static bool equal(Point2f a, Point2f b)
{
    return abs(a.x - b.x) < FLT_EPSILON && abs(a.y - b.y) < FLT_EPSILON;
}

// variable point of the form r * v + c, where r is an independent variable approaching +infinity
struct Point2fEx
{
    Point2f c, v;

    Point2fEx(Point2f c) : c(c), v(0.f, 0.f) { }
    Point2fEx(Point2f c, Point2f v) : c(c), v(v) { }
};

static int counterClockwiseEx(Point2fEx a, Point2fEx b, Point2fEx c)
{
    const static Point2f o(0.f, 0.f);

    if (!equal(a.v, o) && !equal(b.v, o) && !equal(c.v, o)) {
        if (equal(a.c, o) && equal(b.c, o) && equal(c.c, o)) {
            return counterClockwise(a.v, b.v, c.v);
        }
        CV_Error(CV_StsNotImplemented, "");
    }

    if (!equal(a.v, o) && !equal(b.v, o) && equal(c.v, o)) {
        if (!parallel(a.v, b.v)) {
            return counterClockwise(a.v, b.v, o);
        } else {
            if (equal(a.v, b.v)) {
                return counterClockwise(o, a.v, b.c - a.c);
            }
            if (equal(a.c, b.c)) {
                return counterClockwise(a.v, b.v, c.c - a.c);
            }
        }
        CV_Error(CV_StsNotImplemented, "");
    }

    if (equal(a.v, o) && equal(b.v, o) && !equal(c.v, o)) {
        return !parallel(b.c - a.c, c.v) ?
               counterClockwise(o, b.c - a.c, c.v) : counterClockwise(a.c, b.c, c.c);
    }

    if (equal(a.v, o) && equal(b.v, o) && equal(c.v, o)) {
        return counterClockwise(a.c, b.c, c.c);
    }

    return counterClockwiseEx(b, c, a);
}

static int rightOfEx(Point2fEx c, Point2fEx a, Point2fEx b)
{
    return counterClockwiseEx(c, b, a);
}

static int inCircle(Point2f a, Point2f b, Point2f c, Point2f d)
{
    const double eps = FLT_EPSILON * 0.125;

    double val =
            ((double)a.x * a.x + (double)a.y * a.y) * ( c - b ).cross( d - b );
    val -=  ((double)b.x * b.x + (double)b.y * b.y) * ( c - a ).cross( d - a );
    val +=  ((double)c.x * c.x + (double)c.y * c.y) * ( b - a ).cross( d - a );
    val -=  ((double)d.x * d.x + (double)d.y * d.y) * ( b - a ).cross( c - a );

    return val > eps ? 1 : val < -eps ? -1 : 0;
}

static double distance(Point2f a, Point2f b)
{
    return norm(b - a);
}

static int leftOf(Point2f c, Point2f a, Point2f b)
{
    return counterClockwise(c, a, b);
}

static int inCircleEx(Point2fEx a, Point2fEx b, Point2fEx c, Point2fEx d)
{
    const static Point2f o(0.f, 0.f);

    CV_Assert(equal(a.v, o) || equal(b.v, o) || equal(c.v, o) || equal(d.v, o));

    if (!equal(a.v, o) && !equal(b.v, o) && !equal(c.v, o) && equal(d.v, o)) {
        CV_Assert(equal(a.c, o) && equal(b.c, o) && equal(c.c, o));

        return counterClockwiseEx(a, b, c);
    }

    if (!equal(a.v, o) && equal(b.v, o) && !equal(c.v, o) && equal(d.v, o)) {
        CV_Assert(equal(a.c, o) && equal(c.c, o));

        if (!parallel(d.c - b.c, c.v - a.v)) {
            return leftOf(d.c - b.c, o, c.v - a.v);
        } else {
            double od = distance(o, d.c);
            double ob = distance(o, b.c);

            if (abs(od - ob) < FLT_EPSILON) {
                return 0;
            } else {
                return (od < ob && counterClockwiseEx(a, b, c) > 0) ||
                       (od > ob && counterClockwiseEx(a, b, c) < 0) ? 1 : -1;
            }
        }
    }

    if (!equal(a.v, o) && !equal(b.v, o) && equal(c.v, o) && equal(d.v, o)) {
        return -inCircleEx(a, c, b, d);
    }

    if (equal(a.v, o) && equal(b.v, o) && equal(c.v, o) && !equal(d.v, o)) {
        CV_Assert(equal(d.c, o));

        if (!parallel(b.c - a.c, c.c - b.c)) {
            return -counterClockwise(a.c, b.c, c.c);
        } else {
            double ab = distance(a.c, b.c);
            double ac = distance(a.c, c.c);
            double bc = distance(b.c, c.c);

            return ac + bc > ab ? leftOf(d.v, o, b.c - a.c) : leftOf(d.v, o, c.c - b.c);
        }
    }

    if (equal(a.v, o) && equal(b.v, o) && equal(c.v, o) && equal(d.v, o)) {
        return inCircle(a.c, b.c, c.c, d.c);
    }

    return -inCircleEx(b, c, d, a);
}

static int rightOf(Point2f c, Point2f a, Point2f b) {
    return counterClockwise(c, b, a);
}

#define ORIGIN (Point2f(0.f, 0.f))
#define FIRST_EDGE_1_ROTATED(vertex) (rotateEdge(vtx[vertex].firstEdge, 1))
#define EDGE_MIDDLE(edge) ((vtx[edgeOrg(edge)].pt + vtx[edgeDst(edge)].pt) / 2.f)
#define VTX_PT(vertex)                                                                       \
        (vtx[vertex].isbounding() ?                                                              \
                (vtx[vertex].isvirtual() ?                                                       \
                        (Point2fEx(EDGE_MIDDLE(FIRST_EDGE_1_ROTATED(vertex)), vtx[vertex].pt)) : \
                        (Point2fEx(ORIGIN, vtx[vertex].pt))) :                                   \
                (vtx[vertex].pt))


int Subdiv2D::isRightOf(Point2f pt, int edge) const
{
    if (flag) {
        return rightOfEx(pt, VTX_PT(edgeOrg(edge)), VTX_PT(edgeDst(edge)));
    } else {
        return rightOf(pt, vtx[edgeOrg(edge)].pt, vtx[edgeDst(edge)].pt);
    }
}

int Subdiv2D::newEdge()
{
    if( freeQEdge <= 0 )
    {
        qedges.push_back(QuadEdge());
        freeQEdge = (int)(qedges.size()-1);
    }
    int edge = freeQEdge*4;
    freeQEdge = qedges[edge >> 2].next[1];
    qedges[edge >> 2] = QuadEdge(edge);
    return edge;
}

void Subdiv2D::deleteEdge(int edge)
{
    CV_DbgAssert((size_t)(edge >> 2) < (size_t)qedges.size());
    splice( edge, getEdge(edge, PREV_AROUND_ORG) );
    int sedge = symEdge(edge);
    splice(sedge, getEdge(sedge, PREV_AROUND_ORG) );

    edge >>= 2;
    qedges[edge].next[0] = 0;
    qedges[edge].next[1] = freeQEdge;
    freeQEdge = edge;
}

int Subdiv2D::newPoint(Point2f pt, int type, int firstEdge)
{
    if( freePoint == 0 )
    {
        vtx.push_back(Vertex());
        freePoint = (int)(vtx.size()-1);
    }
    int vidx = freePoint;
    freePoint = vtx[vidx].firstEdge;
    vtx[vidx] = Vertex(pt, type, firstEdge);

    return vidx;
}

int Subdiv2D::newPoint(Point2f pt, bool isvirtual, int firstEdge)
{
    return newPoint(pt, isvirtual ? PTTYPE_VORONOI : PTTYPE_DELAUNAY, firstEdge);
}

void Subdiv2D::deletePoint(int vidx)
{
    CV_DbgAssert( (size_t)vidx < vtx.size() );
    vtx[vidx].firstEdge = freePoint;
    vtx[vidx].type = PTTYPE_FREE;
    freePoint = vidx;
}

int Subdiv2D::locate(Point2f pt, int& _edge, int& _vertex)
{
    CV_INSTRUMENT_REGION();

    int edge = recentEdge;
    CV_Assert(edge > 0);

    for (;;) {
        int onext_edge = nextEdge( edge );
        int dprev_edge = getEdge( edge, PREV_AROUND_DST );

        int right_of_curr, right_of_onext, right_of_dprev;
        if (flag) {
            right_of_curr = rightOfEx(pt, VTX_PT(edgeOrg(edge)), VTX_PT(edgeDst(edge)));
            right_of_onext = rightOfEx(pt, VTX_PT(edgeOrg(onext_edge)), VTX_PT(edgeDst(onext_edge)));
            right_of_dprev = rightOfEx(pt, VTX_PT(edgeOrg(dprev_edge)), VTX_PT(edgeDst(dprev_edge)));
        } else {
            right_of_curr = rightOf(pt, vtx[edgeOrg(edge)].pt, vtx[edgeDst(edge)].pt);
            right_of_onext = rightOf(pt, vtx[edgeOrg(onext_edge)].pt, vtx[edgeDst(onext_edge)].pt);
            right_of_dprev = rightOf(pt, vtx[edgeOrg(dprev_edge)].pt, vtx[edgeDst(dprev_edge)].pt);
        }

        if (right_of_curr == 0 && (right_of_onext == 0 || right_of_dprev == 0)) {
            recentEdge = edge;

            _edge = 0;
            _vertex = right_of_onext == 0 ? edgeOrg(edge) : edgeDst(edge);
            return PTLOC_VERTEX;
        }
        else if (right_of_curr > 0) {
            edge = symEdge(edge);
        }
        else if (right_of_onext <= 0) {
            edge = onext_edge;
        }
        else if (right_of_dprev <= 0) {
            edge = dprev_edge;
        }
        else {
            recentEdge = edge;

            _vertex = 0;
            _edge = edge;
            return right_of_curr == 0 ? PTLOC_ON_EDGE : PTLOC_INSIDE;
        }
    }
}


int Subdiv2D::insert(Point2f pt)
{
    CV_INSTRUMENT_REGION();

    int curr_point = 0, curr_edge = 0, deleted_edge = 0;
    int location = locate( pt, curr_edge, curr_point );

    if( location == PTLOC_VERTEX )
        return curr_point;

    if( location == PTLOC_ON_EDGE )
    {
        deleted_edge = curr_edge;
        recentEdge = curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        deleteEdge(deleted_edge);
    }

    assert( curr_edge != 0 );
    validGeometry = false;

    curr_point = newPoint(pt, PTTYPE_DELAUNAY);
    int base_edge = newEdge();
    int first_point = edgeOrg(curr_edge);
    setEdgePoints(base_edge, first_point, curr_point);
    splice(base_edge, curr_edge);

    do
    {
        base_edge = connectEdges( curr_edge, symEdge(base_edge) );
        curr_edge = getEdge(base_edge, PREV_AROUND_ORG);
    }
    while( edgeDst(curr_edge) != first_point );

    curr_edge = getEdge( base_edge, PREV_AROUND_ORG );

    int i, max_edges = (int)(qedges.size()*4);

    for( i = 0; i < max_edges; i++ )
    {
        int temp_dst = 0, curr_org = 0, curr_dst = 0;
        int temp_edge = getEdge( curr_edge, PREV_AROUND_ORG );

        temp_dst = edgeDst( temp_edge );
        curr_org = edgeOrg( curr_edge );
        curr_dst = edgeDst( curr_edge );

        bool swap;
        if (flag) {
            swap = rightOfEx(VTX_PT(temp_dst), VTX_PT(curr_org), VTX_PT(curr_org) ) > 0 &&
                   inCircleEx(VTX_PT(curr_org), VTX_PT(temp_dst), VTX_PT(curr_dst), VTX_PT(curr_point) ) > 0;
        } else {
            swap = rightOf(vtx[temp_dst].pt, vtx[curr_org].pt, vtx[curr_dst].pt) &&
                    inCircle(vtx[curr_org].pt, vtx[temp_dst].pt, vtx[curr_dst].pt, vtx[curr_point].pt);
        }

        if(swap)
        {
            swapEdges( curr_edge );
            curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        }
        else if( curr_org == first_point )
            break;
        else
            curr_edge = getEdge( nextEdge( curr_edge ), PREV_AROUND_LEFT );
    }

    return curr_point;
}

void Subdiv2D::insert(const std::vector<Point2f>& ptvec)
{
    CV_INSTRUMENT_REGION();

    for( size_t i = 0; i < ptvec.size(); i++ )
        insert(ptvec[i]);
}

void Subdiv2D::initDelaunay( Rect rect )
{
    CV_INSTRUMENT_REGION();

    vtx.clear();
    qedges.clear();

    recentEdge = 0;
    validGeometry = false;

    topLeft = Point2f( (float)rect.x, (float)rect.y + (float)rect.height );
    bottomRight = Point2f( (float)rect.x + (float)rect.width, (float)rect.y );

    Point2f ppA, ppB, ppC;
    if (flag) {
        ppA = Point2f(1.f, 0.f);
        ppB = Point2f(cos(2.f * (float) M_PI / 3.f), sin(2.f * (float) M_PI / 3.f));
        ppC = Point2f(cos(4.f * (float) M_PI / 3.f), sin(4.f * (float) M_PI / 3.f));
    } else {
        float big_coord = 3.f * MAX( rect.width, rect.height );
        float rx = (float)rect.x;
        float ry = (float)rect.y;

        ppA = Point2f( rx + big_coord, ry );
        ppB = Point2f( rx, ry + big_coord );
        ppC = Point2f( rx - big_coord, ry - big_coord );
    }

    vtx.push_back(Vertex());
    qedges.push_back(QuadEdge());

    freeQEdge = 0;
    freePoint = 0;

    int pA = newPoint(ppA, PTTYPE_DELAUNAY_BOUNDING);
    int pB = newPoint(ppB, PTTYPE_DELAUNAY_BOUNDING);
    int pC = newPoint(ppC, PTTYPE_DELAUNAY_BOUNDING);

    int edge_AB = newEdge();
    int edge_BC = newEdge();
    int edge_CA = newEdge();

    setEdgePoints( edge_AB, pA, pB );
    setEdgePoints( edge_BC, pB, pC );
    setEdgePoints( edge_CA, pC, pA );

    splice( edge_AB, symEdge( edge_CA ));
    splice( edge_BC, symEdge( edge_AB ));
    splice( edge_CA, symEdge( edge_BC ));

    recentEdge = edge_AB;
}


void Subdiv2D::clearVoronoi()
{
    size_t i, total = qedges.size();

    for( i = 0; i < total; i++ )
        qedges[i].pt[1] = qedges[i].pt[3] = 0;

    total = vtx.size();
    for( i = 0; i < total; i++ )
    {
        if( vtx[i].isvirtual() )
            deletePoint((int)i);
    }

    validGeometry = false;
}


static Point2f computeVoronoiPoint(Point2f org0, Point2f dst0, Point2f org1, Point2f dst1)
{
    double a0 = dst0.x - org0.x;
    double b0 = dst0.y - org0.y;
    double c0 = -0.5*(a0 * (dst0.x + org0.x) + b0 * (dst0.y + org0.y));

    double a1 = dst1.x - org1.x;
    double b1 = dst1.y - org1.y;
    double c1 = -0.5*(a1 * (dst1.x + org1.x) + b1 * (dst1.y + org1.y));

    double det = a0 * b1 - a1 * b0;

    if( det != 0 )
    {
        det = 1. / det;
        return Point2f((float) ((b0 * c1 - b1 * c0) * det),
                       (float) ((a1 * c0 - a0 * c1) * det));
    }

    return Point2f(FLT_MAX, FLT_MAX);
}


void Subdiv2D::calcVoronoi()
{
    // check if it is already calculated
    if( validGeometry )
        return;

    clearVoronoi();

    // loop through all quad-edges, except for the first 3 (#1, #2, #3 - 0 is reserved for "NULL" pointer)
    for (int quad_edge = 4; quad_edge < (int)qedges.size(); ++quad_edge) {
        if (qedges[quad_edge].isfree()) {
            continue;
        }

        for (int i = 0, edge0 = quad_edge * 4; i < 2; ++i, edge0 = symEdge(edge0)) {
            if (!qedges[edge0 >> 2].pt[3 - (edge0 & 2)]) {
                int edge1 = getEdge( edge0, NEXT_AROUND_LEFT );
                int edge2 = getEdge( edge1, NEXT_AROUND_LEFT );

                Point2f org0, dst0, dst1;
                int edge0_org = edgeOrg(edge0, &org0);
                int edge0_dst = edgeDst(edge0, &dst0);
                int edge1_dst = edgeDst(edge1, &dst1);

                int voronoi_point = 0;

                if (!vtx[edge0_org].isbounding() && !vtx[edge0_dst].isbounding() && !vtx[edge1_dst].isbounding()) {
                    voronoi_point = newPoint(computeVoronoiPoint(org0, dst0, dst0, dst1), PTTYPE_VORONOI,
                                             rotateEdge(edge0, 3));
                }

                if (!vtx[edge0_org].isbounding() && !vtx[edge0_dst].isbounding() && vtx[edge1_dst].isbounding()) {
                    if (flag) {
                        Point2f normal(-(dst0 - org0).y, (dst0 - org0).x);
                        voronoi_point = newPoint(normal / norm(normal), PTTYPE_VORONOI_BOUNDING,
                                                 rotateEdge(edge0, 3));
                    } else {
                        voronoi_point = newPoint(computeVoronoiPoint(org0, dst0, dst0, dst1), PTTYPE_VORONOI_BOUNDING,
                                                 rotateEdge(edge0, 3));
                    }
                }

                if (!vtx[edge0_org].isbounding() && vtx[edge0_dst].isbounding() && vtx[edge1_dst].isbounding()) {
                    if (flag) {
                        Point2f normal((dst1 - dst0).y, -(dst1 - dst0).x);
                        voronoi_point = newPoint(normal / norm(normal), PTTYPE_VORONOI_BOUNDING,
                                                 rotateEdge(edge1, 3));
                    } else {
                        voronoi_point = newPoint(computeVoronoiPoint(org0, dst0, dst0, dst1), PTTYPE_VORONOI_BOUNDING,
                                                 rotateEdge(edge1, 3));
                    }
                }

                qedges[edge0 >> 2].pt[3 - (edge0 & 2)] =
                qedges[edge1 >> 2].pt[3 - (edge1 & 2)] =
                qedges[edge2 >> 2].pt[3 - (edge2 & 2)] = voronoi_point;
            }
        }
    }

    validGeometry = true;
}


int Subdiv2D::findNearest(Point2f pt, Point2f* nearestPt)
{
    CV_INSTRUMENT_REGION();

    if( !validGeometry )
        calcVoronoi();

    int vertex, edge;
    int loc = locate( pt, edge, vertex );

    if (loc == PTLOC_VERTEX) {
        if (nearestPt) {
            *nearestPt = vtx[vertex].pt;
        }
        return vertex;
    }
    if (loc == PTLOC_ON_EDGE || loc == PTLOC_INSIDE) {
        vertex = !vtx[edgeOrg(edge)].isbounding() ? edgeOrg(edge) : edgeDst(edge);
    }
    if (loc == PTLOC_INSIDE) {
        int lnext_dst = edgeDst(getEdge(edge, NEXT_AROUND_LEFT));
        vertex = !vtx[lnext_dst].isbounding() ? lnext_dst : vertex;
    }
    if (vtx[vertex].isbounding()) {
        // empty subdivision
        if (nearestPt) {
            nearestPt = NULL;
        }
        return 0;
    }

    for (;;) {
        CV_Assert(!vtx[vertex].isbounding());

        int next_vertex = 0;
        int first_edge = edge = rotateEdge(vtx[vertex].firstEdge, 1);

        do {
            bool outside;
            if (flag) {
                outside = rightOfEx(pt, VTX_PT(edgeOrg(edge)), VTX_PT(edgeDst(edge))) > 0;
            } else {
                outside = rightOf(pt, vtx[edgeOrg(edge)].pt, vtx[edgeDst(edge)].pt);
            }
            if (outside) {
                next_vertex = edgeOrg(rotateEdge(edge, 1));
                break;
            }

            edge = getEdge( edge, NEXT_AROUND_LEFT );
        } while (edge != first_edge);

        if (!next_vertex) {
            break;
        }

        vertex = next_vertex;
    }

    if (nearestPt) {
        *nearestPt = vtx[vertex].pt;
    }
    return vertex;
}

void Subdiv2D::getEdgeList(std::vector<Vec4f>& edgeList) const
{
    edgeList.clear();

    for( size_t i = 4; i < qedges.size(); i++ )
    {
        if( qedges[i].isfree() )
            continue;
        if( qedges[i].pt[0] > 0 && qedges[i].pt[2] > 0 )
        {
            if (flag) {
                if (vtx[qedges[i].pt[0]].isbounding() || vtx[qedges[i].pt[2]].isbounding()) {
                    continue;
                }
            }
            Point2f org = vtx[qedges[i].pt[0]].pt;
            Point2f dst = vtx[qedges[i].pt[2]].pt;
            edgeList.push_back(Vec4f(org.x, org.y, dst.x, dst.y));
        }
    }
}

void Subdiv2D::getLeadingEdgeList(std::vector<int>& leadingEdgeList) const
{
    leadingEdgeList.clear();
    int i, total = (int)(qedges.size()*4);
    std::vector<bool> edgemask(total, false);

    for( i = 4; i < total; i += 2 )
    {
        if( edgemask[i] )
            continue;
        int edge = i;
        edgemask[edge] = true;
        edge = getEdge(edge, NEXT_AROUND_LEFT);
        edgemask[edge] = true;
        edge = getEdge(edge, NEXT_AROUND_LEFT);
        edgemask[edge] = true;
        leadingEdgeList.push_back(i);
    }
}

void Subdiv2D::getTriangleList(std::vector<Vec6f>& triangleList) const
{
    triangleList.clear();
    std::vector<bool> edgemask(qedges.size() * 4, false);

    for( int quad_edge = 4; quad_edge < (int)qedges.size(); ++quad_edge ) {
        if (qedges[quad_edge].isfree()) {
            continue;
        }

        for ( int i = 0, edge_a = quad_edge * 4; i < 2; ++i, edge_a = symEdge(edge_a) ) {
            if( edgemask[edge_a] ) {
                continue;
            }
            int edge_b = getEdge(edge_a, NEXT_AROUND_LEFT);
            int edge_c = getEdge(edge_b, NEXT_AROUND_LEFT);
            edgemask[edge_a] = edgemask[edge_b] = edgemask[edge_c] = true;
            Point2f a, b, c;
            if (    vtx[edgeOrg(edge_a, &a)].isbounding() ||
                    vtx[edgeOrg(edge_b, &b)].isbounding() ||
                    vtx[edgeOrg(edge_c, &c)].isbounding()) {
                continue;
            }
            triangleList.push_back(Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
        }
    }
}

static Point2f intersect(Point2f a, Point2f b, Point2f c, Point2f d)
{
    double a1 = b.y - a.y;
    double b1 = a.x - b.x;
    double c1 = a1 * a.x + b1 * a.y;

    double a2 = d.y - c.y;
    double b2 = c.x - d.x;
    double c2 = a2 * c.x + b2 * c.y;

    double det = a1 * b2 - a2 * b1;

    if (det == 0) {
        return Point2f(FLT_MAX, FLT_MAX);
    }

    return Point2f((float)((b2 * c1 - b1 * c2) / det), (float)((a1 * c2 - a2 * c1) / det));
}

static void cropEdgeEx(Point2fEx &edge_org, Point2fEx &edge_dst, Point2fEx boundary_org, Point2fEx boundary_dst)
{
    static const Point2f o(0.f, 0.f);

    CV_Assert(boundary_org.v == o && boundary_dst.v == o);

    int edge_org_right_of_boundary = rightOfEx(edge_org, boundary_org, boundary_dst);
    int edge_dst_right_of_boundary = rightOfEx(edge_dst, boundary_org, boundary_dst);

    int boundary_org_right_of_edge = rightOfEx(boundary_org, edge_org, edge_dst);
    int boundary_dst_right_of_edge = rightOfEx(boundary_dst, edge_org, edge_dst);

    if (    edge_org_right_of_boundary != edge_dst_right_of_boundary &&
            boundary_org_right_of_edge != boundary_dst_right_of_edge) {

        // either org_ideal == o || dst_ideal == o (or both)
        Point2f x = intersect(
                edge_org.v != o ? edge_dst.c + edge_org.v : edge_org.c,
                edge_dst.v != o ? edge_org.c + edge_dst.v : edge_dst.c,
                boundary_org.c, boundary_dst.c);

        if (edge_org_right_of_boundary > 0) {
            edge_org.c = x;
            edge_org.v = o;
        }
        if (edge_dst_right_of_boundary > 0) {
            edge_dst.c = x;
            edge_dst.v = o;
        }
    }
}

static bool putIfAbsent(Point2f p, std::vector<Point2f> &v) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (equal(p, v[i])) {
                return false;
            }
        }
        v.push_back(p);
        return true;
    }

static void sortAround(Point2f o, std::vector<Point2f> &v, int l, int r)
{
    if (l >= r) {
        return;
    }

    Point2f m = v[l];
    int i = l - 1;
    int j = r + 1;

    while (i < j) {
        do { i++; } while (leftOf(v[i], o, m) < 0);
        do { j--; } while (leftOf(v[j], o, m) > 0);
        if (i < j) {
            std::swap(v[i], v[j]);
        }
    }

    sortAround(o, v, l, j);
    sortAround(o, v, j + 1, r);
}


void Subdiv2D::getVoronoiFacetList(const std::vector<int>& idx,
                                   CV_OUT std::vector<std::vector<Point2f> >& facetList,
                                   CV_OUT std::vector<Point2f>& facetCenters)
{
    static const Point2f o(0.f, 0.f);

    const Point2f topRight(bottomRight.x, topLeft.y);
    const Point2f bottomLeft(topLeft.x, bottomRight.y);

    calcVoronoi();
    facetList.clear();
    facetCenters.clear();

    std::vector<Point2f> buf;

    // if there are multiple owners, the corner is already in the lists of edges of the owners
    int bl_site, br_site, tr_site, tl_site;
    if (flag) {
        bl_site = findNearest(bottomLeft);
        br_site = findNearest(bottomRight);
        tr_site = findNearest(topRight);
        tl_site = findNearest(topLeft);
    }

    size_t i, total;
    if( idx.empty() )
        i = 4, total = vtx.size();
    else
        i = 0, total = idx.size();

    for( ; i < total; i++ )
    {
        int k = idx.empty() ? (int)i : idx[i];

        if( vtx[k].isfree() || vtx[k].isvirtual() || vtx[k].isbounding() )
            continue;
        int edge = rotateEdge(vtx[k].firstEdge, 1), t = edge;

        // gather points
        buf.clear();
        do
        {
            int edge_org = edgeOrg(t);
            int edge_dst = edgeDst(t);

            if (flag) {
                Point2fEx org = VTX_PT(edge_org), dst = VTX_PT(edge_dst);

                cropEdgeEx(org, dst, bottomLeft, bottomRight);
                cropEdgeEx(org, dst, bottomRight, topRight);
                cropEdgeEx(org, dst, topRight, topLeft);
                cropEdgeEx(org, dst, topLeft, bottomLeft);

                // edge is either inclusively inside or exclusively outside of rectangle
                if (org.v == o) {
                    if (    bottomLeft.x <= org.c.x && org.c.x <= topRight.x &&
                            bottomLeft.y <= org.c.y && org.c.y <= topRight.y) {

                        if (buf.empty() || !equal(buf.back(), org.c)) {
                            buf.push_back(org.c);
                        }
                        if (buf.empty() || !equal(buf.back(), dst.c)) {
                            buf.push_back(dst.c);
                        }
                    }
                }
            } else {
                buf.push_back(vtx[edgeOrg(t)].pt);
            }

            t = getEdge( t, NEXT_AROUND_LEFT );
        }
        while( t != edge );

        if (flag) {
            if (buf.size() > 1 && equal(buf.front(), buf.back())) {
                buf.pop_back();
            }

            bool unsorted = false;

            if (k == bl_site) {
                unsorted |= putIfAbsent(bottomLeft, buf);
            }
            if (k == br_site) {
                unsorted |= putIfAbsent(bottomRight, buf);
            }
            if (k == tr_site) {
                unsorted |= putIfAbsent(topRight, buf);
            }
            if (k == tl_site) {
                unsorted |= putIfAbsent(topLeft, buf);
            }

            if (unsorted) {
                // always has non-empty interior
                Point2f center(0.f, 0.f);
                for (size_t j = 0; j < buf.size(); ++j) {
                    center += buf[j];
                }
                center /= (float) buf.size();
                sortAround(center, buf, 0, (int)buf.size() - 1);
            }

        }

        facetList.push_back(buf);
        facetCenters.push_back(vtx[k].pt);
    }
}


void Subdiv2D::checkSubdiv() const
{
    int i, j, total = (int)qedges.size();

    for( i = 0; i < total; i++ )
    {
        const QuadEdge& qe = qedges[i];

        if( qe.isfree() )
            continue;

        for( j = 0; j < 4; j++ )
        {
            int e = (int)(i*4 + j);
            int o_next = nextEdge(e);
            int o_prev = getEdge(e, PREV_AROUND_ORG );
            int d_prev = getEdge(e, PREV_AROUND_DST );
            int d_next = getEdge(e, NEXT_AROUND_DST );

            // check points
            CV_Assert( edgeOrg(e) == edgeOrg(o_next));
            CV_Assert( edgeOrg(e) == edgeOrg(o_prev));
            CV_Assert( edgeDst(e) == edgeDst(d_next));
            CV_Assert( edgeDst(e) == edgeDst(d_prev));

            if( j % 2 == 0 )
            {
                CV_Assert( edgeDst(o_next) == edgeOrg(d_prev));
                CV_Assert( edgeDst(o_prev) == edgeOrg(d_next));
                CV_Assert( getEdge(getEdge(getEdge(e,NEXT_AROUND_LEFT),NEXT_AROUND_LEFT),NEXT_AROUND_LEFT) == e );
                CV_Assert( getEdge(getEdge(getEdge(e,NEXT_AROUND_RIGHT),NEXT_AROUND_RIGHT),NEXT_AROUND_RIGHT) == e);
            }
        }
    }
}

}

/* End of file. */
