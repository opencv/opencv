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


Subdiv2D::Subdiv2D()
{
    reset();
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
        : type(PTTYPE_FREE), firstEdge(0) { }

Subdiv2D::Vertex::Vertex(Point2f pt, int type, int firstEdge)
        : pt(pt), type(type), firstEdge(firstEdge) { }

bool Subdiv2D::Vertex::isvoronoi() const
{
    return type == PTTYPE_VORONOI || type == PTTYPE_VORONOI_IDEAL;
}

bool Subdiv2D::Vertex::isfree() const
{
    return type == PTTYPE_FREE;
}

bool Subdiv2D::Vertex::isideal() const {
    return type == PTTYPE_DELAUNAY_IDEAL || type == PTTYPE_VORONOI_IDEAL;
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

static double distance(Point2f a, Point2f b)
{
    return norm(b - a);
}

static bool parallel(Point2f u, Point2f v)
{
    return abs(u.cross(v)) < FLT_EPSILON;
}

static int counterClockwise(Point2f a, Point2f b, Point2f c)
{
    double cp = (b - a).cross(c - a);
    return cp > FLT_EPSILON ? 1 : (cp < -FLT_EPSILON ? -1 : 0);
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

static int leftOf(Point2f c, Point2f a, Point2f b)
{
    return counterClockwise(c, a, b);
}

/* assuming standard part of ideal points to be (0.f, 0.f) */
static int counterClockwiseEx(Point2f a, Point2f b, Point2f c, bool idealA, bool idealB, bool idealC)
{
    const static Point2f o(0.f, 0.f);

    if (idealA && idealB && idealC) {
        return counterClockwise(a, b, c);
    }

    if (idealA && idealB && !idealC) {
        return !parallel(a, b) ? counterClockwise(a, b, o) : counterClockwise(a, b, c);
    }

    if (!idealA && !idealB && idealC) {
        return !parallel(b - a, c) ? counterClockwise(o, b - a, c) : counterClockwise(a, b, o);
    }

    if (!idealA && !idealB && !idealC) {
        return counterClockwise(a, b, c);
    }

    return counterClockwiseEx(b, c, a, idealB, idealC, idealA);
}

/* assuming standard part of ideal points to be (0.f, 0.f) */
static int inCircleEx(
        Point2f a, Point2f b, Point2f c, Point2f d, bool idealA, bool idealB, bool idealC, bool idealD)
{
    const static Point2f o(0.f, 0.f);

    if (idealA && idealB && idealC && !idealD) {
        return counterClockwiseEx(a, b, c, true, true, true);
    }

    if (idealA && !idealB && idealC && !idealD) {
        if (!parallel(d - b, c - a)) {
            return leftOf(d - b, o, c - a);
        } else {
            double od = distance(o, d);
            double ob = distance(o, b);

            if (abs(od - ob) < FLT_EPSILON) {
                return 0;
            } else {
                return (od < ob && counterClockwiseEx(a, b, c, true, false, true) > 0) ||
                       (od > ob && counterClockwiseEx(a, b, c, true, false, true) < 0) ? 1 : -1;
            }
        }
    }

    if (idealA && idealB && !idealC && !idealD) {
        return -inCircleEx(a, c, b, d, true, false, true, false);
    }

    if (!idealA && !idealB && !idealC && idealD) {
        if (!parallel(b - a, c - b)) {
            return -counterClockwise(a, b, c);
        } else {
            return !c.inside(Rect2f(a, b)) ? leftOf(d, o, b - a) : leftOf(d, o, c - b);
        }
    }

    if (!idealA && !idealB && !idealC && !idealD) {
        return inCircle(a, b, c, d);
    }

    return -inCircleEx(b, c, d, a, idealB, idealC, idealD, idealA);
}

static int rightOfEx(Point2f c, Point2f a, Point2f b, bool idealC, bool idealA, bool idealB)
{
    return counterClockwiseEx(c, b, a, idealC, idealB, idealA);
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

void Subdiv2D::deletePoint(int vidx)
{
    CV_DbgAssert( (size_t)vidx < vtx.size() );
    vtx[vidx].firstEdge = freePoint;
    vtx[vidx].type = PTTYPE_FREE;
    freePoint = vidx;
}

int Subdiv2D::locateInternal(Point2f pt, int &edge, int &vertex)
{
    CV_INSTRUMENT_REGION();

    int curr_edge = recentEdge;
    CV_Assert(curr_edge > 0);

    for (;;) {
        int onext_edge = getEdge(curr_edge, NEXT_AROUND_ORG);
        int dprev_edge = getEdge(curr_edge, PREV_AROUND_DST);

        int curr_org = edgeOrg(curr_edge);
        int curr_dst = edgeDst(curr_edge);
        int onext_org = edgeOrg(onext_edge);
        int onext_dst = edgeDst(onext_edge);
        int dprev_org = edgeOrg(dprev_edge);
        int dprev_dst = edgeDst(dprev_edge);

        int right_of_curr = rightOfEx(
                pt, vtx[curr_org].pt, vtx[curr_dst].pt,
                false, vtx[curr_org].isideal(), vtx[curr_dst].isideal());
        int right_of_onext = rightOfEx(
                pt, vtx[onext_org].pt, vtx[onext_dst].pt,
                false, vtx[onext_org].isideal(), vtx[onext_dst].isideal());
        int right_of_dprev = rightOfEx(
                pt, vtx[dprev_org].pt, vtx[dprev_dst].pt,
                false, vtx[dprev_org].isideal(), vtx[dprev_dst].isideal());

        if (right_of_curr == 0 && (right_of_onext == 0 || right_of_dprev == 0)) {
            recentEdge = curr_edge;

            edge = 0;
            vertex = right_of_onext == 0 ? curr_org : curr_dst;
            return PTLOC_VERTEX;
        }
        else if (right_of_curr > 0) {
            curr_edge = symEdge(curr_edge);
        }
        else if (right_of_onext <= 0) {
            curr_edge = onext_edge;
        }
        else if (right_of_dprev <= 0) {
            curr_edge = dprev_edge;
        }
        else {
            recentEdge = curr_edge;

            vertex = 0;
            edge = curr_edge;
            return right_of_curr == 0 ? PTLOC_EDGE : PTLOC_INSIDE;
        }
    }
}

int Subdiv2D::locate(Point2f pt, int& _edge, int& _vertex)
{
    CV_INSTRUMENT_REGION();

    int result = locateInternal(pt, _edge, _vertex);

    if (result == PTLOC_EDGE || result == PTLOC_INSIDE) {
        int edge_org = edgeOrg(_edge);
        int edge_dst = edgeDst(_edge);
        if (vtx[edge_org].isideal() || vtx[edge_dst].isideal()) {
            _edge = 0;
            return PTLOC_OUTSIDE;
        }
    }

    if (result == PTLOC_INSIDE) {
        int lnext_dst = edgeDst(getEdge(_edge, NEXT_AROUND_LEFT));
        if (vtx[lnext_dst].isideal()) {
            _edge = 0;
            return PTLOC_OUTSIDE;
        }
    }

    return result;
}

int Subdiv2D::insert(Point2f pt)
{
    CV_INSTRUMENT_REGION();

    int curr_point = 0, curr_edge = 0, deleted_edge = 0;
    int location = locateInternal( pt, curr_edge, curr_point );

    if( location == PTLOC_VERTEX )
        return curr_point;

    if( location == PTLOC_EDGE )
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

        if( rightOfEx(
                    vtx[temp_dst].pt, vtx[curr_org].pt, vtx[curr_dst].pt,
                    vtx[temp_dst].isideal(), vtx[curr_org].isideal(), vtx[curr_dst].isideal()) > 0 &&
            inCircleEx(
                    vtx[curr_org].pt, vtx[temp_dst].pt, vtx[curr_dst].pt, vtx[curr_point].pt,
                    vtx[curr_org].isideal(), vtx[temp_dst].isideal(), vtx[curr_dst].isideal(), vtx[curr_point].isideal()) > 0 )
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

void Subdiv2D::reset()
{
    CV_INSTRUMENT_REGION();

    vtx.clear();
    qedges.clear();

    recentEdge = 0;
    validGeometry = false;

    Point2f ppA( 1.f, 0.f );
    Point2f ppB( cos(2.f * (float)M_PI / 3.f), sin(2.f * (float)M_PI / 3.f) );
    Point2f ppC( cos(4.f * (float)M_PI / 3.f), sin(4.f * (float)M_PI / 3.f) );

    vtx.push_back(Vertex());
    qedges.push_back(QuadEdge());

    freeQEdge = 0;
    freePoint = 0;

    int pA = newPoint(ppA, PTTYPE_DELAUNAY_IDEAL);
    int pB = newPoint(ppB, PTTYPE_DELAUNAY_IDEAL);
    int pC = newPoint(ppC, PTTYPE_DELAUNAY_IDEAL);

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
        if( vtx[i].isvoronoi() ) {
            if (vtx[i].isideal() && vtx[i].firstEdge) {
                deletePoint(vtx[i].firstEdge);
            }
            deletePoint((int)i);
        }
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

                if (!vtx[edge0_org].isideal() && !vtx[edge0_dst].isideal() && !vtx[edge1_dst].isideal()) {
                    voronoi_point = newPoint(computeVoronoiPoint(org0, dst0, dst0, dst1), PTTYPE_VORONOI);
                }

                if (!vtx[edge0_org].isideal() && !vtx[edge0_dst].isideal() && vtx[edge1_dst].isideal()) {
                    // ideal point has standard part of (org0 + dst0) / 2.f instead of (0.f, 0.f) expected by CCWEx
                    voronoi_point = newPoint(Point2f(-(dst0 - org0).y, (dst0 - org0).x), PTTYPE_VORONOI_IDEAL,
                                             newPoint((org0 + dst0) / 2.f, PTTYPE_FREE));
                }

                if (!vtx[edge0_org].isideal() && vtx[edge0_dst].isideal() && vtx[edge1_dst].isideal()) {
                    voronoi_point = newPoint(Point2f((dst1 - dst0).y, -(dst1 - dst0).x), PTTYPE_VORONOI_IDEAL);
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
    int location = locateInternal(pt, edge, vertex);

    if (location == PTLOC_VERTEX) {
        if (nearestPt) {
            *nearestPt = vtx[vertex].pt;
        }
        return vertex;
    }

    // we need to start with a PTTYPE_DELAUNAY vertex
    if (location == PTLOC_EDGE && vtx[edgeOrg(edge)].isideal()) {
        edge = symEdge(edge);
    }
    if (location == PTLOC_INSIDE) {
        for (int i = 0; i < 3 && vtx[edgeOrg(edge)].isideal(); i++) {
            edge = getEdge(edge, NEXT_AROUND_LEFT);
        }
        if (vtx[edgeOrg(edge)].isideal()) {
            // empty subdivision
            if (nearestPt) {
                nearestPt = NULL;
            }
            return 0;
        }
    }

    edge = rotateEdge(edge, 1);

    for (;;) {
        vertex = edgeOrg(rotateEdge( edge, 3)); // never PTTYPE_DELAUNAY_IDEAL

        for (;;) {
            int edge_dst = edgeDst(edge);

            // compensate the possible non-zero standard part of (possibly) an ideal point, so that CCWEx could succeed
            Point2f standard_part = vtx[edge_dst].isideal() && vtx[edge_dst].firstEdge ?
                    vtx[vtx[edge_dst].firstEdge].pt : Point2f(0.f, 0.f);

            if (rightOfEx(
                    pt - standard_part, vtx[vertex].pt - standard_part, vtx[edge_dst].pt,
                    false, false, vtx[edge_dst].isideal()) >= 0) {
                break;
            }

            edge = getEdge( edge, NEXT_AROUND_LEFT );
        }

        for (;;) {
            int edge_org = edgeOrg(edge);

            Point2f standard_part = vtx[edge_org].isideal() && vtx[edge_org].firstEdge ?
                    vtx[vtx[edge_org].firstEdge].pt : Point2f(0.f, 0.f);

            if (rightOfEx(
                    pt - standard_part, vtx[vertex].pt - standard_part, vtx[edge_org].pt,
                    false, false, vtx[edge_org].isideal()) < 0) {
                break;
            }

            edge = getEdge( edge, PREV_AROUND_LEFT );
        }

        int edge_org = edgeOrg(edge);
        int edge_dst = edgeDst(edge);

        if (vtx[edge_org].isideal() && vtx[edge_dst].isideal()) {
            Point2f standard_part_org = vtx[edge_org].firstEdge ? vtx[vtx[edge_org].firstEdge].pt : Point2f(0.f, 0.f);
            Point2f standard_part_dst = vtx[edge_dst].firstEdge ? vtx[vtx[edge_dst].firstEdge].pt : Point2f(0.f, 0.f);
            if (standard_part_org != standard_part_dst) {
                // never escapes beyond this edge
                break;
            }
        }

        Point2f standard_part(0.f, 0.f);
        if (vtx[edge_org].isideal() && vtx[edge_org].firstEdge) {
            standard_part = vtx[vtx[edge_org].firstEdge].pt;
        }
        if (vtx[edge_dst].isideal() && vtx[edge_dst].firstEdge) {
            standard_part = vtx[vtx[edge_dst].firstEdge].pt;
        }

        if (rightOfEx(
                pt - standard_part,
                vtx[edge_org].isideal() ? vtx[edge_org].pt : vtx[edge_org].pt - standard_part,
                vtx[edge_dst].isideal() ? vtx[edge_dst].pt : vtx[edge_dst].pt - standard_part,
                false, vtx[edge_org].isideal(), vtx[edge_dst].isideal()) <= 0) {
            break;
        }

        edge = symEdge(edge);
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
            if (vtx[edgeOrg(edge_a, &a)].isideal() ||
                    vtx[edgeOrg(edge_b, &b)].isideal() ||
                    vtx[edgeOrg(edge_c, &c)].isideal()) {
                continue;
            }
            triangleList.push_back(Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
        }
    }
}

void Subdiv2D::getVoronoiFacetList(const std::vector<int>& idx,
                                   CV_OUT std::vector<std::vector<Point2f> >& facetList,
                                   CV_OUT std::vector<Point2f>& facetCenters)
{
    calcVoronoi();
    facetList.clear();
    facetCenters.clear();

    std::vector<Point2f> buf;

    size_t i, total;
    if( idx.empty() )
        i = 4, total = vtx.size();
    else
        i = 0, total = idx.size();

    for( ; i < total; i++ )
    {
        int k = idx.empty() ? (int)i : idx[i];

        if( vtx[k].isfree() || vtx[k].isvoronoi() )
            continue;
        int edge = rotateEdge(vtx[k].firstEdge, 1), t = edge;

        // gather points
        buf.clear();
        do
        {
            buf.push_back(vtx[edgeOrg(t)].pt);
            t = getEdge( t, NEXT_AROUND_LEFT );
        }
        while( t != edge );

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
