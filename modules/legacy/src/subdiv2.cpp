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

CV_IMPL CvSubdiv2D *
cvCreateSubdiv2D( int subdiv_type, int header_size,
                  int vtx_size, int quadedge_size, CvMemStorage * storage )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    if( header_size < (int)sizeof( CvSubdiv2D ) ||
        quadedge_size < (int)sizeof( CvQuadEdge2D ) ||
        vtx_size < (int)sizeof( CvSubdiv2DPoint ))
        CV_Error( CV_StsBadSize, "" );

    return (CvSubdiv2D *)cvCreateGraph( subdiv_type, header_size,
                                        vtx_size, quadedge_size, storage );
}


/****************************************************************************************\
*                                    Quad Edge  algebra                                  *
\****************************************************************************************/

static CvSubdiv2DEdge
cvSubdiv2DMakeEdge( CvSubdiv2D * subdiv )
{
    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    CvQuadEdge2D* edge = (CvQuadEdge2D*)cvSetNew( (CvSet*)subdiv->edges );
    memset( edge->pt, 0, sizeof( edge->pt ));
    CvSubdiv2DEdge edgehandle = (CvSubdiv2DEdge) edge;

    edge->next[0] = edgehandle;
    edge->next[1] = edgehandle + 3;
    edge->next[2] = edgehandle + 2;
    edge->next[3] = edgehandle + 1;

    subdiv->quad_edges++;
    return edgehandle;
}


static CvSubdiv2DPoint *
cvSubdiv2DAddPoint( CvSubdiv2D * subdiv, CvPoint2D32f pt, int is_virtual )
{
    CvSubdiv2DPoint* subdiv_point = (CvSubdiv2DPoint*)cvSetNew( (CvSet*)subdiv );
    if( subdiv_point )
    {
        memset( subdiv_point, 0, subdiv->elem_size );
        subdiv_point->pt = pt;
        subdiv_point->first = 0;
        subdiv_point->flags |= is_virtual ? CV_SUBDIV2D_VIRTUAL_POINT_FLAG : 0;
		subdiv_point->id = -1;
    }

    return subdiv_point;
}


static void
cvSubdiv2DSplice( CvSubdiv2DEdge edgeA, CvSubdiv2DEdge edgeB )
{
    CvSubdiv2DEdge *a_next = &CV_SUBDIV2D_NEXT_EDGE( edgeA );
    CvSubdiv2DEdge *b_next = &CV_SUBDIV2D_NEXT_EDGE( edgeB );
    CvSubdiv2DEdge a_rot = cvSubdiv2DRotateEdge( *a_next, 1 );
    CvSubdiv2DEdge b_rot = cvSubdiv2DRotateEdge( *b_next, 1 );
    CvSubdiv2DEdge *a_rot_next = &CV_SUBDIV2D_NEXT_EDGE( a_rot );
    CvSubdiv2DEdge *b_rot_next = &CV_SUBDIV2D_NEXT_EDGE( b_rot );
    CvSubdiv2DEdge t;

    CV_SWAP( *a_next, *b_next, t );
    CV_SWAP( *a_rot_next, *b_rot_next, t );
}


static void
cvSubdiv2DSetEdgePoints( CvSubdiv2DEdge edge,
                         CvSubdiv2DPoint * org_pt, CvSubdiv2DPoint * dst_pt )
{
    CvQuadEdge2D *quadedge = (CvQuadEdge2D *) (edge & ~3);

    if( !quadedge )
        CV_Error( CV_StsNullPtr, "" );

    quadedge->pt[edge & 3] = org_pt;
    quadedge->pt[(edge + 2) & 3] = dst_pt;
}


static void
cvSubdiv2DDeleteEdge( CvSubdiv2D * subdiv, CvSubdiv2DEdge edge )
{
    CvQuadEdge2D *quadedge = (CvQuadEdge2D *) (edge & ~3);

    if( !subdiv || !quadedge )
        CV_Error( CV_StsNullPtr, "" );

    cvSubdiv2DSplice( edge, cvSubdiv2DGetEdge( edge, CV_PREV_AROUND_ORG ));

    CvSubdiv2DEdge sym_edge = cvSubdiv2DSymEdge( edge );
    cvSubdiv2DSplice( sym_edge, cvSubdiv2DGetEdge( sym_edge, CV_PREV_AROUND_ORG ));

    cvSetRemoveByPtr( (CvSet*)(subdiv->edges), quadedge );
    subdiv->quad_edges--;
}


static CvSubdiv2DEdge
cvSubdiv2DConnectEdges( CvSubdiv2D * subdiv, CvSubdiv2DEdge edgeA, CvSubdiv2DEdge edgeB )
{
    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    CvSubdiv2DEdge new_edge = cvSubdiv2DMakeEdge( subdiv );

    cvSubdiv2DSplice( new_edge, cvSubdiv2DGetEdge( edgeA, CV_NEXT_AROUND_LEFT ));
    cvSubdiv2DSplice( cvSubdiv2DSymEdge( new_edge ), edgeB );

    CvSubdiv2DPoint* dstA = cvSubdiv2DEdgeDst( edgeA );
    CvSubdiv2DPoint* orgB = cvSubdiv2DEdgeOrg( edgeB );
    cvSubdiv2DSetEdgePoints( new_edge, dstA, orgB );

    return new_edge;
}


static void
cvSubdiv2DSwapEdges( CvSubdiv2DEdge edge )
{
    CvSubdiv2DEdge sym_edge = cvSubdiv2DSymEdge( edge );
    CvSubdiv2DEdge a = cvSubdiv2DGetEdge( edge, CV_PREV_AROUND_ORG );
    CvSubdiv2DEdge b = cvSubdiv2DGetEdge( sym_edge, CV_PREV_AROUND_ORG );
    CvSubdiv2DPoint *dstB, *dstA;

    cvSubdiv2DSplice( edge, a );
    cvSubdiv2DSplice( sym_edge, b );

    dstA = cvSubdiv2DEdgeDst( a );
    dstB = cvSubdiv2DEdgeDst( b );
    cvSubdiv2DSetEdgePoints( edge, dstA, dstB );

    cvSubdiv2DSplice( edge, cvSubdiv2DGetEdge( a, CV_NEXT_AROUND_LEFT ));
    cvSubdiv2DSplice( sym_edge, cvSubdiv2DGetEdge( b, CV_NEXT_AROUND_LEFT ));
}


static int
icvIsRightOf( CvPoint2D32f& pt, CvSubdiv2DEdge edge )
{
    CvSubdiv2DPoint *org = cvSubdiv2DEdgeOrg(edge), *dst = cvSubdiv2DEdgeDst(edge);
    double cw_area = cvTriangleArea( pt, dst->pt, org->pt );

    return (cw_area > 0) - (cw_area < 0);
}


CV_IMPL CvSubdiv2DPointLocation
cvSubdiv2DLocate( CvSubdiv2D * subdiv, CvPoint2D32f pt,
                  CvSubdiv2DEdge * _edge, CvSubdiv2DPoint ** _point )
{
    CvSubdiv2DPoint *point = 0;
    int right_of_curr = 0;

    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    if( !CV_IS_SUBDIV2D(subdiv) )
        CV_Error( CV_StsBadFlag, "" );

    int i, max_edges = subdiv->quad_edges * 4;
    CvSubdiv2DEdge edge = subdiv->recent_edge;

    if( max_edges == 0 )
        CV_Error( CV_StsBadSize, "" );
    CV_Assert(edge != 0);

    if( pt.x < subdiv->topleft.x || pt.y < subdiv->topleft.y ||
        pt.x >= subdiv->bottomright.x || pt.y >= subdiv->bottomright.y )
        CV_Error( CV_StsOutOfRange, "" );

    CvSubdiv2DPointLocation location = CV_PTLOC_ERROR;

    right_of_curr = icvIsRightOf( pt, edge );
    if( right_of_curr > 0 )
    {
        edge = cvSubdiv2DSymEdge( edge );
        right_of_curr = -right_of_curr;
    }

    for( i = 0; i < max_edges; i++ )
    {
        CvSubdiv2DEdge onext_edge = cvSubdiv2DNextEdge( edge );
        CvSubdiv2DEdge dprev_edge = cvSubdiv2DGetEdge( edge, CV_PREV_AROUND_DST );

        int right_of_onext = icvIsRightOf( pt, onext_edge );
        int right_of_dprev = icvIsRightOf( pt, dprev_edge );

        if( right_of_dprev > 0 )
        {
            if( right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) )
            {
                location = CV_PTLOC_INSIDE;
                goto exit;
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
        else
        {
            if( right_of_onext > 0 )
            {
                if( right_of_dprev == 0 && right_of_curr == 0 )
                {
                    location = CV_PTLOC_INSIDE;
                    goto exit;
                }
                else
                {
                    right_of_curr = right_of_dprev;
                    edge = dprev_edge;
                }
            }
            else if( right_of_curr == 0 &&
                     icvIsRightOf( cvSubdiv2DEdgeDst( onext_edge )->pt, edge ) >= 0 )
            {
                edge = cvSubdiv2DSymEdge( edge );
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
    }
exit:
    
    subdiv->recent_edge = edge;

    if( location == CV_PTLOC_INSIDE )
    {
        double t1, t2, t3;
        CvPoint2D32f org_pt = cvSubdiv2DEdgeOrg( edge )->pt;
        CvPoint2D32f dst_pt = cvSubdiv2DEdgeDst( edge )->pt;

        t1 = fabs( pt.x - org_pt.x );
        t1 += fabs( pt.y - org_pt.y );
        t2 = fabs( pt.x - dst_pt.x );
        t2 += fabs( pt.y - dst_pt.y );
        t3 = fabs( org_pt.x - dst_pt.x );
        t3 += fabs( org_pt.y - dst_pt.y );

        if( t1 < FLT_EPSILON )
        {
            location = CV_PTLOC_VERTEX;
            point = cvSubdiv2DEdgeOrg( edge );
            edge = 0;
        }
        else if( t2 < FLT_EPSILON )
        {
            location = CV_PTLOC_VERTEX;
            point = cvSubdiv2DEdgeDst( edge );
            edge = 0;
        }
        else if( (t1 < t3 || t2 < t3) &&
                 fabs( cvTriangleArea( pt, org_pt, dst_pt )) < FLT_EPSILON )
        {
            location = CV_PTLOC_ON_EDGE;
            point = 0;
        }
    }

    if( location == CV_PTLOC_ERROR )
    {
        edge = 0;
        point = 0;
    }

    if( _edge )
        *_edge = edge;
    if( _point )
        *_point = point;

    return location;
}


CV_INLINE int
icvIsPtInCircle3( CvPoint2D32f pt, CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c )
{
    const double eps = FLT_EPSILON*0.125;
    double val = ((double)a.x * a.x + (double)a.y * a.y) * cvTriangleArea( b, c, pt );
    val -= ((double)b.x * b.x + (double)b.y * b.y) * cvTriangleArea( a, c, pt );
    val += ((double)c.x * c.x + (double)c.y * c.y) * cvTriangleArea( a, b, pt );
    val -= ((double)pt.x * pt.x + (double)pt.y * pt.y) * cvTriangleArea( a, b, c );

    return val > eps ? 1 : val < -eps ? -1 : 0;
}


CV_IMPL CvSubdiv2DPoint *
cvSubdivDelaunay2DInsert( CvSubdiv2D * subdiv, CvPoint2D32f pt )
{
    CvSubdiv2DPointLocation location = CV_PTLOC_ERROR;

    CvSubdiv2DPoint *curr_point = 0, *first_point = 0;
    CvSubdiv2DEdge curr_edge = 0, deleted_edge = 0, base_edge = 0;
    int i, max_edges;

    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    if( !CV_IS_SUBDIV2D(subdiv) )
        CV_Error( CV_StsBadFlag, "" );

    location = cvSubdiv2DLocate( subdiv, pt, &curr_edge, &curr_point );

    switch (location)
    {
    case CV_PTLOC_ERROR:
        CV_Error( CV_StsBadSize, "" );

    case CV_PTLOC_OUTSIDE_RECT:
        CV_Error( CV_StsOutOfRange, "" );

    case CV_PTLOC_VERTEX:
        break;

    case CV_PTLOC_ON_EDGE:
        deleted_edge = curr_edge;
        subdiv->recent_edge = curr_edge = cvSubdiv2DGetEdge( curr_edge, CV_PREV_AROUND_ORG );
        cvSubdiv2DDeleteEdge( subdiv, deleted_edge );
        /* no break */

    case CV_PTLOC_INSIDE:

        assert( curr_edge != 0 );
        subdiv->is_geometry_valid = 0;

        curr_point = cvSubdiv2DAddPoint( subdiv, pt, 0 );
        base_edge = cvSubdiv2DMakeEdge( subdiv );
        first_point = cvSubdiv2DEdgeOrg( curr_edge );
        cvSubdiv2DSetEdgePoints( base_edge, first_point, curr_point );
        cvSubdiv2DSplice( base_edge, curr_edge );

        do
        {
            base_edge = cvSubdiv2DConnectEdges( subdiv, curr_edge,
                                                cvSubdiv2DSymEdge( base_edge ));
            curr_edge = cvSubdiv2DGetEdge( base_edge, CV_PREV_AROUND_ORG );
        }
        while( cvSubdiv2DEdgeDst( curr_edge ) != first_point );

        curr_edge = cvSubdiv2DGetEdge( base_edge, CV_PREV_AROUND_ORG );

        max_edges = subdiv->quad_edges * 4;

        for( i = 0; i < max_edges; i++ )
        {
            CvSubdiv2DPoint *temp_dst = 0, *curr_org = 0, *curr_dst = 0;
            CvSubdiv2DEdge temp_edge = cvSubdiv2DGetEdge( curr_edge, CV_PREV_AROUND_ORG );

            temp_dst = cvSubdiv2DEdgeDst( temp_edge );
            curr_org = cvSubdiv2DEdgeOrg( curr_edge );
            curr_dst = cvSubdiv2DEdgeDst( curr_edge );

            if( icvIsRightOf( temp_dst->pt, curr_edge ) > 0 &&
                icvIsPtInCircle3( curr_org->pt, temp_dst->pt,
                                  curr_dst->pt, curr_point->pt ) < 0 )
            {
                cvSubdiv2DSwapEdges( curr_edge );
                curr_edge = cvSubdiv2DGetEdge( curr_edge, CV_PREV_AROUND_ORG );
            }
            else if( curr_org == first_point )
            {
                break;
            }
            else
            {
                curr_edge = cvSubdiv2DGetEdge( cvSubdiv2DNextEdge( curr_edge ),
                                               CV_PREV_AROUND_LEFT );
            }
        }
        break;
    default:
        CV_Error_(CV_StsError, ("cvSubdiv2DLocate returned invalid location = %d", location) );
    }

    return curr_point;
}


CV_IMPL void
cvInitSubdivDelaunay2D( CvSubdiv2D * subdiv, CvRect rect )
{
    float big_coord = 3.f * MAX( rect.width, rect.height );
    CvPoint2D32f ppA, ppB, ppC;
    CvSubdiv2DPoint *pA, *pB, *pC;
    CvSubdiv2DEdge edge_AB, edge_BC, edge_CA;
    float rx = (float) rect.x;
    float ry = (float) rect.y;

    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    cvClearSet( (CvSet *) (subdiv->edges) );
    cvClearSet( (CvSet *) subdiv );

    subdiv->quad_edges = 0;
    subdiv->recent_edge = 0;
    subdiv->is_geometry_valid = 0;

    subdiv->topleft = cvPoint2D32f( rx, ry );
    subdiv->bottomright = cvPoint2D32f( rx + rect.width, ry + rect.height );

    ppA = cvPoint2D32f( rx + big_coord, ry );
    ppB = cvPoint2D32f( rx, ry + big_coord );
    ppC = cvPoint2D32f( rx - big_coord, ry - big_coord );

    pA = cvSubdiv2DAddPoint( subdiv, ppA, 0 );
    pB = cvSubdiv2DAddPoint( subdiv, ppB, 0 );
    pC = cvSubdiv2DAddPoint( subdiv, ppC, 0 );

    edge_AB = cvSubdiv2DMakeEdge( subdiv );
    edge_BC = cvSubdiv2DMakeEdge( subdiv );
    edge_CA = cvSubdiv2DMakeEdge( subdiv );

    cvSubdiv2DSetEdgePoints( edge_AB, pA, pB );
    cvSubdiv2DSetEdgePoints( edge_BC, pB, pC );
    cvSubdiv2DSetEdgePoints( edge_CA, pC, pA );

    cvSubdiv2DSplice( edge_AB, cvSubdiv2DSymEdge( edge_CA ));
    cvSubdiv2DSplice( edge_BC, cvSubdiv2DSymEdge( edge_AB ));
    cvSubdiv2DSplice( edge_CA, cvSubdiv2DSymEdge( edge_BC ));

    subdiv->recent_edge = edge_AB;
}


CV_IMPL void
cvClearSubdivVoronoi2D( CvSubdiv2D * subdiv )
{
    int elem_size;
    int i, total;
    CvSeqReader reader;

    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    /* clear pointers to voronoi points */
    total = subdiv->edges->total;
    elem_size = subdiv->edges->elem_size;

    cvStartReadSeq( (CvSeq *) (subdiv->edges), &reader, 0 );

    for( i = 0; i < total; i++ )
    {
        CvQuadEdge2D *quadedge = (CvQuadEdge2D *) reader.ptr;

        quadedge->pt[1] = quadedge->pt[3] = 0;
        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }

    /* remove voronoi points */
    total = subdiv->total;
    elem_size = subdiv->elem_size;

    cvStartReadSeq( (CvSeq *) subdiv, &reader, 0 );

    for( i = 0; i < total; i++ )
    {
        CvSubdiv2DPoint *pt = (CvSubdiv2DPoint *) reader.ptr;

        /* check for virtual point. it is also check that the point exists */
        if( pt->flags & CV_SUBDIV2D_VIRTUAL_POINT_FLAG )
        {
            cvSetRemoveByPtr( (CvSet*)subdiv, pt );
        }
        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }

    subdiv->is_geometry_valid = 0;
}


static void
icvCreateCenterNormalLine( CvSubdiv2DEdge edge, double *_a, double *_b, double *_c )
{
    CvPoint2D32f org = cvSubdiv2DEdgeOrg( edge )->pt;
    CvPoint2D32f dst = cvSubdiv2DEdgeDst( edge )->pt;
    
    double a = dst.x - org.x;
    double b = dst.y - org.y;
    double c = -(a * (dst.x + org.x) + b * (dst.y + org.y));
    
    *_a = a + a;
    *_b = b + b;
    *_c = c;
}


static void
icvIntersectLines3( double *a0, double *b0, double *c0,
                   double *a1, double *b1, double *c1, CvPoint2D32f * point )
{
    double det = a0[0] * b1[0] - a1[0] * b0[0];
    
    if( det != 0 )
    {
        det = 1. / det;
        point->x = (float) ((b0[0] * c1[0] - b1[0] * c0[0]) * det);
        point->y = (float) ((a1[0] * c0[0] - a0[0] * c1[0]) * det);
    }
    else
    {
        point->x = point->y = FLT_MAX;
    }
}


CV_IMPL void
cvCalcSubdivVoronoi2D( CvSubdiv2D * subdiv )
{
    CvSeqReader reader;
    int i, total, elem_size;

    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    /* check if it is already calculated */
    if( subdiv->is_geometry_valid )
        return;

    total = subdiv->edges->total;
    elem_size = subdiv->edges->elem_size;

    cvClearSubdivVoronoi2D( subdiv );

    cvStartReadSeq( (CvSeq *) (subdiv->edges), &reader, 0 );

    if( total <= 3 )
        return;

    /* skip first three edges (bounding triangle) */
    for( i = 0; i < 3; i++ )
        CV_NEXT_SEQ_ELEM( elem_size, reader );

    /* loop through all quad-edges */
    for( ; i < total; i++ )
    {
        CvQuadEdge2D *quadedge = (CvQuadEdge2D *) (reader.ptr);

        if( CV_IS_SET_ELEM( quadedge ))
        {
            CvSubdiv2DEdge edge0 = (CvSubdiv2DEdge) quadedge, edge1, edge2;
            double a0, b0, c0, a1, b1, c1;
            CvPoint2D32f virt_point;
            CvSubdiv2DPoint *voronoi_point;

            if( !quadedge->pt[3] )
            {
                edge1 = cvSubdiv2DGetEdge( edge0, CV_NEXT_AROUND_LEFT );
                edge2 = cvSubdiv2DGetEdge( edge1, CV_NEXT_AROUND_LEFT );

                icvCreateCenterNormalLine( edge0, &a0, &b0, &c0 );
                icvCreateCenterNormalLine( edge1, &a1, &b1, &c1 );

                icvIntersectLines3( &a0, &b0, &c0, &a1, &b1, &c1, &virt_point );
                if( fabs( virt_point.x ) < FLT_MAX * 0.5 &&
                    fabs( virt_point.y ) < FLT_MAX * 0.5 )
                {
                    voronoi_point = cvSubdiv2DAddPoint( subdiv, virt_point, 1 );

                    quadedge->pt[3] =
                        ((CvQuadEdge2D *) (edge1 & ~3))->pt[3 - (edge1 & 2)] =
                        ((CvQuadEdge2D *) (edge2 & ~3))->pt[3 - (edge2 & 2)] = voronoi_point;
                }
            }

            if( !quadedge->pt[1] )
            {
                edge1 = cvSubdiv2DGetEdge( edge0, CV_NEXT_AROUND_RIGHT );
                edge2 = cvSubdiv2DGetEdge( edge1, CV_NEXT_AROUND_RIGHT );

                icvCreateCenterNormalLine( edge0, &a0, &b0, &c0 );
                icvCreateCenterNormalLine( edge1, &a1, &b1, &c1 );

                icvIntersectLines3( &a0, &b0, &c0, &a1, &b1, &c1, &virt_point );

                if( fabs( virt_point.x ) < FLT_MAX * 0.5 &&
                    fabs( virt_point.y ) < FLT_MAX * 0.5 )
                {
                    voronoi_point = cvSubdiv2DAddPoint( subdiv, virt_point, 1 );

                    quadedge->pt[1] =
                        ((CvQuadEdge2D *) (edge1 & ~3))->pt[1 + (edge1 & 2)] =
                        ((CvQuadEdge2D *) (edge2 & ~3))->pt[1 + (edge2 & 2)] = voronoi_point;
                }
            }
        }

        CV_NEXT_SEQ_ELEM( elem_size, reader );
    }

    subdiv->is_geometry_valid = 1;
}


static int
icvIsRightOf2( const CvPoint2D32f& pt, const CvPoint2D32f& org, const CvPoint2D32f& diff )
{
    double cw_area = ((double)org.x - pt.x)*diff.y - ((double)org.y - pt.y)*diff.x;
    return (cw_area > 0) - (cw_area < 0);
}


CV_IMPL CvSubdiv2DPoint*
cvFindNearestPoint2D( CvSubdiv2D* subdiv, CvPoint2D32f pt )
{
    CvSubdiv2DPoint* point = 0;
    CvPoint2D32f start;
    CvPoint2D32f diff;
    CvSubdiv2DPointLocation loc;
    CvSubdiv2DEdge edge; 
    int i;
    
    if( !subdiv )
        CV_Error( CV_StsNullPtr, "" );

    if( !CV_IS_SUBDIV2D( subdiv ))
        CV_Error( CV_StsNullPtr, "" );
    
    if( subdiv->edges->active_count <= 3 )
        return 0;

    if( !subdiv->is_geometry_valid )
        cvCalcSubdivVoronoi2D( subdiv );

    loc = cvSubdiv2DLocate( subdiv, pt, &edge, &point );

    switch( loc )
    {
    case CV_PTLOC_ON_EDGE:
    case CV_PTLOC_INSIDE:
        break;
    default:
        return point;
    }

    point = 0;

    start = cvSubdiv2DEdgeOrg( edge )->pt;
    diff.x = pt.x - start.x;
    diff.y = pt.y - start.y;

    edge = cvSubdiv2DRotateEdge( edge, 1 );

    for( i = 0; i < subdiv->total; i++ )
    {
        CvPoint2D32f t;
        
        for(;;)
        {
            assert( cvSubdiv2DEdgeDst( edge ));
            
            t = cvSubdiv2DEdgeDst( edge )->pt;
            if( icvIsRightOf2( t, start, diff ) >= 0 )
                break;

            edge = cvSubdiv2DGetEdge( edge, CV_NEXT_AROUND_LEFT );
        }

        for(;;)
        {
            assert( cvSubdiv2DEdgeOrg( edge ));

            t = cvSubdiv2DEdgeOrg( edge )->pt;
            if( icvIsRightOf2( t, start, diff ) < 0 )
                break;

            edge = cvSubdiv2DGetEdge( edge, CV_PREV_AROUND_LEFT );
        }

        {
            CvPoint2D32f tempDiff = cvSubdiv2DEdgeDst( edge )->pt;
            t = cvSubdiv2DEdgeOrg( edge )->pt;
            tempDiff.x -= t.x;
            tempDiff.y -= t.y;

            if( icvIsRightOf2( pt, t, tempDiff ) >= 0 )
            {
                point = cvSubdiv2DEdgeOrg( cvSubdiv2DRotateEdge( edge, 3 ));
                break;
            }
        }

        edge = cvSubdiv2DSymEdge( edge );
    }

    return point;
}

CV_IMPL int
icvSubdiv2DCheck( CvSubdiv2D* subdiv )
{
    int i, j, total = subdiv->edges->total;
    CV_Assert( subdiv != 0 );
    
    for( i = 0; i < total; i++ )
    {
        CvQuadEdge2D* edge = (CvQuadEdge2D*)cvGetSetElem(subdiv->edges,i);
        
        if( edge && CV_IS_SET_ELEM( edge ))
        {
            for( j = 0; j < 4; j++ )
            {
                CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge + j;
                CvSubdiv2DEdge o_next = cvSubdiv2DNextEdge(e);
                CvSubdiv2DEdge o_prev = cvSubdiv2DGetEdge(e, CV_PREV_AROUND_ORG );
                CvSubdiv2DEdge d_prev = cvSubdiv2DGetEdge(e, CV_PREV_AROUND_DST );
                CvSubdiv2DEdge d_next = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_DST );

                // check points
                if( cvSubdiv2DEdgeOrg(e) != cvSubdiv2DEdgeOrg(o_next))
                    return 0;
                if( cvSubdiv2DEdgeOrg(e) != cvSubdiv2DEdgeOrg(o_prev))
                    return 0;
                if( cvSubdiv2DEdgeDst(e) != cvSubdiv2DEdgeDst(d_next))
                    return 0;
                if( cvSubdiv2DEdgeDst(e) != cvSubdiv2DEdgeDst(d_prev))
                    return 0;
                if( j % 2 == 0 )
                {
                    if( cvSubdiv2DEdgeDst(o_next) != cvSubdiv2DEdgeOrg(d_prev))
                        return 0;
                    if( cvSubdiv2DEdgeDst(o_prev) != cvSubdiv2DEdgeOrg(d_next))
                        return 0;
                    if( cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(
                        e,CV_NEXT_AROUND_LEFT),CV_NEXT_AROUND_LEFT),CV_NEXT_AROUND_LEFT) != e )
                        return 0;
                    if( cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(cvSubdiv2DGetEdge(
                        e,CV_NEXT_AROUND_RIGHT),CV_NEXT_AROUND_RIGHT),CV_NEXT_AROUND_RIGHT) != e)
                        return 0;
                }
            }
        }
    }

    return 1;
}



static void
draw_subdiv_facet( CvSubdiv2D * subdiv, IplImage * dst, IplImage * src, CvSubdiv2DEdge edge )
{
    CvSubdiv2DEdge t = edge;
    int i, count = 0;
    CvPoint local_buf[100];
    CvPoint *buf = local_buf;

    // count number of edges in facet 
    do
    {
        count++;
        t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
    }
    while( t != edge && count < subdiv->quad_edges * 4 );

    if( count * sizeof( buf[0] ) > sizeof( local_buf ))
    {
        buf = (CvPoint *) malloc( count * sizeof( buf[0] ));
    }

    // gather points
    t = edge;
    for( i = 0; i < count; i++ )
    {
        CvSubdiv2DPoint *pt = cvSubdiv2DEdgeOrg( t );

        if( !pt )
            break;
        assert( fabs( pt->pt.x ) < 10000 && fabs( pt->pt.y ) < 10000 );
        buf[i] = cvPoint( cvRound( pt->pt.x ), cvRound( pt->pt.y ));
        t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
    }

    if( i == count )
    {
        CvSubdiv2DPoint *pt = cvSubdiv2DEdgeDst( cvSubdiv2DRotateEdge( edge, 1 ));
        CvPoint ip = cvPoint( cvRound( pt->pt.x ), cvRound( pt->pt.y ));
        CvScalar color = {{0,0,0,0}};

        //printf("count = %d, (%d,%d)\n", ip.x, ip.y );

        if( 0 <= ip.x && ip.x < src->width && 0 <= ip.y && ip.y < src->height )
        {
            uchar *ptr = (uchar*)(src->imageData + ip.y * src->widthStep + ip.x * 3);
            color = CV_RGB( ptr[2], ptr[1], ptr[0] );
        }

        cvFillConvexPoly( dst, buf, count, color );
        //draw_subdiv_point( dst, pt->pt, CV_RGB(0,0,0));
    }

    if( buf != local_buf )
        free( buf );
}


CV_IMPL void
icvDrawMosaic( CvSubdiv2D * subdiv, IplImage * src, IplImage * dst )
{
    int i, total = subdiv->edges->total;

    cvCalcSubdivVoronoi2D( subdiv );

    //icvSet( dst, 255 );
    for( i = 0; i < total; i++ )
    {
        CvQuadEdge2D *edge = (CvQuadEdge2D *) cvGetSetElem( subdiv->edges, i );

        if( edge && CV_IS_SET_ELEM( edge ))
        {
            CvSubdiv2DEdge e = (CvSubdiv2DEdge) edge;

            // left
            draw_subdiv_facet( subdiv, dst, src, cvSubdiv2DRotateEdge( e, 1 ));
            // right
            draw_subdiv_facet( subdiv, dst, src, cvSubdiv2DRotateEdge( e, 3 ));
        }
    }
}

/* End of file. */
