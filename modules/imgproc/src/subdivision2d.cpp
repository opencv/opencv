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
    CvSubdiv2DPoint *point = 0;
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
        point = curr_point;
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
    validGeometry = false;
    freeQEdge = 0;
    freePoint = 0;
    recentEdge = 0;
}

Subdiv2D::Subdiv2D(Rect rect)
{
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
    type = -1;
}

Subdiv2D::Vertex::Vertex(Point2f _pt, bool _isvirtual, int _firstEdge)
{
    firstEdge = _firstEdge;
    type = (int)_isvirtual;
    pt = _pt;
}

bool Subdiv2D::Vertex::isvirtual() const
{
    return type > 0;
}

bool Subdiv2D::Vertex::isfree() const
{
    return type < 0;
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

int Subdiv2D::isRightOf(Point2f pt, int edge) const
{
    Point2f org, dst;
    edgeOrg(edge, &org);
    edgeDst(edge, &dst);
    double cw_area = cvTriangleArea( pt, dst, org );
    
    return (cw_area > 0) - (cw_area < 0);
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

int Subdiv2D::newPoint(Point2f pt, bool isvirtual, int firstEdge)
{
    if( freePoint == 0 )
    {
        vtx.push_back(Vertex());
        freePoint = (int)(vtx.size()-1);
    }
    int vidx = freePoint;
    freePoint = vtx[vidx].firstEdge;
    vtx[vidx] = Vertex(pt, isvirtual, firstEdge);
    
    return vidx;
}

void Subdiv2D::deletePoint(int vidx)
{
    CV_DbgAssert( (size_t)vidx < vtx.size() );
    vtx[vidx].firstEdge = freePoint;
    vtx[vidx].type = -1;
    freePoint = vidx;
}

int Subdiv2D::locate(Point2f pt, int& _edge, int& _vertex)
{
    int vertex = 0;
    
    int i, maxEdges = (int)(qedges.size() * 4);
    
    if( qedges.size() < (size_t)4 )
        CV_Error( CV_StsError, "Subdivision is empty" );
    
    if( pt.x < topLeft.x || pt.y < topLeft.y || pt.x >= bottomRight.x || pt.y >= bottomRight.y )
        CV_Error( CV_StsOutOfRange, "" );
    
    int edge = recentEdge;
    CV_Assert(edge > 0);
    
    int location = PTLOC_ERROR;
    
    int right_of_curr = isRightOf(pt, edge);
    if( right_of_curr > 0 )
    {
        edge = symEdge(edge);
        right_of_curr = -right_of_curr;
    }
    
    for( i = 0; i < maxEdges; i++ )
    {
        int onext_edge = nextEdge( edge );
        int dprev_edge = getEdge( edge, PREV_AROUND_DST );
        
        int right_of_onext = isRightOf( pt, onext_edge );
        int right_of_dprev = isRightOf( pt, dprev_edge );
        
        if( right_of_dprev > 0 )
        {
            if( right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) )
            {
                location = PTLOC_INSIDE;
                break;
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
                    location = PTLOC_INSIDE;
                    break;
                }
                else
                {
                    right_of_curr = right_of_dprev;
                    edge = dprev_edge;
                }
            }
            else if( right_of_curr == 0 &&
                    isRightOf( vtx[edgeDst(onext_edge)].pt, edge ) >= 0 )
            {
                edge = symEdge( edge );
            }
            else
            {
                right_of_curr = right_of_onext;
                edge = onext_edge;
            }
        }
    }
    
    recentEdge = edge;
    
    if( location == PTLOC_INSIDE )
    {
        Point2f org_pt, dst_pt;
        edgeOrg(edge, &org_pt);
        edgeDst(edge, &dst_pt);
        
        double t1 = fabs( pt.x - org_pt.x );
        t1 += fabs( pt.y - org_pt.y );
        double t2 = fabs( pt.x - dst_pt.x );
        t2 += fabs( pt.y - dst_pt.y );
        double t3 = fabs( org_pt.x - dst_pt.x );
        t3 += fabs( org_pt.y - dst_pt.y );
        
        if( t1 < FLT_EPSILON )
        {
            location = PTLOC_VERTEX;
            vertex = edgeOrg( edge );
            edge = 0;
        }
        else if( t2 < FLT_EPSILON )
        {
            location = PTLOC_VERTEX;
            vertex = edgeDst( edge );
            edge = 0;
        }
        else if( (t1 < t3 || t2 < t3) &&
                fabs( cvTriangleArea( pt, org_pt, dst_pt )) < FLT_EPSILON )
        {
            location = PTLOC_ON_EDGE;
            vertex = 0;
        }
    }
    
    if( location == PTLOC_ERROR )
    {
        edge = 0;
        vertex = 0;
    }
    
    _edge = edge;
    _vertex = vertex;
    
    return location;
}


inline int
isPtInCircle3( Point2f pt, Point2f a, Point2f b, Point2f c)
{
    const double eps = FLT_EPSILON*0.125;
    double val = ((double)a.x * a.x + (double)a.y * a.y) * cvTriangleArea( b, c, pt );
    val -= ((double)b.x * b.x + (double)b.y * b.y) * cvTriangleArea( a, c, pt );
    val += ((double)c.x * c.x + (double)c.y * c.y) * cvTriangleArea( a, b, pt );
    val -= ((double)pt.x * pt.x + (double)pt.y * pt.y) * cvTriangleArea( a, b, c );
    
    return val > eps ? 1 : val < -eps ? -1 : 0;
}


int Subdiv2D::insert(Point2f pt)
{
    int curr_point = 0, curr_edge = 0, deleted_edge = 0;
    int location = locate( pt, curr_edge, curr_point );
    
    if( location == PTLOC_ERROR )
        CV_Error( CV_StsBadSize, "" );
    
    if( location == PTLOC_OUTSIDE_RECT )
        CV_Error( CV_StsOutOfRange, "" );
    
    if( location == PTLOC_VERTEX )
        return curr_point;
    
    if( location == PTLOC_ON_EDGE )
    {
        deleted_edge = curr_edge;
        recentEdge = curr_edge = getEdge( curr_edge, PREV_AROUND_ORG );
        deleteEdge(deleted_edge);
    }
    else if( location == PTLOC_INSIDE )
        ;
    else
        CV_Error_(CV_StsError, ("Subdiv2D::locate returned invalid location = %d", location) );
    
    assert( curr_edge != 0 );
    validGeometry = false;
    
    curr_point = newPoint(pt, false);
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
        
        if( isRightOf( vtx[temp_dst].pt, curr_edge ) > 0 &&
           isPtInCircle3( vtx[curr_org].pt, vtx[temp_dst].pt,
                         vtx[curr_dst].pt, vtx[curr_point].pt ) < 0 )
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

void Subdiv2D::insert(const vector<Point2f>& ptvec)
{
    for( size_t i = 0; i < ptvec.size(); i++ )
        insert(ptvec[i]);
}

void Subdiv2D::initDelaunay( Rect rect )
{
    float big_coord = 3.f * MAX( rect.width, rect.height );
    float rx = (float)rect.x;
    float ry = (float)rect.y;
    
    vtx.clear();
    qedges.clear();
    
    recentEdge = 0;
    validGeometry = false;
    
    topLeft = Point2f( rx, ry );
    bottomRight = Point2f( rx + rect.width, ry + rect.height );
    
    Point2f ppA( rx + big_coord, ry );
    Point2f ppB( rx, ry + big_coord );
    Point2f ppC( rx - big_coord, ry - big_coord );
    
    vtx.push_back(Vertex());
    qedges.push_back(QuadEdge());
    
    freeQEdge = 0;
    freePoint = 0;
    
    int pA = newPoint(ppA, false);
    int pB = newPoint(ppB, false);
    int pC = newPoint(ppC, false);
    
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
    int i, total = (int)qedges.size();
    
    // loop through all quad-edges, except for the first 3 (#1, #2, #3 - 0 is reserved for "NULL" pointer)
    for( i = 4; i < total; i++ )
    {
        QuadEdge& quadedge = qedges[i];
        
        if( quadedge.isfree() )
            continue;
        
        int edge0 = (int)(i*4);
        Point2f org0, dst0, org1, dst1;
        
        if( !quadedge.pt[3] )
        {
            int edge1 = getEdge( edge0, NEXT_AROUND_LEFT );
            int edge2 = getEdge( edge1, NEXT_AROUND_LEFT );
            
            edgeOrg(edge0, &org0);
            edgeDst(edge0, &dst0);
            edgeOrg(edge1, &org1);
            edgeDst(edge1, &dst1);
            
            Point2f virt_point = computeVoronoiPoint(org0, dst0, org1, dst1);
            
            if( fabs( virt_point.x ) < FLT_MAX * 0.5 &&
               fabs( virt_point.y ) < FLT_MAX * 0.5 )
            {
                quadedge.pt[3] = qedges[edge1 >> 2].pt[3 - (edge1 & 2)] =
                qedges[edge2 >> 2].pt[3 - (edge2 & 2)] = newPoint(virt_point, true);
            }
        }
        
        if( !quadedge.pt[1] )
        {
            int edge1 = getEdge( edge0, NEXT_AROUND_RIGHT );
            int edge2 = getEdge( edge1, NEXT_AROUND_RIGHT );
            
            edgeOrg(edge0, &org0);
            edgeDst(edge0, &dst0);
            edgeOrg(edge1, &org1);
            edgeDst(edge1, &dst1);
            
            Point2f virt_point = computeVoronoiPoint(org0, dst0, org1, dst1);
            
            if( fabs( virt_point.x ) < FLT_MAX * 0.5 &&
               fabs( virt_point.y ) < FLT_MAX * 0.5 )
            {
                quadedge.pt[1] = qedges[edge1 >> 2].pt[1 + (edge1 & 2)] = 
                qedges[edge2 >> 2].pt[1 + (edge2 & 2)] = newPoint(virt_point, true);
            }
        }
    }
    
    validGeometry = true;
}


static int
isRightOf2( const Point2f& pt, const Point2f& org, const Point2f& diff )
{
    double cw_area = ((double)org.x - pt.x)*diff.y - ((double)org.y - pt.y)*diff.x;
    return (cw_area > 0) - (cw_area < 0);
}


int Subdiv2D::findNearest(Point2f pt, Point2f* nearestPt)
{
    if( !validGeometry )
        calcVoronoi();
    
    int vertex = 0, edge = 0;
    int loc = locate( pt, edge, vertex );
    
    if( loc != PTLOC_ON_EDGE && loc != PTLOC_INSIDE )
        return vertex;
    
    vertex = 0;
    
    Point2f start;
    edgeOrg(edge, &start);
    Point2f diff = pt - start;
    
    edge = rotateEdge(edge, 1);
    
    int i, total = (int)vtx.size();
    
    for( i = 0; i < total; i++ )
    {
        Point2f t;
        
        for(;;)
        {
            CV_Assert( edgeDst(edge, &t) > 0 );
            if( isRightOf2( t, start, diff ) >= 0 )
                break;
            
            edge = getEdge( edge, NEXT_AROUND_LEFT );
        }
        
        for(;;)
        {
            CV_Assert( edgeOrg( edge, &t ) > 0 );
            
            if( isRightOf2( t, start, diff ) < 0 )
                break;
            
            edge = getEdge( edge, PREV_AROUND_LEFT );
        }
        
        Point2f tempDiff;
        edgeDst(edge, &tempDiff);
        edgeOrg(edge, &t);
        tempDiff -= t;
        
        if( isRightOf2( pt, t, tempDiff ) >= 0 )
        {
            vertex = edgeOrg(rotateEdge( edge, 3 ));
            break;
        }
        
        edge = symEdge( edge );
    }
    
    if( nearestPt && vertex > 0 )
        *nearestPt = vtx[vertex].pt;
    
    return vertex;
}

void Subdiv2D::getEdgeList(vector<Vec4f>& edgeList) const
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

void Subdiv2D::getTriangleList(vector<Vec6f>& triangleList) const
{
    triangleList.clear();
    int i, total = (int)(qedges.size()*4);
    vector<bool> edgemask(total, false);
    
    for( i = 4; i < total; i += 2 )
    {
        if( edgemask[i] )
            continue;
        Point2f a, b, c;
        int edge = i;
        edgeOrg(edge, &a);
        edgemask[edge] = true;
        edge = getEdge(edge, NEXT_AROUND_LEFT);
        edgeOrg(edge, &b);
        edgemask[edge] = true;
        edge = getEdge(edge, NEXT_AROUND_LEFT);
        edgeOrg(edge, &c);
        edgemask[edge] = true;
        triangleList.push_back(Vec6f(a.x, a.y, b.x, b.y, c.x, c.y));
    }
}

void Subdiv2D::getVoronoiFacetList(const vector<int>& idx,
                                   CV_OUT vector<vector<Point2f> >& facetList,
                                   CV_OUT vector<Point2f>& facetCenters)
{
    calcVoronoi();
    facetList.clear();
    facetCenters.clear();
    
    vector<Point2f> buf;
    
    size_t i, total;
    if( idx.empty() )
        i = 4, total = vtx.size();
    else
        i = 0, total = idx.size();
    
    for( ; i < total; i++ )
    {
        int k = idx.empty() ? (int)i : idx[i];
        
        if( vtx[k].isfree() || vtx[k].isvirtual() )
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
