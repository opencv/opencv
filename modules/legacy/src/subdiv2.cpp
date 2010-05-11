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
