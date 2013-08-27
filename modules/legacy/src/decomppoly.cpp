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

#if 0

#include <malloc.h>
//#include "decomppoly.h"

#define ZERO_CLOSE 0.00001f
#define ONE_CLOSE  0.99999f

#define CHECK_COLLINEARITY(vec1_x,vec1_y,vec2_x,vec2_y) \
    if( vec1_x == 0 ) {                                 \
        if( vec1_y * vec2_y > 0 ) {                     \
            return 0;                                   \
        }                                               \
    }                                                   \
    else {                                              \
        if( vec1_x * vec2_x > 0 ) {                     \
            return 0;                                   \
        }                                               \
    }

// determines if edge number one lies in counterclockwise
//  earlier than edge number two
inline int  icvIsFirstEdgeClosier( int x0,
                                   int y0,
                                   int x0_end,
                                   int y0_end,
                                   int x1_end,
                                   int y1_end,
                                   int x2_end,
                                   int y2_end )
{
    int mult, mult1, mult2;
    int vec0_x, vec0_y;
    int vec1_x, vec1_y;
    int vec2_x, vec2_y;

    vec0_x = x0_end - x0;
    vec0_y = y0_end - y0;
    vec1_x = x1_end - x0;
    vec1_y = y1_end - y0;
    vec2_x = x2_end - x0;
    vec2_y = y2_end - y0;

    mult1 = vec1_x * vec0_y - vec0_x * vec1_y;
    mult2 = vec2_x * vec0_y - vec0_x * vec2_y;

    if( mult1 == 0 ) {
        CHECK_COLLINEARITY( vec0_x, vec0_y, vec1_x, vec1_y );
    }
    if( mult2 == 0 ) {
        CHECK_COLLINEARITY( vec0_x, vec0_y, vec2_x, vec2_y );
    }
    if( mult1 > 0 && mult2 < 0 ) {
        return 1;
    }
    if( mult1 < 0 && mult2 > 0 ) {
        return -1;
    }

    mult = vec1_x * vec2_y - vec2_x * vec1_y;
    if( mult == 0 ) {
        CHECK_COLLINEARITY( vec1_x, vec1_y, vec2_x, vec2_y );
    }

    if( mult1 > 0 )
    {
        if( mult > 0 ) {
            return -1;
        }
        else {
            return 1;
        }
    } // if( mult1 > 0 )
    else
    {
        if( mult1 != 0 ) {
            if( mult > 0 ) {
                return 1;
            }
            else {
                return -1;
            }
        } // if( mult1 != 0 )
        else {
            if( mult2 > 0 ) {
                return -1;
            }
            else {
                return 1;
            }
        } // if( mult1 != 0 ) else

    } // if( mult1 > 0 ) else

} // icvIsFirstEdgeClosier

bool icvEarCutTriangulation( CvPoint* contour,
                               int num,
                               int* outEdges,
                               int* numEdges )
{
    int i;
    int notFoundFlag = 0;
    int begIndex = -1;
    int isInternal;
    int currentNum = num;
    int index1, index2, index3;
    int ix0, iy0, ix1, iy1, ix2, iy2;
    int x1, y1, x2, y2, x3, y3;
    int dx1, dy1, dx2, dy2;
    int* pointExist = ( int* )0;
    int det, det1, det2;
    float t1, t2;

    (*numEdges) = 0;

    if( num <= 2 ) {
        return false;
    }

    pointExist = ( int* )malloc( num * sizeof( int ) );

    for( i = 0; i < num; i ++ ) {
        pointExist[i] = 1;
    }

    for( i = 0; i < num; i ++ ) {
        outEdges[ (*numEdges) * 2 ] = i;
        if( i != num - 1 ) {
            outEdges[ (*numEdges) * 2 + 1 ] = i + 1;
        }
        else {
            outEdges[ (*numEdges) * 2 + 1 ] = 0;
        }
        (*numEdges) ++;
    } // for( i = 0; i < num; i ++ )

    // initializing data before while cycle
    index1 = 0;
    index2 = 1;
    index3 = 2;
    x1 = contour[ index1 ].x;
    y1 = contour[ index1 ].y;
    x2 = contour[ index2 ].x;
    y2 = contour[ index2 ].y;
    x3 = contour[ index3 ].x;
    y3 = contour[ index3 ].y;

    while( currentNum > 3 )
    {
        dx1 = x2 - x1;
        dy1 = y2 - y1;
        dx2 = x3 - x2;
        dy2 = y3 - y2;
        if( dx1 * dy2 - dx2 * dy1 < 0 ) // convex condition
        {
            // checking for noncrossing edge
            ix1 = x3 - x1;
            iy1 = y3 - y1;
            isInternal = 1;
            for( i = 0; i < num; i ++ )
            {
                if( i != num - 1 ) {
                    ix2 = contour[ i + 1 ].x - contour[ i ].x;
                    iy2 = contour[ i + 1 ].y - contour[ i ].y;
                }
                else {
                    ix2 = contour[ 0 ].x - contour[ i ].x;
                    iy2 = contour[ 0 ].y - contour[ i ].y;
                }
                ix0 = contour[ i ].x - x1;
                iy0 = contour[ i ].y - y1;

                det  = ix2 * iy1 - ix1 * iy2;
                det1 = ix2 * iy0 - ix0 * iy2;
                if( det != 0.0f )
                {
                    t1 = ( ( float )( det1 ) ) / det;
                    if( t1 > ZERO_CLOSE && t1 < ONE_CLOSE )
                    {
                        det2 = ix1 * iy0 - ix0 * iy1;
                        t2 = ( ( float )( det2 ) ) / det;
                        if( t2 > ZERO_CLOSE && t2 < ONE_CLOSE ) {
                            isInternal = 0;
                        }

                    } // if( t1 > ZERO_CLOSE && t1 < ONE_CLOSE )

                } // if( det != 0.0f )

            } // for( i = 0; i < (*numEdges); i ++ )

            if( isInternal )
            {
                // this edge is internal
                notFoundFlag = 0;
                outEdges[ (*numEdges) * 2     ] = index1;
                outEdges[ (*numEdges) * 2 + 1 ] = index3;
                (*numEdges) ++;
                pointExist[ index2 ] = 0;
                index2 = index3;
                x2 = x3;
                y2 = y3;
                currentNum --;
                if( currentNum >= 3 ) {
                    do {
                        index3 ++;
                        if( index3 == num ) {
                            index3 = 0;
                        }
                    } while( !pointExist[ index3 ] );
                    x3 = contour[ index3 ].x;
                    y3 = contour[ index3 ].y;
                } // if( currentNum >= 3 )

            } // if( isInternal )
            else {
                // this edge intersects some other initial edges
                if( !notFoundFlag ) {
                    notFoundFlag = 1;
                    begIndex = index1;
                }
                index1 = index2;
                x1 = x2;
                y1 = y2;
                index2 = index3;
                x2 = x3;
                y2 = y3;
                do {
                    index3 ++;
                    if( index3 == num ) {
                        index3 = 0;
                    }
                    if( index3 == begIndex ) {
                        if( pointExist ) {
                            free( pointExist );
                        }
                        return false;
                    }
                } while( !pointExist[ index3 ] );
                x3 = contour[ index3 ].x;
                y3 = contour[ index3 ].y;
            } // if( isInternal ) else

        } // if( dx1 * dy2 - dx2 * dy1 < 0 )
        else
        {
            if( !notFoundFlag ) {
                notFoundFlag = 1;
                begIndex = index1;
            }
            index1 = index2;
            x1 = x2;
            y1 = y2;
            index2 = index3;
            x2 = x3;
            y2 = y3;
            do {
                index3 ++;
                if( index3 == num ) {
                    index3 = 0;
                }
                if( index3 == begIndex ) {
                    if( pointExist ) {
                        free( pointExist );
                    }
                    return false;
                }
            } while( !pointExist[ index3 ] );
            x3 = contour[ index3 ].x;
            y3 = contour[ index3 ].y;
        } // if( dx1 * dy2 - dx2 * dy1 < 0 ) else

    } // while( currentNum > 3 )

    if( pointExist ) {
        free( pointExist );
    }

    return true;

} // icvEarCutTriangulation

inline bool icvFindTwoNeighbourEdges( CvPoint* contour,
                                      int* edges,
                                      int numEdges,
                                      int vtxIdx,
                                      int mainEdgeIdx,
                                      int* leftEdgeIdx,
                                      int* rightEdgeIdx )
{
    int i;
    int compRes;
    int vec0_x, vec0_y;
    int x0, y0, x0_end, y0_end;
    int x1_left = 0, y1_left = 0, x1_right = 0, y1_right = 0, x2, y2;

    (*leftEdgeIdx)  = -1;
    (*rightEdgeIdx) = -1;

    if( edges[ mainEdgeIdx * 2 ] == vtxIdx ) {
        x0 = contour[ vtxIdx ].x;
        y0 = contour[ vtxIdx ].y;
        x0_end = contour[ edges[ mainEdgeIdx * 2 + 1 ] ].x;
        y0_end = contour[ edges[ mainEdgeIdx * 2 + 1 ] ].y;
        vec0_x = x0_end - x0;
        vec0_y = y0_end - y0;
    }
    else {
        //x0 = contour[ edges[ mainEdgeIdx * 2 ] ].x;
        //y0 = contour[ edges[ mainEdgeIdx * 2 ] ].y;
        //x0_end = contour[ vtxIdx ].x;
        //y0_end = contour[ vtxIdx ].y;
        x0 = contour[ vtxIdx ].x;
        y0 = contour[ vtxIdx ].y;
        x0_end = contour[ edges[ mainEdgeIdx * 2 ] ].x;
        y0_end = contour[ edges[ mainEdgeIdx * 2 ] ].y;
        vec0_x = x0_end - x0;
        vec0_y = y0_end - y0;
    }

    for( i = 0; i < numEdges; i ++ )
    {
        if( ( i != mainEdgeIdx ) &&
            ( edges[ i * 2 ] == vtxIdx || edges[ i * 2 + 1 ] == vtxIdx ) )
        {
            if( (*leftEdgeIdx) == -1 )
            {
                (*leftEdgeIdx) = (*rightEdgeIdx) = i;
                if( edges[ i * 2 ] == vtxIdx ) {
                    x1_left = x1_right = contour[ edges[ i * 2 + 1 ] ].x;
                    y1_left = y1_right = contour[ edges[ i * 2 + 1 ] ].y;
                }
                else {
                    x1_left = x1_right = contour[ edges[ i * 2 ] ].x;
                    y1_left = y1_right = contour[ edges[ i * 2 ] ].y;
                }

            } // if( (*leftEdgeIdx) == -1 )
            else
            {
                if( edges[ i * 2 ] == vtxIdx ) {
                    x2 = contour[ edges[ i * 2 + 1 ] ].x;
                    y2 = contour[ edges[ i * 2 + 1 ] ].y;
                }
                else {
                    x2 = contour[ edges[ i * 2 ] ].x;
                    y2 = contour[ edges[ i * 2 ] ].y;
                }

                compRes = icvIsFirstEdgeClosier( x0,
                    y0, x0_end, y0_end, x1_left, y1_left, x2, y2 );
                if( compRes == 0 ) {
                    return false;
                }
                if( compRes == -1 ) {
                    (*leftEdgeIdx) = i;
                    x1_left = x2;
                    y1_left = y2;
                } // if( compRes == -1 )
                else {
                    compRes = icvIsFirstEdgeClosier( x0,
                        y0, x0_end, y0_end, x1_right, y1_right, x2, y2 );
                    if( compRes == 0 ) {
                        return false;
                    }
                    if( compRes == 1 ) {
                        (*rightEdgeIdx) = i;
                        x1_right = x2;
                        y1_right = y2;
                    }

                } // if( compRes == -1 ) else

            } // if( (*leftEdgeIdx) == -1 ) else

        } // if( ( i != mainEdgesIdx ) && ...

    } // for( i = 0; i < numEdges; i ++ )

    return true;

} // icvFindTwoNeighbourEdges

bool icvFindReferences( CvPoint* contour,
                        int num,
                        int* outEdges,
                        int* refer,
                        int* numEdges )
{
    int i;
    int currPntIdx;
    int leftEdgeIdx, rightEdgeIdx;

    if( icvEarCutTriangulation( contour, num, outEdges, numEdges ) )
    {
        for( i = 0; i < (*numEdges); i ++ )
        {
            refer[ i * 4     ] = -1;
            refer[ i * 4 + 1 ] = -1;
            refer[ i * 4 + 2 ] = -1;
            refer[ i * 4 + 3 ] = -1;
        } // for( i = 0; i < (*numEdges); i ++ )

        for( i = 0; i < (*numEdges); i ++ )
        {
            currPntIdx = outEdges[ i * 2 ];
            if( !icvFindTwoNeighbourEdges( contour,
                outEdges, (*numEdges), currPntIdx,
                i, &leftEdgeIdx, &rightEdgeIdx ) )
            {
                return false;
            } // if( !icvFindTwoNeighbourEdges( contour, ...
            else
            {
                if( outEdges[ leftEdgeIdx * 2 ] == currPntIdx ) {
                    if( refer[ i * 4     ] == -1 ) {
                        refer[ i * 4     ] = ( leftEdgeIdx << 2 );
                    }
                }
                else {
                    if( refer[ i * 4     ] == -1 ) {
                        refer[ i * 4     ] = ( leftEdgeIdx << 2 ) | 2;
                    }
                }
                if( outEdges[ rightEdgeIdx * 2 ] == currPntIdx ) {
                    if( refer[ i * 4 + 1 ] == -1 ) {
                        refer[ i * 4 + 1 ] = ( rightEdgeIdx << 2 ) | 3;
                    }
                }
                else {
                    if( refer[ i * 4 + 1 ] == -1 ) {
                        refer[ i * 4 + 1 ] = ( rightEdgeIdx << 2 ) | 1;
                    }
                }

            } // if( !icvFindTwoNeighbourEdges( contour, ... ) else

            currPntIdx = outEdges[ i * 2 + 1 ];
            if( i == 18 ) {
                i = i;
            }
            if( !icvFindTwoNeighbourEdges( contour,
                outEdges, (*numEdges), currPntIdx,
                i, &leftEdgeIdx, &rightEdgeIdx ) )
            {
                return false;
            } // if( !icvFindTwoNeighbourEdges( contour, ...
            else
            {
                if( outEdges[ leftEdgeIdx * 2 ] == currPntIdx ) {
                    if( refer[ i * 4 + 3 ] == -1 ) {
                        refer[ i * 4 + 3 ] = ( leftEdgeIdx << 2 );
                    }
                }
                else {
                    if( refer[ i * 4 + 3 ] == -1 ) {
                        refer[ i * 4 + 3 ] = ( leftEdgeIdx << 2 ) | 2;
                    }
                }
                if( outEdges[ rightEdgeIdx * 2 ] == currPntIdx ) {
                    if( refer[ i * 4 + 2 ] == -1 ) {
                        refer[ i * 4 + 2 ] = ( rightEdgeIdx << 2 ) | 3;
                    }
                }
                else {
                    if( refer[ i * 4 + 2 ] == -1 ) {
                        refer[ i * 4 + 2 ] = ( rightEdgeIdx << 2 ) | 1;
                    }
                }

            } // if( !icvFindTwoNeighbourEdges( contour, ... ) else

        } // for( i = 0; i < (*numEdges); i ++ )

    } // if( icvEarCutTriangulation( contour, num, outEdges, numEdges ) )
    else {
        return false;
    } // if( icvEarCutTriangulation( contour, num, outEdges, ... ) else

    return true;

} // icvFindReferences

void cvDecompPoly( CvContour* cont,
                      CvSubdiv2D** subdiv,
                      CvMemStorage* storage )
{
    int*    memory;
    CvPoint*    contour;
    int*        outEdges;
    int*        refer;
    CvSubdiv2DPoint**   pntsPtrs;
    CvQuadEdge2D**      edgesPtrs;
    int numVtx;
    int numEdges;
    int i;
    CvSeqReader reader;
    CvPoint2D32f pnt;
    CvQuadEdge2D* quadEdge;

    numVtx = cont -> total;
    if( numVtx < 3 ) {
        return;
    }

    *subdiv = ( CvSubdiv2D* )0;

    memory = ( int* )malloc( sizeof( int ) * ( numVtx * 2
        + numVtx * numVtx * 2 * 5 )
        + sizeof( CvQuadEdge2D* ) * ( numVtx * numVtx )
        + sizeof( CvSubdiv2DPoint* ) * ( numVtx * 2 ) );
    contour     = ( CvPoint* )memory;
    outEdges    = ( int* )( contour + numVtx );
    refer       = outEdges + numVtx * numVtx * 2;
    edgesPtrs   = ( CvQuadEdge2D** )( refer + numVtx * numVtx * 4 );
    pntsPtrs    = ( CvSubdiv2DPoint** )( edgesPtrs + numVtx * numVtx );

    cvStartReadSeq( ( CvSeq* )cont, &reader, 0 );
    for( i = 0; i < numVtx; i ++ )
    {
        CV_READ_SEQ_ELEM( (contour[ i ]), reader );
    } // for( i = 0; i < numVtx; i ++ )

    if( !icvFindReferences( contour, numVtx, outEdges, refer, &numEdges ) )
    {
        free( memory );
        return;
    } // if( !icvFindReferences( contour, numVtx, outEdges, refer, ...

    *subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D,
                                sizeof( CvSubdiv2D ),
                                sizeof( CvSubdiv2DPoint ),
                                sizeof( CvQuadEdge2D ),
                                storage );

    for( i = 0; i < numVtx; i ++ )
    {
        pnt.x = ( float )contour[ i ].x;
        pnt.y = ( float )contour[ i ].y;
        pntsPtrs[ i ] = cvSubdiv2DAddPoint( *subdiv, pnt, 0 );
    } // for( i = 0; i < numVtx; i ++ )

    for( i = 0; i < numEdges; i ++ )
    {
        edgesPtrs[ i ] = ( CvQuadEdge2D* )
            ( cvSubdiv2DMakeEdge( *subdiv ) & 0xfffffffc );
    } // for( i = 0; i < numEdges; i ++ )

    for( i = 0; i < numEdges; i ++ )
    {
        quadEdge = edgesPtrs[ i ];
        quadEdge -> next[ 0 ] =
            ( ( CvSubdiv2DEdge )edgesPtrs[ refer[ i * 4     ] >> 2 ] )
            | ( refer[ i * 4     ] & 3 );
        quadEdge -> next[ 1 ] =
            ( ( CvSubdiv2DEdge )edgesPtrs[ refer[ i * 4 + 1 ] >> 2 ] )
            | ( refer[ i * 4 + 1 ] & 3 );
        quadEdge -> next[ 2 ] =
            ( ( CvSubdiv2DEdge )edgesPtrs[ refer[ i * 4 + 2 ] >> 2 ] )
            | ( refer[ i * 4 + 2 ] & 3 );
        quadEdge -> next[ 3 ] =
            ( ( CvSubdiv2DEdge )edgesPtrs[ refer[ i * 4 + 3 ] >> 2 ] )
            | ( refer[ i * 4 + 3 ] & 3 );
        quadEdge -> pt[ 0 ] = pntsPtrs[ outEdges[ i * 2     ] ];
        quadEdge -> pt[ 1 ] = ( CvSubdiv2DPoint* )0;
        quadEdge -> pt[ 2 ] = pntsPtrs[ outEdges[ i * 2 + 1 ] ];
        quadEdge -> pt[ 3 ] = ( CvSubdiv2DPoint* )0;
    } // for( i = 0; i < numEdges; i ++ )

    (*subdiv) -> topleft.x = ( float )cont -> rect.x;
    (*subdiv) -> topleft.y = ( float )cont -> rect.y;
    (*subdiv) -> bottomright.x =
        ( float )( cont -> rect.x + cont -> rect.width );
    (*subdiv) -> bottomright.y =
        ( float )( cont -> rect.y + cont -> rect.height );

    free( memory );
    return;

} // cvDecompPoly

#endif

// End of file decomppoly.cpp
