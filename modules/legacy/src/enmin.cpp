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

//#include "windows.h"

//#define ALPHA_EXPANSION

#ifndef ALPHA_EXPANSION
    #define ALPHA_BETA_EXCHANGE
#endif

#define MAX_LABEL 20

#define CV_MODULE(xxx) \
    ( (xxx) < 0 ? -(xxx) : (xxx) )

#define CV_MAX3(xxx1,xxx2,xxx3) \
    ( (xxx1) > (xxx2) && (xxx1) > (xxx3) ? (xxx1) : \
        (xxx2) > (xxx3) ? (xxx2) : (xxx3) )

#define CV_MIN2(xxx1,xxx2) \
    ( (xxx1) < (xxx2) ? (xxx1) : (xxx2) )

#define getSizeForGraph(xxxType) \
    ( sizeof(xxxType) < 8 ? 8 : sizeof(xxxType) + 4 - sizeof(xxxType) % 4 )

#define INT_INFINITY 1000000000
#define MAX_DIFFERENCE 10


// struct Vertex is used for storing vertices of graph
//      coord       - coordinate corresponding pixel on the real image line
struct Vertex
{
    CvGraphVtx vtx;
    int coord;
};

// struct Edge is used for storing edges of graph
//      weight      - weight of the edge ( maximum flow via the edge )
//      flow        - current flow via the edge
//      srcVtx      - coordinate of source vertex on the real image line
//      destVtx     - coordinate of destination vertex on the real image line
struct Edge
{
    CV_GRAPH_EDGE_FIELDS()
    int weight;
    int flow;
    int srcVtx;
    int destVtx;
};

// function vFunc is energy function which determines the difference
//   between two labels ( alpha and beta )
//      alpha       - label number one
//      beta        - label number two
inline int vFunc( int alpha, int beta )
{
    if( alpha == beta )
        return 0;
    else
        return /*1*//*5*/10;
}

// function dFunc is energy function which determines energy of interaction
//   between pixel ( having coordinate xCoord ) and label
//          leftLine        - line of left image
//          rightLine       - line of right image
//          xCoord          - coordinate of pixel on the left image
//          label           - label corresponding to the pixel
//          width           - width of the image line in pixels
inline int dFunc( unsigned char* leftLine,
                  unsigned char* rightLine,
                  int xCoord,
                  int label,
                  int width)
{
    assert( xCoord >= 0 && xCoord < width );
    int r, g, b;
    int yCoord = xCoord + label;

    if( yCoord >= width )
        yCoord = width;
    if( yCoord < 0 )
        yCoord = 0;

    r = leftLine[ 3 * xCoord     ] - rightLine[ 3 * yCoord     ];
    g = leftLine[ 3 * xCoord + 1 ] - rightLine[ 3 * yCoord + 1 ];
    b = leftLine[ 3 * xCoord + 2 ] - rightLine[ 3 * yCoord + 2 ];

    r = CV_MODULE( r );
    g = CV_MODULE( g );
    b = CV_MODULE( b );

    return CV_MAX3( r, g, b );
}

// function allocTempMem allocates all temporary memory needed for work
//   of some function
//      memPtr          - pointer to pointer to the large block of memory
//      verticesPtr     - pointer to pointer to block of memory for
//                        temporary storing vertices
//      width           - width of line in pixels
void allocTempMem( int** memPtr,
                   int** verticesPtr,
                   int width )
{
    int* tempPtr = ( int* ) malloc( ( width + 2 ) * 7 * sizeof( int ) );
    *verticesPtr = tempPtr;
    *memPtr = *verticesPtr + width + 2;
}

// function freeTempMem frees all allocated by allocTempMem function memory
//      memPtr          - pointer to pointer to the large block of memory
//      verticesPtr     - pointer to pointer to block of memory for
//                        temporary storing vertices
void freeTempMem( int** memPtr,
                  int** verticesPtr )
{
    free( ( void* )( *verticesPtr ) );
    *verticesPtr = NULL;
    *memPtr = NULL;
}

// function makeGraph creates initial graph to find maximum flow in it
//      graphPtr        - pointer to pointer to CvGraph structure to be filled
//      leftLine        - pointer to the left image line
//      rightLine       - pointer to the right image line
//      alpha           - label number one for doing exchange
//      beta            - label number two for doing exchange
//      corr            - pointer to array of correspondences ( each element
//                        of array includes disparity of pixel on right image
//                        for pixel each on left image ). This pointer direct
//                        to correspondence ofr one line only
//      width           - width of image lines in pixels
//      storage         - pointer to CvMemStorage structure which contains
//                        memory storage
void makeGraph( CvGraph** graphPtr,
                unsigned char* leftLine,
                unsigned char* rightLine,
                int alpha,
                int beta,
                int* corr,
                int width,
                CvMemStorage* storage )
{
    int i;

    if( *graphPtr )  {
        cvClearGraph( *graphPtr );
    }
    /*else {*/
        *graphPtr = cvCreateGraph( CV_SEQ_KIND_GRAPH | CV_GRAPH_FLAG_ORIENTED,
                                   sizeof( CvGraph ),
                                   getSizeForGraph( Vertex ),
                                   getSizeForGraph( Edge ),
                                   storage );
    /*}*/

    CvGraph* graph = *graphPtr;

    #ifdef ALPHA_BETA_EXCHANGE

    CvGraphVtx* newVtxPtr;
    for( i = 0; i < width; i ++ )
    {
        if( corr[i] == alpha || corr[i] == beta ) {
            cvGraphAddVtx( graph, NULL, &newVtxPtr );
            ( ( Vertex* )newVtxPtr ) -> coord = i;
        }
    } /* for( i = 0; i < width; i ++ ) */
    cvGraphAddVtx( graph, NULL, &newVtxPtr );
    if( newVtxPtr )
        ( ( Vertex* )newVtxPtr ) -> coord = -2; /* adding alpha vertex */
    cvGraphAddVtx( graph, NULL, &newVtxPtr );
    if( newVtxPtr )
        ( ( Vertex* )newVtxPtr ) -> coord = -1; /* adding beta vertex */

    int alphaVtx = graph -> total - 2;
    int betaVtx = graph -> total - 1;
    CvGraphEdge* newEdgePtr;
    CvGraphVtx* vtxPtr;
    if( graph -> total > 2 )
    {
        for( i = 0; i < alphaVtx; i ++ )
        {
            vtxPtr = cvGetGraphVtx( graph, i );

            /* adding edge oriented from alpha vertex to current vertex */
            cvGraphAddEdge( graph, alphaVtx, i, NULL, &newEdgePtr );
            ( ( Edge* )newEdgePtr ) -> weight = dFunc( leftLine,
                rightLine,
                ( ( Vertex* )vtxPtr ) -> coord,
                alpha,
                width );
            ( ( Edge* )newEdgePtr ) -> flow = 0;
            if( i != 0 ) {
                CvGraphVtx* tempVtxPtr = cvGetGraphVtx( graph, i - 1 );
                /* if vertices are neighbours */
                if( ( ( Vertex* )tempVtxPtr ) -> coord + 1 ==
                    ( ( Vertex* )vtxPtr ) -> coord )
                {
                    ( ( Edge* )newEdgePtr ) -> weight +=
                        vFunc( corr[ ( ( Vertex* )tempVtxPtr ) -> coord ],
                               alpha );
                    /* adding neighbour edge oriented from current vertex
                       to the previous one */
                    CvGraphEdge* tempEdgePtr;
                    cvGraphAddEdge( graph, i, i - 1, NULL, &tempEdgePtr );
                    ( ( Edge* )tempEdgePtr ) -> weight = vFunc( alpha, beta );
                    ( ( Edge* )tempEdgePtr ) -> flow = 0;
                    ( ( Edge* )tempEdgePtr ) -> srcVtx =
                        ( ( Vertex* )vtxPtr ) -> coord;
                    ( ( Edge* )tempEdgePtr ) -> destVtx =
                        ( ( Vertex* )tempVtxPtr ) -> coord;
                }
            } /* if( i != 0 ) */
            if( i != alphaVtx - 1 ) {
                CvGraphVtx* tempVtxPtr = cvGetGraphVtx( graph, i + 1 );
                /* if vertices are neighbours */
                if( ( ( Vertex* )tempVtxPtr ) -> coord - 1 ==
                    ( ( Vertex* )vtxPtr ) -> coord )
                {
                    ( ( Edge* )newEdgePtr ) -> weight +=
                        vFunc( corr[ ( ( Vertex* )tempVtxPtr ) -> coord ],
                               alpha );
                    /* adding neighbour edge oriented from current vertex
                       to the next one */
                    CvGraphEdge* tempEdgePtr;
                    cvGraphAddEdge( graph, i, i + 1, NULL, &tempEdgePtr );
                    ( ( Edge* )tempEdgePtr ) -> weight = vFunc( alpha, beta );
                    ( ( Edge* )tempEdgePtr ) -> flow = 0;
                    ( ( Edge* )tempEdgePtr ) -> srcVtx =
                        ( ( Vertex* )vtxPtr ) -> coord;
                    ( ( Edge* )tempEdgePtr ) -> destVtx =
                        ( ( Vertex* )tempVtxPtr ) -> coord;
                }
            } /* if( i != alphaVtx - 1 ) */
            ( ( Edge* )newEdgePtr ) -> flow = 0;
            ( ( Edge* )newEdgePtr ) -> srcVtx = -1; /* source vertex is alpha
                                                       vertex */
            ( ( Edge* )newEdgePtr ) -> destVtx = ( ( Vertex* )vtxPtr ) -> coord;

            /* adding edge oriented from current vertex to beta vertex */
            cvGraphAddEdge( graph, i, betaVtx, NULL, &newEdgePtr );
            ( ( Edge* )newEdgePtr ) -> weight = dFunc( leftLine,
                rightLine,
                ( ( Vertex* )vtxPtr ) -> coord,
                beta,
                width );
            ( ( Edge* )newEdgePtr ) -> flow = 0;
            if( i != 0 ) {
                CvGraphVtx* tempVtxPtr = cvGetGraphVtx( graph, i - 1 );
                /* if vertices are neighbours */
                if( ( ( Vertex* )tempVtxPtr ) -> coord + 1 ==
                    ( ( Vertex* )vtxPtr ) -> coord )
                {
                    ( ( Edge* )newEdgePtr ) -> weight +=
                        vFunc( corr[ ( ( Vertex* )tempVtxPtr ) -> coord ],
                               beta );
                }
            } /* if( i != 0 ) */
            if( i != alphaVtx - 1 ) {
                CvGraphVtx* tempVtxPtr = cvGetGraphVtx( graph, i + 1 );
                /* if vertices are neighbours */
                if( ( ( Vertex* )tempVtxPtr ) -> coord - 1 ==
                    ( ( Vertex* )vtxPtr ) -> coord )
                {
                    ( ( Edge* )newEdgePtr ) -> weight +=
                        vFunc( corr[ ( ( Vertex* )tempVtxPtr ) -> coord ],
                               beta );
                }
            } /* if( i != alphaVtx - 1 ) */
            ( ( Edge* )newEdgePtr ) -> flow = 0;
            ( ( Edge* )newEdgePtr ) -> srcVtx =
                ( ( Vertex* )vtxPtr ) -> coord;
            ( ( Edge* )newEdgePtr ) -> destVtx = -2; /* destination vertex is
                                                        beta vertex */

        } /* for( i = 0; i < graph -> total - 2; i ++ ) */

    } /* if( graph -> total > 2 ) */

    #endif /* #ifdef ALPHA_BETA_EXCHANGE */

    #ifdef ALPHA_EXPANSION
    #endif /* #ifdef ALPHA_EXPANSION */

} /* makeGraph */

// function makeHelpGraph creates help graph using initial graph
//      graph           - pointer to initial graph ( represented by CvGraph
//                        structure )
//      hlpGraphPtr     - pointer to pointer to new help graph
//      storage         - pointer to CvStorage structure
//      mem             - pointer to memory allocated by allocTempMem function
//      vertices        - pointer to memory allocated by allocTempMem function
//      verticesCountPtr- pointer to value containing number of vertices
//                        in vertices array
//      width           - width of image line in pixels
int makeHelpGraph( CvGraph* graph,
                   CvGraph** hlpGraphPtr,
                   CvMemStorage* storage,
                   int* mem,
                   int* vertices,
                   int* verticesCountPtr,
                   int width )
{
    int u, v;
    int* order = mem;
    int* lengthArr = order + width + 2;
    int s = graph -> total - 2; /* source vertex */
    int t = graph -> total - 1; /* terminate vertex */
    int orderFirst;
    int orderCount;
    int &verticesCount = *verticesCountPtr;
    CvGraph* hlpGraph;

    if( *hlpGraphPtr )  {
        cvClearGraph( *hlpGraphPtr );
    }
    else {
        *hlpGraphPtr = cvCreateGraph( CV_SEQ_KIND_GRAPH |
                                          CV_GRAPH_FLAG_ORIENTED,
                                      sizeof( CvGraph ),
                                      getSizeForGraph( Vertex ),
                                      getSizeForGraph( Edge ),
                                      storage );
    }

    hlpGraph = *hlpGraphPtr;

    /* initialization */
    for( u = 0; u < graph -> total; u ++ )
    {
        lengthArr[ u ] = INT_INFINITY;
        cvGraphAddVtx( hlpGraph, NULL, NULL );
    } /* for( u = 0; u < graph -> total - 1; u ++ ) */

    orderFirst = 0;
    orderCount = 0;
    verticesCount = 0;
    lengthArr[ s ] = 0;

    /* add vertex s to order */
    order[ orderCount ] = s;
    orderCount ++;

    while( orderCount != orderFirst )
    {
        /* getting u from order */
        u = order[ orderFirst ];
        orderFirst ++;

        /* adding u to vertex array */
        vertices[ verticesCount ] = u;
        verticesCount ++;

        int ofs;
        CvGraphVtx* graphVtx = cvGetGraphVtx( graph, u );

        /* processing all vertices outgoing from vertex u */
        CvGraphEdge* graphEdge = graphVtx -> first;
        while( graphEdge )
        {
            int tempVtxIdx = cvGraphVtxIdx( graph, graphEdge -> vtx[1] );
            ofs = tempVtxIdx == u;
            if( !ofs )
            {
                v = tempVtxIdx;

                CvGraphEdge* tempGraphEdge = cvFindGraphEdge( graph, u, v );
                if( ( lengthArr[ u ] < lengthArr[ v ] )
                    && ( lengthArr[ v ] <= lengthArr[ t ] )
                    && ( ( ( Edge* )tempGraphEdge ) -> flow <
                        ( ( Edge* )tempGraphEdge ) -> weight ) )
                {
                    if( lengthArr[ v ] == INT_INFINITY )
                    {
                        /* adding vertex v to order */
                        order[ orderCount ] = v;
                        orderCount ++;

                        lengthArr[ v ] = lengthArr[ u ] + 1;
                        CvGraphEdge* tempGraphEdge2;

                        cvGraphAddEdge( hlpGraph, u, v, NULL, &tempGraphEdge2 );
                        ( ( Edge* )tempGraphEdge2 ) -> flow = 0;

                        ( ( Edge* )tempGraphEdge2 ) -> weight =
                            ( ( Edge* )tempGraphEdge ) -> weight -
                            ( ( Edge* )tempGraphEdge ) -> flow;

                    } /* if( length[ v ] == INT_INFINITY ) */

                } /* if( ( lengthArr[ u ] < lengthArr[ v ] ) ... */

            } /* if( !ofs ) */

            graphEdge = graphEdge -> next[ ofs ];

        } /* while( graphEdge ) */

        /* processing all vertices incoming to vertex u */
        graphEdge = graphVtx -> first;
        while( graphEdge )
        {
            int tempVtxIdx = cvGraphVtxIdx( graph, graphEdge -> vtx[1] );
            ofs = tempVtxIdx == u;
            if( ofs )
            {
                tempVtxIdx = cvGraphVtxIdx( graph, graphEdge -> vtx[0] );
                v = tempVtxIdx;

                CvGraphEdge* tempGraphEdge = cvFindGraphEdge( graph, v, u );
                if( ( lengthArr[ u ] < lengthArr[ v ] )
                    && ( lengthArr[ v ] <= lengthArr[ t ] )
                    && ( ( ( Edge* )tempGraphEdge ) -> flow > 0 ) )
                {
                    if( lengthArr[ v ] == INT_INFINITY )
                    {
                        /* adding vertex v to order */
                        order[ orderCount ] = v;
                        orderCount ++;

                        lengthArr[ v ] = lengthArr[ u ] + 1;
                        CvGraphEdge* tempGraphEdge3 = cvFindGraphEdge( hlpGraph, u, v );

                        if( tempGraphEdge3 == NULL ||
                            ( ( Edge* )tempGraphEdge3 ) -> weight == 0 )
                        {
                            CvGraphEdge* tempGraphEdge2;
                            cvGraphAddEdge( hlpGraph, u, v, NULL,
                                &tempGraphEdge2 );
                            ( ( Edge* )tempGraphEdge2 ) -> flow = 0;
                            ( ( Edge* )tempGraphEdge2 ) -> weight = 0;
                        } /* if( tempGraphEdge3 == NULL || ... */

                        ( ( Edge* )tempGraphEdge3 ) -> weight +=
                            ( ( Edge* )tempGraphEdge ) -> flow;

                    } /* if( length[ v ] == INT_INFINITY ) */

                } /* if( ( lengthArr[ u ] < lengthArr[ v ] ) ... */

            } /* if( ofs ) */

            graphEdge = graphEdge -> next[ ofs ];

        } /* while( graphEdge ) */
  
    } /* while( orderCount != orderFirst ) */

    int i;
    for( i = 0; i < hlpGraph -> total - 2; i ++ )
    {
        CvGraphVtx* hlpGraphVtxTemp = cvGetGraphVtx( hlpGraph, i );
        if( hlpGraphVtxTemp ) {
            if( !hlpGraphVtxTemp -> first ) {
                cvGraphRemoveVtxByPtr( hlpGraph, hlpGraphVtxTemp );
            }
        }
    } /* for( i = 0; i < hlpGraph -> total - 2; i ++ ) */

    return lengthArr[ t ];

} /* makeHelpGraph */

// function makePseudoMaxFlow increases flow in graph by using hlpGraph
//      graph           - pointer to initial graph
//      hlpGraph        - pointer to help graph
//      vertices        - pointer to vertices array
//      verticesCount   - number of vertices in vertices array
//      mem             - pointer to memory allocated by allocTempMem function
//      width           - width of image line in pixels
void makePseudoMaxFlow( CvGraph* graph,
                        CvGraph* hlpGraph,
                        int* vertices,
                        int verticesCount,
                        int* mem,
                        int width )
{
    int stekCount;
    int orderFirst;
    int orderCount;
    int i;
    int v, u;
    int* stek = mem;
    int* order = stek + width + 2;
    int* incomFlow = order + width + 2;
    int* outgoFlow = incomFlow + width + 2;
    int* flow = outgoFlow + width + 2;
    int* cargo = flow+ width + 2;
    int s = graph -> total - 2; /* source vertex */
    int t = graph -> total - 1; /* terminate vertex */
    int realVerticesCount = verticesCount;

    stekCount = 0;

    for( i = 0; i < verticesCount; i ++ )
    {
        v = vertices[ i ];

        incomFlow[ v ] = outgoFlow[ v ] = 0;

        if( v == s ) {
            incomFlow[ v ] = INT_INFINITY;
        } /* if( v == s ) */
        else {
            CvGraphVtx* hlpGraphVtx = cvGetGraphVtx( hlpGraph, v );
            CvGraphEdge* hlpGraphEdge = hlpGraphVtx -> first;
            int ofs;

            while( hlpGraphEdge )
            {
                int vtxIdx = cvGraphVtxIdx( hlpGraph,
                    hlpGraphEdge -> vtx[1] );
                ofs = vtxIdx == v;

                if( ofs )
                {
                    incomFlow[ v ] += ( ( Edge* )hlpGraphEdge ) -> weight;
                } /* if( ofs ) */

                hlpGraphEdge = hlpGraphEdge -> next[ ofs ];
            } /* while( hlpGraphEdge ) */

        } /* if( v == s ) else */

        if( v == t ) {
            outgoFlow[ v ] = INT_INFINITY;
        } /* if( v == t ) */
        else {
            CvGraphVtx* hlpGraphVtx = cvGetGraphVtx( hlpGraph, v );
            CvGraphEdge* hlpGraphEdge = hlpGraphVtx -> first;
            int ofs;

            while( hlpGraphEdge )
            {
                int vtxIdx = cvGraphVtxIdx( hlpGraph,
                    hlpGraphEdge -> vtx[1] );
                ofs = vtxIdx == v;

                if( !ofs )
                {
                    outgoFlow[ v ] += ( ( Edge* )hlpGraphEdge ) -> weight;
                } /* if( ofs ) */

                hlpGraphEdge = hlpGraphEdge -> next[ ofs ];
            } /* while( hlpGraphEdge ) */

        } /* if( v == t ) else */

        flow[ v ] = CV_MIN2( incomFlow[ v ], outgoFlow[ v ] );

        if( !flow[ v ] ) {
            stek[ stekCount ] = v;
            stekCount ++;
        } /* if( !flow[ v ] ) */

    } /* for( i = 0; i < verticesCount; i ++ ) */

    for( i = 0; i < verticesCount; i ++ )
    {
        v = vertices[ i ];
        cargo[ v ] = 0;
    } /* for( i = 0; i < verticesCount; i ++ ) */

    while( realVerticesCount > 2 )
    {
        /* deleting all vertices included in stek */
        while( stekCount )
        {
            v = stek[ stekCount - 1 ];
            stekCount --;

            /* deleting edges incoming to v and outgoing from v */
            int ofs;
            CvGraphVtx* hlpGraphVtx = cvGetGraphVtx( hlpGraph, v );
            CvGraphEdge* hlpGraphEdge;
            if( hlpGraphVtx ) {
                hlpGraphEdge = hlpGraphVtx -> first;
            }
            else {
                hlpGraphEdge = NULL;
            }
            while( hlpGraphEdge )
            {
                CvGraphVtx* hlpGraphVtx2 = hlpGraphEdge -> vtx[ 1 ];
                int hlpGraphVtxIdx2 = cvGraphVtxIdx( hlpGraph,
                    hlpGraphVtx2 );
                ofs = hlpGraphVtxIdx2 == v;

                if( ofs )
                {
                    /* hlpGraphEdge is incoming edge */
                    CvGraphVtx* hlpGraphVtx3 = hlpGraphEdge -> vtx[0];
                    u = cvGraphVtxIdx( hlpGraph,
                                       hlpGraphVtx3 );
                    outgoFlow[ u ] -= ( ( Edge* )hlpGraphEdge ) -> weight
                        - ( ( Edge* )hlpGraphEdge ) -> flow;
                    cvGraphRemoveEdgeByPtr( hlpGraph,
                                            hlpGraphVtx3,
                                            hlpGraphVtx2 );
                    if( flow[ u ] != 0 ) {
                        flow[ u ] = CV_MIN2( incomFlow[u], outgoFlow[u] );
                        if( flow[ u ] == 0 ) {
                            stek[ stekCount ] = u;
                            stekCount ++;
                        }
                    }
                } /* if( ofs ) */
                else
                {
                    /* hlpGraphEdge is outgoing edge */
                    CvGraphVtx* hlpGraphVtx3 = hlpGraphEdge -> vtx[1];
                    int u = cvGraphVtxIdx( hlpGraph,
                                              hlpGraphVtx3 );
                    incomFlow[ u ] -= ( ( Edge* )hlpGraphEdge ) -> weight
                        - ( ( Edge* )hlpGraphEdge ) -> flow;
                    cvGraphRemoveEdgeByPtr( hlpGraph,
                                            hlpGraphVtx2,
                                            hlpGraphVtx3 );
                    if( flow[ u ] != 0 ) {
                        flow[ u ] = CV_MIN2( incomFlow[u], outgoFlow[u] );
                        if( flow[ u ] == 0 ) {
                            stek[ stekCount ] = u;
                            stekCount ++;
                        }
                    }
                } /* if( ofs ) else */

                hlpGraphEdge = hlpGraphEdge -> next[ ofs ];

            } /* while( hlpGraphEdge ) */

            /* deleting vertex v */
            cvGraphRemoveVtx( hlpGraph, v );
            realVerticesCount --;

        } /* while( stekCount ) */

        if( realVerticesCount > 2 ) /* the flow is not max still */
        {
            int p = INT_INFINITY;
            int r = -1;
            CvGraphVtx* graphVtx;

            if( realVerticesCount == 3 ) {
                r = r;
            }
            for( i = 0; i < hlpGraph -> total - 2; i ++ )
            {
                graphVtx = cvGetGraphVtx( hlpGraph, i );
                if( graphVtx ) {
                    v = cvGraphVtxIdx( hlpGraph, graphVtx );
                    if( flow[ v ] < p ) {
                        r = v;
                        p = flow[ v ];
                    }
                }

            } /* for( i = 0; i < hlpGraph -> total - 2; i ++ ) */

            /* building of size p flow from r to t */
            orderCount = orderFirst = 0;
            order[ orderCount ] = r;
            orderCount ++;

            v = order[ orderFirst ];
            orderFirst ++;

            cargo[ r ] = p;
            do /* while( v != t ) */
            {
                incomFlow[ v ] -= cargo[ v ];
                outgoFlow[ v ] -= cargo[ v ];
                flow[ v ] -= cargo[ v ];

                if( flow[ v ] == 0 ) {
                    stek[ stekCount ] = v;
                    stekCount ++;
                }

                if( v == t ) {
                    cargo[ v ] = p;
                }
                else
                {
                    int ofs;
                    CvGraphVtx* hlpGraphVtx2;
                    CvGraphVtx* hlpGraphVtx = cvGetGraphVtx( hlpGraph, v );
                    CvGraphEdge* hlpGraphEdge = hlpGraphVtx -> first;
                    CvGraphEdge* hlpGraphEdge2 = NULL;

                    while( hlpGraphEdge && cargo[ v ] > 0 )
                    {
                        hlpGraphVtx2 = hlpGraphEdge -> vtx[ 1 ];
                        u = cvGraphVtxIdx( hlpGraph, hlpGraphVtx2 );
                        ofs = u == v;

                        if( !ofs )
                        {
                            if( cargo[ u ] == 0 ) {
                                order[ orderCount ] = u;
                                orderCount ++;
                            }
                            int delta = ( ( Edge* )hlpGraphEdge ) -> weight
                                - ( ( Edge* )hlpGraphEdge ) -> flow;
                            delta = CV_MIN2( cargo[ v ], delta );
                            ( ( Edge* )hlpGraphEdge ) -> flow += delta;
                            cargo[ v ] -= delta;
                            cargo[ u ] += delta;
                            if( ( ( Edge* )hlpGraphEdge ) -> weight ==
                                ( ( Edge* )hlpGraphEdge ) -> flow )
                            {
                                /* deleting hlpGraphEdge */
                                hlpGraphEdge2 = hlpGraphEdge -> next[ ofs ];
                                CvGraphEdge* graphEdge =
                                    cvFindGraphEdge( graph, v, u );
                                ( ( Edge* )graphEdge ) -> flow +=
                                    ( ( Edge* )hlpGraphEdge ) -> flow;
                                cvGraphRemoveEdgeByPtr( hlpGraph,
                                    hlpGraphEdge -> vtx[0],
                                    hlpGraphEdge -> vtx[1] );
                            }
                        } /* if( !ofs ) */

                        if( hlpGraphEdge2 ) {
                            hlpGraphEdge = hlpGraphEdge2;
                            hlpGraphEdge2 = NULL;
                        }
                        else {
                            hlpGraphEdge = hlpGraphEdge -> next[ ofs ];
                        }
                    } /* while( hlpGraphEdge && cargo[ v ] > 0 ) */

                } /* if( v == t ) else */

                v = order[ orderFirst ];
                orderFirst ++;

            } while( v != t ); /* do */

            /* building of size p flow from s to r */
            orderCount = orderFirst = 0;
            order[ orderCount ] = r;
            orderCount ++;

            v = order[ orderFirst ];
            orderFirst ++;

            cargo[ r ] = p;
            do /* while( v != s ) */
            {
                if( v != r )
                {
                    incomFlow[ v ] -= cargo[ v ];
                    outgoFlow[ v ] -= cargo[ v ];
                    flow[ v ] -= cargo[ v ];
                    if( flow[ v ] == 0 ) {
                        stek[ stekCount ] = v;
                        stekCount ++;
                    }
                } /* if( v != r ) */

                if( v == s ) {
                    cargo[ v ] = 0;
                } /* if( v == s ) */
                else
                {
                    int ofs;

                    CvGraphVtx* hlpGraphVtx = cvGetGraphVtx( hlpGraph, v );
                    CvGraphEdge* hlpGraphEdge = hlpGraphVtx -> first;
                    CvGraphEdge* hlpGraphEdge2 = NULL;
                    while( hlpGraphEdge && cargo[ v ] > 0 )
                    {
                        u = cvGraphVtxIdx( hlpGraph,
                                hlpGraphEdge -> vtx[ 1 ] );
                        ofs = u == v;

                        if( ofs )
                        {
                            u = cvGraphVtxIdx( hlpGraph,
                                    hlpGraphEdge -> vtx[ 0 ] );

                            if( cargo[ u ] == 0 ) {
                                order[ orderCount ] = u;
                                orderCount ++;
                            }

                            int delta = ( ( Edge* )hlpGraphEdge ) -> weight
                                - ( ( Edge* )hlpGraphEdge ) -> flow;

                            delta = CV_MIN2( cargo[ v ], delta );
                            
                            (( ( Edge* )hlpGraphEdge ) -> flow) += delta;

                            cargo[ v ] -= delta;
                            cargo[ u ] += delta;

                            if( ( ( Edge* )hlpGraphEdge ) -> weight ==
                                ( ( Edge* )hlpGraphEdge ) -> flow )
                            {
                                hlpGraphEdge2 = hlpGraphEdge -> next[ ofs ];
                                CvGraphEdge* graphEdge =
                                    cvFindGraphEdge( graph, u, v );
                                ( ( Edge* )graphEdge ) -> flow +=
                                    ( ( Edge* )hlpGraphEdge ) -> flow;
                                cvGraphRemoveEdgeByPtr( hlpGraph,
                                    hlpGraphEdge -> vtx[0],
                                    hlpGraphEdge -> vtx[1] );
                            }
                        } /* if( ofs ) */

                        if( hlpGraphEdge2 ) {
                            hlpGraphEdge = hlpGraphEdge2;
                            hlpGraphEdge2 = NULL;
                        }
                        else {
                            hlpGraphEdge = hlpGraphEdge -> next[ ofs ];
                        }
                    } /* while( hlpGraphEdge && cargo[ v ] > 0 ) */

                } /* if( v == s ) else */

                v = order[ orderFirst ]; //added
                orderFirst ++; //added

            } while( v != s ); /* do */

        } /* if( hlpGraph -> total > 2 ) */

    } /* while( hlpGraph -> total > 2 ) */

} /* makePseudoMaxFlow */


// function oneStep produces one alpha-beta exchange for one line of images
//      leftLine        - pointer to the left image line
//      rightLine       - pointer to the right image line
//      alpha           - label number one
//      beta            - label number two
//      corr            - pointer to correspondence array for this line
//      width           - width of image line in pixels
//      mem             - pointer to memory allocated by allocTempMem function
//      vertices        - pointer to vertices array allocated by allocTempMem
//                        function
//      storage         - pointer to CvMemStorage structure
bool oneStep( unsigned char* leftLine,
              unsigned char* rightLine,
              int alpha,
              int beta,
              int* corr,
              int width,
              int* mem,
              int* vertices,
              CvMemStorage* storage )
{
    CvGraph* graph = NULL;
    CvGraph* hlpGraph = NULL;
    CvMemStoragePos storagePos;
    int i;
    bool change = false;
    cvSaveMemStoragePos( storage, &storagePos );

    int verticesCount;

    makeGraph( &graph, leftLine, rightLine, alpha, beta, corr, width, storage );

    int s = graph -> total - 2; /* source vertex - alpha vertex */
    //int t = graph -> total - 1; /* terminate vertex - beta vertex */

    int length = makeHelpGraph( graph,
                                &hlpGraph,
                                storage,
                                mem,
                                vertices,
                                &verticesCount,
                                width );
    while( length != INT_INFINITY )
    {
        change = true;
        makePseudoMaxFlow( graph,
                           hlpGraph,
                           vertices,
                           verticesCount,
                           mem,
                           width );
        cvClearGraph( hlpGraph );
        length = makeHelpGraph( graph,
                                &hlpGraph,
                                storage,
                                mem,
                                vertices,
                                &verticesCount,
                                width );
    } /* while( length != INT_INFINITY ) */

    int coord;
    CvGraphVtx* graphVtx;
    for( i = 0; i < s; i ++ )
    {
        CvGraphEdge* graphEdge = cvFindGraphEdge( graph, s, i );

        if( ( ( Edge* )graphEdge ) -> weight ==
            ( ( Edge* )graphEdge ) -> flow )
        { /* this vertex must have alpha label */
            graphVtx = cvGetGraphVtx( graph, i );
            coord = ( ( Vertex* )graphVtx )-> coord;
            if( corr[ coord ] != alpha ) {
                corr[ coord ] = alpha; //added
                change = true;
            }
            else {
                corr[ coord ] = alpha;
            }
        } /* if( ( ( Edge* )graphEdge ) -> weight == ... */
        else
        { /* this vertex must have beta label */
            graphVtx = cvGetGraphVtx( graph, i );
            coord = ( ( Vertex* )graphVtx )-> coord;
            if( corr[ coord ] != beta ) {
                corr[ coord ] = beta; //added
                change = true;
            }
            else {
                corr[ coord ] = beta;
            }
        } /* if( ( ( Edge* )graphEdge ) -> weight == ... else */

    } /* for( i = 0; i < s; i ++ ) */

    cvClearGraph( hlpGraph );
    cvClearGraph( graph );

    cvRestoreMemStoragePos( storage, &storagePos );

    return change;

} /* oneStep */

// function initCorr fills correspondence array with initial values
//      corr                - pointer to correspondence array for this line
//      width               -  width of image line in pixels
//      maxPixelDifference  - maximum value of difference between the same
//                            point painted on two images
void initCorr( int* corr, int width, int maxPixelDifference )
{
    int i;
    int pixelDifferenceRange = maxPixelDifference * 2 + 1;

    for( i = 0; i < width; i ++ )
    {
        corr[ i ] = i % pixelDifferenceRange - maxPixelDifference;
    }
} /* initCorr */

// function oneLineCorr fully computes one line of images
//      leftLine                - pointer to the left image line
//      rightLine               - pointer to the right image line
//      corr                    - pointer to the correspondence array for one
//                                line
//      mem                     - pointer to memory allocated by allocTempMem
//                                function
//      vertices                - pointer to memory allocated by allocTempMem
//                                function
//      width                   - width of image line in pixels
//      maxPixelDifference      - maximum value of pixel differences in pixels
//      storage                 - pointer to CvMemStorage struct which
//                                contains memory storage
void oneLineCorr( unsigned char* leftLine,
                  unsigned char* rightLine,
                  int* corr,
                  int* mem,
                  int* vertices,
                  int width,
                  int maxPixelDifference,
                  CvMemStorage* storage )
{
    int result = 1;
    int count = 0;
    int i, j;

    initCorr( corr, width, maxPixelDifference );
    while( result )
    {
        result = 0;

        for( i = - maxPixelDifference; i < maxPixelDifference; i ++ )
        for( j = i + 1; j <= maxPixelDifference; j ++ )
        {
            result += (int)oneStep( leftLine,
                               rightLine,
                               i,
                               j,
                               corr,
                               width,
                               mem,
                               vertices,
                               storage );
        } /* for( j = i + 1; j < width; j ++ ) */

        count ++;
        if( count > /*0*//*1*/2 ) {
            break;
        }

    } /* while( result ) */

} /* oneLineCorr */

// function allLinesCorr computes all lines on the images
//      leftImage           - pointer to the left image
//      leftLineStep        - size of line on the left image in bytes
//      rightImage          - pointer to the right image
//      rightLineStep       - size of line on the right image in bytes
//      width               - width of line in pixels
//      height              - height of image in pixels
//      corr                - pointer to correspondence array for all lines
//      maxPixelDifference  - maximum value of difference between the same
//                            point painted on two images
//      storage             - pointer to CvMemStorage which contains memory
//                            storage
void allLinesCorr( unsigned char* leftImage,
                   int leftLineStep,
                   unsigned char* rightImage,
                   int rightLineStep,
                   int width,
                   int height,
                   int* corr,
                   int maxPixelDifference,
                   CvMemStorage* storage )
{
    int i;
    unsigned char* leftLine = leftImage;
    unsigned char* rightLine = rightImage;
    int* mem;
    int* vertices;

    allocTempMem( &mem,
                  &vertices,
                  width );

    for( i = 0; i < height; i ++ )
    {
        oneLineCorr( leftLine,
                     rightLine,
                     corr + i * width,
                     mem,
                     vertices,
                     width,
                     maxPixelDifference,
                     storage );
        leftLine += leftLineStep;
        rightLine += rightLineStep;
    } /* for( i = 0; i < height; i ++ ) */

    freeTempMem( &mem,
                 &vertices );

} /* allLinesCorr */

// This function produces morphing of two images into one image, which includes morphed
// image or depth map
//      _leftImage              - pointer to left image
//      _leftLineStep           - size of line on left image in bytes
//      _rightImage             - pointer to right image
//      _rightLineStep          - size of line on right image in bytes
//      _resultImage            - pointer to result morphed image
//      _resultLineStep         - size of line on result image in bytes
//      _corrArray              - pointer to array with correspondences
//      _numCorrArray           - pointer to array with numbers correspondeces on each line
//      width                   - width of images
//      height                  - height of images
//      alpha                   - position of virtual camera ( 0 corresponds to left image, 1 - to right one )
//      imageNeed               - defines your wishes. if you want to see normal morphed image you have to set
//                                this parameter to morphNormalImage ( this is default value ), else if you want
//                                to see depth map you have to set this parameter to morphDepthMap and set the
//                                next parameter ( maxPixelDifference ) to real value
//      maxPixelDifference      - maximum value of pixel difference on two images
void CCvGraphCutMorpher::Morph( unsigned char* _leftImage,
            int _leftLineStep,
            unsigned char* _rightImage,
            int _rightLineStep,
            unsigned char* _resultImage,
            int _resultLineStep,
            int* _corrArray,
            int width,
            int height,
            float alpha,
            morphImageType imageNeed,
            int maxDifference
          )
{
    unsigned char* leftArray    = _leftImage;
    unsigned char* middleArray  = _resultImage;
    unsigned char* rightArray   = _rightImage;
    int leftLineSize            = _leftLineStep;
    int middleLineSize          = _resultLineStep;
    int rightLineSize           = _rightLineStep;

    int lineNumber;
    unsigned char* leftTemp;
    unsigned char* middleTemp;
    unsigned char* rightTemp;
    int leftPixel;
    int prevLeftPixel;
    int middlePixel;
    int prevMiddlePixel;
    int rightPixel;
    int prevRightPixel;
    int leftPixel3;
    int middlePixel3;
    int rightPixel3;
    int i;
    int j;
    int tempIndex;
    int* result;
    int number;
    float alpha1        = 1.0f - alpha;
    
    for( lineNumber = 0; lineNumber < height; lineNumber ++ )
    {
        leftTemp    = leftArray + leftLineSize * lineNumber;
        middleTemp  = middleArray + middleLineSize * lineNumber;
        rightTemp   = rightArray + rightLineSize * lineNumber;
        memset( ( void* )middleTemp, 0, middleLineSize );

        result = _corrArray + width * lineNumber;
        number = width;
        
        prevLeftPixel   = 0;
        prevRightPixel  = prevLeftPixel + result[ 0 ];
        if( prevRightPixel >= width ) {
            prevRightPixel = width - 1;
        }
        else if ( prevRightPixel < 0 ) {
            prevRightPixel = 0;
        }
        prevMiddlePixel =
            (int)( prevLeftPixel * alpha1 + prevRightPixel * alpha );
        for( i = 0; i < number - 1; i ++ )
        {
            leftPixel       = i;
            rightPixel      = i + result[ i ];
            if( rightPixel >= width ) {
                rightPixel = width - 1;
            }
            else if( rightPixel < 0 ) {
                rightPixel = 0;
            }
            middlePixel     =
                (int)( leftPixel * alpha1 + rightPixel * alpha );
            leftPixel3      = leftPixel * 3;
            middlePixel3    = middlePixel * 3;
            rightPixel3     = rightPixel * 3;
            
            if( imageNeed == morphDepthMap ) {
                int t   = leftPixel - rightPixel + maxDifference;
                t       = t < 0 ? -t : t;
                t       = t * 255 / maxDifference / 2;
                middleTemp[ middlePixel3 ]      = ( unsigned char )t;
                middleTemp[ middlePixel3 + 1 ]  = ( unsigned char )t;
                middleTemp[ middlePixel3 + 2 ]  = ( unsigned char )t;
            } // if( imageNeed == morphDepthMap )
            else
            {
                middleTemp[ middlePixel3 ] =
                    (unsigned char)( leftTemp[ leftPixel3 ] * alpha1
                    + rightTemp[ rightPixel3 ] * alpha );
                middleTemp[ middlePixel3 + 1 ] =
                    (unsigned char)( leftTemp[ leftPixel3 + 1 ] * alpha1
                    + rightTemp[ rightPixel3 + 1 ] * alpha );
                middleTemp[ middlePixel3 + 2 ] =
                    (unsigned char)( leftTemp[ leftPixel3 + 2 ] * alpha1
                    + rightTemp[ rightPixel3 + 2 ] * alpha );

                if( middlePixel - prevMiddlePixel > 1 ) // occlusion
                {
                    if( leftPixel - prevLeftPixel > 1 )
                    {
                        int LenSrc  = leftPixel - prevLeftPixel - 2;
                        int LenDest = middlePixel - prevMiddlePixel - 1;
                        for( j = prevMiddlePixel + 1; j < middlePixel; j ++ )
                        {
                            tempIndex   = prevLeftPixel + 1 + LenSrc *
                                ( j - prevMiddlePixel - 1 ) / LenDest;
                            middleTemp[ j * 3 ]     =
                                leftTemp[ tempIndex * 3 ];
                            middleTemp[ j * 3 + 1 ] =
                                leftTemp[ tempIndex * 3 + 1 ];
                            middleTemp[ j * 3 + 2 ] =
                                leftTemp[ tempIndex * 3 + 2 ];
                        }
                    } // if( leftPixel - prevLeftPixel > 1 )
                    else
                    {
                        int LenSrc  = rightPixel - prevRightPixel - 2;
                        int LenDest = middlePixel - prevMiddlePixel - 1;
                        for( j = prevMiddlePixel + 1; j < middlePixel; j ++ )
                        {
                            tempIndex   = prevRightPixel + 1 + LenSrc *
                                ( j - prevMiddlePixel - 1 ) / LenDest;
                            middleTemp[ j * 3 ]     =
                                rightTemp[ tempIndex * 3 ];
                            middleTemp[ j * 3 + 1 ] =
                                rightTemp[ tempIndex * 3 + 1 ];
                            middleTemp[ j * 3 + 2 ] =
                                rightTemp[ tempIndex * 3 + 2 ];
                        }
                    } // if( leftPixel - prevLeftPixel > 1 ) else
                    
                } // if( middlePixel - prevMiddlePixel > 1 )

            } // if( imageNeed == morphDepthMap ) else

            if( middlePixel > prevMiddlePixel ) {
                if( leftPixel > prevLeftPixel )
                    prevLeftPixel   = leftPixel;
                if( rightPixel > prevRightPixel )
                    prevRightPixel  = rightPixel;
                prevMiddlePixel = middlePixel;
            }
        } // for( i = number - 1; i >= 0; i -- )
        
    } // for( lineNumber = 0; lineNumber < LeftImage -> m_Raster -> GetHeight() )

} // Morph

bool  CCvGraphCutMorpher::OnCalculateStereo()
{
    CvSize imageSizeLeft = GetImageSize( m_left_img ),
           imageSizeRight = GetImageSize( m_right_img );

    if( ( imageSizeLeft.width != imageSizeRight.width )
        || ( imageSizeLeft.height != imageSizeRight.height ) )
    {
        return false;
    }

    if( m_corr ) {
        free( m_corr );
        m_corr = NULL;
    }
    m_corr = ( int* ) malloc( m_left_img -> width
        * m_left_img -> height
        * sizeof( int ) );

    if( !m_storage ) {
        m_storage = cvCreateMemStorage( 0 );
        m_isStorageAllocated = true;
    }
    // Find correspondence for full image and store it to corr array
    allLinesCorr( ( unsigned char* )m_left_img -> imageData,
                  m_left_img -> widthStep,
                  ( unsigned char* )m_right_img -> imageData,
                  m_right_img -> widthStep,
                  m_left_img -> width,
                  m_left_img -> height,
                  m_corr,
                  m_maxPixelDifference,
                  m_storage );

    m_isStereoReady = true;

    return true;
}

bool  CCvGraphCutMorpher::OnCalculateVirtualImage()
{
    // Output image to ResultImage window
    Morph( ( unsigned char* )m_left_img -> imageData,
           m_left_img ->widthStep,
           ( unsigned char* )m_right_img -> imageData,
           m_right_img -> widthStep,
           ( unsigned char* )m_virtual_img -> imageData,
           m_virtual_img -> widthStep,
           m_corr,
           m_left_img -> width,
           m_left_img -> height,
           m_pan );

    m_isVirtualImageReady = true;

    return true;
}

bool  CCvGraphCutMorpher::OnCalculateDisparity()
{
    Morph( ( unsigned char* )m_left_img -> imageData,
           m_left_img ->widthStep,
           ( unsigned char* )m_right_img -> imageData,
           m_right_img -> widthStep,
           ( unsigned char* )m_disparity_img -> imageData,
           m_disparity_img -> widthStep,
           m_corr,
           m_left_img -> width,
           m_left_img -> height,
           m_pan,
           morphDepthMap,
           m_maxPixelDifference );

    return true;
}

bool  CCvGraphCutMorpher::OnCalculateDisparityImage()
{
    Morph( ( unsigned char* )m_left_img -> imageData,
           m_left_img ->widthStep,
           ( unsigned char* )m_right_img -> imageData,
           m_right_img -> widthStep,
           ( unsigned char* )m_disparity_img -> imageData,
           m_disparity_img -> widthStep,
           m_corr,
           m_left_img -> width,
           m_left_img -> height,
           m_pan,
           morphDepthMap,
           m_maxPixelDifference );

    return true;
}

CCvGraphCutMorpher::CCvGraphCutMorpher()
{
    m_maxPixelDifference = MAX_DIFFERENCE;
    m_corr = 0;
    m_isStereoReady = false;
    m_isVirtualImageReady = false;
    m_isDisparityReady = false;
    m_storage = NULL;
    m_isStorageAllocated = false;
}

#endif

/* End of file */
