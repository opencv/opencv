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

#include <float.h>
#include <limits.h>
#include <stdio.h>


#include "_cvutils.h"
#include "_cvwrap.h"

/*typedef struct CvCliqueFinder
{
    CvGraph* graph;
    int**    adj_matr;
    int N; //graph size

    // stacks, counters etc/
    int k; //stack size
    int* current_comp;
    int** All;

    int* ne;
    int* ce;
    int* fixp; //node with minimal disconnections
    int* nod;
    int* s; //for selected candidate
    int status;
    int best_score;

} CvCliqueFinder;
*/

#define GO 1
#define BACK 2
#define PEREBOR 3
#define NEXT PEREBOR
#define END 4


#define  CV_GET_ADJ_VTX( vertex, edge ) \
(                                       \
    assert(edge->vtx[0]==vertex||edge->vtx[1] == vertex ), \
    (edge->vtx[0] == vertex)?edge->vtx[1]:edge->vtx[0]     \
)


#define NUMBER( v ) ((v)->flags >> 1 )

void _MarkNodes( CvGraph* graph )
{
    //set number of vertices to their flags
    for( int i = 0; i < graph->total; i++ )
    {
        CvGraphVtx* ver = cvGetGraphVtx( graph, i );
        if( ver )
        {
            ver->flags = i<<1;
        }
    }
}

void _FillAdjMatrix( CvGraph* graph, int** connected, int reverse )
{
    //assume all vertices are marked
    for( int i = 0; i < graph->total; i++ )
    {
        for( int j = 0; j < graph->total; j++ )
        {
            connected[i][j] = 0|reverse;
        }
        //memset( connected[i], 0, sizeof(int)*graph->total );
        CvGraphVtx* ver = cvGetGraphVtx( graph, i );
        if( ver )
        {
            connected[i][i] = 1;
            for( CvGraphEdge* e = ver->first; e ; e = CV_NEXT_GRAPH_EDGE( e, ver ) )
            {
               CvGraphVtx* v = CV_GET_ADJ_VTX( ver, e );
               connected[i][NUMBER(v)] = 1^reverse;
            }
        }
    }
}


void cvStartFindCliques( CvGraph* graph, CvCliqueFinder* finder, int reverse, int weighted, int weighted_edges )
{
    int i;

    if (weighted)
    {
        finder->weighted = 1;
        finder->best_weight = 0;
        finder->vertex_weights = (float*)malloc( sizeof(float)*(graph->total+1));
        finder->cur_weight = (float*)malloc( sizeof(float)*(graph->total+1));
        finder->cand_weight = (float*)malloc( sizeof(float)*(graph->total+1));

        finder->cur_weight[0] = 0;
        finder->cand_weight[0] = 0;
        for( i = 0 ; i < graph->total; i++ )
        {
            CvGraphWeightedVtx* ver = (CvGraphWeightedVtx*)cvGetGraphVtx( graph, i );
            assert(ver);
            assert(ver->weight>=0);
            finder->vertex_weights[i] = ver->weight;
            finder->cand_weight[0] += ver->weight;
        }
    }
    else finder->weighted = 0;

    if (weighted_edges)
    {
        finder->weighted_edges = 1;
        //finder->best_weight = 0;
        finder->edge_weights = (float*)malloc( sizeof(float)*(graph->total)*(graph->total));
        //finder->cur_weight = (float*)malloc( sizeof(float)*(graph->total+1));
        //finder->cand_weight = (float*)malloc( sizeof(float)*(graph->total+1));

        //finder->cur_weight[0] = 0;
        //finder->cand_weight[0] = 0;
        memset( finder->edge_weights, 0, sizeof(float)*(graph->total)*(graph->total) );
        for( i = 0 ; i < graph->total; i++ )
        {
            CvGraphVtx* ver1 = cvGetGraphVtx( graph, i );
            if(!ver1) continue;
            for( int j = i ; j < graph->total; j++ )
            {
                CvGraphVtx* ver2 = cvGetGraphVtx( graph, j );
                if(!ver2) continue;
                CvGraphEdge* edge = cvFindGraphEdgeByPtr( graph, ver1, ver2 );
                if( edge )
                {
                    assert( ((CvGraphWeightedEdge*)edge)->weight >= 0 );
                    finder->edge_weights[ i * graph->total + j ] =
                    finder->edge_weights[ j * graph->total + i ] = ((CvGraphWeightedEdge*)edge)->weight;
                }
            }
        }
    }
    else finder->weighted_edges = 0;


    //int* Compsub; //current component (vertex stack)
    finder->k = 0; //counter of steps
    int N = finder->N = graph->total;
    finder->current_comp = new int[N];
    finder->All = new int*[N];
    for( i = 0 ; i < finder->N; i++ )
    {
        finder->All[i] = new int[N];
    }

    finder->ne = new int[N+1];
    finder->ce = new int[N+1];
    finder->fixp = new int[N+1]; //node with minimal disconnections
    finder->nod = new int[N+1];
    finder->s = new int[N+1]; //for selected candidate

    //form adj matrix
    finder->adj_matr = new int*[N]; //assume filled with 0
    for( i = 0 ; i < N; i++ )
    {
        finder->adj_matr[i] = new int[N];
    }

    //set number to vertices
    _MarkNodes( graph );
    _FillAdjMatrix( graph, finder->adj_matr, reverse );

    //init all arrays
    int k = finder->k = 0; //stack size
    memset( finder->All[k], 0, sizeof(int) * N );
    for( i = 0; i < N; i++ )  finder->All[k][i] = i;
    finder->ne[0] = 0;
    finder->ce[0] = N;

    finder->status = GO;
    finder->best_score = 0;

}

void cvEndFindCliques( CvCliqueFinder* finder )
{
    int i;

    //int* Compsub; //current component (vertex stack)
    delete finder->current_comp;
    for( i = 0 ; i < finder->N; i++ )
    {
        delete finder->All[i];
    }
    delete finder->All;

    delete finder->ne;
    delete finder->ce;
    delete finder->fixp; //node with minimal disconnections
    delete finder->nod;
    delete finder->s; //for selected candidate

    //delete adj matrix
    for( i = 0 ; i < finder->N; i++ )
    {
        delete finder->adj_matr[i];
    }
    delete finder->adj_matr;

    if(finder->weighted)
    {
        free(finder->vertex_weights);
        free(finder->cur_weight);
        free(finder->cand_weight);
    }
    if(finder->weighted_edges)
    {
        free(finder->edge_weights);
    }
}

int cvFindNextMaximalClique( CvCliqueFinder* finder )
{
    int**  connected = finder->adj_matr;
//    int N = finder->N; //graph size

    // stacks, counters etc/
    int k = finder->k; //stack size
    int* Compsub = finder->current_comp;
    int** All = finder->All;

    int* ne = finder->ne;
    int* ce = finder->ce;
    int* fixp = finder->fixp; //node with minimal disconnections
    int* nod = finder->nod;
    int* s = finder->s; //for selected candidate

    //START
    while( k >= 0)
    {
        int* old = All[k];
        switch(finder->status)
        {
        case GO://Forward step
        /* we have all sets and will choose fixed point */
            {
                //check potential size of clique
                if( (!finder->weighted) && (k + ce[k] - ne[k] < finder->best_score) )
                {
                    finder->status  = BACK;
                    break;
                }
                //check potential weight
                if( finder->weighted && !finder->weighted_edges &&
                    finder->cur_weight[k] + finder->cand_weight[k] < finder->best_weight )
                {
                    finder->status  = BACK;
                    break;
                }

                int minnod = ce[k];
                nod[k] = 0;

                //for every vertex of All determine counter value and choose minimum
                for( int i = 0; i < ce[k] && minnod != 0; i++)
                {
                    int p = old[i]; //current point
                    int count = 0;  //counter
                    int pos = 0;

                    /* Count disconnections with candidates */
                    for (int j = ne[k]; j < ce[k] && (count < minnod); j++)
                    {
                        if ( !connected[p][old[j]] )
                        {
                            count++;
                            /* Save position of potential candidate */
                            pos = j;
                        }
                    }

                    /* Test new minimum */
                    if (count < minnod)
                    {
                        fixp[k] = p;     //set current point as fixed
                        minnod = count;  //new value for minnod
                        if (i < ne[k])      //if current fixed point belongs to 'not'
                        {
                            s[k] = pos;     //s - selected candidate
                        }
                        else
                        {
                            s[k] = i;        //selected candidate is fixed point itself
                            /* preincr */
                            nod[k] = 1;      //nod is aux variable, 1 means fixp == s
                        }
                    }
                }//for

                nod[k] = minnod + nod[k];
                finder->status = NEXT;//go to backtrackcycle
            }
            break;
        case NEXT:
            //here we will look for candidate to translate into not
            //s[k] now contains index of choosen candidate
            {
                int* new_ = All[k+1];
                if( nod[k] != 0 )
                {
                    //swap selected and first candidate
                    int i, p = old[s[k]];
                    old[s[k]] = old[ne[k]];
                    int sel = old[ne[k]] = p;

                    int newne = 0;
                    //fill new set 'not'
                    for ( i = 0; i < ne[k]; i++)
                    {
                        if (connected[sel][old[i]])
                        {
                            new_[newne] = old[i];
                            newne++;

                        }
                    }
                    //fill new set 'candidate'
                    int newce = newne;
                    i++;//skip selected candidate

                    float candweight = 0;

                    for (; i < ce[k]; i++)
                    {
                        if (connected[sel][old[i]])
                        {
                            new_[newce] = old[i];

                            if( finder->weighted )
                                candweight += finder->vertex_weights[old[i]];

                            newce++;
                        }
                    }

                    nod[k]--;

                    //add selected to stack
                    Compsub[k] = sel;

                    k++;
                    assert( k <= finder->N );
                    if( finder->weighted )
                    {
                        //update weights of current clique and candidates
                        finder->cur_weight[k] = finder->cur_weight[k-1] + finder->vertex_weights[sel];
                        finder->cand_weight[k] = candweight;
                    }
                    if( finder->weighted_edges )
                    {
                        //update total weight by edge weights
                        float added = 0;
                        for( int ind = 0; ind < k-1; ind++ )
                        {
                            added += finder->edge_weights[ Compsub[ind] * finder->N + sel ];
                        }
                        finder->cur_weight[k] += added;
                    }

                    //check if 'not' and 'cand' are both empty
                    if( newce == 0 )
                    {
                        finder->best_score = MAX(finder->best_score, k );

                        if( finder->weighted )
                            finder->best_weight = MAX( finder->best_weight, finder->cur_weight[k] );
                        /*FILE* file  = fopen("cliques.txt", "a" );

                        for (int t=0; t<k; t++)
                        {
                          fprintf(file, "%d ", Compsub[t]);
                        }
                        fprintf(file, "\n");

                        fclose(file);
                        */

                        //output new clique//************************
                        finder->status = BACK;
                        finder->k = k;

                        return CLIQUE_FOUND;

                    }
                    else //check nonempty set of candidates
                    if( newne < newce )
                    {
                        //go further
                        ne[k] = newne;
                        ce[k] = newce;
                        finder->status  = GO;
                        break;
                    }

                }
                else
                    finder->status  = BACK;

            }
            break;

        case BACK:
            {
                //decrease stack
                k--;
                old = All[k];
                if( k < 0 ) break;

                //add to not
                ne[k]++;

                if( nod[k] > 0 )
                {
                    //select next candidate
                    for( s[k] = ne[k]; s[k] < ce[k]; s[k]++ )
                    {
                        if( !connected[fixp[k]][old[s[k]]])
                            break;
                    }
                    assert( s[k] < ce[k] );
                    finder->status = NEXT;
                }
                else
                    finder->status = BACK;

            }
            break;
        case END: assert(0);

        }
    }//end while

    finder->status = END;
    return CLIQUE_END;
}




void cvBronKerbosch( CvGraph* graph )
{
    int* Compsub; //current component (vertex stack)
    int k = 0; //counter of steps
    int N = graph->total;
    int i;
    Compsub = new int[N];
    int** All = new int*[N];
    for( i = 0 ; i < N; i++ )
    {
        All[i] = new int[N];
    }

    int* ne = new int[N];
    int* ce = new int[N];
    int* fixp = new int[N]; //node with minimal disconnections
    int* nod = new int[N];
    int* s = new int[N]; //for selected candidate

    //form adj matrix
    int** connected = new int*[N]; //assume filled with 0
    for( i = 0 ; i < N; i++ )
    {
        connected[i] = new int[N];
    }



    //set number to vertices
    _MarkNodes( graph );
    _FillAdjMatrix( graph, connected, 0 );

    //init all arrays
    k = 0; //stack size
    memset( All[k], 0, sizeof(int) * N );
    for( i = 0; i < N; i++ )  All[k][i] = i;
    ne[0] = 0;
    ce[0] = N;

    int status = GO;
    int best_score = 0;

    //START
    while( k >= 0)
    {
        int* old = All[k];
        switch(status)
        {
        case GO://Forward step
        /* we have all sets and will choose fixed point */
            {

                if( k + ce[k] - ne[k] < best_score )
                {
                    status  = BACK;
                    break;
                }

                int minnod = ce[k];
                nod[k] = 0;

                //for every vertex of All determine counter value and choose minimum
                for( int i = 0; i < ce[k] && minnod != 0; i++)
                {
                    int p = old[i]; //current point
                    int count = 0;  //counter
                    int pos = 0;

                    /* Count disconnections with candidates */
                    for (int j = ne[k]; j < ce[k] && (count < minnod); j++)
                    {
                        if ( !connected[p][old[j]] )
                        {
                            count++;
                            /* Save position of potential candidate */
                            pos = j;
                        }
                    }

                    /* Test new minimum */
                    if (count < minnod)
                    {
                        fixp[k] = p;     //set current point as fixed
                        minnod = count;  //new value for minnod
                        if (i < ne[k])      //if current fixed point belongs to 'not'
                        {
                            s[k] = pos;     //s - selected candidate
                        }
                        else
                        {
                            s[k] = i;        //selected candidate is fixed point itself
                            /* preincr */
                            nod[k] = 1;      //nod is aux variable, 1 means fixp == s
                        }
                    }
                }//for

                nod[k] = minnod + nod[k];
                status = NEXT;//go to backtrackcycle
            }
            break;
        case NEXT:
            //here we will look for candidate to translate into not
            //s[k] now contains index of choosen candidate
            {
                int* new_ = All[k+1];
                if( nod[k] != 0 )
                {
                    //swap selected and first candidate
                    int p = old[s[k]];
                    old[s[k]] = old[ne[k]];
                    int sel = old[ne[k]] = p;

                    int newne = 0;
                    //fill new set 'not'
                    for ( i = 0; i < ne[k]; i++)
                    {
                        if (connected[sel][old[i]])
                        {
                            new_[newne] = old[i];
                            newne++;

                        }
                    }
                    //fill new set 'candidate'
                    int newce = newne;
                    i++;//skip selected candidate
                    for (; i < ce[k]; i++)
                    {
                        if (connected[sel][old[i]])
                        {
                            new_[newce] = old[i];
                            newce++;
                        }
                    }

                    nod[k]--;

                    //add selected to stack
                    Compsub[k] = sel;
                    k++;

                    //check if 'not' and 'cand' are both empty
                    if( newce == 0 )
                    {
                        best_score = MAX(best_score, k );

                        FILE* file  = fopen("cliques.txt", "a" );

                        for (int t=0; t<k; t++)
                        {
                          fprintf(file, "%d ", Compsub[t]);
                        }
                        fprintf(file, "\n");

                        fclose(file);

                        /*for( int t = 0; t < k; t++ )
                        {
                            printf("%d ", Compsub[t] );
                        }
                        printf("\n"); */

                        //printf("found %d\n", k);

                        //output new clique//************************
                        status = BACK;
                    }
                    else //check nonempty set of candidates
                    if( newne < newce )
                    {
                        //go further
                        ne[k] = newne;
                        ce[k] = newce;
                        status  = GO;
                        break;
                    }

                }
                else
                    status  = BACK;

            }
            break;

        case BACK:
            {
                //decrease stack
                k--;
                old = All[k];
                if( k < 0 ) break;

                //add to not
                ne[k]++;

                if( nod[k] > 0 )
                {
                    //select next candidate
                    for( s[k] = ne[k]; s[k] < ce[k]; s[k]++ )
                    {
                        if( !connected[fixp[k]][old[s[k]]])
                            break;
                    }
                    assert( s[k] < ce[k] );
                    status = NEXT;
                }
                else
                    status = BACK;

            }
            break;


        }
    }//end while

}//end cvBronKerbosch

#endif
