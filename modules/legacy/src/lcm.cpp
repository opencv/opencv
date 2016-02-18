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

/* Hybrid linear-contour model reconstruction */
#include "precomp.hpp"

#define CV_IMPL CV_EXTERN_C

const float LCM_CONST_ZERO = 1e-6f;

/****************************************************************************************\
*                                    Auxiliary struct definitions                                 *
\****************************************************************************************/
typedef struct CvLCM
{
    CvGraph* Graph;
    CvVoronoiDiagram2D* VoronoiDiagram;
    CvMemStorage* ContourStorage;
    CvMemStorage* EdgeStorage;
    float maxWidth;
} CvLCM;

typedef struct CvLCMComplexNodeData
{
    CvVoronoiNode2D edge_node;
    CvPoint2D32f site_first_pt;
    CvPoint2D32f site_last_pt;
    CvVoronoiSite2D* site_first;
    CvVoronoiSite2D* site_last;
    CvVoronoiEdge2D* edge;
} CvLCMComplexNodeData;

typedef struct CvLCMData
{
    CvVoronoiNode2D* pnode;
    CvVoronoiSite2D* psite;
    CvVoronoiEdge2D* pedge;
} CvLCMData;


/****************************************************************************************\
*                                    Function definitions                                *
\****************************************************************************************/

#define _CV_READ_SEQ_ELEM( elem, reader, type )                       \
{                                                              \
    assert( (reader).seq->elem_size == sizeof(*elem));         \
    elem = (type)(reader).ptr;                                 \
    CV_NEXT_SEQ_ELEM( sizeof(*elem), reader )                  \
}

#define _CV_IS_SITE_REFLEX( SITE )  ((SITE) ->node[0] == (SITE) ->node[1])
#define _CV_IS_EDGE_REFLEX( EDGE )  (( (EDGE)->site[0]->node[0] == (EDGE)->site[0]->node[0] ) || \
                                      ( (EDGE)->site[1]->node[0] == (EDGE)->site[1]->node[0] ) )

#define _CV_INITIALIZE_CVLCMDATA(STRUCT,SITE,EDGE,NODE)\
{ (STRUCT)->psite = SITE ; (STRUCT)->pedge = EDGE; (STRUCT)->pnode = NODE;}
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructLCM
//    Purpose: Function constructs hybrid model
//    Context:
//    Parameters:
//      LCM : in&out.
//    Returns: 1, if hybrid model was successfully constructed
//             0, if some error occurs
//F*/
CV_IMPL
int _cvConstructLCM(CvLCM* LCM);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructLCMComplexNode
//    Purpose: Function constructs Complex Node (node, which consists of
//             two points and more) of hybrid model
//    Context:
//    Parameters:
//      pLCM : in&out.
//      pLCMEdge: in, input edge of hybrid model
//      pLCMInputData: in, input parameters
//    Returns: pointer to constructed node
//F*/
CV_IMPL
CvLCMNode* _cvConstructLCMComplexNode(CvLCM* pLCM,
                                      CvLCMEdge* pLCMEdge,
                                      CvLCMData* pLCMInputData);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructLCMSimpleNode
//    Purpose: Function constructs Simple Node (node, which consists of
//             one point) of hybrid model
//    Context:
//    Parameters:
//      pLCM : in&out.
//      pLCMEdge: in, input edge of hybrid model
//      pLCMInputData: in, input parameters
//    Returns: pointer to constructed node
//F*/
CV_IMPL
CvLCMNode* _cvConstructLCMSimpleNode(CvLCM* pLCM,
                                    CvLCMEdge* pLCMEdge,
                                    CvLCMData* pLCMInputData);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructLCMSimpleNode
//    Purpose: Function constructs Edge of hybrid model
//    Context:
//    Parameters:
//      pLCM : in&out.
//      pLCMInputData: in, input parameters
//    Returns: pointer to constructed edge
//F*/
CV_IMPL
CvLCMEdge* _cvConstructLCMEdge(CvLCM* pLCM,
                               CvLCMData* pLCMInputData);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvTreatExeptionalCase
//    Purpose: Function treats triangles and regular polygons
//    Context:
//    Parameters:
//      pLCM : in, information about graph
//      pLCMInputData: in, input parameters
//    Returns: pointer to graph node
//F*/
CV_IMPL
CvLCMNode* _cvTreatExeptionalCase(CvLCM* pLCM,
                                  CvLCMData* pLCMInputData);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvNodeMultyplicity
//    Purpose: Function seeks all non-boundary edges incident to
//              given node and correspondent incident sites
//    Context:
//    Parameters:
//      pEdge : in, original edge
//      pNode : in, given node
//      LinkedEdges : out, matrix of incident edges
//      LinkedSites : out, matrix of incident sites
//      pSite: in, original site (pNode must be the begin point of pEdge
//              for this pSite, this property hold out far all edges)
//    Returns: number of incident edges (must be less than 10)
//F*/
CV_IMPL
int _cvNodeMultyplicity(CvVoronoiSite2D* pSite,
                        CvVoronoiEdge2D* pEdge,
                        CvVoronoiNode2D* pNode,
                        CvVoronoiEdge2D** LinkedEdges,
                        CvVoronoiSite2D** LinkedSites);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvCreateLCMNode
//    Purpose: Function create graph node
//    Context:
//    Parameters:
//      pLCM : in, information about graph
//    Returns: pointer to graph node
//F*/
CV_IMPL
CvLCMNode* _cvCreateLCMNode(CvLCM* pLCM);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvCreateLCMEdge
//    Purpose: Function create graph edge
//    Context:
//    Parameters:
//      pLCM : in, information about graph
//    Returns: pointer to graph edge
//F*/
CV_IMPL
CvLCMEdge* _cvCreateLCMEdge(CvLCM* pLCM);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvCreateLCMNode
//    Purpose: Function establishs the connection between node and ege
//    Context:
//    Parameters:
//      LCMNode : in, graph node
//      LCMEdge : in, graph edge
//      LCMEdge_prev : in&out, previous edge, connected with given node
//      index: in,
//      i    : =0, if node is initial for edge
//             =1, if node  is terminal for edge
//    Returns:
//F*/
CV_IMPL
void _cvAttachLCMEdgeToLCMNode(CvLCMNode* LCMNode,
                               CvLCMEdge* LCMEdge,
                               CvLCMEdge* &LCMEdge_prev,
                               int index,
                               int i);
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvProjectionPointToSegment
//    Purpose: Function computes the ortogonal projection of PointO to
//             to segment[PointA, PointB]
//    Context:
//    Parameters:
//      PointO, PointA,PointB: in, given points
//      PrPoint : out, projection
//      dist : distance from PointO to PrPoint
//    Returns:
//F*/
CV_IMPL
void _cvProjectionPointToSegment(CvPoint2D32f* PointO,
                                 CvPoint2D32f* PointA,
                                 CvPoint2D32f* PointB,
                                 CvPoint2D32f* PrPoint,
                                 float* dist);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvPrepareData
//    Purpose: Function fills up the struct CvLCMComplexNodeData
//    Context:
//    Parameters:
//      pLCMData : in
//      pLCMCCNData : out
//    Returns:
//F*/
CV_IMPL
void _cvPrepareData(CvLCMComplexNodeData* pLCMCCNData,
                    CvLCMData* pLCMData);

/****************************************************************************************\
*                                    Function realization                               *
\****************************************************************************************/

CV_IMPL CvGraph* cvLinearContorModelFromVoronoiDiagram(CvVoronoiDiagram2D* VoronoiDiagram,
                                                       float maxWidth)
{
    CvMemStorage* LCMstorage;
    CvSet* SiteSet;
    CvLCM LCM = {NULL, VoronoiDiagram,NULL,NULL,maxWidth};

    CV_FUNCNAME( "cvLinearContorModelFromVoronoiDiagram" );
     __BEGIN__;

    if( !VoronoiDiagram )
        CV_ERROR( CV_StsBadArg,"Voronoi Diagram is not defined" );
    if( maxWidth < 0 )
        CV_ERROR( CV_StsBadArg,"Treshold parameter must be non negative" );

    for(SiteSet = VoronoiDiagram->sites;
        SiteSet != NULL;
        SiteSet = (CvSet*)SiteSet->h_next)
        {
            if(SiteSet->v_next)
                CV_ERROR( CV_StsBadArg,"Can't operate with multiconnected domains" );
            if(SiteSet->total > 70000)
                CV_ERROR( CV_StsBadArg,"Can't operate with large domains" );
        }


    LCMstorage = cvCreateMemStorage(0);
    LCM.EdgeStorage = cvCreateChildMemStorage(LCMstorage);
    LCM.ContourStorage = cvCreateChildMemStorage(LCMstorage);
    LCM.Graph = cvCreateGraph(CV_SEQ_KIND_GRAPH|CV_GRAPH_FLAG_ORIENTED,
                              sizeof(CvGraph),
                              sizeof(CvLCMNode),
                              sizeof(CvLCMEdge),
                              LCMstorage);
    if(!_cvConstructLCM(&LCM))
        cvReleaseLinearContorModelStorage(&LCM.Graph);


    __END__;
    return LCM.Graph;
}//end of cvLinearContorModelFromVoronoiDiagram

CV_IMPL int cvReleaseLinearContorModelStorage(CvGraph** Graph)
{
    CvSeq* LCMNodeSeq, *LCMEdgeSeq;
    CvLCMNode* pLCMNode;
    CvLCMEdge* pLCMEdge;

    /*CV_FUNCNAME( "cvReleaseLinearContorModelStorage" );*/
     __BEGIN__;

    if(!Graph || !(*Graph))
        return 0;

    LCMNodeSeq = (CvSeq*)(*Graph);
    LCMEdgeSeq = (CvSeq*)(*Graph)->edges;
    if(LCMNodeSeq->total > 0)
    {
        pLCMNode = (CvLCMNode*)cvGetSeqElem(LCMNodeSeq,0);
        if(pLCMNode->contour->storage)
            cvReleaseMemStorage(&pLCMNode->contour->storage);
    }
    if(LCMEdgeSeq->total > 0)
    {
        pLCMEdge = (CvLCMEdge*)cvGetSeqElem(LCMEdgeSeq,0);
        if(pLCMEdge->chain->storage)
            cvReleaseMemStorage(&pLCMEdge->chain->storage);
    }
    if((*Graph)->storage)
        cvReleaseMemStorage(&(*Graph)->storage);
    *Graph = NULL;


    __END__;
    return 1;
}//end of cvReleaseLinearContorModelStorage

int _cvConstructLCM(CvLCM* LCM)
{
    CvVoronoiSite2D* pSite = 0;
    CvVoronoiEdge2D* pEdge = 0, *pEdge1;
    CvVoronoiNode2D* pNode, *pNode1;

    CvVoronoiEdge2D* LinkedEdges[10];
    CvVoronoiSite2D* LinkedSites[10];

    CvSeqReader reader;
    CvLCMData LCMdata;
    int i;

    for(CvSet* SiteSet = LCM->VoronoiDiagram->sites;
        SiteSet != NULL;
        SiteSet = (CvSet*)SiteSet->h_next)
    {
        cvStartReadSeq((CvSeq*)SiteSet, &reader);
        for(i = 0; i < SiteSet->total; i++)
        {
            _CV_READ_SEQ_ELEM(pSite,reader,CvVoronoiSite2D*);
            if(pSite->node[0] == pSite->node[1])
                continue;
            pEdge = CV_LAST_VORONOIEDGE2D(pSite);
            pNode = CV_VORONOIEDGE2D_BEGINNODE(pEdge,pSite);
            if(pNode->radius > LCM->maxWidth)
                goto PREPARECOMPLEXNODE;

            pEdge1 = CV_PREV_VORONOIEDGE2D(pEdge,pSite);
            pNode1 = CV_VORONOIEDGE2D_BEGINNODE(pEdge1,pSite);
            if(pNode1->radius > LCM->maxWidth)
                goto PREPARECOMPLEXNODE;
            if(pNode1->radius == 0)
                continue;
            if(_cvNodeMultyplicity(pSite, pEdge,pNode,LinkedEdges,LinkedSites) == 1)
                goto PREPARESIMPLENODE;
        }
// treate triangle or regular polygon
        _CV_INITIALIZE_CVLCMDATA(&LCMdata,pSite,pEdge,CV_VORONOIEDGE2D_ENDNODE(pEdge,pSite));
        if(!_cvTreatExeptionalCase(LCM,&LCMdata))
            return 0;
        continue;

PREPARECOMPLEXNODE:
        _CV_INITIALIZE_CVLCMDATA(&LCMdata,pSite,pEdge,CV_VORONOIEDGE2D_ENDNODE(pEdge,pSite));
        if(!_cvConstructLCMComplexNode(LCM,NULL,&LCMdata))
            return 0;
        continue;

PREPARESIMPLENODE:
        _CV_INITIALIZE_CVLCMDATA(&LCMdata,pSite,pEdge,CV_VORONOIEDGE2D_ENDNODE(pEdge,pSite));
        if(!_cvConstructLCMSimpleNode(LCM,NULL,&LCMdata))
            return 0;
        continue;
    }
    return 1;
}//end of _cvConstructLCM

CvLCMNode* _cvConstructLCMComplexNode(CvLCM* pLCM,
                                      CvLCMEdge* pLCMEdge,
                                      CvLCMData* pLCMInputData)
{
    CvLCMNode* pLCMNode;
    CvLCMEdge* pLCMEdge_prev = NULL;
    CvSeqWriter writer;
    CvVoronoiSite2D* pSite, *pSite_first, *pSite_last;
    CvVoronoiEdge2D* pEdge, *pEdge_stop;
    CvVoronoiNode2D* pNode0, *pNode1;
    CvLCMData LCMOutputData;
    CvLCMComplexNodeData LCMCCNData;
    int index = 0;

    _cvPrepareData(&LCMCCNData,pLCMInputData);

    pLCMNode = _cvCreateLCMNode(pLCM);
    _cvAttachLCMEdgeToLCMNode(pLCMNode,pLCMEdge,pLCMEdge_prev,1,1);
    cvStartAppendToSeq((CvSeq*)pLCMNode->contour,&writer);
    CV_WRITE_SEQ_ELEM(LCMCCNData.site_last_pt, writer);
    index++;

    if(pLCMEdge)
    {
        CV_WRITE_SEQ_ELEM(LCMCCNData.edge_node.pt, writer );
        CV_WRITE_SEQ_ELEM(LCMCCNData.site_first_pt, writer );
        index+=2;
    }

    pSite_first = LCMCCNData.site_first;
    pSite_last = LCMCCNData.site_last;
    pEdge = LCMCCNData.edge;

    for(pSite = pSite_first;
        pSite != pSite_last;
        pSite = CV_NEXT_VORONOISITE2D(pSite),
        pEdge = CV_PREV_VORONOIEDGE2D(CV_LAST_VORONOIEDGE2D(pSite),pSite))
    {
        pEdge_stop = CV_FIRST_VORONOIEDGE2D(pSite);
        for(;pEdge && pEdge != pEdge_stop;
             pEdge = CV_PREV_VORONOIEDGE2D(pEdge,pSite))
        {
            pNode0 = CV_VORONOIEDGE2D_BEGINNODE(pEdge,pSite);
            pNode1 = CV_VORONOIEDGE2D_ENDNODE(pEdge,pSite);
            if(pNode0->radius <= pLCM->maxWidth && pNode1->radius <= pLCM->maxWidth)
            {
                _CV_INITIALIZE_CVLCMDATA(&LCMOutputData,pSite,pEdge,pNode1);
                _cvPrepareData(&LCMCCNData,&LCMOutputData);
                CV_WRITE_SEQ_ELEM(LCMCCNData.site_first_pt, writer);
                CV_WRITE_SEQ_ELEM(LCMCCNData.edge_node.pt, writer );
                index+=2;
                pLCMEdge = _cvConstructLCMEdge(pLCM,&LCMOutputData);
                _cvAttachLCMEdgeToLCMNode(pLCMNode,pLCMEdge,pLCMEdge_prev,index - 1,0);
                CV_WRITE_SEQ_ELEM(LCMCCNData.site_last_pt, writer);
                index++;

                pSite = CV_TWIN_VORONOISITE2D(pSite,pEdge);
                pEdge_stop = CV_FIRST_VORONOIEDGE2D(pSite);
                if(pSite == pSite_last)
                    break;
            }
        }
        if(pSite == pSite_last)
            break;

        CV_WRITE_SEQ_ELEM(pSite->node[1]->pt, writer);
        index++;
    }

    if(pLCMEdge_prev)
        pLCMEdge_prev->next[(pLCMEdge_prev == (CvLCMEdge*)pLCMNode->first)] = pLCMNode->first;
    cvEndWriteSeq(&writer);
    return pLCMNode;
}//end of _cvConstructLCMComplexNode

CvLCMNode* _cvConstructLCMSimpleNode(CvLCM* pLCM,
                                     CvLCMEdge* pLCMEdge,
                                     CvLCMData* pLCMInputData)
{
    CvVoronoiEdge2D* pEdge = pLCMInputData->pedge;
    CvVoronoiSite2D* pSite = pLCMInputData->psite;
    CvVoronoiNode2D* pNode = CV_VORONOIEDGE2D_BEGINNODE(pEdge,pSite);

    CvVoronoiEdge2D* LinkedEdges[10];
    CvVoronoiSite2D* LinkedSites[10];
    int multyplicity = _cvNodeMultyplicity(pSite,pEdge,pNode,LinkedEdges,LinkedSites);
    if(multyplicity == 2)
    {
        pLCMInputData->pedge = LinkedEdges[1];
        pLCMInputData->psite = CV_TWIN_VORONOISITE2D(LinkedSites[1],LinkedEdges[1]);
        return NULL;
    }

    CvLCMEdge* pLCMEdge_prev = NULL;
    CvLCMNode* pLCMNode;
    CvLCMData LCMOutputData;

    pLCMNode = _cvCreateLCMNode(pLCM);
    cvSeqPush((CvSeq*)pLCMNode->contour,&pNode->pt);
    _cvAttachLCMEdgeToLCMNode(pLCMNode,pLCMEdge,pLCMEdge_prev,0,1);

    for(int i = (int)(pLCMEdge != NULL);i < multyplicity; i++)
    {
        pEdge = LinkedEdges[i];
        pSite = LinkedSites[i];
        _CV_INITIALIZE_CVLCMDATA(&LCMOutputData,CV_TWIN_VORONOISITE2D(pSite,pEdge),pEdge,pNode);
        pLCMEdge = _cvConstructLCMEdge(pLCM,&LCMOutputData);
        _cvAttachLCMEdgeToLCMNode(pLCMNode,pLCMEdge,pLCMEdge_prev,0,0);
    }
    pLCMEdge_prev->next[(pLCMEdge_prev == (CvLCMEdge*)pLCMNode->first)] = pLCMNode->first;
    return pLCMNode;
}//end of _cvConstructLCMSimpleNode

CvLCMEdge* _cvConstructLCMEdge(CvLCM* pLCM,
                               CvLCMData* pLCMInputData)
{
    CvVoronoiEdge2D* pEdge = pLCMInputData->pedge;
    CvVoronoiSite2D* pSite = pLCMInputData->psite;
    float width = 0;

    CvLCMData LCMData;
    CvVoronoiNode2D* pNode0,*pNode1;

    CvLCMEdge* pLCMEdge = _cvCreateLCMEdge(pLCM);

    CvSeqWriter writer;
    cvStartAppendToSeq(pLCMEdge->chain,&writer );

    pNode0 = pNode1 = pLCMInputData->pnode;
    CV_WRITE_SEQ_ELEM(pNode0->pt, writer);
    width += pNode0->radius;

    for(int counter = 0;
            counter < pLCM->VoronoiDiagram->edges->total;
            counter++)
    {
        pNode1 = CV_VORONOIEDGE2D_BEGINNODE(pEdge,pSite);
        if(pNode1->radius >= pLCM->maxWidth)
            goto CREATECOMPLEXNODE;

        CV_WRITE_SEQ_ELEM(pNode1->pt,writer);
        width += pNode1->radius;
        _CV_INITIALIZE_CVLCMDATA(&LCMData,pSite,pEdge,pNode1);
        if(_cvConstructLCMSimpleNode(pLCM,pLCMEdge,&LCMData))
            goto LCMEDGEEXIT;

        pEdge = LCMData.pedge; pSite = LCMData.psite;
        pNode0 = pNode1;
    }
    return NULL;

CREATECOMPLEXNODE:
    _CV_INITIALIZE_CVLCMDATA(&LCMData,pSite,pEdge,pNode0);
    CV_WRITE_SEQ_ELEM(LCMData.pnode->pt,writer);
    width += LCMData.pnode->radius;
    _cvConstructLCMComplexNode(pLCM,pLCMEdge,&LCMData);

LCMEDGEEXIT:
    cvEndWriteSeq(&writer);
    pLCMEdge->width = width/pLCMEdge->chain->total;
    return pLCMEdge;
}//end of _cvConstructLCMEdge

CvLCMNode* _cvTreatExeptionalCase(CvLCM* pLCM,
                                  CvLCMData* pLCMInputData)
{
    CvVoronoiEdge2D* pEdge = pLCMInputData->pedge;
    CvVoronoiSite2D* pSite = pLCMInputData->psite;
    CvVoronoiNode2D* pNode = CV_VORONOIEDGE2D_BEGINNODE(pEdge,pSite);
    CvLCMNode* pLCMNode = _cvCreateLCMNode(pLCM);
    cvSeqPush((CvSeq*)pLCMNode->contour,&pNode->pt);
    return pLCMNode;
}//end of _cvConstructLCMEdge

CV_INLINE
CvLCMNode* _cvCreateLCMNode(CvLCM* pLCM)
{
    CvLCMNode* pLCMNode;
    cvSetAdd((CvSet*)pLCM->Graph, NULL, (CvSetElem**)&pLCMNode );
    pLCMNode->contour = (CvContour*)cvCreateSeq(0, sizeof(CvContour),
                                                   sizeof(CvPoint2D32f),pLCM->ContourStorage);
    pLCMNode->first = NULL;
    return pLCMNode;
}//end of _cvCreateLCMNode

CV_INLINE
CvLCMEdge* _cvCreateLCMEdge(CvLCM* pLCM)
{
    CvLCMEdge* pLCMEdge;
    cvSetAdd( (CvSet*)(pLCM->Graph->edges), 0, (CvSetElem**)&pLCMEdge );
    pLCMEdge->chain = cvCreateSeq(0, sizeof(CvSeq),sizeof(CvPoint2D32f),pLCM->EdgeStorage);
    pLCMEdge->next[0] = pLCMEdge->next[1] = NULL;
    pLCMEdge->vtx[0] =  pLCMEdge->vtx[1] = NULL;
    pLCMEdge->index1 =  pLCMEdge->index2 = -1;
    return pLCMEdge;
}//end of _cvCreateLCMEdge

CV_INLINE
void _cvAttachLCMEdgeToLCMNode(CvLCMNode* LCMNode,
                               CvLCMEdge* LCMEdge,
                               CvLCMEdge* &LCMEdge_prev,
                               int index,
                               int i)
{
    if(!LCMEdge)
        return;
    if(i==0)
        LCMEdge->index1 = index;
    else
        LCMEdge->index2 = index;

    LCMEdge->vtx[i] = (CvGraphVtx*)LCMNode;
    if(!LCMEdge_prev)
        LCMNode->first = (CvGraphEdge*)LCMEdge;
    else
//      LCMEdge_prev->next[(LCMEdge_prev == (CvLCMEdge*)LCMNode->first)] = (CvGraphEdge*)LCMEdge;
        LCMEdge_prev->next[(LCMEdge_prev->vtx[0] != (CvGraphVtx*)LCMNode)] = (CvGraphEdge*)LCMEdge;

    LCMEdge->next[i] = LCMNode->first;
    LCMEdge_prev = LCMEdge;
}//end of _cvAttachLCMEdgeToLCMNode


int _cvNodeMultyplicity(CvVoronoiSite2D* pSite,
                        CvVoronoiEdge2D* pEdge,
                        CvVoronoiNode2D* pNode,
                        CvVoronoiEdge2D** LinkedEdges,
                        CvVoronoiSite2D** LinkedSites)
{
    if(!pNode->radius)
        return -1;
    assert(pNode == CV_VORONOIEDGE2D_BEGINNODE(pEdge,pSite));

    int multyplicity = 0;
    CvVoronoiEdge2D* pEdge_cur = pEdge;
    do
    {
        if(pEdge_cur->node[0]->radius && pEdge_cur->node[1]->radius)
        {
            LinkedEdges[multyplicity] = pEdge_cur;
            LinkedSites[multyplicity] = pSite;
            multyplicity++;
        }
        pEdge_cur = CV_PREV_VORONOIEDGE2D(pEdge_cur,pSite);
        pSite = CV_TWIN_VORONOISITE2D(pSite,pEdge_cur);
    }while(pEdge_cur != pEdge);
    return multyplicity;
}//end of _cvNodeMultyplicity


CV_INLINE
void _cvPrepareData(CvLCMComplexNodeData* pLCMCCNData,
                    CvLCMData* pLCMData)
{
    pLCMCCNData->site_first = pLCMData->psite;
    pLCMCCNData->site_last = CV_TWIN_VORONOISITE2D(pLCMData->psite,pLCMData->pedge);
    if(pLCMData->pedge == CV_LAST_VORONOIEDGE2D(pLCMData->psite))
    {
        pLCMCCNData->edge = CV_PREV_VORONOIEDGE2D(pLCMData->pedge,pLCMData->psite);
        pLCMCCNData->edge_node = *pLCMData->pnode;
        pLCMCCNData->site_first_pt = pLCMData->psite->node[0]->pt;
        pLCMCCNData->site_last_pt = pLCMData->psite->node[0]->pt;
    }
    else
    {
        pLCMCCNData->edge = pLCMData->pedge;
        pLCMCCNData->edge_node = *pLCMData->pnode;
        _cvProjectionPointToSegment(&pLCMCCNData->edge_node.pt,
                                  &pLCMCCNData->site_first->node[0]->pt,
                                  &pLCMCCNData->site_first->node[1]->pt,
                                  &pLCMCCNData->site_first_pt,
                                  NULL);
        _cvProjectionPointToSegment(&pLCMCCNData->edge_node.pt,
                                  &pLCMCCNData->site_last->node[0]->pt,
                                  &pLCMCCNData->site_last->node[1]->pt,
                                  &pLCMCCNData->site_last_pt,
                                  NULL);
    }
}//end of _cvPrepareData


void _cvProjectionPointToSegment(CvPoint2D32f* PointO,
                                 CvPoint2D32f* PointA,
                                 CvPoint2D32f* PointB,
                                 CvPoint2D32f* PrPoint,
                                 float* dist)
{
    float scal_AO_AB, scal_AB_AB;
    CvPoint2D32f VectorAB = {PointB->x - PointA->x, PointB->y - PointA->y};
    scal_AB_AB = VectorAB.x*VectorAB.x + VectorAB.y*VectorAB.y;
    if(scal_AB_AB < LCM_CONST_ZERO)
    {
        *PrPoint = *PointA;
        if(dist)
            *dist = (float)sqrt( (double)(PointO->x -PointA->x)*(PointO->x -PointA->x) + (PointO->y - PointA->y)*(PointO->y - PointA->y));
        return;
    }

    CvPoint2D32f VectorAO = {PointO->x - PointA->x, PointO->y - PointA->y};
    scal_AO_AB = VectorAO.x*VectorAB.x + VectorAO.y*VectorAB.y;

    if(dist)
    {
        float vector_AO_AB = (float)fabs(VectorAO.x*VectorAB.y - VectorAO.y*VectorAB.x);
        *dist = (float)(vector_AO_AB/sqrt((double)scal_AB_AB));
    }

    float alfa = scal_AO_AB/scal_AB_AB;
    PrPoint->x = PointO->x - VectorAO.x + alfa*VectorAB.x;
    PrPoint->y = PointO->y - VectorAO.y + alfa*VectorAB.y;
    return;
}//end of _cvProjectionPointToSegment
