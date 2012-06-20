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

/* Reconstruction of contour skeleton */
#include "precomp.hpp"
#include <time.h>

#define NEXT_SEQ(seq,seq_first) ((seq) == (seq_first) ? seq->v_next : seq->h_next)
#define SIGN(x) ( x<0 ? -1:( x>0 ? 1:0 ) )

const float LEE_CONST_ZERO = 1e-6f;
const float LEE_CONST_DIFF_POINTS = 1e-2f;
const float LEE_CONST_ACCEPTABLE_ERROR = 1e-4f;

/****************************************************************************************\
*                                    Auxiliary struct definitions                                 *
\****************************************************************************************/

template<class T>
struct CvLeePoint
{
    T x,y;
};

typedef CvLeePoint<float> CvPointFloat;
typedef CvLeePoint<float> CvDirection;

struct CvVoronoiSiteInt;
struct CvVoronoiEdgeInt;
struct CvVoronoiNodeInt;
struct CvVoronoiParabolaInt;
struct CvVoronoiChainInt;
struct CvVoronoiHoleInt;

struct CvVoronoiDiagramInt
{
    CvSeq* SiteSeq;
    CvSeq* EdgeSeq;
    CvSeq* NodeSeq;
    CvSeq* ChainSeq;
    CvSeq* ParabolaSeq;
    CvSeq* DirectionSeq;
    CvSeq* HoleSeq;
    CvVoronoiSiteInt* reflex_site;
    CvVoronoiHoleInt* top_hole;
};

struct CvVoronoiStorageInt
{
    CvMemStorage* SiteStorage;
    CvMemStorage* EdgeStorage;
    CvMemStorage* NodeStorage;
    CvMemStorage* ChainStorage;
    CvMemStorage* ParabolaStorage;
    CvMemStorage* DirectionStorage;
    CvMemStorage* HoleStorage;
};

struct CvVoronoiNodeInt
{
    CvPointFloat  node;
    float         radius;
};

struct CvVoronoiSiteInt
{
    CvVoronoiNodeInt* node1;
    CvVoronoiNodeInt* node2;
    CvVoronoiEdgeInt* edge1;
    CvVoronoiEdgeInt* edge2;
    CvVoronoiSiteInt* next_site;
    CvVoronoiSiteInt* prev_site;
    CvDirection* direction;
};

struct CvVoronoiEdgeInt
{
    CvVoronoiNodeInt* node1;
    CvVoronoiNodeInt* node2;
    CvVoronoiSiteInt* site;
    CvVoronoiEdgeInt* next_edge;
    CvVoronoiEdgeInt* prev_edge;
    CvVoronoiEdgeInt* twin_edge;
    CvVoronoiParabolaInt* parabola;
    CvDirection*  direction;
};

struct CvVoronoiParabolaInt
{
    float map[6];
    float a;
    CvVoronoiNodeInt* focus;
    CvVoronoiSiteInt* directrice;
};

struct CvVoronoiChainInt
{
    CvVoronoiSiteInt * first_site;
    CvVoronoiSiteInt * last_site;
    CvVoronoiChainInt* next_chain;
};

struct CvVoronoiHoleInt
{
    CvSeq* SiteSeq;
    CvSeq* ChainSeq;
    CvVoronoiSiteInt* site_top;
    CvVoronoiSiteInt* site_nearest;
    CvVoronoiSiteInt* site_opposite;
    CvVoronoiNodeInt* node;
    CvVoronoiHoleInt* next_hole;
    bool error;
    float x_coord;
};

typedef CvVoronoiSiteInt* pCvVoronoiSite;
typedef CvVoronoiEdgeInt* pCvVoronoiEdge;
typedef CvVoronoiNodeInt* pCvVoronoiNode;
typedef CvVoronoiParabolaInt* pCvVoronoiParabola;
typedef CvVoronoiChainInt* pCvVoronoiChain;
typedef CvVoronoiHoleInt* pCvVoronoiHole;
typedef CvPointFloat* pCvPointFloat;
typedef CvDirection* pCvDirection;

/****************************************************************************************\
*                                    Function definitions                                *
\****************************************************************************************/

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvLee
//    Purpose: Compute Voronoi Diagram for one given polygon with holes
//    Context:
//    Parameters:
//      ContourSeq : in, vertices of polygon.
//      VoronoiDiagramInt : in&out, pointer to struct, which contains the
//                       description of Voronoi Diagram.
//      VoronoiStorage: in, storage for Voronoi Diagram.
//      contour_type: in, type of vertices.
//                    The possible values are CV_LEE_INT,CV_LEE_FLOAT,CV_LEE_DOUBLE.
//      contour_orientation: in, orientation of polygons.
//                           = 1, if contour is left - oriented in left coordinat system
//                           =-1, if contour is left - oriented in right coordinat system
//      attempt_number: in, number of unsuccessful attemts made by program to compute
//                          the Voronoi Diagram befor return the error
//
//    Returns: 1, if Voronoi Diagram was succesfully computed
//             0, if some error occures
//F*/
static int  _cvLee(CvSeq* ContourSeq,
                      CvVoronoiDiagramInt* pVoronoiDiagramInt,
                      CvMemStorage* VoronoiStorage,
                      CvLeeParameters contour_type,
                      int contour_orientation,
                      int attempt_number);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstuctSites
//    Purpose : Compute sites for given polygon with holes
//                     (site is an edge of polygon or a reflex vertex).
//    Context:
//    Parameters:
//            ContourSeq : in, vertices of polygon
//       pVoronoiDiagram : in, pointer to struct, which contains the
//                          description of Voronoi Diagram
//           contour_type: in, type of vertices.  The possible values are
//                          CV_LEE_INT,CV_LEE_FLOAT,CV_LEE_DOUBLE.
//    contour_orientation: in, orientation of polygons.
//                           = 1, if contour is left - oriented in left coordinat system
//                           =-1, if contour is left - oriented in right coordinat system
//     Return: 1, if sites were succesfully constructed
//             0, if some error occures
//F*/
static int _cvConstuctSites(CvSeq* ContourSeq,
                            CvVoronoiDiagramInt* pVoronoiDiagram,
                            CvLeeParameters contour_type,
                            int contour_orientation);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructChains
//    Purpose : Compute chains for given polygon with holes.
//    Context:
//    Parameters:
//       pVoronoiDiagram : in, pointer to struct, which contains the
//                          description of Voronoi Diagram
//     Return: 1, if chains were succesfully constructed
//             0, if some error occures
//F*/
static int _cvConstructChains(CvVoronoiDiagramInt* pVoronoiDiagram);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructSkeleton
//    Purpose: Compute skeleton for given collection of sites, using Lee algorithm
//    Context:
//    Parameters:
//      VoronoiDiagram : in, pointer to struct, which contains the
//                       description of Voronoi Diagram.
//    Returns: 1, if skeleton was succesfully computed
//             0, if some error occures
//F*/
static int _cvConstructSkeleton(CvVoronoiDiagramInt* pVoronoiDiagram);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:  Andrey Sobolev
//    Name:    _cvConstructSiteTree
//    Purpose: Construct tree of sites (by analogy with contour tree).
//    Context:
//    Parameters:
//      VoronoiDiagram : in, pointer to struct, which contains the
//                       description of Voronoi Diagram.
//    Returns:
//F*/
static void _cvConstructSiteTree(CvVoronoiDiagramInt* pVoronoiDiagram);

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Author:   Andrey Sobolev
//    Name:     _cvReleaseVoronoiStorage
//    Purpose : Function realease storages.
//                  The storages are divided into two groups:
//                  SiteStorage, EdgeStorage, NodeStorage form the first group;
//                  ChainStorage,ParabolaStorage,DirectionStorage,HoleStorage form the second group.
//    Context:
//    Parameters:
//        pVoronoiStorage: in,
//        group1,group2: in, if group1<>0 then storages from first group released
//                           if group2<>0 then storages from second group released
//    Return    :
//F*/
static void _cvReleaseVoronoiStorage(CvVoronoiStorageInt* pVoronoiStorage, int group1, int group2);

/*F///////////////////////////////////////////////////////////////////////////////////////
//  Author:  Andrey Sobolev
//  Name:  _cvConvert
//  Purpose :  Function convert internal representation of VD (via
//                  structs CvVoronoiSiteInt, CvVoronoiEdgeInt,CvVoronoiNodeInt) into
//                  external representation of VD (via structs CvVoronoiSite2D, CvVoronoiEdge2D,
//                  CvVoronoiNode2D)
//    Context:
//    Parameters:
//        VoronoiDiagram: in
//        VoronoiStorage: in
//        change_orientation: in, if = -1 then the convertion is accompanied with change
//                            of orientation
//
//     Return: 1, if convertion was succesfully completed
//             0, if some error occures
//F*/
/*
static int _cvConvert(CvVoronoiDiagram2D* VoronoiDiagram,
                      CvMemStorage* VoronoiStorage,
                      int change_orientation);
*/
static int _cvConvert(CvVoronoiDiagram2D* VoronoiDiagram,
                       CvVoronoiDiagramInt VoronoiDiagramInt,
                       CvSet* &NewSiteSeqPrev,
                       CvSeqWriter &NodeWriter,
                       CvSeqWriter &EdgeWriter,
                       CvMemStorage* VoronoiStorage,
                       int change_orientation);

/*F///////////////////////////////////////////////////////////////////////////////////////
//  Author:  Andrey Sobolev
//  Name:  _cvConvertSameOrientation
//  Purpose :  Function convert internal representation of VD (via
//                  structs CvVoronoiSiteInt, CvVoronoiEdgeInt,CvVoronoiNodeInt) into
//                  external representation of VD (via structs CvVoronoiSite2D, CvVoronoiEdge2D,
//                  CvVoronoiNode2D) without change of orientation
//    Context:
//    Parameters:
//        VoronoiDiagram: in
//        VoronoiStorage: in
/
//     Return: 1, if convertion was succesfully completed
//             0, if some error occures
//F*/
/*
static int _cvConvertSameOrientation(CvVoronoiDiagram2D* VoronoiDiagram,
                                      CvMemStorage* VoronoiStorage);
*/
static int _cvConvertSameOrientation(CvVoronoiDiagram2D* VoronoiDiagram,
                       CvVoronoiDiagramInt VoronoiDiagramInt,
                       CvSet* &NewSiteSeqPrev,
                       CvSeqWriter &NodeWriter,
                       CvSeqWriter &EdgeWriter,
                       CvMemStorage* VoronoiStorage);

/*F///////////////////////////////////////////////////////////////////////////////////////
//  Author:  Andrey Sobolev
//  Name:  _cvConvertChangeOrientation
//  Purpose :  Function convert internal representation of VD (via
//                  structs CvVoronoiSiteInt, CvVoronoiEdgeInt,CvVoronoiNodeInt) into
//                  external representation of VD (via structs CvVoronoiSite2D, CvVoronoiEdge2D,
//                  CvVoronoiNode2D) with change of orientation
//    Context:
//    Parameters:
//        VoronoiDiagram: in
//        VoronoiStorage: in
/
//     Return: 1, if convertion was succesfully completed
//             0, if some error occures
//F*/
/*
static int _cvConvertChangeOrientation(CvVoronoiDiagram2D* VoronoiDiagram,
                                      CvMemStorage* VoronoiStorage);
                                      */
static int _cvConvertChangeOrientation(CvVoronoiDiagram2D* VoronoiDiagram,
                       CvVoronoiDiagramInt VoronoiDiagramInt,
                       CvSet* &NewSiteSeqPrev,
                       CvSeqWriter &NodeWriter,
                       CvSeqWriter &EdgeWriter,
                       CvMemStorage* VoronoiStorage);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute  sites for external polygon.
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     ContourSeq : in, vertices of polygon
     pReflexSite: out, pointer to reflex site,if any exist,else NULL
     orientation: in, orientation of contour ( = 1 or = -1)
     type:        in, type of vertices. The possible values are (int)1,
                   (float)1,(double)1.
     Return:    1, if sites were succesfully constructed
                0, if some error occures    :
    --------------------------------------------------------------------------*/
template<class T>
int _cvConstructExtSites(CvVoronoiDiagramInt* pVoronoiDiagram,
                         CvSeq* ContourSeq,
                         int orientation,
                         T /*type*/);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute  sites for internal polygon (for hole).
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     CurrSiteSeq: in, the sequence for sites to be constructed
     CurrContourSeq : in, vertices of polygon
     pTopSite: out, pointer to the most left site of polygon (it is the most left
                vertex of polygon)
     orientation: in, orientation of contour ( = 1 or = -1)
     type:        in, type of vertices. The possible values are (int)1,
                   (float)1,(double)1.
     Return:    1, if sites were succesfully constructed
                0, if some error occures    :
    --------------------------------------------------------------------------*/
template<class T>
int _cvConstructIntSites(CvVoronoiDiagramInt* pVoronoiDiagram,
                         CvSeq* CurrSiteSeq,
                         CvSeq* CurrContourSeq,
                         pCvVoronoiSite &pTopSite,
                         int orientation,
                         T /*type*/);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the simple chains of sites for external polygon.
    Arguments
     pVoronoiDiagram : in&out, pointer to struct, which contains the
                        description of Voronoi Diagram

    Return: 1, if chains were succesfully constructed
            0, if some error occures
    --------------------------------------------------------------------------*/
static int _cvConstructExtChains(CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the simple chains of sites for internal polygon (= hole)
    Arguments
    pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
    CurrSiteSeq : in, the sequence of sites
   CurrChainSeq : in,the sequence for chains to be constructed
       pTopSite : in, the most left site of hole

      Return    :
    --------------------------------------------------------------------------*/
static void _cvConstructIntChains(CvVoronoiDiagramInt* pVoronoiDiagram,
                                  CvSeq* CurrChainSeq,
                                  CvSeq* CurrSiteSeq,
                                  pCvVoronoiSite pTopSite);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the initial Voronoi Diagram for single site
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pSite: in, pointer to site

     Return :
    --------------------------------------------------------------------------*/
static void _cvConstructEdges(pCvVoronoiSite pSite,CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function moves each node on small random value. The nodes are taken
                    from pVoronoiDiagram->NodeSeq.
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     begin,end: in, the first and the last nodes in pVoronoiDiagram->NodeSeq,
                    which moves
     shift: in, the value of maximal shift.
     Return :
    --------------------------------------------------------------------------*/
static void _cvRandomModification(CvVoronoiDiagramInt* pVoronoiDiagram, int begin, int end, float shift);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the internal Voronoi Diagram for external polygon.
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     Return     : 1, if VD was constructed succesfully
                  0, if some error occure
    --------------------------------------------------------------------------*/
static int _cvConstructExtVD(CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the external Voronoi Diagram for each internal polygon (hole).
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     Return     :
    --------------------------------------------------------------------------*/
static void _cvConstructIntVD(CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function joins the Voronoi Diagrams of different
                    chains into one Voronoi Diagram
    Arguments
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pChain1,pChain1: in, given chains
     Return     : 1, if joining was succesful
                  0, if some error occure
    --------------------------------------------------------------------------*/
static int _cvJoinChains(pCvVoronoiChain pChain1,
                         pCvVoronoiChain pChain2,
                         CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function finds the nearest site for top vertex
                 (= the most left vertex) of each hole
    Arguments
        pVoronoiDiagram : in, pointer to struct, which contains the
                         description of Voronoi Diagram
     Return     :
    --------------------------------------------------------------------------*/
static void _cvFindNearestSite(CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function seeks for site, which has common bisector in
                  final VD with top vertex of given hole. It stores in pHole->opposite_site.
                   The search begins from  Hole->nearest_site and realizes in clockwise
                   direction around the top vertex of given hole.
    Arguments
        pVoronoiDiagram : in, pointer to struct, which contains the
                          description of Voronoi Diagram
          pHole : in, given hole
     Return     : 1, if the search was succesful
                  0, if some error occure
    --------------------------------------------------------------------------*/
static int _cvFindOppositSiteCW(pCvVoronoiHole pHole, CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function seeks for site, which has common bisector in
                  final VD with top vertex of given hole. It stores in pHole->opposite_site.
                   The search begins from  Hole->nearest_site and realizes in counterclockwise
                   direction around the top vertex of given hole.
    Arguments
        pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
          pHole : in, given hole
     Return     : 1, if the search was succesful
                  0, if some error occure
    --------------------------------------------------------------------------*/
static int _cvFindOppositSiteCCW(pCvVoronoiHole pHole,CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function merges external VD of hole and internal VD, which was
                  constructed ealier.
    Arguments
pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
          pHole : in, given hole
     Return     : 1, if merging was succesful
                  0, if some error occure
    --------------------------------------------------------------------------*/
static int _cvMergeVD(pCvVoronoiHole pHole,CvVoronoiDiagramInt* pVoronoiDiagram);


/* ///////////////////////////////////////////////////////////////////////////////////////
//                               Computation of bisectors                               //
/////////////////////////////////////////////////////////////////////////////////////// */

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the bisector of two sites
    Arguments
     pSite_left,pSite_right: in, given sites
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
void _cvCalcEdge(pCvVoronoiSite pSite_left,
                pCvVoronoiSite pSite_right,
                pCvVoronoiEdge pEdge,
                CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the bisector of point and site
    Arguments
     pSite      : in, site
     pNode      : in, point
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
void _cvCalcEdge(pCvVoronoiSite pSite,
                pCvVoronoiNode pNode,
                pCvVoronoiEdge pEdge,
                CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the bisector of point and site
    Arguments
     pSite      : in, site
     pNode      : in, point
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
void _cvCalcEdge(pCvVoronoiNode pNode,
                pCvVoronoiSite pSite,
                pCvVoronoiEdge pEdge,
                CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the direction of bisector of two segments
    Arguments
     pDirection1: in, direction of first segment
     pDirection2: in, direction of second segment
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvCalcEdgeLL(pCvDirection pDirection1,
                  pCvDirection pDirection2,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the bisector of two points
    Arguments
     pPoint1, pPoint2: in, given points
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvCalcEdgePP(pCvPointFloat pPoint1,
                  pCvPointFloat pPoint2,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the bisector of segment and point. Since
                    it is parabola, it is defined by its focus (site - point)
                    and directrice(site-segment)
    Arguments
     pFocus    : in, point, which defines the focus of parabola
     pDirectrice: in, site - segment, which defines the directrice of parabola
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvCalcEdgePL(pCvVoronoiNode pFocus,
                  pCvVoronoiSite pDirectrice,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Compute the bisector of segment and point. Since
                    it is parabola, it is defined by its focus (site - point)
                    and directrice(site-segment)
    Arguments
     pFocus    : in, point, which defines the focus of parabola
     pDirectrice: in, site - segment, which defines the directrice of parabola
     pVoronoiDiagram : in, pointer to struct, which contains the
                        description of Voronoi Diagram
     pEdge      : out, bisector
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvCalcEdgeLP(pCvVoronoiSite pDirectrice,
                  pCvVoronoiNode pFocus,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram);

/* ///////////////////////////////////////////////////////////////////////////////////////
//                  Computation of intersections of bisectors                           //
/////////////////////////////////////////////////////////////////////////////////////// */

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1,pEdge2: in, two edges
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvCalcEdgeIntersection(pCvVoronoiEdge pEdge1,
                              pCvVoronoiEdge pEdge2,
                              CvPointFloat* pPoint,
                              float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in, straight ray
    pEdge2: in, straight ray or segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
     Return : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvLine_LineIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in, straight ray
    pEdge2: in, parabolic ray or segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvLine_ParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in, straight ray
    pEdge2: in, parabolic segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvLine_CloseParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in, straight ray
    pEdge2: in, parabolic ray
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvLine_OpenParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in,  parabolic ray
    pEdge2: in,  straight ray or segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvPar_LineIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);
/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in,  parabolic ray
    pEdge2: in,  straight ray
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvPar_OpenLineIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in,  parabolic ray
    pEdge2: in,  straight segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvPar_CloseLineIntersection(pCvVoronoiEdge pEdge1,
                                    pCvVoronoiEdge pEdge2,
                                    pCvPointFloat  pPoint,
                                    float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in,  parabolic ray
    pEdge2: in,  parabolic ray or segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvPar_ParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius);


/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in,  parabolic ray
    pEdge2: in,  parabolic ray
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvPar_OpenParIntersection(pCvVoronoiEdge pEdge1,
                            pCvVoronoiEdge pEdge2,
                            pCvPointFloat  pPoint,
                            float &Radius);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes intersection of two edges. Intersection
                    must be the nearest to the marked point on pEdge1
                    (this marked point is pEdge1->node1->node).
    Arguments
    pEdge1 : in,  parabolic ray
    pEdge2: in,  parabolic segment
        pPoint: out, intersection of pEdge1 and pEdge2
        Radius: out, distance between pPoint and sites, assosiated
                    with pEdge1 and pEdge2 (pPoint is situated on the equal
                    distance from site, assosiated with pEdge1 and from
                    site,assosiated with pEdge2)
      Return    : distance between pPoint and marked point on pEdge1 or
                : -1, if edges have no intersections
    --------------------------------------------------------------------------*/
static
float _cvPar_CloseParIntersection(pCvVoronoiEdge pEdge1,
                            pCvVoronoiEdge pEdge2,
                            pCvPointFloat  pPoint,
                            float &Radius);

/* ///////////////////////////////////////////////////////////////////////////////////////
//                           Subsidiary functions                                       //
/////////////////////////////////////////////////////////////////////////////////////// */

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge1  : in
        pEdge2  : out
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvMakeTwinEdge(pCvVoronoiEdge pEdge2,
                     pCvVoronoiEdge pEdge1);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge   : in&out
        pEdge_left_prev : in&out
        pSite_left : in&out
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvStickEdgeLeftBegin(pCvVoronoiEdge pEdge,
                          pCvVoronoiEdge pEdge_left_prev,
                          pCvVoronoiSite pSite_left);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge   : in&out
        pEdge_right_next : in&out
        pSite_right : in&out
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvStickEdgeRightBegin(pCvVoronoiEdge pEdge,
                          pCvVoronoiEdge pEdge_right_next,
                          pCvVoronoiSite pSite_right);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge   : in&out
        pEdge_left_next : in&out
        pSite_left : in&out
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvStickEdgeLeftEnd(pCvVoronoiEdge pEdge,
                        pCvVoronoiEdge pEdge_left_next,
                        pCvVoronoiSite pSite_left);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge   : in&out
        pEdge_right_prev : in&out
        pSite_right : in&out
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvStickEdgeRightEnd(pCvVoronoiEdge pEdge,
                         pCvVoronoiEdge pEdge_right_prev,
                         pCvVoronoiSite pSite_right);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge_left_cur  : in
        pEdge_left : in
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvTwinNULLLeft(pCvVoronoiEdge pEdge_left_cur,
                    pCvVoronoiEdge pEdge_left);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description :
    Arguments
        pEdge_right_cur : in
        pEdge_right : in
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvTwinNULLRight(pCvVoronoiEdge pEdge_right_cur,
                     pCvVoronoiEdge pEdge_right);


/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function initializes the struct CvVoronoiNode
    Arguments
        pNode   : out
         pPoint : in,
        radius  : in
     Return     :
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
void _cvInitVoronoiNode(pCvVoronoiNode pNode,
                       T pPoint, float radius = 0);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function initializes the struct CvVoronoiSite
    Arguments
        pSite   : out
         pNode1,pNode2,pPrev_site : in
     Return     :
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvInitVoronoiSite(pCvVoronoiSite pSite,
                       pCvVoronoiNode pNode1,
                       pCvVoronoiNode pNode2,
                       pCvVoronoiSite pPrev_site);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function pushs the element in the end of the sequence
                    end returns its adress
    Arguments
            Seq : in, pointer to the sequence
           Elem : in, element
     Return     : pointer to the element in the sequence
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
T _cvSeqPush(CvSeq* Seq, T pElem);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function pushs the element in the begin of the sequence
                    end returns its adress
    Arguments
            Seq : in, pointer to the sequence
           Elem : in, element
     Return     : pointer to the element in the sequence
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
T _cvSeqPushFront(CvSeq* Seq, T pElem);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function pushs the element pHole in pHoleHierarchy->HoleSeq
                     so as all elements in this sequence would be normalized
                     according to field .x_coord of element. pHoleHierarchy->TopHole
                     points to hole with smallest x_coord.
    Arguments
pHoleHierarchy  : in, pointer to the structur
          pHole : in, element
     Return     : pointer to the element in the sequence
    --------------------------------------------------------------------------*/
CV_INLINE
void _cvSeqPushInOrder(CvVoronoiDiagramInt* pVoronoiDiagram, pCvVoronoiHole pHole);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function intersects given edge pEdge (and his twin edge)
                    by point pNode on two parts
    Arguments
        pEdge   : in, given edge
          pNode : in, given point
        EdgeSeq : in
     Return     : one of parts
    --------------------------------------------------------------------------*/
CV_INLINE
pCvVoronoiEdge _cvDivideRightEdge(pCvVoronoiEdge pEdge,
                                 pCvVoronoiNode pNode,
                                 CvSeq* EdgeSeq);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function intersects given edge pEdge (and his twin edge)
                    by point pNode on two parts
    Arguments
        pEdge   : in, given edge
          pNode : in, given point
        EdgeSeq : in
     Return     : one of parts
    --------------------------------------------------------------------------*/
CV_INLINE
pCvVoronoiEdge _cvDivideLeftEdge(pCvVoronoiEdge pEdge,
                                pCvVoronoiNode pNode,
                                CvSeq* EdgeSeq);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : function pushs the element in the end of the sequence
                    end returns its adress
    Arguments
          writer: in, writer associated with sequence
          pElem : in, element
     Return     : pointer to the element in the sequence
    --------------------------------------------------------------------------*/
template<class T> CV_INLINE
T _cvWriteSeqElem(T pElem, CvSeqWriter &writer);

/* ///////////////////////////////////////////////////////////////////////////////////////
//                           Mathematical functions                                     //
/////////////////////////////////////////////////////////////////////////////////////// */

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function changes x and y
    Arguments
            x,y : in&out
      Return    :
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
void _cvSwap(T &x, T &y);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the inverse map to the
                    given ortogonal affine map
    Arguments
            A   : in, given ortogonal affine map
            B   : out, inverse map
      Return    : 1, if inverse map exist
                  0, else
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
int _cvCalcOrtogInverse(T* B, T* A);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the composition of two affine maps
    Arguments
            A,B : in, given affine maps
            Result: out, composition of A and B (Result = AB)
      Return    :
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
void _cvCalcComposition(T* Result,T* A,T* B);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the image of point under
                    given affin map
    Arguments
            A   : in, affine maps
        pPoint  : in, pointer to point
        pImgPoint:out, pointer to image of point
      Return    :
    --------------------------------------------------------------------------*/
template<class T> CV_INLINE
void _cvCalcPointImage(pCvPointFloat pImgPoint,pCvPointFloat pPoint,T* A);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the image of vector under
                    given affin map
    Arguments
            A   : in, affine maps
        pVector : in, pointer to vector
        pImgVector:out, pointer to image of vector
      Return    :
    --------------------------------------------------------------------------*/
template<class T> CV_INLINE
void _cvCalcVectorImage(pCvDirection pImgVector,pCvDirection pVector,T* A);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the distance between the point
                    and site. Internal function.
    Arguments
        pPoint  : in, point
        pSite   : in, site
      Return    : distance
    --------------------------------------------------------------------------*/
CV_INLINE
float _cvCalcDist(pCvPointFloat pPoint, pCvVoronoiSite pSite);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the distance between two points
    Arguments
    pPoint1,pPoint2 : in, two points
      Return    : distance
    --------------------------------------------------------------------------*/
CV_INLINE
float _cvPPDist(pCvPointFloat pPoint1,pCvPointFloat pPoint2);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function computes the distance betwin point and
                    segment. Internal function.
    Arguments
          pPoint: in, point
    pPoint1,pPoint2 : in, segment [pPoint1,pPoint2]
       Return   : distance
    --------------------------------------------------------------------------*/
CV_INLINE
float _cvPLDist(pCvPointFloat pPoint,pCvPointFloat pPoint1,pCvDirection pDirection);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function solves the squar equation with real coefficients
                    T - float or double
    Arguments
     c2,c1,c0: in, real coefficients of polynom
               X: out, array of roots
     Return     : number of roots
    --------------------------------------------------------------------------*/
template <class T>
int _cvSolveEqu2thR(T c2, T c1, T c0, T* X);

/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : Function solves the linear equation with real or complex coefficients
                    T - float or double or complex
    Arguments
        c1,c0: in, real or complex coefficients of polynom
               X: out, array of roots
     Return     : number of roots
    --------------------------------------------------------------------------*/
template <class T> CV_INLINE
int _cvSolveEqu1th(T c1, T c0, T* X);

/****************************************************************************************\
*                             Storage Block Increase                                    *
\****************************************************************************************/
/*--------------------------------------------------------------------------
    Author      : Andrey Sobolev
    Description : For each sequence function creates the memory block sufficient to store
                  all elements of sequnce
    Arguments
        pVoronoiDiagramInt: in, pointer to struct, which contains the
                            description of Voronoi Diagram.
        vertices_number: in, number of vertices in polygon
     Return     :
    --------------------------------------------------------------------------*/
static void _cvSetSeqBlockSize(CvVoronoiDiagramInt* pVoronoiDiagramInt,int vertices_number)
{
    int N = 2*vertices_number;
    cvSetSeqBlockSize(pVoronoiDiagramInt->SiteSeq,N*pVoronoiDiagramInt->SiteSeq->elem_size);
    cvSetSeqBlockSize(pVoronoiDiagramInt->EdgeSeq,3*N*pVoronoiDiagramInt->EdgeSeq->elem_size);
    cvSetSeqBlockSize(pVoronoiDiagramInt->NodeSeq,5*N*pVoronoiDiagramInt->NodeSeq->elem_size);
    cvSetSeqBlockSize(pVoronoiDiagramInt->ParabolaSeq,N*pVoronoiDiagramInt->ParabolaSeq->elem_size);
    cvSetSeqBlockSize(pVoronoiDiagramInt->DirectionSeq,3*N*pVoronoiDiagramInt->DirectionSeq->elem_size);
    cvSetSeqBlockSize(pVoronoiDiagramInt->ChainSeq,N*pVoronoiDiagramInt->DirectionSeq->elem_size);
    cvSetSeqBlockSize(pVoronoiDiagramInt->HoleSeq,100*pVoronoiDiagramInt->HoleSeq->elem_size);
}

/****************************************************************************************\
*                                    Function realization                               *
\****************************************************************************************/


CV_IMPL int   cvVoronoiDiagramFromContour(CvSeq* ContourSeq,
                                           CvVoronoiDiagram2D** VoronoiDiagram,
                                           CvMemStorage* VoronoiStorage,
                                           CvLeeParameters contour_type,
                                           int contour_orientation,
                                           int attempt_number)
{
    CV_FUNCNAME( "cvVoronoiDiagramFromContour" );

    __BEGIN__;

    CvSet* SiteSeq = NULL;
    CvSeq* CurContourSeq = NULL;
    CvVoronoiDiagramInt VoronoiDiagramInt;
    CvSeqWriter NodeWriter, EdgeWriter;
    CvMemStorage* storage;

    memset( &VoronoiDiagramInt, 0, sizeof(VoronoiDiagramInt) );

    if( !ContourSeq )
        CV_ERROR( CV_StsBadArg,"Contour sequence is empty" );

    if(!VoronoiStorage)
        CV_ERROR( CV_StsBadArg,"Storage is not initialized" );

    if( contour_type < 0 || contour_type > 2)
        CV_ERROR( CV_StsBadArg,"Unsupported parameter: type" );

    if( contour_orientation != 1 &&  contour_orientation != -1)
        CV_ERROR( CV_StsBadArg,"Unsupported parameter: orientation" );

    storage = cvCreateChildMemStorage(VoronoiStorage);
    (*VoronoiDiagram) = (CvVoronoiDiagram2D*)cvCreateSet(0,sizeof(CvVoronoiDiagram2D),sizeof(CvVoronoiNode2D), storage);
    storage = cvCreateChildMemStorage(VoronoiStorage);
    (*VoronoiDiagram)->edges = cvCreateSet(0,sizeof(CvSet),sizeof(CvVoronoiEdge2D), storage);
    cvStartAppendToSeq((CvSeq*)(*VoronoiDiagram)->edges, &EdgeWriter);
    cvStartAppendToSeq((CvSeq*)(*VoronoiDiagram), &NodeWriter);

        for(CurContourSeq = ContourSeq;\
            CurContourSeq != NULL;\
            CurContourSeq = CurContourSeq->h_next)
        {
            if(_cvLee(CurContourSeq, &VoronoiDiagramInt,VoronoiStorage,contour_type,contour_orientation,attempt_number))

            {
                if(!_cvConvert(*VoronoiDiagram,VoronoiDiagramInt,SiteSeq,NodeWriter,EdgeWriter,VoronoiStorage,contour_orientation))
                    goto exit;
            }
            else if(CurContourSeq->total >= 3)
                goto exit;
        }

        cvEndWriteSeq(&EdgeWriter);
        cvEndWriteSeq(&NodeWriter);
        if(SiteSeq != NULL)
            return 1;


    __END__;
    return 0;
}//end of cvVoronoiDiagramFromContour

CV_IMPL int   cvVoronoiDiagramFromImage(IplImage* pImage,
                                         CvSeq** ContourSeq,
                                         CvVoronoiDiagram2D** VoronoiDiagram,
                                         CvMemStorage* VoronoiStorage,
                                         CvLeeParameters regularization_method,
                                         float approx_precision)
{
    CV_FUNCNAME( "cvVoronoiDiagramFromContour" );
    int RESULT = 0;

    __BEGIN__;

    IplImage* pWorkImage = NULL;
    CvSize image_size;
    int i, multiplicator = 3;

    int approx_method;
    CvMemStorage* ApproxContourStorage = NULL;
    CvSeq* ApproxContourSeq = NULL;

    if( !ContourSeq )
        CV_ERROR( CV_StsBadArg,"Contour sequence is not initialized" );

    if( (*ContourSeq)->total != 0)
        CV_ERROR( CV_StsBadArg,"Contour sequence is not empty" );

    if(!VoronoiStorage)
        CV_ERROR( CV_StsBadArg,"Storage is not initialized" );

    if(!pImage)
        CV_ERROR( CV_StsBadArg,"Image is not initialized" );

    if(pImage->nChannels != 1 || pImage->depth != 8)
        CV_ERROR( CV_StsBadArg,"Unsupported image format" );

    if(approx_precision<0 && approx_precision != CV_LEE_AUTO)
        CV_ERROR( CV_StsBadArg,"Unsupported presision value" );

    switch(regularization_method)
    {
        case CV_LEE_ERODE:  image_size.width = pImage->width;
                            image_size.height = pImage->height;
                            pWorkImage = cvCreateImage(image_size,8,1);
                            cvErode(pImage,pWorkImage,0,1);
                            approx_method = CV_CHAIN_APPROX_TC89_L1;
                            break;
        case CV_LEE_ZOOM:   image_size.width = multiplicator*pImage->width;
                            image_size.height = multiplicator*pImage->height;
                            pWorkImage = cvCreateImage(image_size,8,1);
                            cvResize(pImage, pWorkImage, CV_INTER_NN);
                            approx_method = CV_CHAIN_APPROX_TC89_L1;
                            break;
        case CV_LEE_NON:    pWorkImage = pImage;
                            approx_method = CV_CHAIN_APPROX_TC89_L1;
                            break;
        default:            CV_ERROR( CV_StsBadArg,"Unsupported regularisation method" );
                            break;

    }

    cvFindContours(pWorkImage, (*ContourSeq)->storage, ContourSeq, \
                            sizeof(CvContour), CV_RETR_CCOMP, approx_method );

    if(pWorkImage && pWorkImage != pImage )
        cvReleaseImage(&pWorkImage);

    ApproxContourStorage = cvCreateMemStorage(0);
    if(approx_precision > 0)
    {
        ApproxContourSeq = cvApproxPoly(*ContourSeq, sizeof(CvContour), ApproxContourStorage,\
                                        CV_POLY_APPROX_DP,approx_precision,1);

        RESULT = cvVoronoiDiagramFromContour(ApproxContourSeq,VoronoiDiagram,VoronoiStorage,CV_LEE_INT,-1,10);
    }
    else if(approx_precision == CV_LEE_AUTO)
    {
        ApproxContourSeq = *ContourSeq;
        for(i = 1; i < 50; i++)
        {
            RESULT = cvVoronoiDiagramFromContour(ApproxContourSeq,VoronoiDiagram,VoronoiStorage,CV_LEE_INT,-1,1);
            if(RESULT)
                break;
            ApproxContourSeq = cvApproxPoly(ApproxContourSeq, sizeof(CvContour),ApproxContourStorage,\
                                            CV_POLY_APPROX_DP,(float)i,1);
        }
    }
    else
        RESULT = cvVoronoiDiagramFromContour(*ContourSeq,VoronoiDiagram,VoronoiStorage,CV_LEE_INT,-1,10);
/*
    if(ApproxContourSeq)
    {
        cvClearMemStorage( (*ContourSeq)->storage );
        *ContourSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvPoint),(*ContourSeq)->storage);
        CvSeqReader reader;
        CvSeqWriter writer;
        CvPoint Point;
        cvStartAppendToSeq(*ContourSeq, &writer);
        cvStartReadSeq(ApproxContourSeq, &reader);
        for(int i = 0;i < ApproxContourSeq->total;i++)
        {
            CV_READ_SEQ_ELEM(Point,reader);
            Point.y = 600 - Point.y;
            CV_WRITE_SEQ_ELEM(Point,writer);
        }
        cvEndWriteSeq(&writer);
    }
    */

    cvReleaseMemStorage(&ApproxContourStorage);


    __END__;
    return RESULT;
}//end of cvVoronoiDiagramFromImage

CV_IMPL void cvReleaseVoronoiStorage(CvVoronoiDiagram2D* VoronoiDiagram,
                                     CvMemStorage** pVoronoiStorage)
{
    /*CV_FUNCNAME( "cvReleaseVoronoiStorage" );*/
    __BEGIN__;

    CvSeq* Seq;

    if(VoronoiDiagram->storage)
        cvReleaseMemStorage(&VoronoiDiagram->storage);
    for(Seq = (CvSeq*)VoronoiDiagram->sites; Seq != NULL; Seq = Seq->h_next)
        if(Seq->storage)
            cvReleaseMemStorage(&Seq->storage);
    for(Seq = (CvSeq*)VoronoiDiagram->edges; Seq != NULL; Seq = Seq->h_next)
        if(Seq->storage)
            cvReleaseMemStorage(&Seq->storage);

    if(*pVoronoiStorage)
        cvReleaseMemStorage(pVoronoiStorage);

    __END__;
}//end of cvReleaseVoronoiStorage

static int  _cvLee(CvSeq* ContourSeq,
                    CvVoronoiDiagramInt* pVoronoiDiagramInt,
                    CvMemStorage* VoronoiStorage,
                    CvLeeParameters contour_type,
                    int contour_orientation,
                    int attempt_number)
{
    //orientation = 1 for left coordinat system
    //orientation = -1 for right coordinat system
    if(ContourSeq->total<3)
        return 0;

    int attempt = 0;
    CvVoronoiStorageInt VoronoiStorageInt;

    srand((int)cvGetTickCount());

NEXTATTEMPT:
    VoronoiStorageInt.SiteStorage = cvCreateChildMemStorage(VoronoiStorage);
    VoronoiStorageInt.NodeStorage = cvCreateChildMemStorage(VoronoiStorage);
    VoronoiStorageInt.EdgeStorage = cvCreateChildMemStorage(VoronoiStorage);
    VoronoiStorageInt.ParabolaStorage = cvCreateMemStorage(0);
    VoronoiStorageInt.ChainStorage = cvCreateMemStorage(0);
    VoronoiStorageInt.DirectionStorage = cvCreateMemStorage(0);
    VoronoiStorageInt.HoleStorage = cvCreateMemStorage(0);

    pVoronoiDiagramInt->SiteSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiSiteInt),VoronoiStorageInt.SiteStorage);
    pVoronoiDiagramInt->NodeSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiNodeInt),VoronoiStorageInt.NodeStorage);
    pVoronoiDiagramInt->EdgeSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiEdgeInt),VoronoiStorageInt.EdgeStorage);
    pVoronoiDiagramInt->ChainSeq  = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiChainInt),VoronoiStorageInt.ChainStorage);
    pVoronoiDiagramInt->DirectionSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvDirection),VoronoiStorageInt.DirectionStorage);
    pVoronoiDiagramInt->ParabolaSeq =  cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiParabolaInt),VoronoiStorageInt.ParabolaStorage);
    pVoronoiDiagramInt->HoleSeq =  cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiHoleInt),VoronoiStorageInt.HoleStorage);

    _cvSetSeqBlockSize(pVoronoiDiagramInt,ContourSeq->total);

    if(!_cvConstuctSites(ContourSeq, pVoronoiDiagramInt, contour_type,contour_orientation))
    {
        attempt = attempt_number;
        goto FAULT;
    }
    _cvRandomModification(pVoronoiDiagramInt, 0,pVoronoiDiagramInt->NodeSeq->total,0.2f);

    if(!_cvConstructChains(pVoronoiDiagramInt))
    {
        attempt = attempt_number;
        goto FAULT;
    }

    if(!_cvConstructSkeleton(pVoronoiDiagramInt))
        goto FAULT;

    _cvConstructSiteTree(pVoronoiDiagramInt);

//SUCCESS:
    _cvReleaseVoronoiStorage(&VoronoiStorageInt,0,1);
    return 1;

FAULT:
    _cvReleaseVoronoiStorage(&VoronoiStorageInt,1,1);
    if(++attempt < attempt_number)
        goto NEXTATTEMPT;

    return 0;
}// end of _cvLee

static int _cvConstuctSites(CvSeq* ContourSeq,
                            CvVoronoiDiagramInt* pVoronoiDiagram,
                            CvLeeParameters contour_type,
                            int contour_orientation)
{
    pVoronoiDiagram->reflex_site = NULL;
    pVoronoiDiagram->top_hole = NULL;
    int result = 0;

    switch(contour_type)
    {
        case CV_LEE_INT :    result = _cvConstructExtSites(pVoronoiDiagram,ContourSeq,contour_orientation,(int)1);
                             break;
        case CV_LEE_FLOAT :  result = _cvConstructExtSites(pVoronoiDiagram,ContourSeq,contour_orientation,(float)1);
                             break;
        case CV_LEE_DOUBLE : result = _cvConstructExtSites(pVoronoiDiagram,ContourSeq,contour_orientation,(double)1);
                             break;
        default:             return 0;
    }

    if(!result)
        return 0;

    CvSeq* CurSiteSeq;
    CvVoronoiHoleInt Hole = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,false,0};
    pCvVoronoiSite pTopSite = 0;
    for(CvSeq* CurContourSeq = ContourSeq->v_next;\
        CurContourSeq != NULL;\
        CurContourSeq = CurContourSeq->h_next)
    {
        if(CurContourSeq->total == 0)
            continue;

        CurSiteSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiSiteInt),pVoronoiDiagram->SiteSeq->storage);
        switch(contour_type)
        {
            case CV_LEE_INT :   result = _cvConstructIntSites(pVoronoiDiagram,CurSiteSeq,CurContourSeq,pTopSite,contour_orientation,(int)1);
                                break;
            case CV_LEE_FLOAT : result = _cvConstructIntSites(pVoronoiDiagram,CurSiteSeq,CurContourSeq,pTopSite,contour_orientation,(float)1);
                                break;
            case CV_LEE_DOUBLE :result = _cvConstructIntSites(pVoronoiDiagram,CurSiteSeq,CurContourSeq,pTopSite,contour_orientation,(double)1);
                                break;
            default:            result = 0;
        }
        if(!result)
            continue;

        Hole.SiteSeq = CurSiteSeq;
        Hole.site_top = pTopSite;
        Hole.x_coord = pTopSite->node1->node.x;
        Hole.error = false;
        _cvSeqPushInOrder(pVoronoiDiagram, &Hole);
    }
    return 1;
}//end of _cvConstuctSites

static int _cvConstructChains(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    if(!_cvConstructExtChains(pVoronoiDiagram))
        return 0;

    CvSeq* CurrChainSeq;
    for(pCvVoronoiHole pHole = pVoronoiDiagram->top_hole;\
        pHole != NULL; \
        pHole = pHole->next_hole)
        {
            pHole->error = false;
            CurrChainSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvVoronoiChainInt),pVoronoiDiagram->ChainSeq->storage);
            _cvConstructIntChains(pVoronoiDiagram,CurrChainSeq,pHole->SiteSeq,pHole->site_top);
            pHole->ChainSeq = CurrChainSeq;
        }
    return 1;
}//end of _cvConstructChains

static int _cvConstructSkeleton(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    if(!_cvConstructExtVD(pVoronoiDiagram))
        return 0;
    _cvConstructIntVD(pVoronoiDiagram);
    _cvFindNearestSite(pVoronoiDiagram);

    float dx,dy;
    int result;
    for(pCvVoronoiHole pHole = pVoronoiDiagram->top_hole;\
        pHole != NULL; pHole = pHole->next_hole)
    {
        if(pHole->error)
            continue;
        dx = pHole->node->node.x - pHole->site_top->node1->node.x;
        dy = pHole->node->node.y - pHole->site_top->node1->node.y;

        if(fabs(dy) < 0.01 && dx < 0)
            pHole->site_opposite = pHole->site_nearest;
        else
        {
            if(dy > 0)
                result = _cvFindOppositSiteCCW(pHole,pVoronoiDiagram);
            else
                result = _cvFindOppositSiteCW(pHole,pVoronoiDiagram);

            if(!result)
            {
                pHole->error = true;
                continue;
            }
        }

        if(!_cvMergeVD(pHole,pVoronoiDiagram))
            return 0;
    }
    return 1;
}//end of _cvConstructSkeleton

static void _cvConstructSiteTree(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    CvSeq* CurSeq = pVoronoiDiagram->SiteSeq;
    for(pCvVoronoiHole pHole = pVoronoiDiagram->top_hole;\
        pHole != NULL; pHole = pHole->next_hole)
    {
        if(pHole->error)
            continue;
        if(CurSeq == pVoronoiDiagram->SiteSeq)
        {
            CurSeq->v_next = pHole->SiteSeq;
            pHole->SiteSeq->v_prev = CurSeq;
        }
        else
        {
            CurSeq->h_next = pHole->SiteSeq;
            pHole->SiteSeq->h_prev = CurSeq;
            pHole->SiteSeq->v_prev = pVoronoiDiagram->SiteSeq;
        }
        CurSeq = pHole->SiteSeq;
    }
    CurSeq->h_next = NULL;
}//end of _cvConstructSiteTree

static void _cvRandomModification(CvVoronoiDiagramInt* pVoronoiDiagram, int begin, int end, float shift)
{
    CvSeqReader Reader;
    pCvVoronoiNode pNode;
    const float rnd_const = shift/RAND_MAX;
    int i;

    cvStartReadSeq(pVoronoiDiagram->NodeSeq, &Reader,0);
    for( i = begin; i < end; i++)
    {
        pNode = (pCvVoronoiNode)Reader.ptr;
        pNode->node.x = (float)cvFloor(pNode->node.x) + rand()*rnd_const;
        pNode->node.y = (float)cvFloor(pNode->node.y) + rand()*rnd_const;
        CV_NEXT_SEQ_ELEM( sizeof(CvVoronoiNodeInt), Reader );
    }

    for(pCvVoronoiHole pHole = pVoronoiDiagram->top_hole;\
        pHole != NULL;\
        pHole = pHole->next_hole)
    {
        pHole->site_top->node1->node.x = (float)cvFloor(pHole->site_top->node1->node.x);
    }

}//end of _cvRandomModification

static void _cvReleaseVoronoiStorage(CvVoronoiStorageInt* pVoronoiStorage, int group1, int group2)
{
    //if group1 = 1 then SiteSeq, NodeSeq, EdgeSeq released
    //if group2 = 1 then DirectionSeq, ParabolaSeq, ChainSeq, HoleSeq released
    if(group1 == 1)
    {
        if(pVoronoiStorage->SiteStorage!=NULL)
            cvReleaseMemStorage(&pVoronoiStorage->SiteStorage);
        if(pVoronoiStorage->EdgeStorage!=NULL)
            cvReleaseMemStorage(&pVoronoiStorage->EdgeStorage);
        if(pVoronoiStorage->NodeStorage!=NULL)
            cvReleaseMemStorage(&pVoronoiStorage->NodeStorage);
    }
    if(group2 == 1)
    {
        if(pVoronoiStorage->ParabolaStorage!=NULL)
            cvReleaseMemStorage(&pVoronoiStorage->ParabolaStorage);
        if(pVoronoiStorage->ChainStorage!=NULL)
            cvReleaseMemStorage(&pVoronoiStorage->ChainStorage);
        if(pVoronoiStorage->DirectionStorage!=NULL)
            cvReleaseMemStorage(&pVoronoiStorage->DirectionStorage);
        if(pVoronoiStorage->HoleStorage != NULL)
            cvReleaseMemStorage(&pVoronoiStorage->HoleStorage);
    }
}// end of _cvReleaseVoronoiStorage

static int _cvConvert(CvVoronoiDiagram2D* VoronoiDiagram,
                       CvVoronoiDiagramInt VoronoiDiagramInt,
                       CvSet* &SiteSeq,
                       CvSeqWriter &NodeWriter,
                       CvSeqWriter &EdgeWriter,
                       CvMemStorage* VoronoiStorage,
                       int contour_orientation)
{
    if(contour_orientation == 1)
        return _cvConvertSameOrientation(VoronoiDiagram,VoronoiDiagramInt,SiteSeq,NodeWriter,
                                        EdgeWriter,VoronoiStorage);
    else
        return _cvConvertChangeOrientation(VoronoiDiagram,VoronoiDiagramInt,SiteSeq,NodeWriter,
                                        EdgeWriter,VoronoiStorage);
}// end of _cvConvert

static int _cvConvertSameOrientation(CvVoronoiDiagram2D* VoronoiDiagram,
                                      CvVoronoiDiagramInt VoronoiDiagramInt,
                                      CvSet* &NewSiteSeqPrev,
                                      CvSeqWriter &NodeWriter,
                                      CvSeqWriter &EdgeWriter,
                                      CvMemStorage* VoronoiStorage)
{
    CvSeq* SiteSeq = VoronoiDiagramInt.SiteSeq;
    CvSeq* EdgeSeq = VoronoiDiagramInt.EdgeSeq;
    CvSeq* NodeSeq = VoronoiDiagramInt.NodeSeq;


    CvMemStorage *NewSiteStorage = cvCreateChildMemStorage(VoronoiStorage);
    CvSet *NewSiteSeq = NULL,*CurrNewSiteSeq = NULL, *PrevNewSiteSeq = NULL;;
    CvSeqWriter SiteWriter;

    CvVoronoiSite2D NewSite = {{0,0},{0,0},{0,0}},NewSite_prev = {{0,0},{0,0},{0,0}};
    CvVoronoiSite2D *pNewSite, *pNewSite_prev = &NewSite_prev;
    pCvVoronoiSite pSite,pFirstSite;

    CvVoronoiEdge2D NewEdge = {{0,0},{0,0},{0,0,0,0}};
    CvVoronoiEdge2D *pNewEdge1, *pNewEdge2;
    pCvVoronoiEdge pEdge;

    CvVoronoiNode2D* pNode1, *pNode2;
    CvVoronoiNode2D Node;
    Node.next_free = NULL;

    for(CvSeq* CurrSiteSeq = SiteSeq;\
        CurrSiteSeq != NULL;\
        CurrSiteSeq = NEXT_SEQ(CurrSiteSeq,SiteSeq))
    {
        CurrNewSiteSeq = cvCreateSet(0,sizeof(CvSet),sizeof(CvVoronoiSite2D), NewSiteStorage);
        if(!NewSiteSeq)
            NewSiteSeq = PrevNewSiteSeq = CurrNewSiteSeq;
        else if(PrevNewSiteSeq->v_prev == NULL)
        {
            PrevNewSiteSeq->v_next = (CvSeq*)CurrNewSiteSeq;
            CurrNewSiteSeq->v_prev = (CvSeq*)PrevNewSiteSeq;
        }
        else
        {
            PrevNewSiteSeq->h_next = (CvSeq*)CurrNewSiteSeq;
            CurrNewSiteSeq->h_prev = (CvSeq*)PrevNewSiteSeq;
            CurrNewSiteSeq->v_prev = (CvSeq*)PrevNewSiteSeq->v_prev;
        }
        PrevNewSiteSeq = CurrNewSiteSeq;

        pSite = pFirstSite = (pCvVoronoiSite)cvGetSeqElem(CurrSiteSeq, 0);
        while(pSite->prev_site->node1 == pSite->prev_site->node2)\
            pSite = pSite->next_site;
        pFirstSite = pSite;

        pNewSite_prev = &NewSite_prev;
        cvStartAppendToSeq((CvSeq*)CurrNewSiteSeq, &SiteWriter);
        do
        {
            pNewSite = _cvWriteSeqElem(&NewSite,SiteWriter);
            pNewSite->next[0] = pNewSite_prev;
            pNewSite_prev->next[1] = pNewSite;
            pEdge = pSite->edge1;
            if(!pEdge || !pEdge->node1 || !pEdge->node2)
                return 0;

            if(pEdge->site == NULL)
            {
                pNewEdge1 = (CvVoronoiEdge2D*)pEdge->twin_edge;
                pNewEdge1->site[1] = pNewSite;
                pNewSite->node[0] = pNewEdge1->node[0];
            }
            else
            {
                pNewEdge1 = _cvWriteSeqElem(&NewEdge,EdgeWriter);
                pNewEdge1->site[0] = pNewSite;

                pNode1 = _cvWriteSeqElem(&Node,NodeWriter);
                pNode2 = _cvWriteSeqElem(&Node,NodeWriter);
                pNode1->pt.x = pEdge->node1->node.x;
                pNode1->pt.y = pEdge->node1->node.y;
                pNode1->radius = pEdge->node1->radius;
                pNode2->pt.x = pEdge->node2->node.x;
                pNode2->pt.y = pEdge->node2->node.y;
                pNode2->radius = pEdge->node2->radius;
                pNewEdge1->node[0] = pNode1;
                pNewEdge1->node[1] = pNode2;

                pNewSite->node[0] = pNewEdge1->node[1];

                if(!pNewEdge1->node[0] || !pNewEdge1->node[1])
                    return 0;
                pEdge->twin_edge->site = NULL;
                pEdge->twin_edge->twin_edge = (pCvVoronoiEdge)pNewEdge1;
            }
            pNewSite->edge[1] = pNewEdge1;
            pEdge = pEdge->prev_edge;
            while((pEdge != NULL && CurrSiteSeq->total>1) ||
                  (pEdge != pSite->edge2 && CurrSiteSeq->total == 1))
            {
                if(pEdge->site == NULL)
                {
                    pNewEdge2 = (CvVoronoiEdge2D*)pEdge->twin_edge;
                    pNewEdge2->site[1] = pNewSite;
                    if(CV_VORONOIEDGE2D_BEGINNODE(pNewEdge1,pNewSite) != pNewEdge2->node[0])
                    {
                        cvFlushSeqWriter(&NodeWriter);
//                      cvSetRemove((CvSet*)VoronoiDiagram,VoronoiDiagram->total-1);
                        pNewEdge1->node[0] = pNewEdge2->node[0];
                    }
                }
                else
                {
                    pNewEdge2 = _cvWriteSeqElem(&NewEdge,EdgeWriter);
                    pNewEdge2->site[0] = pNewSite;

                    pNode1 = _cvWriteSeqElem(&Node,NodeWriter);
                    pNode1->pt.x = pEdge->node1->node.x;
                    pNode1->pt.y = pEdge->node1->node.y;
                    pNode1->radius = pEdge->node1->radius;
                    pNewEdge2->node[0] = pNode1;

                    if(pNewEdge1->site[0] == pNewSite)
                        pNewEdge2->node[1] = pNewEdge1->node[0];
                    else
                        pNewEdge2->node[1] = pNewEdge1->node[1];

                    if(!pNewEdge1->node[0] || !pNewEdge1->node[1])
                        return 0;
                    pEdge->twin_edge->site = NULL;
                    pEdge->twin_edge->twin_edge = (pCvVoronoiEdge)pNewEdge2;
                }
                if(pNewEdge1->site[0] == pNewSite)
                    pNewEdge1->next[2] = pNewEdge2;
                else
                    pNewEdge1->next[3] = pNewEdge2;

                if(pNewEdge2->site[0] == pNewSite)
                    pNewEdge2->next[0] = pNewEdge1;
                else
                    pNewEdge2->next[1] = pNewEdge1;

                pEdge = pEdge->prev_edge;
                pNewEdge1 = pNewEdge2;
            }
            pNewSite->edge[0] = pNewEdge1;
            pNewSite->node[1] = pNewEdge1->node[0];

            if(pSite->node1 == pSite->node2 && pSite != pSite->next_site && pNewEdge1->node[0] != pNewEdge1->node[1])
            {
                cvFlushSeqWriter(&NodeWriter);
//              cvSetRemove((CvSet*)VoronoiDiagram,VoronoiDiagram->total-1);
                pNewSite->node[1] = pNewEdge1->node[0] = pNewSite->node[0];
            }

            pNewSite_prev = pNewSite;
            pSite = pSite->next_site;
        }while(pSite != pFirstSite);
        pNewSite->node[1] = pNewEdge1->node[1];
        if(pSite == pSite->next_site)
        {
            Node.pt.x = pSite->node1->node.x;
            Node.pt.y = pSite->node1->node.y;
            Node.radius = 0;
            pNewSite->node[0] = pNewSite->node[1] = _cvWriteSeqElem(&Node,NodeWriter);
        }

        cvEndWriteSeq(&SiteWriter);
        pNewSite = (CvVoronoiSite2D*)cvGetSetElem(CurrNewSiteSeq, 0);
        pNewSite->next[0] = pNewSite_prev;
        pNewSite_prev->next[1] = pNewSite;
    }

    cvReleaseMemStorage(&(SiteSeq->storage));
    cvReleaseMemStorage(&(EdgeSeq->storage));
    cvReleaseMemStorage(&(NodeSeq->storage));

    if(NewSiteSeqPrev == NULL)
        VoronoiDiagram->sites = NewSiteSeq;
    else
    {
        NewSiteSeqPrev->h_next = (CvSeq*)NewSiteSeq;
        NewSiteSeq->h_prev = (CvSeq*)NewSiteSeqPrev;
    }

    NewSiteSeqPrev = NewSiteSeq;
    return 1;
}//end of _cvConvertSameOrientation

static int _cvConvertChangeOrientation(CvVoronoiDiagram2D* VoronoiDiagram,
                                        CvVoronoiDiagramInt VoronoiDiagramInt,
                                        CvSet* &NewSiteSeqPrev,
                                        CvSeqWriter &NodeWriter,
                                        CvSeqWriter &EdgeWriter,
                                        CvMemStorage* VoronoiStorage)
{
    // pNewSite->edge[1] = pSite->edge2
    // pNewSite->edge[0] = pSite->edge1

    CvSeq* SiteSeq = VoronoiDiagramInt.SiteSeq;
    CvSeq* EdgeSeq = VoronoiDiagramInt.EdgeSeq;
    CvSeq* NodeSeq = VoronoiDiagramInt.NodeSeq;


    CvMemStorage *NewSiteStorage = cvCreateChildMemStorage(VoronoiStorage);
    CvSet *NewSiteSeq = NULL,*CurrNewSiteSeq = NULL, *PrevNewSiteSeq = NULL;;
    CvSeqWriter SiteWriter;

    CvVoronoiSite2D NewSite = {{0,0},{0,0},{0,0}},NewSite_prev = {{0,0},{0,0},{0,0}};
    CvVoronoiSite2D *pNewSite, *pNewSite_prev = &NewSite_prev;
    pCvVoronoiSite pSite,pFirstSite;

    CvVoronoiEdge2D NewEdge = {{0,0},{0,0},{0,0,0,0}};
    CvVoronoiEdge2D *pNewEdge1, *pNewEdge2;
    pCvVoronoiEdge pEdge;

    CvVoronoiNode2D* pNode1, *pNode2;
    CvVoronoiNode2D Node;
    Node.next_free = NULL;

    for(CvSeq* CurrSiteSeq = SiteSeq;\
        CurrSiteSeq != NULL;\
        CurrSiteSeq = NEXT_SEQ(CurrSiteSeq,SiteSeq))
    {
        CurrNewSiteSeq = cvCreateSet(0,sizeof(CvSet),sizeof(CvVoronoiSite2D), NewSiteStorage);
        if(!NewSiteSeq)
            NewSiteSeq = PrevNewSiteSeq = CurrNewSiteSeq;
        else if(PrevNewSiteSeq->v_prev == NULL)
        {
            PrevNewSiteSeq->v_next = (CvSeq*)CurrNewSiteSeq;
            CurrNewSiteSeq->v_prev = (CvSeq*)PrevNewSiteSeq;
        }
        else
        {
            PrevNewSiteSeq->h_next = (CvSeq*)CurrNewSiteSeq;
            CurrNewSiteSeq->h_prev = (CvSeq*)PrevNewSiteSeq;
            CurrNewSiteSeq->v_prev = (CvSeq*)PrevNewSiteSeq->v_prev;
        }
        PrevNewSiteSeq = CurrNewSiteSeq;

        pSite = (pCvVoronoiSite)cvGetSeqElem(CurrSiteSeq, 0);
        while(pSite->next_site->node1 == pSite->next_site->node2)\
            pSite = pSite->next_site;
        pFirstSite = pSite;

        pNewSite_prev = &NewSite_prev;
        cvStartAppendToSeq((CvSeq*)CurrNewSiteSeq, &SiteWriter);
        do
        {
            pNewSite = _cvWriteSeqElem(&NewSite,SiteWriter);
            pNewSite->next[0] = pNewSite_prev;
            pNewSite_prev->next[1] = pNewSite;

            pEdge = pSite->edge2;
            if(!pEdge || !pEdge->node1 || !pEdge->node2)
                return 0;

            if(pEdge->site == NULL)
            {
                pNewEdge1 = (CvVoronoiEdge2D*)pEdge->twin_edge;
                pNewEdge1->site[1] = pNewSite;
                pNewSite->node[0] = pNewEdge1->node[0];
            }
            else
            {
                pNewEdge1 = _cvWriteSeqElem(&NewEdge,EdgeWriter);
                pNewEdge1->site[0] = pNewSite;

                pNode1 = _cvWriteSeqElem(&Node,NodeWriter);
                pNode2 = _cvWriteSeqElem(&Node,NodeWriter);
                pNode1->pt.x = pEdge->node1->node.x;
                pNode1->pt.y = pEdge->node1->node.y;
                pNode1->radius = pEdge->node1->radius;
                pNode2->pt.x = pEdge->node2->node.x;
                pNode2->pt.y = pEdge->node2->node.y;
                pNode2->radius = pEdge->node2->radius;
                pNewEdge1->node[0] = pNode2;
                pNewEdge1->node[1] = pNode1;

                pNewSite->node[0] = pNewEdge1->node[1];

                if(!pNewEdge1->node[0] || !pNewEdge1->node[1])
                    return 0;
                pEdge->twin_edge->site = NULL;
                pEdge->twin_edge->twin_edge = (pCvVoronoiEdge)pNewEdge1;
            }
            pNewSite->edge[1] = pNewEdge1;


            pEdge = pEdge->next_edge;
            while((pEdge != NULL && CurrSiteSeq->total>1) ||
                  (pEdge != pSite->edge1 && CurrSiteSeq->total == 1))
            {
                if(pEdge->site == NULL)
                {
                    pNewEdge2 = (CvVoronoiEdge2D*)pEdge->twin_edge;
                    pNewEdge2->site[1] = pNewSite;
                    if(CV_VORONOIEDGE2D_BEGINNODE(pNewEdge1,pNewSite) != pNewEdge2->node[0])
                    {
                        cvFlushSeqWriter(&NodeWriter);
//                      cvSetRemove((CvSet*)VoronoiDiagram,VoronoiDiagram->total-1);
                        pNewEdge1->node[0] = pNewEdge2->node[0];
                    }
                }
                else
                {
                    pNewEdge2 = _cvWriteSeqElem(&NewEdge,EdgeWriter);
                    pNewEdge2->site[0] = pNewSite;

                    pNode2 = _cvWriteSeqElem(&Node,NodeWriter);
                    pNode2->pt.x = pEdge->node2->node.x;
                    pNode2->pt.y = pEdge->node2->node.y;
                    pNode2->radius = pEdge->node2->radius;
                    pNewEdge2->node[0] = pNode2;

                    if(pNewEdge1->site[0] == pNewSite)
                        pNewEdge2->node[1] = pNewEdge1->node[0];
                    else
                        pNewEdge2->node[1] = pNewEdge1->node[1];

                    if(!pNewEdge1->node[0] || !pNewEdge1->node[1])
                        return 0;
                    pEdge->twin_edge->site = NULL;
                    pEdge->twin_edge->twin_edge = (pCvVoronoiEdge)pNewEdge2;
                }
                if(pNewEdge1->site[0] == pNewSite)
                    pNewEdge1->next[2] = pNewEdge2;
                else
                    pNewEdge1->next[3] = pNewEdge2;

                if(pNewEdge2->site[0] == pNewSite)
                    pNewEdge2->next[0] = pNewEdge1;
                else
                    pNewEdge2->next[1] = pNewEdge1;

                pEdge = pEdge->next_edge;
                pNewEdge1 = pNewEdge2;
            }
            pNewSite->edge[0] = pNewEdge1;
            pNewSite->node[1] = pNewEdge1->node[0];

            if(pSite->node1 == pSite->node2 && pSite != pSite->next_site && pNewEdge1->node[0] != pNewEdge1->node[1])
            {
                cvFlushSeqWriter(&NodeWriter);
//              cvSetRemove((CvSet*)VoronoiDiagram,VoronoiDiagram->total-1);
                pNewSite->node[1] = pNewEdge1->node[0] = pNewSite->node[0];
            }

            pNewSite_prev = pNewSite;
            pSite = pSite->prev_site;
        }while(pSite != pFirstSite);
        pNewSite->node[1] = pNewEdge1->node[1];
        if(pSite == pSite->next_site)
        {
            Node.pt.x = pSite->node1->node.x;
            Node.pt.y = pSite->node1->node.y;
            Node.radius = 0;
            pNewSite->node[0] = pNewSite->node[1] = _cvWriteSeqElem(&Node,NodeWriter);
        }

        cvEndWriteSeq(&SiteWriter);
        pNewSite = (CvVoronoiSite2D*)cvGetSetElem(CurrNewSiteSeq, 0);
        pNewSite->next[0] = pNewSite_prev;
        pNewSite_prev->next[1] = pNewSite;
    }

    cvReleaseMemStorage(&(SiteSeq->storage));
    cvReleaseMemStorage(&(EdgeSeq->storage));
    cvReleaseMemStorage(&(NodeSeq->storage));

    if(NewSiteSeqPrev == NULL)
        VoronoiDiagram->sites = NewSiteSeq;
    else
    {
        NewSiteSeqPrev->h_next = (CvSeq*)NewSiteSeq;
        NewSiteSeq->h_prev = (CvSeq*)NewSiteSeqPrev;
    }
    NewSiteSeqPrev = NewSiteSeq;
    return 1;
}//end of _cvConvert

template<class T>
int _cvConstructExtSites(CvVoronoiDiagramInt* pVoronoiDiagram,
                         CvSeq* ContourSeq,
                         int orientation,
                         T /*type*/)
{
    const double angl_eps = 0.03;
    CvSeq* SiteSeq = pVoronoiDiagram->SiteSeq;
    CvSeq* NodeSeq = pVoronoiDiagram->NodeSeq;
    //CvSeq* DirectionSeq = pVoronoiDiagram->DirectionSeq;
    CvPointFloat Vertex1,Vertex2,Vertex3;
    CvLeePoint<T> VertexT1,VertexT2,VertexT3;

    CvSeqReader ContourReader;
    CvVoronoiSiteInt Site = {NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    CvVoronoiSiteInt SiteTemp = {NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    CvVoronoiNodeInt Node;
    pCvVoronoiNode pNode1,pNode2;
    pCvVoronoiSite pSite = &SiteTemp,pSite_prev = &SiteTemp;
    pCvVoronoiSite pReflexSite = NULL;
    int NReflexSite = 0;

    float delta_x_rc, delta_x_cl, delta_y_rc, delta_y_cl;
    float norm_cl,norm_rc, mult_cl_rc;
    float _sin, _cos;
    int i;

    if(orientation == 1)
    {
        cvStartReadSeq(ContourSeq, &ContourReader,0);
        CV_READ_SEQ_ELEM(VertexT1,ContourReader);
        CV_READ_SEQ_ELEM(VertexT2,ContourReader);
    }
    else
    {
        cvStartReadSeq(ContourSeq, &ContourReader,1);
        CV_REV_READ_SEQ_ELEM(VertexT1,ContourReader);
        CV_REV_READ_SEQ_ELEM(VertexT2,ContourReader);
    }

    Vertex1.x = (float)VertexT1.x;
    Vertex1.y = (float)VertexT1.y;
    Vertex2.x = (float)VertexT2.x;
    Vertex2.y = (float)VertexT2.y;

    _cvInitVoronoiNode(&Node, &Vertex2);
    pNode1 = _cvSeqPush(NodeSeq, &Node);

    delta_x_cl = Vertex2.x - Vertex1.x;
    delta_y_cl = Vertex2.y - Vertex1.y;
    norm_cl = (float)sqrt((double)delta_x_cl*delta_x_cl + delta_y_cl*delta_y_cl);

    for( i = 0;i<ContourSeq->total;i++)
    {
        if(orientation == 1)
        {
            CV_READ_SEQ_ELEM(VertexT3,ContourReader);
        }
        else
        {
            CV_REV_READ_SEQ_ELEM(VertexT3,ContourReader);
        }

        Vertex3.x = (float)VertexT3.x;
        Vertex3.y = (float)VertexT3.y;

        _cvInitVoronoiNode(&Node, &Vertex3);
        pNode2 = _cvSeqPush(NodeSeq, &Node);

        delta_x_rc = Vertex3.x - Vertex2.x;
        delta_y_rc = Vertex3.y - Vertex2.y;
        norm_rc = (float)sqrt((double)delta_x_rc*delta_x_rc + delta_y_rc*delta_y_rc);
        if(norm_rc==0)
            continue;

        mult_cl_rc = norm_cl*norm_rc;
        _sin = (delta_y_rc* delta_x_cl - delta_x_rc* delta_y_cl)/mult_cl_rc;
        _cos = -(delta_x_cl*delta_x_rc + delta_y_cl*delta_y_rc)/mult_cl_rc;

        if((_sin > angl_eps) || (_sin > 0 && _cos > 0))
        {
            pSite = _cvSeqPush(SiteSeq, &Site);
            _cvInitVoronoiSite(pSite,pNode1,pNode2,pSite_prev);
            pSite_prev->next_site = pSite;
        }
        else if((_sin < -angl_eps) || (_sin < 0 && _cos > 0))
        {
            pSite = _cvSeqPush(SiteSeq, &Site);
            _cvInitVoronoiSite(pSite,pNode1,pNode1,pSite_prev);
            pReflexSite = pSite;
            NReflexSite++;
            pSite_prev->next_site = pSite;

            pSite_prev = pSite;
            pSite = _cvSeqPush(SiteSeq, &Site);
            _cvInitVoronoiSite(pSite,pNode1,pNode2,pSite_prev);
            pSite_prev->next_site = pSite;
        }
        else
        {
            Vertex2 = Vertex3;
            delta_y_rc = delta_y_cl + delta_y_rc;
            delta_x_rc = delta_x_cl + delta_x_rc;
            pSite->node2 = pNode2;

            norm_rc = (float)sqrt((double)delta_y_rc*delta_y_rc + delta_x_rc*delta_x_rc);
        }
        Vertex2=Vertex3;
        delta_y_cl= delta_y_rc;
        delta_x_cl= delta_x_rc;
        norm_cl = norm_rc;
        pSite_prev = pSite;
        pNode1 = pNode2;
    }

    if(SiteTemp.next_site==NULL)
        return 0;

    if(ContourSeq->total - NReflexSite<2)
        return 0;

    if(SiteSeq->total<3)
        return 0;

    pSite->node2 = SiteTemp.next_site->node1;
    pSite->next_site = SiteTemp.next_site;
    SiteTemp.next_site->prev_site = pSite;

    i = 0;
    if(pReflexSite!=NULL)
        for(i=0; i<SiteSeq->total; i++)
        {
            if(pReflexSite->next_site->next_site->node1 !=
              pReflexSite->next_site->next_site->node2)
              break;
            else
                pReflexSite = pReflexSite->next_site->next_site;
        }
    pVoronoiDiagram->reflex_site = pReflexSite;
    return (i<SiteSeq->total);
}//end of _cvConstructExtSites

template<class T>
int _cvConstructIntSites(CvVoronoiDiagramInt* pVoronoiDiagram,
                                 CvSeq* CurrSiteSeq,
                                 CvSeq* CurrContourSeq,
                                 pCvVoronoiSite &pTopSite,
                                 int orientation,
                                 T /*type*/)
{
    const double angl_eps = 0.03;
    float min_x = (float)999999999;
    CvSeq* SiteSeq = CurrSiteSeq;
    CvSeq* NodeSeq = pVoronoiDiagram->NodeSeq;
    //CvSeq* DirectionSeq = pVoronoiDiagram->DirectionSeq;
    CvPointFloat Vertex1,Vertex2,Vertex3;
    CvLeePoint<T> VertexT1,VertexT2,VertexT3;

    CvSeqReader ContourReader;
    CvVoronoiSiteInt Site = {NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    CvVoronoiSiteInt SiteTemp = {NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    CvVoronoiNodeInt Node;
    pCvVoronoiNode pNode1,pNode2;
    pCvVoronoiSite pSite = &SiteTemp,pSite_prev = &SiteTemp;
    pTopSite = NULL;
    int NReflexSite = 0;

    if(CurrContourSeq->total == 1)
    {
        cvStartReadSeq(CurrContourSeq, &ContourReader,0);
        CV_READ_SEQ_ELEM(VertexT1,ContourReader);
        Vertex1.x = (float)VertexT1.x;
        Vertex1.y = (float)VertexT1.y;

        _cvInitVoronoiNode(&Node, &Vertex1);
        pNode1 = _cvSeqPush(NodeSeq, &Node);
        pTopSite = pSite = _cvSeqPush(SiteSeq, &Site);
        _cvInitVoronoiSite(pSite,pNode1,pNode1,pSite);
        pSite->next_site = pSite;
        return 1;
    }

    float delta_x_rc, delta_x_cl, delta_y_rc, delta_y_cl;
    float norm_cl,norm_rc, mult_cl_rc;
    float _sin, _cos;
    int i;

    if(orientation == 1)
    {
        cvStartReadSeq(CurrContourSeq, &ContourReader,0);
        CV_READ_SEQ_ELEM(VertexT1,ContourReader);
        CV_READ_SEQ_ELEM(VertexT2,ContourReader);
    }
    else
    {
        cvStartReadSeq(CurrContourSeq, &ContourReader,1);
        CV_REV_READ_SEQ_ELEM(VertexT1,ContourReader);
        CV_REV_READ_SEQ_ELEM(VertexT2,ContourReader);
    }

    Vertex1.x = (float)VertexT1.x;
    Vertex1.y = (float)VertexT1.y;
    Vertex2.x = (float)VertexT2.x;
    Vertex2.y = (float)VertexT2.y;

    _cvInitVoronoiNode(&Node, &Vertex2);
    pNode1 = _cvSeqPush(NodeSeq, &Node);

    delta_x_cl = Vertex2.x - Vertex1.x;
    delta_y_cl = Vertex2.y - Vertex1.y;
    norm_cl = (float)sqrt((double)delta_x_cl*delta_x_cl + delta_y_cl*delta_y_cl);
    for( i = 0;i<CurrContourSeq->total;i++)
    {
        if(orientation == 1)
        {
            CV_READ_SEQ_ELEM(VertexT3,ContourReader);
        }
        else
        {
            CV_REV_READ_SEQ_ELEM(VertexT3,ContourReader);
        }
        Vertex3.x = (float)VertexT3.x;
        Vertex3.y = (float)VertexT3.y;

        _cvInitVoronoiNode(&Node, &Vertex3);
        pNode2 = _cvSeqPush(NodeSeq, &Node);

        delta_x_rc = Vertex3.x - Vertex2.x;
        delta_y_rc = Vertex3.y - Vertex2.y;
        norm_rc = (float)sqrt((double)delta_x_rc*delta_x_rc + delta_y_rc*delta_y_rc);
        if(norm_rc==0)
            continue;

        mult_cl_rc = norm_cl*norm_rc;
        _sin = (delta_y_rc* delta_x_cl - delta_x_rc* delta_y_cl)/mult_cl_rc;
        _cos = -(delta_x_cl*delta_x_rc + delta_y_cl*delta_y_rc)/mult_cl_rc;
        if((_sin > angl_eps) || (_sin > 0 && _cos > 0))
        {
            pSite = _cvSeqPush(SiteSeq, &Site);
            _cvInitVoronoiSite(pSite,pNode1,pNode2,pSite_prev);
            pSite_prev->next_site = pSite;
        }
        else if((_sin < -angl_eps) || (_sin < 0 && _cos > 0) || (_sin == 0 && CurrContourSeq->total == 2))
        {
            pSite = _cvSeqPush(SiteSeq, &Site);
            _cvInitVoronoiSite(pSite,pNode1,pNode1,pSite_prev);
            if(pNode1->node.x < min_x)
            {
                min_x = pNode1->node.x;
                pTopSite = pSite;
            }
            NReflexSite++;
            pSite_prev->next_site = pSite;

            pSite_prev = pSite;
            pSite = _cvSeqPush(SiteSeq, &Site);
            _cvInitVoronoiSite(pSite,pNode1,pNode2,pSite_prev);
            pSite_prev->next_site = pSite;
        }
        else
        {
            Vertex2 = Vertex3;
            delta_y_rc = delta_y_cl + delta_y_rc;
            delta_x_rc = delta_x_cl + delta_x_rc;
            norm_rc = (float)sqrt((double)delta_y_rc*delta_y_rc + delta_x_rc*delta_x_rc);
            pSite->node2 = pNode2;
        }

        Vertex1=Vertex2;
        Vertex2=Vertex3;
        delta_y_cl= delta_y_rc;
        delta_x_cl= delta_x_rc;
        norm_cl = norm_rc;
        pSite_prev = pSite;
        pNode1 = pNode2;
    }

    if(SiteTemp.next_site==NULL)
        return 0;

    if((NReflexSite < 3 && CurrContourSeq->total > 2) || NReflexSite < 2)
        return 0;

    pSite->node2 = SiteTemp.next_site->node1;
    pSite->next_site = SiteTemp.next_site;
    SiteTemp.next_site->prev_site = pSite;

    return 1;
}//end of _cvConstructIntSites

static int _cvConstructExtChains(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    CvSeq* SiteSeq = pVoronoiDiagram->SiteSeq;
    CvSeq* ChainSeq = pVoronoiDiagram->ChainSeq;

    CvVoronoiChainInt Chain;
    pCvVoronoiChain pChain,pChainFirst;
    pCvVoronoiSite pSite, pSite_prev, pSiteFirst,pReflexSite = pVoronoiDiagram->reflex_site;

    if(pReflexSite==NULL)
        pSite = pSiteFirst = (pCvVoronoiSite)cvGetSeqElem(SiteSeq, 0);
    else
    {
        while(pReflexSite->next_site->next_site->node1==
              pReflexSite->next_site->next_site->node2)
            pReflexSite = pReflexSite->next_site->next_site;

        pSite = pSiteFirst = pReflexSite->next_site;
    }

    Chain.last_site = pSite;
    _cvConstructEdges(pSite,pVoronoiDiagram);
    pSite_prev = pSite;
    pSite = pSite->prev_site;
    do
    {
        if(pSite->node1!=pSite->node2)
        {
            Chain.first_site = pSite_prev;
            pChain = _cvSeqPushFront(ChainSeq,&Chain);

            _cvConstructEdges(pSite,pVoronoiDiagram);
            Chain.last_site = pSite;
            Chain.next_chain = pChain;
        }
        else
        {
            pSite=pSite->prev_site;
            _cvConstructEdges(pSite,pVoronoiDiagram);
            _cvConstructEdges(pSite->next_site,pVoronoiDiagram);
        }
        pSite_prev = pSite;
        pSite = pSite->prev_site;
    }while(pSite!=pSiteFirst);

    Chain.first_site = pSite_prev;
    pChain = _cvSeqPushFront(ChainSeq,&Chain);
    pChainFirst = (pCvVoronoiChain)cvGetSeqElem(ChainSeq,ChainSeq->total - 1);
    pChainFirst->next_chain = pChain;
    if(ChainSeq->total < 3)
        return 0;
    else
        return 1;
}// end of _cvConstructExtChains

static void _cvConstructIntChains(CvVoronoiDiagramInt* pVoronoiDiagram,
                                   CvSeq* CurrChainSeq,
                                   CvSeq* CurrSiteSeq,
                                   pCvVoronoiSite pTopSite)
{
    CvSeq* ChainSeq = CurrChainSeq;

    if(CurrSiteSeq->total == 1)
        return;

    CvVoronoiChainInt Chain = {NULL,NULL,NULL};
    pCvVoronoiChain pChain,pChainFirst;;
    pCvVoronoiSite pSite, pSite_prev, pSiteFirst;
    pSite = pSiteFirst = pTopSite->next_site;

    Chain.last_site = pSite;
    _cvConstructEdges(pSite,pVoronoiDiagram);
    pSite_prev = pSite;
    pSite = pSite->prev_site;
    do
    {
        if(pSite->node1!=pSite->node2)
        {
            Chain.first_site = pSite_prev;
            pChain = _cvSeqPushFront(ChainSeq,&Chain);

            _cvConstructEdges(pSite,pVoronoiDiagram);
            Chain.last_site = pSite;
            Chain.next_chain = pChain;
        }
        else
        {
            pSite=pSite->prev_site;
            if(pSite != pSiteFirst)
                _cvConstructEdges(pSite,pVoronoiDiagram);
            _cvConstructEdges(pSite->next_site,pVoronoiDiagram);
        }
        pSite_prev = pSite;
        pSite = pSite->prev_site;
    }while(pSite!=pSiteFirst && pSite!= pSiteFirst->prev_site);

    if(pSite == pSiteFirst->prev_site && ChainSeq->total == 0)
        return;

    Chain.first_site = pSite_prev;
    if(pSite == pSiteFirst->prev_site)
    {
        pChainFirst = (pCvVoronoiChain)cvGetSeqElem(ChainSeq,ChainSeq->total - 1);
        pChainFirst->last_site = Chain.last_site;
        pChainFirst->next_chain = Chain.next_chain;
        return;
    }
    else
    {
        pChain = _cvSeqPushFront(ChainSeq,&Chain);
        pChainFirst = (pCvVoronoiChain)cvGetSeqElem(ChainSeq,ChainSeq->total - 1);
        pChainFirst->next_chain = pChain;
        return;
    }
}// end of _cvConstructIntChains

CV_INLINE void _cvConstructEdges(pCvVoronoiSite pSite,CvVoronoiDiagramInt* pVoronoiDiagram)
{
    CvSeq* EdgeSeq = pVoronoiDiagram->EdgeSeq;
    CvSeq* DirectionSeq = pVoronoiDiagram->DirectionSeq;
    CvVoronoiEdgeInt Edge = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    pCvVoronoiEdge pEdge1,pEdge2;
    CvDirection EdgeDirection,SiteDirection;
    float site_lengh;

    Edge.site = pSite;
    if(pSite->node1!=pSite->node2)
    {
        SiteDirection.x = pSite->node2->node.x - pSite->node1->node.x;
        SiteDirection.y = pSite->node2->node.y - pSite->node1->node.y;
        site_lengh = (float)sqrt((double)SiteDirection.x*SiteDirection.x + SiteDirection.y * SiteDirection.y);
        SiteDirection.x /= site_lengh;
        SiteDirection.y /= site_lengh;
        EdgeDirection.x = -SiteDirection.y;
        EdgeDirection.y = SiteDirection.x;
        Edge.direction = _cvSeqPush(DirectionSeq,&EdgeDirection);
        pSite->direction = _cvSeqPush(DirectionSeq,&SiteDirection);

        pEdge1 = _cvSeqPush(EdgeSeq,&Edge);
        pEdge2 = _cvSeqPush(EdgeSeq,&Edge);
    }
    else
    {
        pCvVoronoiSite pSite_prev = pSite->prev_site;
        pCvVoronoiSite pSite_next = pSite->next_site;

        pEdge1 = _cvSeqPush(EdgeSeq,&Edge);
        pEdge2 = _cvSeqPush(EdgeSeq,&Edge);

        pEdge2->direction = pSite_next->edge1->direction;
        pEdge2->twin_edge = pSite_next->edge1;
        pSite_next->edge1->twin_edge = pEdge2;

        pEdge1->direction = pSite_prev->edge2->direction;
        pEdge1->twin_edge = pSite_prev->edge2;
        pSite_prev->edge2->twin_edge = pEdge1;
    }

        pEdge2->node1 = pSite->node2;
        pEdge1->node2 = pSite->node1;
        pSite->edge1 = pEdge1;
        pSite->edge2 = pEdge2;
        pEdge2->next_edge = pEdge1;
        pEdge1->prev_edge = pEdge2;
}// end of _cvConstructEdges

static int _cvConstructExtVD(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvVoronoiSite pSite_right = 0,pSite_left = 0;
    pCvVoronoiEdge pEdge_left,pEdge_right;
    pCvVoronoiChain pChain1, pChain2;

    pChain1 = (pCvVoronoiChain)cvGetSeqElem(pVoronoiDiagram->ChainSeq,0);
    do
    {
        pChain2 = pChain1->next_chain;
        if(pChain2->next_chain==pChain1)
        {
            pSite_right = pChain1->first_site;
            pSite_left = pChain2->last_site;
            pEdge_left = pSite_left->edge2->next_edge;
            pEdge_right = pSite_right->edge1->prev_edge;
            pEdge_left->prev_edge = NULL;
            pEdge_right->next_edge = NULL;
            pSite_right->edge1 = NULL;
            pSite_left->edge2 = NULL;
        }

        if(!_cvJoinChains(pChain1,pChain2,pVoronoiDiagram))
            return 0;

        pChain1->last_site = pChain2->last_site;
        pChain1->next_chain = pChain2->next_chain;
        pChain1 = pChain1->next_chain;
    }while(pChain1->next_chain != pChain1);

    pCvVoronoiNode pEndNode = pSite_left->node2;
    if(pSite_right->edge1==NULL)
        return 0;
    else
        pSite_right->edge1->node2 = pEndNode;

    if(pSite_left->edge2==NULL)
        return 0;
    else
        pSite_left->edge2->node1 = pEndNode;

    return 1;
}//end of _cvConstructExtVD

static int _cvJoinChains(pCvVoronoiChain pChain1,
                          pCvVoronoiChain pChain2,
                          CvVoronoiDiagramInt* pVoronoiDiagram)
{
    const double dist_eps = 0.05;
    if(pChain1->first_site == NULL || pChain1->last_site == NULL ||
        pChain2->first_site == NULL || pChain2->last_site == NULL)
        return 0;

    CvVoronoiEdgeInt EdgeNULL = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};

    CvSeq* NodeSeq = pVoronoiDiagram->NodeSeq;
    CvSeq* EdgeSeq = pVoronoiDiagram->EdgeSeq;

    pCvVoronoiSite pSite_left = pChain1->last_site;
    pCvVoronoiSite pSite_right = pChain2->first_site;

    pCvVoronoiEdge pEdge_left = pSite_left->edge2->next_edge;
    pCvVoronoiEdge pEdge_right = pSite_right->edge1->prev_edge;

    pCvVoronoiEdge pEdge_left_cur = pEdge_left;
    pCvVoronoiEdge pEdge_right_cur = pEdge_right;

    pCvVoronoiEdge pEdge_left_prev = NULL;
    pCvVoronoiEdge pEdge_right_next = NULL;

    pCvVoronoiNode pNode_siteleft = pChain1->first_site->node1;
    pCvVoronoiNode pNode_siteright = pChain2->last_site->node2;
    /*CvVoronoiSiteInt Site_left_chain = {pNode_siteleft,pNode_siteleft,NULL,NULL,NULL,NULL};
    CvVoronoiSiteInt Site_right_chain = {pNode_siteright,pNode_siteright,NULL,NULL,NULL,NULL};*/

    pCvVoronoiEdge pEdge1,pEdge2;
    CvPointFloat Point1 = {0,0}, Point2 = {0,0};

    float radius1,radius2,dist1,dist2;
    bool left = true,right = true;

    CvVoronoiNodeInt Node;
    pCvVoronoiNode pNode_begin = pSite_left->node2;

    pEdge1 = pSite_left->edge2;
    pEdge1->node2 = NULL;
    pEdge2 = pSite_right->edge1;
    pEdge2->node1 = NULL;

    for(;;)
    {

        if(left)
            pEdge1->node1 = pNode_begin;
        if(right)
            pEdge2->node2 = pNode_begin;

        pEdge_left = pEdge_left_cur;
        pEdge_right = pEdge_right_cur;

        if(left&&right)
        {
            _cvCalcEdge(pSite_left,pSite_right,pEdge1,pVoronoiDiagram);
            _cvMakeTwinEdge(pEdge2,pEdge1);
            _cvStickEdgeLeftBegin(pEdge1,pEdge_left_prev,pSite_left);
            _cvStickEdgeRightBegin(pEdge2,pEdge_right_next,pSite_right);
        }
        else if(!left)
        {
            _cvCalcEdge(pNode_siteleft,pSite_right,pEdge2,pVoronoiDiagram);
            _cvStickEdgeRightBegin(pEdge2,pEdge_right_next,pSite_right);
        }
        else if(!right)
        {
            _cvCalcEdge(pSite_left,pNode_siteright,pEdge1,pVoronoiDiagram);
            _cvStickEdgeLeftBegin(pEdge1,pEdge_left_prev,pSite_left);
        }

        dist1=dist2=-1;
        radius1 = -1; radius2 = -2;

        while(pEdge_left!=NULL)
        {
            if(pEdge_left->node2 == NULL)
            {
                pEdge_left_cur = pEdge_left = pEdge_left->next_edge;
                if(pEdge_left == NULL)
                    break;
            }

            if(left)
                dist1 = _cvCalcEdgeIntersection(pEdge1, pEdge_left, &Point1,radius1);
            else
                dist1 = _cvCalcEdgeIntersection(pEdge2, pEdge_left, &Point1,radius1);

            if(dist1>=0)
                break;

            pEdge_left = pEdge_left->next_edge;
        }

        while(pEdge_right!=NULL)
        {
            if(pEdge_right->node1 == NULL)
            {
                pEdge_right_cur = pEdge_right = pEdge_right->prev_edge;
                if(pEdge_right == NULL)
                    break;
            }

            if(left)
                dist2 = _cvCalcEdgeIntersection(pEdge1, pEdge_right, &Point2, radius2);
            else
                dist2 = _cvCalcEdgeIntersection(pEdge2, pEdge_right, &Point2, radius2);

            if(dist2>=0)
                break;

            pEdge_right = pEdge_right->prev_edge;
        }

        if(dist1<0&&dist2<0)
        {
            if(left)
            {
                pEdge_left = pSite_left->edge1;
                if(pEdge_left==NULL)
                    _cvStickEdgeLeftEnd(pEdge1,NULL,pSite_left);
                else
                {
                    while(pEdge_left->node1!=NULL
                        &&pEdge_left->node1==pEdge_left->prev_edge->node2)
                    {
                        pEdge_left = pEdge_left->prev_edge;
                        if(pEdge_left==NULL || pEdge_left->prev_edge == NULL)
                            return 0;
                    }
                    _cvStickEdgeLeftEnd(pEdge1,pEdge_left,pSite_left);
                }
            }
            if(right)
            {
                pEdge_right = pSite_right->edge2;
                if(pEdge_right==NULL)
                    _cvStickEdgeRightEnd(pEdge2,NULL,pSite_right);
                else
                {
                    while(pEdge_right->node2!=NULL
                        &&  pEdge_right->node2==pEdge_right->next_edge->node1)
                    {
                        pEdge_right = pEdge_right->next_edge;
                        if(pEdge_right==NULL || pEdge_right->next_edge == NULL )
                            return 0;
                    }
                    _cvStickEdgeRightEnd(pEdge2,pEdge_right,pSite_right);
                }
            }
            return 1;
        }

        if(fabs(dist1 - dist2)<dist_eps)
        {
            pNode_begin = _cvSeqPush(NodeSeq,&Node);
            _cvInitVoronoiNode(pNode_begin, &Point2,radius2);

            pEdge1->node2 = pNode_begin;
            pEdge2->node1 = pNode_begin;

            _cvStickEdgeLeftEnd(pEdge1,pEdge_left,pSite_left);
            _cvTwinNULLLeft(pEdge_left_cur,pEdge_left);

            _cvStickEdgeRightEnd(pEdge2,pEdge_right,pSite_right);
            _cvTwinNULLRight(pEdge_right_cur,pEdge_right);

            if(pEdge_left->twin_edge!=NULL&&pEdge_right->twin_edge!=NULL)
            {
                pEdge_left_prev = pEdge_left->twin_edge;
                if(!pEdge_left_prev)
                    return 0;
                pEdge_left_cur = pEdge_left_prev->next_edge;
                pEdge_right_next = pEdge_right->twin_edge;
                if(!pEdge_right_next)
                    return 0;
                pEdge_right_cur = pEdge_right_next->prev_edge;
                pSite_right = pEdge_right_next->site;
                pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                pSite_left = pEdge_left_prev->site;
                pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                continue;
            }

            if(pEdge_left->twin_edge==NULL&&pEdge_right->twin_edge!=NULL)
            {
                pEdge_right_next = pEdge_right->twin_edge;
                if(!pEdge_right_next)
                    return 0;
                pEdge_right_cur = pEdge_right_next->prev_edge;
                pSite_right = pEdge_right_next->site;
                pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                pEdge_left_cur = NULL;
                left = false;
                continue;
            }

            if(pEdge_left->twin_edge!=NULL&&pEdge_right->twin_edge==NULL)
            {
                pEdge_left_prev = pEdge_left->twin_edge;
                if(!pEdge_left_prev)
                    return 0;
                pEdge_left_cur = pEdge_left_prev->next_edge;
                pSite_left = pEdge_left_prev->site;
                pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                pEdge_right_cur = NULL;
                right = false;
                continue;
            }
            if(pEdge_left->twin_edge==NULL&&pEdge_right->twin_edge==NULL)
                return 1;
        }

        if((dist1<dist2&&dist1>=0)||(dist1>=0&&dist2<0))
        {

            pNode_begin = _cvSeqPush(NodeSeq,&Node);
            _cvInitVoronoiNode(pNode_begin, &Point1,radius1);
            pEdge1->node2 = pNode_begin;
            _cvTwinNULLLeft(pEdge_left_cur,pEdge_left);
            _cvStickEdgeLeftEnd(pEdge1,pEdge_left,pSite_left);
            if(right)
            {
                pEdge2->node1 = pNode_begin;
                pEdge_right_next = pEdge2;
                pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                if(pEdge_left->twin_edge!=NULL)
                {
                    pEdge_left_prev = pEdge_left->twin_edge;
                    if(!pEdge_left_prev)
                        return 0;
                    pEdge_left_cur = pEdge_left_prev->next_edge;
                    pSite_left = pEdge_left_prev->site;
                    pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                    continue;
                }
                else
                {
                    pEdge_left_cur = NULL;
                    left = false;
                    continue;
                }
            }
            else
            {
                if(pEdge_left->twin_edge!=NULL)
                {
                    pEdge_left_prev = pEdge_left->twin_edge;
                    if(!pEdge_left_prev)
                        return 0;
                    pEdge_left_cur = pEdge_left_prev->next_edge;
                    pSite_left = pEdge_left_prev->site;
                    pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                    continue;
                }
                else
                    return 1;

            }

        }

        if((dist1>dist2&&dist2>=0)||(dist1<0&&dist2>=0))
        {
            pNode_begin = _cvSeqPush(NodeSeq,&Node);
            _cvInitVoronoiNode(pNode_begin, &Point2,radius2);
            pEdge2->node1 = pNode_begin;
            _cvTwinNULLRight(pEdge_right_cur,pEdge_right);
            _cvStickEdgeRightEnd(pEdge2,pEdge_right,pSite_right);
            if(left)
            {
                pEdge1->node2 = pNode_begin;
                pEdge_left_prev = pEdge1;
                pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                if(pEdge_right->twin_edge!=NULL)
                {
                    pEdge_right_next = pEdge_right->twin_edge;
                    if(!pEdge_right_next)
                        return 0;
                    pEdge_right_cur = pEdge_right_next->prev_edge;
                    pSite_right = pEdge_right_next->site;
                    pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                    continue;
                }
                else
                {
                    pEdge_right_cur = NULL;
                    right = false;
                    continue;
                }
            }
            else
            {
                if(pEdge_right->twin_edge!=NULL)
                {
                    pEdge_right_next = pEdge_right->twin_edge;
                    if(!pEdge_right_next)
                        return 0;
                    pEdge_right_cur = pEdge_right_next->prev_edge;
                    pSite_right = pEdge_right_next->site;
                    pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);
                    continue;
                }
                else
                    return 1;
            }

        }
    }

}// end of _cvJoinChains

static void _cvFindNearestSite(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvVoronoiHole pCurrHole, pHole = pVoronoiDiagram->top_hole;
    pCvPointFloat pTopPoint,pPoint1,pPoint2;
    CvPointFloat Direction;
    pCvVoronoiSite pSite;
    CvVoronoiNodeInt Node;
    CvSeq* CurrSeq;
    float min_distance,distance;
    int i;
    for(;pHole != NULL; pHole = pHole->next_hole)
    {
        if(pHole->error)
            continue;
        pTopPoint = &pHole->site_top->node1->node;
        pCurrHole = NULL;
        CurrSeq = pVoronoiDiagram->SiteSeq;
        min_distance = (float)3e+34;
        while(pCurrHole != pHole)
        {
            if(pCurrHole && pCurrHole->error)
                continue;
            pSite = (pCvVoronoiSite)cvGetSeqElem(CurrSeq,0);
            if(CurrSeq->total == 1)
            {
                distance = _cvCalcDist(pTopPoint, pSite);
                if(distance < min_distance)
                {
                    min_distance = distance;
                    pHole->site_nearest = pSite;
                }
            }
            else
            {
                for(i = 0; i < CurrSeq->total;i++, pSite = pSite->next_site)
                {
                    if(pSite->node1 != pSite->node2)
                    {
                        pPoint1 = &pSite->node1->node;
                        pPoint2 = &pSite->node2->node;

                        Direction.x = -pSite->direction->y;
                        Direction.y = pSite->direction->x;

                        if(
                                 (pTopPoint->x - pPoint2->x)*Direction.y -
                                 (pTopPoint->y - pPoint2->y)*Direction.x > 0
                            ||
                                 (pTopPoint->x - pPoint1->x)*Direction.y -
                                 (pTopPoint->y - pPoint1->y)*Direction.x < 0
                            ||
                                 (pTopPoint->x - pPoint1->x)*pSite->direction->y -
                                 (pTopPoint->y - pPoint1->y)*pSite->direction->x > 0
                           )
                                continue;

                        distance = _cvCalcDist(pTopPoint, pSite);
                    }
                    else
                    {
                        pPoint1 = &pSite->node1->node;
                        if(
                                 (pTopPoint->x - pPoint1->x)*pSite->edge2->direction->y -
                                 (pTopPoint->y - pPoint1->y)*pSite->edge2->direction->x > 0
                            ||
                                 (pTopPoint->x - pPoint1->x)*pSite->edge1->direction->y -
                                 (pTopPoint->y - pPoint1->y)*pSite->edge1->direction->x < 0
                           )
                                continue;

                        distance = _cvCalcDist(pTopPoint, pSite);
                    }


                    if(distance < min_distance)
                    {
                        min_distance = distance;
                        pHole->site_nearest = pSite;
                    }
                }
            }

            if(pCurrHole == NULL)
                pCurrHole = pVoronoiDiagram->top_hole;
            else
                pCurrHole = pCurrHole->next_hole;

            CurrSeq = pCurrHole->SiteSeq;
        }
        pHole->x_coord = min_distance;

        if(pHole->site_nearest->node1 == pHole->site_nearest->node2)
        {
            Direction.x = (pHole->site_nearest->node1->node.x - pHole->site_top->node1->node.x)/2;
            Direction.y = (pHole->site_nearest->node1->node.y - pHole->site_top->node1->node.y)/2;
        }
        else
        {

            Direction.x = pHole->site_nearest->direction->y * min_distance / 2;
            Direction.y = - pHole->site_nearest->direction->x * min_distance / 2;
        }

        Node.node.x = pHole->site_top->node1->node.x + Direction.x;
        Node.node.y = pHole->site_top->node1->node.y + Direction.y;
        pHole->node = _cvSeqPush(pVoronoiDiagram->NodeSeq, &Node);
    }
}//end of _cvFindNearestSite

static void _cvConstructIntVD(CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvVoronoiChain pChain1, pChain2;
    pCvVoronoiHole pHole;
    int i;

    pHole = pVoronoiDiagram->top_hole;

    for(;pHole != NULL; pHole = pHole->next_hole)
    {
        if(pHole->ChainSeq->total == 0)
            continue;
        pChain1 = (pCvVoronoiChain)cvGetSeqElem(pHole->ChainSeq,0);
        for(i = pHole->ChainSeq->total; i > 0;i--)
        {
            pChain2 = pChain1->next_chain;
            if(!_cvJoinChains(pChain1,pChain2,pVoronoiDiagram))
            {
                pHole->error = true;
                break;
            }

            pChain1->last_site = pChain2->last_site;
            pChain1->next_chain = pChain2->next_chain;
            pChain1 = pChain1->next_chain;
        }
    }
}// end of _cvConstructIntVD

static int _cvFindOppositSiteCW(pCvVoronoiHole pHole, CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvVoronoiSite pSite_left = pHole->site_nearest;
    pCvVoronoiSite pSite_right = pHole->site_top;
    pCvVoronoiNode pNode = pHole->node;

    CvDirection Direction = {-1,0};
    CvVoronoiEdgeInt Edge_right = {NULL,pSite_right->node1,pSite_right,NULL,NULL,NULL,NULL,&Direction};

    pCvVoronoiEdge pEdge_left = pSite_left->edge2->next_edge;
    pCvVoronoiEdge pEdge_right = &Edge_right;

    CvVoronoiEdgeInt Edge     = {NULL,pNode,pSite_right,NULL,NULL,NULL,NULL,NULL};
    CvVoronoiEdgeInt Edge_cur = {NULL,NULL, NULL,       NULL,NULL,NULL,NULL,NULL};
    pCvVoronoiEdge pEdge = &Edge;

    float radius1, radius2,dist1, dist2;
    CvPointFloat Point1 = {0,0}, Point2 = {0,0};

    for(;;)
    {
        pEdge->direction = NULL;
        pEdge->parabola = NULL;
        _cvCalcEdge(pSite_left,pSite_right,pEdge,pVoronoiDiagram);

        dist1=dist2=-1;
        radius1 = -1; radius2 = -2;
        while(pEdge_left!=NULL)
        {
            dist1 = _cvCalcEdgeIntersection(pEdge, pEdge_left, &Point1,radius1);
            if(dist1>=0)
                break;
            pEdge_left = pEdge_left->next_edge;
        }

        dist2 = _cvCalcEdgeIntersection(pEdge, pEdge_right, &Point2, radius2);
        if(dist2>=0 && dist1 >= dist2)
        {
            pHole->site_opposite = pSite_left;
            pNode->node = Point2;
            return 1;
        }

        if(dist1<0)
            return 0;

        Edge_cur = *pEdge_left->twin_edge;
        Edge_cur.node1 = pNode;
        pEdge_left = &Edge_cur;

        pSite_left = pEdge_left->site;
        pNode->node = Point1;
    }
}//end of _cvFindOppositSiteCW

static int _cvFindOppositSiteCCW(pCvVoronoiHole pHole,CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvVoronoiSite pSite_right = pHole->site_nearest;
    pCvVoronoiSite pSite_left = pHole->site_top;
    pCvVoronoiNode pNode = pHole->node;

    CvDirection Direction = {-1,0};
    CvVoronoiEdgeInt Edge_left = {pSite_left->node1,NULL,pSite_left,NULL,NULL,NULL, NULL, &Direction};

    pCvVoronoiEdge pEdge_left = &Edge_left;
    pCvVoronoiEdge pEdge_right = pSite_right->edge1->prev_edge;

    CvVoronoiEdgeInt Edge     = {NULL,pNode,pSite_left,NULL,NULL,NULL,NULL,NULL};
    CvVoronoiEdgeInt Edge_cur = {NULL,NULL, NULL,      NULL,NULL,NULL,NULL,NULL};
    pCvVoronoiEdge pEdge = &Edge;

    double dist1, dist2;
    float radius1, radius2;
    CvPointFloat Point1 = {0,0}, Point2 = {0,0};

    for(;;)
    {
        pEdge->direction = NULL;
        pEdge->parabola = NULL;
        _cvCalcEdge(pSite_left,pSite_right,pEdge,pVoronoiDiagram);

        dist1=dist2=-1;
        radius1 = -1; radius2 = -2;
        while(pEdge_right!=NULL)
        {
            dist1 = _cvCalcEdgeIntersection(pEdge, pEdge_right, &Point1,radius2);
            if(dist1>=0)
                break;
            pEdge_right = pEdge_right->prev_edge;
        }

        dist2 = _cvCalcEdgeIntersection(pEdge, pEdge_left, &Point2, radius1);
        if(dist2>=0 && dist1 > dist2)
        {
            pHole->site_opposite = pSite_right;
            pNode->node = Point2;
            return 1;
        }

        if(dist1<0)
            return 0;

        Edge_cur = *pEdge_right->twin_edge;
        Edge_cur.node2 = pNode;
        pEdge_right = &Edge_cur;

        pSite_right = pEdge_right->site;
        pNode->node = Point1;
    }
}//end of _cvFindOppositSiteCCW

static int _cvMergeVD(pCvVoronoiHole pHole,CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvVoronoiSite pSite_left_first = pHole->site_top;
    pCvVoronoiSite pSite_right_first = pHole->site_opposite;
    pCvVoronoiNode pNode_begin = pHole->node;
    if(pSite_left_first == NULL || pSite_right_first == NULL || pNode_begin == NULL)
        return 0;

    pCvVoronoiSite pSite_left = pSite_left_first;
    pCvVoronoiSite pSite_right = pSite_right_first;

    const double dist_eps = 0.05;
    CvVoronoiEdgeInt EdgeNULL = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};

    CvSeq* NodeSeq = pVoronoiDiagram->NodeSeq;
    CvSeq* EdgeSeq = pVoronoiDiagram->EdgeSeq;

    pCvVoronoiEdge pEdge_left = NULL;
    if(pSite_left->edge2 != NULL)
        pEdge_left = pSite_left->edge2->next_edge;

    pCvVoronoiEdge pEdge_right = pSite_right->edge1;
    pCvVoronoiEdge pEdge_left_cur = pEdge_left;
    pCvVoronoiEdge pEdge_right_cur = pEdge_right;

    pCvVoronoiEdge pEdge_left_prev = NULL;
    pCvVoronoiEdge pEdge_right_next = NULL;

    pCvVoronoiEdge pEdge1,pEdge2,pEdge1_first, pEdge2_first;
    CvPointFloat Point1 = {0,0}, Point2 = {0,0};

    float radius1,radius2,dist1,dist2;

    CvVoronoiNodeInt Node;

    pEdge1_first = pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);;
    pEdge2_first = pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);;
    pEdge1->site = pSite_left_first;
    pEdge2->site = pSite_right_first;

    do
    {
        pEdge1->node1 = pEdge2->node2 = pNode_begin;

        pEdge_left = pEdge_left_cur;
        pEdge_right = pEdge_right_cur->prev_edge;

        _cvCalcEdge(pSite_left,pSite_right,pEdge1,pVoronoiDiagram);
        _cvMakeTwinEdge(pEdge2,pEdge1);

        if(pEdge_left_prev != NULL)
            _cvStickEdgeLeftBegin(pEdge1,pEdge_left_prev,pSite_left);
        if(pEdge_right_next != NULL)
            _cvStickEdgeRightBegin(pEdge2,pEdge_right_next,pSite_right);

        dist1=dist2=-1;
        radius1 = -1; radius2 = -2;

//LEFT:
        while(pEdge_left!=NULL)
        {
            if(pEdge_left->node2 == NULL)
                pEdge_left_cur = pEdge_left = pEdge_left->next_edge;

            dist1 = _cvCalcEdgeIntersection(pEdge1, pEdge_left, &Point1,radius1);
            if(dist1>=0)
                goto RIGHT;
            pEdge_left = pEdge_left->next_edge;
        }

RIGHT:
        while(pEdge_right!=NULL)
        {
            dist2 = _cvCalcEdgeIntersection(pEdge1, pEdge_right, &Point2,radius2);
            if(dist2>=0)
                goto RESULTHANDLING;

            pEdge_right = pEdge_right->prev_edge;
        }
        pEdge_right = pEdge_right_cur;
        dist2 = _cvCalcEdgeIntersection(pEdge1, pEdge_right, &Point2, radius2);

RESULTHANDLING:
        if(dist1<0&&dist2<0)
            return 0;

        if(fabs(dist1 - dist2)<dist_eps)
        {
            pNode_begin = _cvSeqPush(NodeSeq,&Node);
            _cvInitVoronoiNode(pNode_begin, &Point2,radius2);

            pEdge1->node2 = pNode_begin;
            pEdge2->node1 = pNode_begin;

            pEdge_right_cur = _cvDivideRightEdge(pEdge_right,pNode_begin,EdgeSeq);

            _cvStickEdgeLeftEnd(pEdge1,pEdge_left,pSite_left);
            _cvStickEdgeRightEnd(pEdge2,pEdge_right,pSite_right);

            pEdge_left_prev = pEdge_left->twin_edge;
            if(!pEdge_left_prev)
                return 0;
            pEdge_left_cur = pEdge_left_prev->next_edge;
            pSite_left = pEdge_left_prev->site;
            pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);

            pEdge_right_next = pEdge_right->twin_edge;
            if(!pEdge_right_next)
                return 0;
            pSite_right = pEdge_right_next->site;
            pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);

            continue;
        }

        if((dist1<dist2&&dist1>=0)||(dist1>=0&&dist2<0))
        {
            pNode_begin = _cvSeqPush(NodeSeq,&Node);
            _cvInitVoronoiNode(pNode_begin, &Point1,radius1);

            pEdge1->node2 = pNode_begin;
            _cvStickEdgeLeftEnd(pEdge1,pEdge_left,pSite_left);

            pEdge2->node1 = pNode_begin;
            pEdge_right_next = pEdge2;
            pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);

            pEdge_left_prev = pEdge_left->twin_edge;
            if(!pEdge_left_prev)
                return 0;
            pEdge_left_cur = pEdge_left_prev->next_edge;
            pSite_left = pEdge_left_prev->site;
            pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);

            continue;
        }

        if((dist1>dist2&&dist2>=0)||(dist1<0&&dist2>=0))
        {
            pNode_begin = _cvSeqPush(NodeSeq,&Node);
            _cvInitVoronoiNode(pNode_begin, &Point2,radius2);

            pEdge_right_cur = _cvDivideRightEdge(pEdge_right,pNode_begin,EdgeSeq);

            pEdge2->node1 = pNode_begin;
            _cvStickEdgeRightEnd(pEdge2,pEdge_right,pSite_right);

            pEdge1->node2 = pNode_begin;
            pEdge_left_prev = pEdge1;
            pEdge1 = _cvSeqPush(EdgeSeq, &EdgeNULL);

            pEdge_right_next = pEdge_right->twin_edge;
            if(!pEdge_right_next)
                return 0;
            pSite_right = pEdge_right_next->site;
            pEdge2 = _cvSeqPush(EdgeSeq, &EdgeNULL);

            continue;
        }

    }while(!(pSite_left == pSite_left_first && pSite_right == pSite_right_first));

        pEdge1_first->node1 = pNode_begin;
        pEdge2_first->node2 = pNode_begin;
        _cvStickEdgeLeftBegin(pEdge1_first,pEdge_left_prev,pSite_left_first);
        _cvStickEdgeRightBegin(pEdge2_first,pEdge_right_next,pSite_right_first);

        if(pSite_left_first->edge2 == NULL)
            pSite_left_first->edge2 = pSite_left_first->edge1 = pEdge1_first;
        return 1;
}// end of _cvMergeVD


/* ///////////////////////////////////////////////////////////////////////////////////////
//                               Computation of bisectors                               //
/////////////////////////////////////////////////////////////////////////////////////// */

void _cvCalcEdge(pCvVoronoiSite pSite_left,
                 pCvVoronoiSite pSite_right,
                 pCvVoronoiEdge pEdge,
                 CvVoronoiDiagramInt* pVoronoiDiagram)
{
    if((pSite_left->node1!=pSite_left->node2)&&
        (pSite_right->node1!=pSite_right->node2))
        _cvCalcEdgeLL(pSite_left->direction,
                     pSite_right->direction,
                     pEdge,pVoronoiDiagram);

    else if((pSite_left->node1!=pSite_left->node2)&&
            (pSite_right->node1 == pSite_right->node2))
        _cvCalcEdgeLP(pSite_left,pSite_right->node1,pEdge,pVoronoiDiagram);

    else if((pSite_left->node1==pSite_left->node2)&&
            (pSite_right->node1!=pSite_right->node2))
        _cvCalcEdgePL(pSite_left->node1,pSite_right,pEdge,pVoronoiDiagram);

    else
        _cvCalcEdgePP(&(pSite_left->node1->node),
                     &(pSite_right->node1->node),
                     pEdge,pVoronoiDiagram);
}//end of _cvCalcEdge

void _cvCalcEdge(pCvVoronoiSite pSite,
                 pCvVoronoiNode pNode,
                 pCvVoronoiEdge pEdge,
                 CvVoronoiDiagramInt* pVoronoiDiagram)
{
    if(pSite->node1!=pSite->node2)
        _cvCalcEdgeLP(pSite, pNode, pEdge,pVoronoiDiagram);
    else
        _cvCalcEdgePP(&(pSite->node1->node),
                     &pNode->node,
                     pEdge,pVoronoiDiagram);
}//end of _cvCalcEdge

void _cvCalcEdge(pCvVoronoiNode pNode,
                         pCvVoronoiSite pSite,
                         pCvVoronoiEdge pEdge,
                         CvVoronoiDiagramInt* pVoronoiDiagram)
{
    if(pSite->node1!=pSite->node2)
        _cvCalcEdgePL(pNode,pSite,pEdge,pVoronoiDiagram);
    else
        _cvCalcEdgePP(&pNode->node,&pSite->node1->node,pEdge,pVoronoiDiagram);
}//end of _cvCalcEdge

CV_INLINE
void _cvCalcEdgeLL(pCvDirection pDirection1,
                  pCvDirection pDirection2,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram)
{
    CvDirection Direction = {pDirection2->x - pDirection1->x, pDirection2->y - pDirection1->y};
    if((fabs(Direction.x)<LEE_CONST_ZERO)&&(fabs(Direction.y)<LEE_CONST_ZERO))
            Direction = *pDirection2;
    pEdge->direction = _cvSeqPush(pVoronoiDiagram->DirectionSeq,&Direction);;
}//end of _cvCalcEdgeLL

CV_INLINE
void _cvCalcEdgePP(pCvPointFloat pPoint1,
                  pCvPointFloat pPoint2,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram)
{
    CvDirection Direction = {pPoint1->y - pPoint2->y,pPoint2->x - pPoint1->x};
    pEdge->direction = _cvSeqPush(pVoronoiDiagram->DirectionSeq,&Direction);
}//end of _cvCalcEdgePP

CV_INLINE
void _cvCalcEdgePL(pCvVoronoiNode pFocus,
                  pCvVoronoiSite pDirectrice,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvPointFloat pPoint0 = &pFocus->node;
    pCvPointFloat pPoint1 = &pDirectrice->node1->node;

    CvDirection Vector01 = {pPoint0->x - pPoint1->x,pPoint0->y - pPoint1->y};
    float half_h = (Vector01.y*pDirectrice->direction->x - Vector01.x*pDirectrice->direction->y)/2;
    CvDirection Normal = {-pDirectrice->direction->y,pDirectrice->direction->x};
    if(half_h < LEE_CONST_ZERO)
    {
        pEdge->direction = _cvSeqPush(pVoronoiDiagram->DirectionSeq,&Normal);
        return;
    }
    CvVoronoiParabolaInt Parabola;
    pCvVoronoiParabola  pParabola = _cvSeqPush(pVoronoiDiagram->ParabolaSeq,&Parabola);
    float* map = pParabola->map;

    map[1] = Normal.x;
    map[4] = Normal.y;
    map[0] = Normal.y;
    map[3] = -Normal.x;
    map[2] = pPoint0->x - Normal.x*half_h;
    map[5] = pPoint0->y - Normal.y*half_h;

    pParabola->a = 1/(4*half_h);
    pParabola->focus = pFocus;
    pParabola->directrice = pDirectrice;
    pEdge->parabola = pParabola;
}//end of _cvCalcEdgePL

CV_INLINE
void _cvCalcEdgeLP(pCvVoronoiSite pDirectrice,
                  pCvVoronoiNode pFocus,
                  pCvVoronoiEdge pEdge,
                  CvVoronoiDiagramInt* pVoronoiDiagram)
{
    pCvPointFloat pPoint0 = &pFocus->node;
    pCvPointFloat pPoint1 = &pDirectrice->node1->node;

    CvDirection Vector01 = {pPoint0->x - pPoint1->x,pPoint0->y - pPoint1->y};
    float half_h = (Vector01.y*pDirectrice->direction->x - Vector01.x*pDirectrice->direction->y)/2;
    CvDirection Normal = {-pDirectrice->direction->y,pDirectrice->direction->x};
    if(half_h < LEE_CONST_ZERO)
    {
        pEdge->direction = _cvSeqPush(pVoronoiDiagram->DirectionSeq,&Normal);
        return;
    }
    CvVoronoiParabolaInt Parabola;
    pCvVoronoiParabola  pParabola = _cvSeqPush(pVoronoiDiagram->ParabolaSeq,&Parabola);
    float* map = pParabola->map;

    map[1] = Normal.x;
    map[4] = Normal.y;
    map[0] = -Normal.y;
    map[3] = Normal.x;
    map[2] = pPoint0->x - Normal.x*half_h;
    map[5] = pPoint0->y - Normal.y*half_h;

    pParabola->a = 1/(4*half_h);
    pParabola->focus = pFocus;
    pParabola->directrice = pDirectrice;
    pEdge->parabola = pParabola;
}//end of _cvCalcEdgeLP

/* ///////////////////////////////////////////////////////////////////////////////////////
//                  Computation of intersections of bisectors                           //
/////////////////////////////////////////////////////////////////////////////////////// */

static
float _cvCalcEdgeIntersection(pCvVoronoiEdge pEdge1,
                              pCvVoronoiEdge pEdge2,
                              CvPointFloat* pPoint,
                              float &Radius)
{
    if((pEdge1->parabola==NULL)&&(pEdge2->parabola==NULL))
        return _cvLine_LineIntersection(pEdge1,pEdge2,pPoint,Radius);
    if((pEdge1->parabola==NULL)&&(pEdge2->parabola!=NULL))
        return _cvLine_ParIntersection(pEdge1,pEdge2,pPoint,Radius);
    if((pEdge1->parabola!=NULL)&&(pEdge2->parabola==NULL))
        return _cvPar_LineIntersection(pEdge1,pEdge2,pPoint,Radius);
    //if((pEdge1->parabola!=NULL)&&(pEdge2->parabola!=NULL))
        return _cvPar_ParIntersection(pEdge1,pEdge2,pPoint,Radius);
    //return -1;
}//end of _cvCalcEdgeIntersection

static
float _cvLine_LineIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius)
{
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        return -1;

    CvPointFloat Point1,Point3;
    float det;
    float k,m;
    float x21,x43,y43,y21,x31,y31;

    if(pEdge1->node1!=NULL)
    {
        Point1.x = pEdge1->node1->node.x;
        Point1.y = pEdge1->node1->node.y;
    }
    else
    {
        Point1.x = pEdge1->node2->node.x;
        Point1.y = pEdge1->node2->node.y;
    }
    x21 = pEdge1->direction->x;
    y21 = pEdge1->direction->y;

    if(pEdge2->node2==NULL)
    {
        Point3.x = pEdge2->node1->node.x;
        Point3.y = pEdge2->node1->node.y;
        x43 = pEdge2->direction->x;
        y43 = pEdge2->direction->y;

    }
    else if(pEdge2->node1==NULL)
    {
        Point3.x = pEdge2->node2->node.x;
        Point3.y = pEdge2->node2->node.y;
        x43 = pEdge2->direction->x;
        y43 = pEdge2->direction->y;
    }
    else
    {
        Point3.x = pEdge2->node1->node.x;
        Point3.y = pEdge2->node1->node.y;
        x43 = pEdge2->node2->node.x - Point3.x;
        y43 = pEdge2->node2->node.y - Point3.y;
    }

    x31 = Point3.x - Point1.x;
    y31 = Point3.y - Point1.y;

    det = y21*x43 - x21*y43;
    if(fabs(det) < LEE_CONST_ZERO)
        return -1;

    k = (x43*y31 - y43*x31)/det;
    m = (x21*y31 - y21*x31)/det;

    if(k<-LEE_CONST_ACCEPTABLE_ERROR||m<-LEE_CONST_ACCEPTABLE_ERROR)
        return -1;
    if(((pEdge1->node2!=NULL)&&(pEdge1->node1!=NULL))&&(k>1.f+LEE_CONST_ACCEPTABLE_ERROR))
        return -1;
    if(((pEdge2->node2!=NULL)&&(pEdge2->node1!=NULL))&&(m>1.f+LEE_CONST_ACCEPTABLE_ERROR))
        return -1;

    pPoint->x = (float)(k*x21) + Point1.x;
    pPoint->y = (float)(k*y21) + Point1.y;

    Radius = _cvCalcDist(pPoint,pEdge1->site);
    return _cvPPDist(pPoint,&Point1);;
}//end of _cvLine_LineIntersection

static
float _cvLine_ParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius)
{
    if(pEdge2->node1==NULL||pEdge2->node2==NULL)
        return _cvLine_OpenParIntersection(pEdge1,pEdge2,pPoint,Radius);
    return _cvLine_CloseParIntersection(pEdge1,pEdge2,pPoint,Radius);
}//end of _cvLine_ParIntersection

static
float _cvLine_OpenParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius)
{
    int IntersectionNumber = 1;
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        IntersectionNumber = 2;

    pCvPointFloat pRayPoint1;
    if(pEdge1->node1!=NULL)
        pRayPoint1 = &(pEdge1->node1->node);
    else
        pRayPoint1 = &(pEdge1->node2->node);

    pCvDirection pDirection = pEdge1->direction;
    float* Parabola = pEdge2->parabola->map;

    pCvPointFloat pParPoint1;
    if(pEdge2->node1==NULL)
        pParPoint1 = &(pEdge2->node2->node);
    else
        pParPoint1 = &(pEdge2->node1->node);

    float InversParabola[6]={0,0,0,0,0,0};
    _cvCalcOrtogInverse(InversParabola, Parabola);

    CvPointFloat  Point,ParPoint1_img,RayPoint1_img;
    CvDirection Direction_img;
    _cvCalcPointImage(&RayPoint1_img, pRayPoint1, InversParabola);
    _cvCalcVectorImage(&Direction_img,pDirection, InversParabola);

    float c2 = pEdge2->parabola->a*Direction_img.x;
    float c1 = -Direction_img.y;
    float c0 = Direction_img.y* RayPoint1_img.x - Direction_img.x*RayPoint1_img.y;
    float X[2];
    int N = _cvSolveEqu2thR(c2,c1,c0,X);
    if(N==0)
        return -1;

    _cvCalcPointImage(&ParPoint1_img, pParPoint1, InversParabola);
    int sign_x = SIGN(Direction_img.x);
    int sign_y = SIGN(Direction_img.y);
    if(N==1)
    {
        if(X[0]<ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            return -1;
        float pr0 = (X[0]-RayPoint1_img.x)*sign_x + \
                    (pEdge2->parabola->a*X[0]*X[0]-RayPoint1_img.y)*sign_y;
        if(pr0 <= -LEE_CONST_ACCEPTABLE_ERROR)
            return -1;
    }
    else
    {
        if(X[1]<ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            return -1;
        float pr0 = (X[0]-RayPoint1_img.x)*sign_x + \
                        (pEdge2->parabola->a*X[0]*X[0]-RayPoint1_img.y)*sign_y;
        float pr1 = (X[1]-RayPoint1_img.x)*sign_x + \
                        (pEdge2->parabola->a*X[1]*X[1]-RayPoint1_img.y)*sign_y;

        if(pr0 <= -LEE_CONST_ACCEPTABLE_ERROR &&pr1 <= -LEE_CONST_ACCEPTABLE_ERROR)
            return -1;

        if(pr0 >- LEE_CONST_ACCEPTABLE_ERROR && pr1 >- LEE_CONST_ACCEPTABLE_ERROR)
        {
            if(X[0] >= ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            {
                if(pr0>pr1)
                    _cvSwap(X[0],X[1]);
            }
            else
            {
                N=1;
                X[0] = X[1];
            }
        }
        else if(pr0 >- LEE_CONST_ACCEPTABLE_ERROR)
        {
            N = 1;
            if(X[0] < ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
                return -1;
        }
        else if(pr1 >- LEE_CONST_ACCEPTABLE_ERROR)
        {
            N=1;
            X[0] = X[1];
        }
        else
            return -1;
    }

    Point.x = X[(N-1)*(IntersectionNumber - 1)];
    Point.y = pEdge2->parabola->a*Point.x*Point.x;

    Radius = Point.y + 1.f/(4*pEdge2->parabola->a);
    _cvCalcPointImage(pPoint,&Point,Parabola);
    float dist = _cvPPDist(pPoint, pRayPoint1);
    if(IntersectionNumber == 2 && dist < LEE_CONST_DIFF_POINTS)
        return -1;
    else
        return dist;
}// end of _cvLine_OpenParIntersection

static
float _cvLine_CloseParIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius)
{
    int IntersectionNumber = 1;
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        IntersectionNumber = 2;

    pCvPointFloat pRayPoint1;
    if(pEdge1->node1!=NULL)
        pRayPoint1 = &(pEdge1->node1->node);
    else
        pRayPoint1 = &(pEdge1->node2->node);

    pCvDirection pDirection = pEdge1->direction;
    float* Parabola = pEdge2->parabola->map;

    pCvPointFloat pParPoint1,pParPoint2;
    pParPoint2 = &(pEdge2->node1->node);
    pParPoint1 = &(pEdge2->node2->node);


    float InversParabola[6]={0,0,0,0,0,0};
    _cvCalcOrtogInverse(InversParabola, Parabola);

    CvPointFloat  Point,ParPoint1_img,ParPoint2_img,RayPoint1_img;
    CvDirection Direction_img;
    _cvCalcPointImage(&RayPoint1_img, pRayPoint1, InversParabola);
    _cvCalcVectorImage(&Direction_img,pDirection, InversParabola);

    float c2 = pEdge2->parabola->a*Direction_img.x;
    float c1 = -Direction_img.y;
    float c0 = Direction_img.y* RayPoint1_img.x - Direction_img.x*RayPoint1_img.y;
    float X[2];
    int N = _cvSolveEqu2thR(c2,c1,c0,X);
    if(N==0)
        return -1;

    _cvCalcPointImage(&ParPoint1_img, pParPoint1, InversParabola);
    _cvCalcPointImage(&ParPoint2_img, pParPoint2, InversParabola);
    if(ParPoint1_img.x>ParPoint2_img.x)
        _cvSwap(ParPoint1_img,ParPoint2_img);
    int sign_x = SIGN(Direction_img.x);
    int sign_y = SIGN(Direction_img.y);
    if(N==1)
    {
        if((X[0]<ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR) ||
           (X[0]>ParPoint2_img.x + LEE_CONST_ACCEPTABLE_ERROR))
            return -1;
        float pr0 = (X[0]-RayPoint1_img.x)*sign_x + \
                    (pEdge2->parabola->a*X[0]*X[0]-RayPoint1_img.y)*sign_y;
        if(pr0 <= -LEE_CONST_ACCEPTABLE_ERROR)
            return -1;
    }
    else
    {
        if((X[1]<ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR) ||
           (X[0]>ParPoint2_img.x + LEE_CONST_ACCEPTABLE_ERROR))
            return -1;

        if((X[0]<ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR) &&
           (X[1]>ParPoint2_img.x + LEE_CONST_ACCEPTABLE_ERROR))
            return -1;

        float pr0 = (X[0]-RayPoint1_img.x)*sign_x + \
                    (pEdge2->parabola->a*X[0]*X[0]-RayPoint1_img.y)*sign_y;
        float pr1 = (X[1]-RayPoint1_img.x)*sign_x + \
                    (pEdge2->parabola->a*X[1]*X[1]-RayPoint1_img.y)*sign_y;

        if(pr0 <= -LEE_CONST_ACCEPTABLE_ERROR && pr1 <= -LEE_CONST_ACCEPTABLE_ERROR)
            return -1;

        if(pr0 > -LEE_CONST_ACCEPTABLE_ERROR && pr1 > -LEE_CONST_ACCEPTABLE_ERROR)
        {
            if(X[0] >= ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            {
                if(X[1] <= ParPoint2_img.x + LEE_CONST_ACCEPTABLE_ERROR)
                {
                    if(pr0>pr1)
                        _cvSwap(X[0], X[1]);
                }
                else
                    N=1;
            }
            else
            {
                N=1;
                X[0] = X[1];
            }
        }
        else if(pr0 > -LEE_CONST_ACCEPTABLE_ERROR)
        {

            if(X[0] >= ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
                N=1;
            else
                return -1;
        }
        else if(pr1 > -LEE_CONST_ACCEPTABLE_ERROR)
        {
            if(X[1] <= ParPoint2_img.x + LEE_CONST_ACCEPTABLE_ERROR)
            {
                N=1;
                X[0] = X[1];
            }
            else
                return -1;
        }
        else
            return -1;
    }

    Point.x = X[(N-1)*(IntersectionNumber - 1)];
    Point.y = pEdge2->parabola->a*Point.x*Point.x;
    Radius = Point.y + 1.f/(4*pEdge2->parabola->a);
    _cvCalcPointImage(pPoint,&Point,Parabola);
    float dist = _cvPPDist(pPoint, pRayPoint1);
    if(IntersectionNumber == 2 && dist < LEE_CONST_DIFF_POINTS)
        return -1;
    else
        return dist;
}// end of _cvLine_CloseParIntersection

static
float _cvPar_LineIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius)
{
    if(pEdge2->node1==NULL||pEdge2->node2==NULL)
        return _cvPar_OpenLineIntersection(pEdge1,pEdge2,pPoint,Radius);
    return _cvPar_CloseLineIntersection(pEdge1,pEdge2,pPoint,Radius);
}//end _cvPar_LineIntersection

static
float _cvPar_OpenLineIntersection(pCvVoronoiEdge pEdge1,
                                pCvVoronoiEdge pEdge2,
                                pCvPointFloat  pPoint,
                                float &Radius)
{
    int i, IntersectionNumber = 1;
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        IntersectionNumber = 2;

    float* Parabola = pEdge1->parabola->map;
    pCvPointFloat pParPoint1;
    if(pEdge1->node1!=NULL)
        pParPoint1 = &(pEdge1->node1->node);
    else
        pParPoint1 = &(pEdge1->node2->node);

    pCvPointFloat pRayPoint1;
    if(pEdge2->node1==NULL)
        pRayPoint1 = &(pEdge2->node2->node);
    else
        pRayPoint1 = &(pEdge2->node1->node);
    pCvDirection pDirection = pEdge2->direction;


    float InversParabola[6]={0,0,0,0,0,0};
    _cvCalcOrtogInverse(InversParabola, Parabola);

    CvPointFloat  Point = {0,0},ParPoint1_img,RayPoint1_img;
    CvDirection Direction_img;
    _cvCalcVectorImage(&Direction_img,pDirection, InversParabola);
    _cvCalcPointImage(&RayPoint1_img, pRayPoint1, InversParabola);


    float q = RayPoint1_img.y - pEdge1->parabola->a*RayPoint1_img.x*RayPoint1_img.x;
    if((pEdge2->site->node1 == pEdge2->site->node2 && q < 0) ||
        (pEdge2->site->node1 != pEdge2->site->node2 && q > 0))
        return -1;

    float c2 = pEdge1->parabola->a*Direction_img.x;
    float c1 = -Direction_img.y;
    float c0 = Direction_img.y* RayPoint1_img.x - Direction_img.x*RayPoint1_img.y;
    float X[2];
    int N = _cvSolveEqu2thR(c2,c1,c0,X);
    if(N==0)
        return -1;

    _cvCalcPointImage(&ParPoint1_img, pParPoint1, InversParabola);
    int sign_x = SIGN(Direction_img.x);
    int sign_y = SIGN(Direction_img.y);
    float pr;

    if(N==2 && IntersectionNumber == 2)
        _cvSwap(X[0], X[1]);

    for( i=0;i<N;i++)
    {
        if(X[i]<=ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            continue;
        pr = (X[i]-RayPoint1_img.x)*sign_x +
                        (pEdge1->parabola->a*X[i]*X[i]-RayPoint1_img.y)*sign_y;
        if(pr <= -LEE_CONST_ACCEPTABLE_ERROR)
            continue;
        else
        {
            Point.x = X[i];
            break;
        }
    }

    if(i==N)
        return -1;

    Point.y = pEdge1->parabola->a*Point.x*Point.x;
    Radius = Point.y + 1.f/(4*pEdge1->parabola->a);
    _cvCalcPointImage(pPoint,&Point,Parabola);
    float dist = Point.x - ParPoint1_img.x;
    if(IntersectionNumber == 2 && dist < LEE_CONST_DIFF_POINTS)
        return -1;
    else
        return dist;
}// end of _cvPar_OpenLineIntersection

static
float _cvPar_CloseLineIntersection(pCvVoronoiEdge pEdge1,
                                    pCvVoronoiEdge pEdge2,
                                    pCvPointFloat  pPoint,
                                    float &Radius)
{
    int i, IntersectionNumber = 1;
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        IntersectionNumber = 2;

    float* Parabola = pEdge1->parabola->map;
    pCvPointFloat pParPoint1;
    if(pEdge1->node1!=NULL)
        pParPoint1 = &(pEdge1->node1->node);
    else
        pParPoint1 = &(pEdge1->node2->node);

    pCvPointFloat pRayPoint1,pRayPoint2;
    pRayPoint2 = &(pEdge2->node1->node);
    pRayPoint1 = &(pEdge2->node2->node);

    pCvDirection pDirection = pEdge2->direction;
    float InversParabola[6]={0,0,0,0,0,0};
    _cvCalcOrtogInverse(InversParabola, Parabola);

    CvPointFloat  Point={0,0},ParPoint1_img,RayPoint1_img,RayPoint2_img;
    CvDirection Direction_img;
    _cvCalcPointImage(&RayPoint1_img, pRayPoint1, InversParabola);
    _cvCalcPointImage(&RayPoint2_img, pRayPoint2, InversParabola);

    float q;
    if(Radius == -1)
    {
         q = RayPoint1_img.y - pEdge1->parabola->a*RayPoint1_img.x*RayPoint1_img.x;
         if((pEdge2->site->node1 == pEdge2->site->node2 && q < 0) ||
            (pEdge2->site->node1 != pEdge2->site->node2 && q > 0))
                return -1;
    }
    if(Radius == -2)
    {
         q = RayPoint2_img.y - pEdge1->parabola->a*RayPoint2_img.x*RayPoint2_img.x;
        if((pEdge2->site->node1 == pEdge2->site->node2 && q < 0) ||
            (pEdge2->site->node1 != pEdge2->site->node2 && q > 0))
                return -1;
    }

    _cvCalcPointImage(&ParPoint1_img, pParPoint1, InversParabola);
    _cvCalcVectorImage(&Direction_img,pDirection, InversParabola);

    float c2 = pEdge1->parabola->a*Direction_img.x;
    float c1 = -Direction_img.y;
    float c0 = Direction_img.y* RayPoint1_img.x - Direction_img.x*RayPoint1_img.y;
    float X[2];
    int N = _cvSolveEqu2thR(c2,c1,c0,X);
    if(N==0)
        return -1;
    int sign_x = SIGN(RayPoint2_img.x - RayPoint1_img.x);
    int sign_y = SIGN(RayPoint2_img.y - RayPoint1_img.y);
    float pr_dir = (RayPoint2_img.x - RayPoint1_img.x)*sign_x +
                   (RayPoint2_img.y - RayPoint1_img.y)*sign_y;
    float pr;

    if(N==2 && IntersectionNumber == 2)
        _cvSwap(X[0], X[1]);

    for( i =0;i<N;i++)
    {
        if(X[i] <= ParPoint1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            continue;
        pr = (X[i]-RayPoint1_img.x)*sign_x + \
             (pEdge1->parabola->a*X[i]*X[i]-RayPoint1_img.y)*sign_y;
        if(pr <= -LEE_CONST_ACCEPTABLE_ERROR || pr>=pr_dir + LEE_CONST_ACCEPTABLE_ERROR)
            continue;
        else
        {
            Point.x = X[i];
            break;
        }
    }

    if(i==N)
        return -1;

    Point.y = pEdge1->parabola->a*Point.x*Point.x;
    Radius = Point.y + 1.f/(4*pEdge1->parabola->a);
    _cvCalcPointImage(pPoint,&Point,Parabola);
    float dist = Point.x - ParPoint1_img.x;
    if(IntersectionNumber == 2 && dist < LEE_CONST_DIFF_POINTS)
        return -1;
    else
        return dist;
}// end of _cvPar_CloseLineIntersection

static
float _cvPar_ParIntersection(pCvVoronoiEdge pEdge1,
                            pCvVoronoiEdge pEdge2,
                            pCvPointFloat  pPoint,
                            float &Radius)
{
    if(pEdge2->node1==NULL||pEdge2->node2==NULL)
        return _cvPar_OpenParIntersection(pEdge1,pEdge2,pPoint,Radius);
    return _cvPar_CloseParIntersection(pEdge1,pEdge2,pPoint,Radius);
}// end of _cvPar_ParIntersection

static
float _cvPar_OpenParIntersection(pCvVoronoiEdge pEdge1,
                            pCvVoronoiEdge pEdge2,
                            pCvPointFloat  pPoint,
                            float &Radius)
{
    int i, IntersectionNumber = 1;
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        IntersectionNumber = 2;

    float* Parabola1 = pEdge1->parabola->map;
    pCvPointFloat pPar1Point1;
    if(pEdge1->node1!=NULL)
        pPar1Point1 = &(pEdge1->node1->node);
    else
        pPar1Point1 = &(pEdge1->node2->node);

    float* Parabola2 = pEdge2->parabola->map;
    pCvPointFloat pPar2Point1;
    if(pEdge2->node1!=NULL)
        pPar2Point1 = &(pEdge2->node1->node);
    else
        pPar2Point1 = &(pEdge2->node2->node);

    CvPointFloat Point;
    CvDirection Direction;
    if(pEdge1->parabola->directrice==pEdge2->parabola->directrice)  //common site is segment -> different focuses
    {
        pCvPointFloat pFocus1 = &(pEdge1->parabola->focus->node);
        pCvPointFloat pFocus2 = &(pEdge2->parabola->focus->node);

        Point.x = (pFocus1->x + pFocus2->x)/2;
        Point.y = (pFocus1->y + pFocus2->y)/2;
        Direction.x = pFocus1->y - pFocus2->y;
        Direction.y = pFocus2->x - pFocus1->x;
    }
    else//common site is focus -> different directrices
    {
        pCvVoronoiSite pDirectrice1 = pEdge1->parabola->directrice;
        pCvPointFloat pPoint1 = &(pDirectrice1->node1->node);
        pCvDirection pVector21 = pDirectrice1->direction;

        pCvVoronoiSite pDirectrice2 = pEdge2->parabola->directrice;
        pCvPointFloat pPoint3 = &(pDirectrice2->node1->node);
        pCvDirection pVector43 = pDirectrice2->direction;

        Direction.x = pVector43->x - pVector21->x;
        Direction.y = pVector43->y - pVector21->y;

        if((fabs(Direction.x) < LEE_CONST_ZERO) &&
           (fabs(Direction.y) < LEE_CONST_ZERO))
                Direction = *pVector43;

        float det = pVector21->y * pVector43->x - pVector21->x * pVector43->y;
        if(fabs(det) < LEE_CONST_ZERO)
        {
            Point.x = (pPoint1->x + pPoint3->x)/2;
            Point.y = (pPoint1->y + pPoint3->y)/2;
        }
        else
        {
            float d1 = pVector21->y*pPoint1->x - pVector21->x*pPoint1->y;
            float d2 = pVector43->y*pPoint3->x - pVector43->x*pPoint3->y;
            Point.x = (float)((pVector43->x*d1 - pVector21->x*d2)/det);
            Point.y = (float)((pVector43->y*d1 - pVector21->y*d2)/det);
        }
    }

    float InversParabola2[6]={0,0,0,0,0,0};
    _cvCalcOrtogInverse(InversParabola2, Parabola2);

    CvPointFloat  Par2Point1_img,Point_img;
    CvDirection Direction_img;
    _cvCalcVectorImage(&Direction_img,&Direction, InversParabola2);
    _cvCalcPointImage(&Point_img, &Point, InversParabola2);

    float a1 = pEdge1->parabola->a;
    float a2 = pEdge2->parabola->a;
    float c2 = a2*Direction_img.x;
    float c1 = -Direction_img.y;
    float c0 = Direction_img.y* Point_img.x - Direction_img.x*Point_img.y;
    float X[2];
    int N = _cvSolveEqu2thR(c2,c1,c0,X);

    if(N==0)
        return -1;

    _cvCalcPointImage(&Par2Point1_img, pPar2Point1, InversParabola2);

    if(X[N-1]<Par2Point1_img.x)
        return -1;

    if(X[0]<Par2Point1_img.x)
    {
        X[0] = X[1];
        N=1;
    }

    float InversParabola1[6]={0,0,0,0,0,0};
    CvPointFloat Par1Point1_img;
    _cvCalcOrtogInverse(InversParabola1, Parabola1);
    _cvCalcPointImage(&Par1Point1_img, pPar1Point1, InversParabola1);
    float InvPar1_Par2[6];
    _cvCalcComposition(InvPar1_Par2,InversParabola1,Parabola2);
    for(i=0;i<N;i++)
        X[i] = (InvPar1_Par2[1]*a2*X[i] + InvPar1_Par2[0])*X[i] +  InvPar1_Par2[2];

    if(N!=1)
    {
        if((X[0]>X[1] && IntersectionNumber == 1)||
            (X[0]<X[1] && IntersectionNumber == 2))
            _cvSwap(X[0], X[1]);
    }

    for(i = 0;i<N;i++)
    {
        Point.x = X[i];
        Point.y = a1*Point.x*Point.x;
        if(Point.x < Par1Point1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            continue;
        else
            break;
    }

    if(i==N)
        return -1;

    Radius = Point.y + 1.f/(4*pEdge1->parabola->a);
    _cvCalcPointImage(pPoint,&Point,Parabola1);
    float dist = Point.x - Par1Point1_img.x;
    if(IntersectionNumber == 2 && dist < LEE_CONST_DIFF_POINTS)
        return -1;
    else
        return dist;
}// end of _cvPar_OpenParIntersection

static
float _cvPar_CloseParIntersection(pCvVoronoiEdge pEdge1,
                                  pCvVoronoiEdge pEdge2,
                                  pCvPointFloat  pPoint,
                                  float &Radius)
{
    int i, IntersectionNumber = 1;
    if(((pEdge1->node1 == pEdge2->node1 ||
        pEdge1->node1 == pEdge2->node2) &&
        pEdge1->node1 != NULL)||
       ((pEdge1->node2 == pEdge2->node1 ||
        pEdge1->node2 == pEdge2->node2) &&
        pEdge1->node2 != NULL))
        IntersectionNumber = 2;

    float* Parabola1 = pEdge1->parabola->map;
    float* Parabola2 = pEdge2->parabola->map;
    pCvPointFloat pPar1Point1;
    if(pEdge1->node1!=NULL)
        pPar1Point1 = &(pEdge1->node1->node);
    else
        pPar1Point1 = &(pEdge1->node2->node);

    pCvPointFloat pPar2Point1 = &(pEdge2->node1->node);
    pCvPointFloat pPar2Point2 = &(pEdge2->node2->node);

    CvPointFloat Point;
    CvDirection Direction;
    if(pEdge1->parabola->directrice==pEdge2->parabola->directrice)  //common site is segment -> different focuses
    {
        pCvPointFloat pFocus1 = &(pEdge1->parabola->focus->node);
        pCvPointFloat pFocus2 = &(pEdge2->parabola->focus->node);

        Point.x = (pFocus1->x + pFocus2->x)/2;
        Point.y = (pFocus1->y + pFocus2->y)/2;
        Direction.x = pFocus1->y - pFocus2->y;
        Direction.y = pFocus2->x - pFocus1->x;
    }
    else//common site is focus -> different directrices
    {
        pCvVoronoiSite pDirectrice1 = pEdge1->parabola->directrice;
        pCvPointFloat pPoint1 = &(pDirectrice1->node1->node);
        pCvDirection pVector21 = pDirectrice1->direction;

        pCvVoronoiSite pDirectrice2 = pEdge2->parabola->directrice;
        pCvPointFloat pPoint3 = &(pDirectrice2->node1->node);
        pCvDirection pVector43 = pDirectrice2->direction;

        Direction.x = pVector43->x - pVector21->x;
        Direction.y = pVector43->y - pVector21->y;

        if((fabs(Direction.x) < LEE_CONST_ZERO) &&
           (fabs(Direction.y) < LEE_CONST_ZERO))
                Direction = *pVector43;

        float det = pVector21->y * pVector43->x - pVector21->x * pVector43->y;
        if(fabs(det) < LEE_CONST_ZERO)
        {
            Point.x = (pPoint1->x + pPoint3->x)/2;
            Point.y = (pPoint1->y + pPoint3->y)/2;
        }
        else
        {
            float d1 = pVector21->y*pPoint1->x - pVector21->x*pPoint1->y;
            float d2 = pVector43->y*pPoint3->x - pVector43->x*pPoint3->y;
            Point.x = (float)((pVector43->x*d1 - pVector21->x*d2)/det);
            Point.y = (float)((pVector43->y*d1 - pVector21->y*d2)/det);
        }
    }



    float InversParabola2[6]={0,0,0,0,0,0};
    _cvCalcOrtogInverse(InversParabola2, Parabola2);

    CvPointFloat  Par2Point1_img,Par2Point2_img,Point_img;
    CvDirection Direction_img;
    _cvCalcVectorImage(&Direction_img,&Direction, InversParabola2);
    _cvCalcPointImage(&Point_img, &Point, InversParabola2);

    float a1 = pEdge1->parabola->a;
    float a2 = pEdge2->parabola->a;
    float c2 = a2*Direction_img.x;
    float c1 = -Direction_img.y;
    float c0 = Direction_img.y* Point_img.x - Direction_img.x*Point_img.y;
    float X[2];
    int N = _cvSolveEqu2thR(c2,c1,c0,X);

    if(N==0)
        return -1;

    _cvCalcPointImage(&Par2Point1_img, pPar2Point1, InversParabola2);
    _cvCalcPointImage(&Par2Point2_img, pPar2Point2, InversParabola2);
    if(Par2Point1_img.x>Par2Point2_img.x)
        _cvSwap(Par2Point1_img,Par2Point2_img);

    if(X[0]>Par2Point2_img.x||X[N-1]<Par2Point1_img.x)
        return -1;

    if(X[0]<Par2Point1_img.x)
    {
        if(X[1]<Par2Point2_img.x)
        {
            X[0] = X[1];
            N=1;
        }
        else
            return -1;
    }
    else if(X[N-1]>Par2Point2_img.x)
            N=1;

    float InversParabola1[6]={0,0,0,0,0,0};
    CvPointFloat Par1Point1_img;
    _cvCalcOrtogInverse(InversParabola1, Parabola1);
    _cvCalcPointImage(&Par1Point1_img, pPar1Point1, InversParabola1);
    float InvPar1_Par2[6];
    _cvCalcComposition(InvPar1_Par2,InversParabola1,Parabola2);
    for(i=0;i<N;i++)
        X[i] = (InvPar1_Par2[1]*a2*X[i] + InvPar1_Par2[0])*X[i] +  InvPar1_Par2[2];

    if(N!=1)
    {
        if((X[0]>X[1] && IntersectionNumber == 1)||
            (X[0]<X[1] && IntersectionNumber == 2))
            _cvSwap(X[0], X[1]);
    }


    for(i = 0;i<N;i++)
    {
        Point.x = (float)X[i];
        Point.y = (float)a1*Point.x*Point.x;
        if(Point.x < Par1Point1_img.x - LEE_CONST_ACCEPTABLE_ERROR)
            continue;
        else
            break;
    }

    if(i==N)
        return -1;

    Radius = Point.y + 1.f/(4*a1);
    _cvCalcPointImage(pPoint,&Point,Parabola1);
    float dist = Point.x - Par1Point1_img.x;
    if(IntersectionNumber == 2 && dist < LEE_CONST_DIFF_POINTS)
        return -1;
    else
        return dist;
}// end of _cvPar_CloseParIntersection

/* ///////////////////////////////////////////////////////////////////////////////////////
//                           Subsidiary functions                                       //
/////////////////////////////////////////////////////////////////////////////////////// */

CV_INLINE
void _cvMakeTwinEdge(pCvVoronoiEdge pEdge2,
                    pCvVoronoiEdge pEdge1)
{
    pEdge2->direction = pEdge1->direction;
    pEdge2->parabola = pEdge1->parabola;
    pEdge2->node1 = pEdge1->node2;
    pEdge2->twin_edge = pEdge1;
    pEdge1->twin_edge = pEdge2;
}//end of _cvMakeTwinEdge

CV_INLINE
void _cvStickEdgeLeftBegin(pCvVoronoiEdge pEdge,
                          pCvVoronoiEdge pEdge_left_prev,
                          pCvVoronoiSite pSite_left)
{
    pEdge->prev_edge = pEdge_left_prev;
    pEdge->site = pSite_left;
    if(pEdge_left_prev == NULL)
        pSite_left->edge2 = pEdge;
    else
    {
        pEdge_left_prev->node2 = pEdge->node1;
        pEdge_left_prev->next_edge = pEdge;
    }
}//end of _cvStickEdgeLeftBegin

CV_INLINE
void _cvStickEdgeRightBegin(pCvVoronoiEdge pEdge,
                          pCvVoronoiEdge pEdge_right_next,
                          pCvVoronoiSite pSite_right)
{
    pEdge->next_edge = pEdge_right_next;
    pEdge->site = pSite_right;
    if(pEdge_right_next == NULL)
        pSite_right->edge1 = pEdge;
    else
    {
        pEdge_right_next->node1 = pEdge->node2;
        pEdge_right_next->prev_edge = pEdge;
    }
}// end of _cvStickEdgeRightBegin

CV_INLINE
void _cvStickEdgeLeftEnd(pCvVoronoiEdge pEdge,
                        pCvVoronoiEdge pEdge_left_next,
                        pCvVoronoiSite pSite_left)
{
    pEdge->next_edge = pEdge_left_next;
    if(pEdge_left_next == NULL)
        pSite_left->edge1 = pEdge;
    else
    {
        pEdge_left_next->node1 = pEdge->node2;
        pEdge_left_next->prev_edge = pEdge;
    }
}//end of _cvStickEdgeLeftEnd

CV_INLINE
void _cvStickEdgeRightEnd(pCvVoronoiEdge pEdge,
                         pCvVoronoiEdge pEdge_right_prev,
                         pCvVoronoiSite pSite_right)
{
    pEdge->prev_edge = pEdge_right_prev;
    if(pEdge_right_prev == NULL)
        pSite_right->edge2 = pEdge;
    else
    {
        pEdge_right_prev->node2 = pEdge->node1;
        pEdge_right_prev->next_edge = pEdge;
    }
}//end of _cvStickEdgeRightEnd

template <class T> CV_INLINE
void _cvInitVoronoiNode(pCvVoronoiNode pNode,
                        T pPoint,
                        float radius)
{
    pNode->node.x = (float)pPoint->x;
    pNode->node.y = (float)pPoint->y;
    pNode->radius = radius;
}//end of _cvInitVoronoiNode

CV_INLINE
void _cvInitVoronoiSite(pCvVoronoiSite pSite,
                       pCvVoronoiNode pNode1,
                       pCvVoronoiNode pNode2,
                       pCvVoronoiSite pPrev_site)
{
    pSite->node1 = pNode1;
    pSite->node2 = pNode2;
    pSite->prev_site = pPrev_site;
}//end of _cvInitVoronoiSite

template <class T> CV_INLINE
T _cvSeqPush(CvSeq* Seq, T pElem)
{
    cvSeqPush(Seq, pElem);
    return (T)(Seq->ptr - Seq->elem_size);
//  return (T)cvGetSeqElem(Seq, Seq->total - 1,NULL);
}//end of _cvSeqPush

template <class T> CV_INLINE
T _cvSeqPushFront(CvSeq* Seq, T pElem)
{
    cvSeqPushFront(Seq,pElem);
    return (T)Seq->first->data;
//  return (T)cvGetSeqElem(Seq,0,NULL);
}//end of _cvSeqPushFront

CV_INLINE
void _cvTwinNULLLeft(pCvVoronoiEdge pEdge_left_cur,
                    pCvVoronoiEdge pEdge_left)
{
    while(pEdge_left_cur!=pEdge_left)
    {
        if(pEdge_left_cur->twin_edge!=NULL)
            pEdge_left_cur->twin_edge->twin_edge = NULL;
        pEdge_left_cur = pEdge_left_cur->next_edge;
    }
}//end of _cvTwinNULLLeft

CV_INLINE
void _cvTwinNULLRight(pCvVoronoiEdge pEdge_right_cur,
                     pCvVoronoiEdge pEdge_right)
{
    while(pEdge_right_cur!=pEdge_right)
    {
        if(pEdge_right_cur->twin_edge!=NULL)
            pEdge_right_cur->twin_edge->twin_edge = NULL;
        pEdge_right_cur = pEdge_right_cur->prev_edge;
    }
}//end of _cvTwinNULLRight

CV_INLINE
void _cvSeqPushInOrder(CvVoronoiDiagramInt* pVoronoiDiagram, pCvVoronoiHole pHole)
{
    pHole = _cvSeqPush(pVoronoiDiagram->HoleSeq, pHole);
    if(pVoronoiDiagram->HoleSeq->total == 1)
    {
        pVoronoiDiagram->top_hole = pHole;
        return;
    }

    pCvVoronoiHole pTopHole = pVoronoiDiagram->top_hole;
    pCvVoronoiHole pCurrHole;
    if(pTopHole->x_coord >= pHole->x_coord)
    {
        pHole->next_hole = pTopHole;
        pVoronoiDiagram->top_hole = pHole;
        return;
    }

    for(pCurrHole = pTopHole; \
        pCurrHole->next_hole != NULL; \
        pCurrHole = pCurrHole->next_hole)
        if(pCurrHole->next_hole->x_coord >= pHole->x_coord)
            break;
    pHole->next_hole = pCurrHole->next_hole;
    pCurrHole->next_hole = pHole;
}//end of _cvSeqPushInOrder

CV_INLINE
pCvVoronoiEdge _cvDivideRightEdge(pCvVoronoiEdge pEdge,pCvVoronoiNode pNode, CvSeq* EdgeSeq)
{
    CvVoronoiEdgeInt Edge1 = *pEdge;
    CvVoronoiEdgeInt Edge2 = *pEdge->twin_edge;
    pCvVoronoiEdge pEdge1, pEdge2;

    pEdge1 = _cvSeqPush(EdgeSeq, &Edge1);
    pEdge2 = _cvSeqPush(EdgeSeq, &Edge2);

    if(pEdge1->next_edge != NULL)
        pEdge1->next_edge->prev_edge = pEdge1;
    pEdge1->prev_edge = NULL;

    if(pEdge2->prev_edge != NULL)
        pEdge2->prev_edge->next_edge = pEdge2;
    pEdge2->next_edge = NULL;

    pEdge1->node1 = pEdge2->node2= pNode;
    pEdge1->twin_edge = pEdge2;
    pEdge2->twin_edge = pEdge1;
    return pEdge2;
}//end of _cvDivideRightEdge

CV_INLINE
pCvVoronoiEdge _cvDivideLeftEdge(pCvVoronoiEdge pEdge,pCvVoronoiNode pNode, CvSeq* EdgeSeq)
{
    CvVoronoiEdgeInt Edge1 = *pEdge;
    CvVoronoiEdgeInt Edge2 = *pEdge->twin_edge;
    pCvVoronoiEdge pEdge1, pEdge2;

    pEdge1 = _cvSeqPush(EdgeSeq, &Edge1);
    pEdge2 = _cvSeqPush(EdgeSeq, &Edge2);

    if(pEdge2->next_edge != NULL)
        pEdge2->next_edge->prev_edge = pEdge2;
    pEdge2->prev_edge = NULL;

    if(pEdge1->prev_edge != NULL)
        pEdge1->prev_edge->next_edge = pEdge1;
    pEdge1->next_edge = NULL;

    pEdge1->node2 = pEdge2->node1= pNode;
    pEdge1->twin_edge = pEdge2;
    pEdge2->twin_edge = pEdge1;
    return pEdge2;
}//end of _cvDivideLeftEdge

template<class T> CV_INLINE
T _cvWriteSeqElem(T pElem, CvSeqWriter &writer)
{
    if( writer.ptr >= writer.block_max )
          cvCreateSeqBlock( &writer);

    T ptr = (T)writer.ptr;
    memcpy(writer.ptr, pElem, sizeof(*pElem));
    writer.ptr += sizeof(*pElem);
    return ptr;
}//end of _cvWriteSeqElem

/* ///////////////////////////////////////////////////////////////////////////////////////
//                           Mathematical functions                                     //
/////////////////////////////////////////////////////////////////////////////////////// */

template<class T> CV_INLINE
void _cvCalcPointImage(pCvPointFloat pImgPoint,pCvPointFloat pPoint,T* A)
{
    pImgPoint->x = (float)(A[0]*pPoint->x + A[1]*pPoint->y + A[2]);
    pImgPoint->y = (float)(A[3]*pPoint->x + A[4]*pPoint->y + A[5]);
}//end of _cvCalcPointImage

template <class T> CV_INLINE
void _cvSwap(T &x, T &y)
{
    T z; z=x; x=y; y=z;
}//end of _cvSwap

template <class T> CV_INLINE
int _cvCalcOrtogInverse(T* B, T* A)
{
    int sign_det = SIGN(A[0]*A[4] - A[1]*A[3]);

    if(sign_det)
    {
        B[0] =  A[4]*sign_det;
        B[1] = -A[1]*sign_det;
        B[3] = -A[3]*sign_det;
        B[4] =  A[0]*sign_det;
        B[2] = - (B[0]*A[2]+B[1]*A[5]);
        B[5] = - (B[3]*A[2]+B[4]*A[5]);
        return 1;
    }
    else
        return 0;
}//end of _cvCalcOrtogInverse

template<class T> CV_INLINE
void _cvCalcVectorImage(pCvDirection pImgVector,pCvDirection pVector,T* A)
{
    pImgVector->x = A[0]*pVector->x + A[1]*pVector->y;
    pImgVector->y = A[3]*pVector->x + A[4]*pVector->y;
}//end of _cvCalcVectorImage

template <class T> CV_INLINE
void _cvCalcComposition(T* Result,T* A,T* B)
{
    Result[0] = A[0]*B[0] + A[1]*B[3];
    Result[1] = A[0]*B[1] + A[1]*B[4];
    Result[3] = A[3]*B[0] + A[4]*B[3];
    Result[4] = A[3]*B[1] + A[4]*B[4];
    Result[2] = A[0]*B[2] + A[1]*B[5] + A[2];
    Result[5] = A[3]*B[2] + A[4]*B[5] + A[5];
}//end of _cvCalcComposition

CV_INLINE
float _cvCalcDist(pCvPointFloat pPoint, pCvVoronoiSite pSite)
{
    if(pSite->node1==pSite->node2)
        return _cvPPDist(pPoint,&(pSite->node1->node));
    else
        return _cvPLDist(pPoint,&(pSite->node1->node),pSite->direction);
}//end of _cvCalcComposition

CV_INLINE
float _cvPPDist(pCvPointFloat pPoint1,pCvPointFloat pPoint2)
{
    float delta_x = pPoint1->x - pPoint2->x;
    float delta_y = pPoint1->y - pPoint2->y;
    return (float)sqrt((double)delta_x*delta_x + delta_y*delta_y);
}//end of _cvPPDist


CV_INLINE
float _cvPLDist(pCvPointFloat pPoint,pCvPointFloat pPoint1,pCvDirection pDirection)
{
    return (float)fabs(pDirection->x*(pPoint->y - pPoint1->y) -
                     pDirection->y*(pPoint->x - pPoint1->x));
}//end of _cvPLDist

template <class T>
int _cvSolveEqu2thR(T c2, T c1, T c0, T* X)
{
    const T eps = (T)1e-6;
    if(fabs(c2)<eps)
        return _cvSolveEqu1th(c1,c0,X);

    T Discr = c1*c1 - c2*c0*4;
    if(Discr<-eps)
        return 0;
    Discr = (T)sqrt((double)fabs(Discr));

    if(fabs(Discr)<eps)
    {
        X[0] = -c1/(c2*2);
        if(fabs(X[0])<eps)
            X[0]=0;
        return 1;
    }
    else
    {
        if(c1 >=0)
        {
            if(c2>0)
            {
                X[0] = (-c1 - Discr)/(2*c2);
                X[1] = -2*c0/(c1+Discr);
                return 2;
            }
            else
            {
                X[1] = (-c1 - Discr)/(2*c2);
                X[0] = -2*c0/(c1+Discr);
                return 2;
            }
        }
        else
        {
            if(c2>0)
            {
                X[1] = (-c1 + Discr)/(2*c2);
                X[0] = -2*c0/(c1-Discr);
                return 2;
            }
            else
            {
                X[0] = (-c1 + Discr)/(2*c2);
                X[1] = -2*c0/(c1-Discr);
                return 2;
            }
        }
    }
}//end of _cvSolveEqu2thR

template <class T> CV_INLINE
int _cvSolveEqu1th(T c1, T c0, T* X)
{
    const T eps = (T)1e-6;
    if(fabs(c1)<eps)
        return 0;

    X[0] = -c0/c1;
    return 1;
}//end of _cvSolveEqu1th
