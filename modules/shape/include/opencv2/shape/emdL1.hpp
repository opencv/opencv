/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifndef __OPENCV_EMD_L1_HPP__
#define __OPENCV_EMD_L1_HPP__

#include <vector>

namespace cv
{
/****************************************************************************************\
*                                   For EMDL1 Framework                                 *
\****************************************************************************************/
typedef struct EMDEdge * PEmdEdge;
typedef struct EMDNode * PEmdNode;
typedef struct EMDNode
{
    int pos[3]; // grid position
    double d; // initial value
    int u;
    // tree maintainance
    int iLevel; // level in the tree, 0 means root
    PEmdNode pParent; // pointer to its parent
    PEmdEdge pChild;
    PEmdEdge pPEdge; // point to the edge coming out from its parent
}EMDNode;
typedef struct EMDEdge
{
    double flow; // initial value
    int iDir; // 1:outward, 0:inward
    // tree maintainance
    PEmdNode pParent; // point to its parent
    PEmdNode pChild; // the child node
    PEmdEdge pNxt; // next child/edge
}EMDEdge;
typedef std::vector<EMDNode> EMDNodeArray;
typedef std::vector<EMDEdge> EMDEdgeArray;
typedef std::vector<EMDNodeArray> EMDNodeArray2D;
typedef std::vector<EMDEdgeArray> EMDEdgeArray2D;
typedef std::vector<double> EMDTYPEArray;
typedef std::vector<EMDTYPEArray> EMDTYPEArray2D;

/****************************************************************************************\
*                                   EMDL1 Functions                                     *
\****************************************************************************************/
class EmdL1
{
public:
    EmdL1();
    ~EmdL1();
    double getEMDL1(double *H1, double *H2, int n1, int n2, int n3=0);
    void setMaxIteration(int nMaxIt){ m_nMaxIt=nMaxIt; }
private:
    //-- SubFunctions called in the EMD algorithm
    bool initMemory(int n1=0, int n2=0, int n3=0);
    bool initialize(double *H1, double *H2);
    bool greedySolution();
    bool greedySolution2();
    bool greedySolution3();
    void initBVTree(); // initialize BVTree from the initial BF solution
    void updateSubtree(PEmdNode pRoot);
    bool isOptimal();
    void findNewSolution();
    void findLoopFromEnterBV();
    double compuTotalFlow(); // Computing the total flow as the final distance
private:
    int m_nDim;
    int m_n1, m_n2, m_n3; // the hitogram contains m_n1 rows and m_n2 columns
    int m_nNBV; // number of Non-Basic Variables (NBV)
    int m_nMaxIt;
    EMDNodeArray2D m_Nodes; // all nodes
    EMDEdgeArray2D m_EdgesRight; // all edges to right
    EMDEdgeArray2D m_EdgesUp; // all edges to upward
    std::vector<EMDNodeArray2D>	m_3dNodes; // all nodes for 3D
    std::vector<EMDEdgeArray2D>	m_3dEdgesRight; // all edges to right, 3D
    std::vector<EMDEdgeArray2D>	m_3dEdgesUp; // all edges to upward, 3D
    std::vector<EMDEdgeArray2D>	m_3dEdgesDeep; // all edges to deep, 3D
    std::vector<PEmdEdge> m_NBVEdges; // pointers to all NON-BV edges
    std::vector<PEmdNode> m_auxQueue; // auxiliary node queue
    PEmdNode m_pRoot; // root of the BV Tree
    PEmdEdge m_pEnter; // Enter BV edge
    int m_iEnter; // Enter BV edge, index in m_NBVEdges
    PEmdEdge m_pLeave; // Leave BV edge
    int m_nItr; // number of iteration
    // auxiliary variables for searching a new loop
    std::vector<PEmdEdge> m_fromLoop;
    std::vector<PEmdEdge> m_toLoop;
    int	m_iFrom;
    int m_iTo;
};

}//namespace cv

#endif 
