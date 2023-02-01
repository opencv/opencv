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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include <stdlib.h>
#include <math.h>
#include <vector>

/****************************************************************************************\
*                                   For EMDL1 Framework                                 *
\****************************************************************************************/
typedef struct cvEMDEdge* cvPEmdEdge;
typedef struct cvEMDNode* cvPEmdNode;
struct cvEMDNode
{
    int pos[3]; // grid position
    float d; // initial value
    int u;
    // tree maintenance
    int iLevel; // level in the tree, 0 means root
    cvPEmdNode pParent; // pointer to its parent
    cvPEmdEdge pChild;
    cvPEmdEdge pPEdge; // point to the edge coming out from its parent
};
struct cvEMDEdge
{
    float flow; // initial value
    int iDir; // 1:outward, 0:inward
    // tree maintenance
    cvPEmdNode pParent; // point to its parent
    cvPEmdNode pChild; // the child node
    cvPEmdEdge pNxt; // next child/edge
};
typedef std::vector<cvEMDNode> cvEMDNodeArray;
typedef std::vector<cvEMDEdge> cvEMDEdgeArray;
typedef std::vector<cvEMDNodeArray> cvEMDNodeArray2D;
typedef std::vector<cvEMDEdgeArray> cvEMDEdgeArray2D;
typedef std::vector<float> floatArray;
typedef std::vector<floatArray> floatArray2D;

/****************************************************************************************\
*                                   EMDL1 Class                                         *
\****************************************************************************************/
class EmdL1
{
public:
    EmdL1()
    {
        m_pRoot	= NULL;
        binsDim1 = 0;
        binsDim2 = 0;
        binsDim3 = 0;
        dimension = 0;
        nMaxIt = 500;

        m_pLeave = 0;
        m_iEnter = 0;
        nNBV = 0;
        m_nItr = 0;
        m_iTo = 0;
        m_iFrom = 0;
        m_pEnter = 0;
    }

    ~EmdL1()
    {
    }

    float getEMDL1(cv::Mat &sig1, cv::Mat &sig2);
    void setMaxIteration(int _nMaxIt);

private:
    //-- SubFunctions called in the EMD algorithm
    bool initBaseTrees(int n1=0, int n2=0, int n3=0);
    bool fillBaseTrees(float *H1, float *H2);
    bool greedySolution();
    bool greedySolution2();
    bool greedySolution3();
    void initBVTree();
    void updateSubtree(cvPEmdNode pRoot);
    bool isOptimal();
    void findNewSolution();
    void findLoopFromEnterBV();
    float compuTotalFlow();

private:
    int dimension;
    int binsDim1, binsDim2, binsDim3; // the histogram contains m_n1 rows and m_n2 columns
    int nNBV; // number of Non-Basic Variables (NBV)
    int nMaxIt;
    cvEMDNodeArray2D m_Nodes; // all nodes
    cvEMDEdgeArray2D m_EdgesRight; // all edges to right
    cvEMDEdgeArray2D m_EdgesUp; // all edges to upward
    std::vector<cvEMDNodeArray2D>	m_3dNodes; // all nodes for 3D
    std::vector<cvEMDEdgeArray2D>	m_3dEdgesRight; // all edges to right, 3D
    std::vector<cvEMDEdgeArray2D>	m_3dEdgesUp; // all edges to upward, 3D
    std::vector<cvEMDEdgeArray2D>	m_3dEdgesDeep; // all edges to deep, 3D
    std::vector<cvPEmdEdge> m_NBVEdges; // pointers to all NON-BV edges
    std::vector<cvPEmdNode> m_auxQueue; // auxiliary node queue
    cvPEmdNode m_pRoot; // root of the BV Tree
    cvPEmdEdge m_pEnter; // Enter BV edge
    int m_iEnter; // Enter BV edge, index in m_NBVEdges
    cvPEmdEdge m_pLeave; // Leave BV edge
    int m_nItr; // number of iteration
    // auxiliary variables for searching a new loop
    std::vector<cvPEmdEdge> m_fromLoop;
    std::vector<cvPEmdEdge> m_toLoop;
    int	m_iFrom;
    int m_iTo;
};
