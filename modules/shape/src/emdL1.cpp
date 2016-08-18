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

/*
 * Implementation of an optimized EMD for histograms based in
 * the papers "EMD-L1: An efficient and Robust Algorithm
 * for comparing histogram-based descriptors", by Haibin Ling and
 * Kazunori Okuda; and "The Earth Mover's Distance is the Mallows
 * Distance: Some Insights from Statistics", by Elizaveta Levina and
 * Peter Bickel, based on HAIBIN LING AND KAZUNORI OKADA implementation.
 */

#include "precomp.hpp"
#include "emdL1_def.hpp"
#include <limits>

/****************************************************************************************\
*                                   EMDL1 Class                                         *
\****************************************************************************************/

float EmdL1::getEMDL1(cv::Mat &sig1, cv::Mat &sig2)
{
    // Initialization
    CV_Assert((sig1.rows==sig2.rows) && (sig1.cols==sig2.cols) && (!sig1.empty()) && (!sig2.empty()));
    if(!initBaseTrees(sig1.rows, 1))
        return -1;

    float *H1=new float[sig1.rows], *H2 = new float[sig2.rows];
    for (int ii=0; ii<sig1.rows; ii++)
    {
        H1[ii]=sig1.at<float>(ii,0);
        H2[ii]=sig2.at<float>(ii,0);
    }

    fillBaseTrees(H1,H2); // Initialize histograms
    greedySolution(); // Construct an initial Basic Feasible solution
    initBVTree(); // Initialize BVTree

    // Iteration
    bool bOptimal = false;
    m_nItr = 0;
    while(!bOptimal && m_nItr<nMaxIt)
    {
        // Derive U=(u_ij) for row i and column j
        if(m_nItr==0) updateSubtree(m_pRoot);
        else updateSubtree(m_pEnter->pChild);

        // Optimality test
        bOptimal = isOptimal();

        // Find new solution
        if(!bOptimal)
            findNewSolution();
        ++m_nItr;
    }
    delete [] H1;
    delete [] H2;
    // Output the total flow
    return compuTotalFlow();
}

void EmdL1::setMaxIteration(int _nMaxIt)
{
    nMaxIt=_nMaxIt;
}

//-- SubFunctions called in the EMD algorithm
bool EmdL1::initBaseTrees(int n1, int n2, int n3)
{
    if(binsDim1==n1 && binsDim2==n2 && binsDim3==n3)
        return true;
    binsDim1 = n1;
    binsDim2 = n2;
    binsDim3 = n3;
    if(binsDim1==0 || binsDim2==0) dimension = 0;
    else dimension	= (binsDim3==0)?2:3;

    if(dimension==2)
    {
        m_Nodes.resize(binsDim1);
        m_EdgesUp.resize(binsDim1);
        m_EdgesRight.resize(binsDim1);
        for(int i1=0; i1<binsDim1; i1++)
        {
            m_Nodes[i1].resize(binsDim2);
            m_EdgesUp[i1].resize(binsDim2);
            m_EdgesRight[i1].resize(binsDim2);
        }
        m_NBVEdges.resize(binsDim1*binsDim2*4+2);
        m_auxQueue.resize(binsDim1*binsDim2+2);
        m_fromLoop.resize(binsDim1*binsDim2+2);
        m_toLoop.resize(binsDim1*binsDim2+2);
    }
    else if(dimension==3)
    {
        m_3dNodes.resize(binsDim1);
        m_3dEdgesUp.resize(binsDim1);
        m_3dEdgesRight.resize(binsDim1);
        m_3dEdgesDeep.resize(binsDim1);
        for(int i1=0; i1<binsDim1; i1++)
        {
            m_3dNodes[i1].resize(binsDim2);
            m_3dEdgesUp[i1].resize(binsDim2);
            m_3dEdgesRight[i1].resize(binsDim2);
            m_3dEdgesDeep[i1].resize(binsDim2);
            for(int i2=0; i2<binsDim2; i2++)
            {
                m_3dNodes[i1][i2].resize(binsDim3);
                m_3dEdgesUp[i1][i2].resize(binsDim3);
                m_3dEdgesRight[i1][i2].resize(binsDim3);
                m_3dEdgesDeep[i1][i2].resize(binsDim3);
            }
        }
        m_NBVEdges.resize(binsDim1*binsDim2*binsDim3*6+4);
        m_auxQueue.resize(binsDim1*binsDim2*binsDim3+4);
        m_fromLoop.resize(binsDim1*binsDim2*binsDim3+4);
        m_toLoop.resize(binsDim1*binsDim2*binsDim3+2);
    }
    else
        return false;

    return true;
}

bool EmdL1::fillBaseTrees(float *H1, float *H2)
{
    //- Set global counters
    m_pRoot	= NULL;
    // Graph initialization
    float *p1 = H1;
    float *p2 = H2;
    if(dimension==2)
    {
        for(int c=0; c<binsDim2; c++)
        {
            for(int r=0; r<binsDim1; r++)
            {
                //- initialize nodes and links
                m_Nodes[r][c].pos[0] = r;
                m_Nodes[r][c].pos[1] = c;
                m_Nodes[r][c].d = *(p1++)-*(p2++);
                m_Nodes[r][c].pParent = NULL;
                m_Nodes[r][c].pChild = NULL;
                m_Nodes[r][c].iLevel = -1;

                //- initialize edges
                // to the right
                m_EdgesRight[r][c].pParent = &(m_Nodes[r][c]);
                m_EdgesRight[r][c].pChild = &(m_Nodes[r][(c+1)%binsDim2]);
                m_EdgesRight[r][c].flow	= 0;
                m_EdgesRight[r][c].iDir	= 1;
                m_EdgesRight[r][c].pNxt	= NULL;

                // to the upward
                m_EdgesUp[r][c].pParent	= &(m_Nodes[r][c]);
                m_EdgesUp[r][c].pChild	= &(m_Nodes[(r+1)%binsDim1][c]);
                m_EdgesUp[r][c].flow = 0;
                m_EdgesUp[r][c].iDir = 1;
                m_EdgesUp[r][c].pNxt = NULL;
            }
        }
    }
    else if(dimension==3)
    {
        for(int z=0; z<binsDim3; z++)
        {
            for(int c=0; c<binsDim2; c++)
            {
                for(int r=0; r<binsDim1; r++)
                {
                    //- initialize nodes and edges
                    m_3dNodes[r][c][z].pos[0] = r;
                    m_3dNodes[r][c][z].pos[1] = c;
                    m_3dNodes[r][c][z].pos[2] = z;
                    m_3dNodes[r][c][z].d = *(p1++)-*(p2++);
                    m_3dNodes[r][c][z].pParent = NULL;
                    m_3dNodes[r][c][z].pChild = NULL;
                    m_3dNodes[r][c][z].iLevel = -1;

                    //- initialize edges
                    // to the upward
                    m_3dEdgesUp[r][c][z].pParent= &(m_3dNodes[r][c][z]);
                    m_3dEdgesUp[r][c][z].pChild	= &(m_3dNodes[(r+1)%binsDim1][c][z]);
                    m_3dEdgesUp[r][c][z].flow = 0;
                    m_3dEdgesUp[r][c][z].iDir = 1;
                    m_3dEdgesUp[r][c][z].pNxt = NULL;

                    // to the right
                    m_3dEdgesRight[r][c][z].pParent	= &(m_3dNodes[r][c][z]);
                    m_3dEdgesRight[r][c][z].pChild	= &(m_3dNodes[r][(c+1)%binsDim2][z]);
                    m_3dEdgesRight[r][c][z].flow	= 0;
                    m_3dEdgesRight[r][c][z].iDir	= 1;
                    m_3dEdgesRight[r][c][z].pNxt	= NULL;

                    // to the deep
                    m_3dEdgesDeep[r][c][z].pParent	= &(m_3dNodes[r][c][z]);
                    m_3dEdgesDeep[r][c][z].pChild	= &(m_3dNodes[r][c])[(z+1)%binsDim3];
                    m_3dEdgesDeep[r][c][z].flow = 0;
                    m_3dEdgesDeep[r][c][z].iDir = 1;
                    m_3dEdgesDeep[r][c][z].pNxt = NULL;
                }
            }
        }
    }
    return true;
}

bool EmdL1::greedySolution()
{
    return dimension==2?greedySolution2():greedySolution3();
}

bool EmdL1::greedySolution2()
{
    //- Prepare auxiliary array, D=H1-H2
    int	c,r;
    floatArray2D D(binsDim1);
    for(r=0; r<binsDim1; r++)
    {
        D[r].resize(binsDim2);
        for(c=0; c<binsDim2; c++) D[r][c] = m_Nodes[r][c].d;
    }
    // compute integrated values along each dimension
    std::vector<float> d2s(binsDim2);
    d2s[0] = 0;
    for(c=0; c<binsDim2-1; c++)
    {
        d2s[c+1] = d2s[c];
        for(r=0; r<binsDim1; r++) d2s[c+1]-= D[r][c];
    }

    std::vector<float> d1s(binsDim1);
    d1s[0] = 0;
    for(r=0; r<binsDim1-1; r++)
    {
        d1s[r+1] = d1s[r];
        for(c=0; c<binsDim2; c++) d1s[r+1]-= D[r][c];
    }

    //- Greedy algorithm for initial solution
    cvPEmdEdge pBV;
    float dFlow;
    bool bUpward = false;
    nNBV = 0; // number of NON-BV edges

    for(c=0; c<binsDim2-1; c++)
    for(r=0; r<binsDim1; r++)
    {
        dFlow = D[r][c];
        bUpward = (r<binsDim1-1) && (fabs(dFlow+d2s[c+1]) > fabs(dFlow+d1s[r+1]));	// Move upward or right

        // modify basic variables, record BV and related values
        if(bUpward)
        {
            // move to up
            pBV	= &(m_EdgesUp[r][c]);
            m_NBVEdges[nNBV++]	= &(m_EdgesRight[r][c]);
            D[r+1][c] += dFlow;		// auxilary matrix maintanence
            d1s[r+1] += dFlow;		// auxilary matrix maintanence
        }
        else
        {
            // move to right, no other choice
            pBV	= &(m_EdgesRight[r][c]);
            if(r<binsDim1-1)
                m_NBVEdges[nNBV++]	= &(m_EdgesUp[r][c]);

            D[r][c+1] += dFlow;		// auxilary matrix maintanence
            d2s[c+1] += dFlow;		// auxilary matrix maintanence
        }
        pBV->pParent->pChild = pBV;
        pBV->flow = fabs(dFlow);
        pBV->iDir = dFlow>0;		// 1:outward, 0:inward
    }

    //- rightmost column, no choice but move upward
    c = binsDim2-1;
    for(r=0; r<binsDim1-1; r++)
    {
        dFlow = D[r][c];
        pBV = &(m_EdgesUp[r][c]);
        D[r+1][c] += dFlow;		// auxilary matrix maintanence
        pBV->pParent->pChild= pBV;
        pBV->flow = fabs(dFlow);
        pBV->iDir = dFlow>0;		// 1:outward, 0:inward
    }
    return true;
}

bool EmdL1::greedySolution3()
{
    //- Prepare auxiliary array, D=H1-H2
    int i1,i2,i3;
    std::vector<floatArray2D> D(binsDim1);
    for(i1=0; i1<binsDim1; i1++)
    {
        D[i1].resize(binsDim2);
        for(i2=0; i2<binsDim2; i2++)
        {
            D[i1][i2].resize(binsDim3);
            for(i3=0; i3<binsDim3; i3++)
                D[i1][i2][i3] = m_3dNodes[i1][i2][i3].d;
        }
    }

    // compute integrated values along each dimension
    std::vector<float> d1s(binsDim1);
    d1s[0] = 0;
    for(i1=0; i1<binsDim1-1; i1++)
    {
        d1s[i1+1] = d1s[i1];
        for(i2=0; i2<binsDim2; i2++)
        {
            for(i3=0; i3<binsDim3; i3++)
                d1s[i1+1] -= D[i1][i2][i3];
        }
    }

    std::vector<float> d2s(binsDim2);
    d2s[0] = 0;
    for(i2=0; i2<binsDim2-1; i2++)
    {
        d2s[i2+1] = d2s[i2];
        for(i1=0; i1<binsDim1; i1++)
        {
            for(i3=0; i3<binsDim3; i3++)
                d2s[i2+1] -= D[i1][i2][i3];
        }
    }

    std::vector<float> d3s(binsDim3);
    d3s[0] = 0;
    for(i3=0; i3<binsDim3-1; i3++)
    {
        d3s[i3+1]	= d3s[i3];
        for(i1=0; i1<binsDim1; i1++)
        {
            for(i2=0; i2<binsDim2; i2++)
                d3s[i3+1] -= D[i1][i2][i3];
        }
    }

    //- Greedy algorithm for initial solution
    cvPEmdEdge pBV;
    float dFlow, f1,f2,f3;
    nNBV = 0; // number of NON-BV edges
    for(i3=0; i3<binsDim3; i3++)
    {
        for(i2=0; i2<binsDim2; i2++)
        {
            for(i1=0; i1<binsDim1; i1++)
            {
                if(i3==binsDim3-1 && i2==binsDim2-1 && i1==binsDim1-1) break;

                //- determine which direction to move, either right or upward
                dFlow = D[i1][i2][i3];
                f1 = (i1<(binsDim1-1))?fabs(dFlow+d1s[i1+1]):std::numeric_limits<float>::max();
                f2 = (i2<(binsDim2-1))?fabs(dFlow+d2s[i2+1]):std::numeric_limits<float>::max();
                f3 = (i3<(binsDim3-1))?fabs(dFlow+d3s[i3+1]):std::numeric_limits<float>::max();

                if(f1<f2 && f1<f3)
                {
                    pBV	= &(m_3dEdgesUp[i1][i2][i3]); // up
                    if(i2<binsDim2-1) m_NBVEdges[nNBV++] = &(m_3dEdgesRight[i1][i2][i3]);	// right
                    if(i3<binsDim3-1) m_NBVEdges[nNBV++] = &(m_3dEdgesDeep[i1][i2][i3]); // deep
                    D[i1+1][i2][i3]	+= dFlow; // maintain auxilary matrix
                    d1s[i1+1] += dFlow;
                }
                else if(f2<f3)
                {
                    pBV	= &(m_3dEdgesRight[i1][i2][i3]); // right
                    if(i1<binsDim1-1) m_NBVEdges[nNBV++] = &(m_3dEdgesUp[i1][i2][i3]); // up
                    if(i3<binsDim3-1) m_NBVEdges[nNBV++] = &(m_3dEdgesDeep[i1][i2][i3]); // deep
                    D[i1][i2+1][i3]	+= dFlow; // maintain auxilary matrix
                    d2s[i2+1] += dFlow;
                }
                else
                {
                    pBV	= &(m_3dEdgesDeep[i1][i2][i3]); // deep
                    if(i2<binsDim2-1) m_NBVEdges[nNBV++] = &(m_3dEdgesRight[i1][i2][i3]);	// right
                    if(i1<binsDim1-1) m_NBVEdges[nNBV++] = &(m_3dEdgesUp[i1][i2][i3]); // up
                    D[i1][i2][i3+1]	+= dFlow; // maintain auxilary matrix
                    d3s[i3+1] += dFlow;
                }

                pBV->flow = fabs(dFlow);
                pBV->iDir = dFlow>0; // 1:outward, 0:inward
                pBV->pParent->pChild= pBV;
            }
        }
    }
    return true;
}

void EmdL1::initBVTree()
{
    // initialize BVTree from the initial BF solution
    //- Using the center of the graph as the root
    int r = (int)(0.5*binsDim1-.5);
    int c = (int)(0.5*binsDim2-.5);
    int z = (int)(0.5*binsDim3-.5);
    m_pRoot	= dimension==2 ? &(m_Nodes[r][c]) : &(m_3dNodes[r][c][z]);
    m_pRoot->u = 0;
    m_pRoot->iLevel	= 0;
    m_pRoot->pParent= NULL;
    m_pRoot->pPEdge	= NULL;

    //- Prepare a queue
    m_auxQueue[0] = m_pRoot;
    int nQueue = 1; // length of queue
    int iQHead = 0; // head of queue

    //- Recursively build subtrees
    cvPEmdEdge pCurE=NULL, pNxtE=NULL;
    cvPEmdNode pCurN=NULL, pNxtN=NULL;
    int	nBin = binsDim1*binsDim2*std::max(binsDim3,1);
    while(iQHead<nQueue && nQueue<nBin)
    {
        pCurN = m_auxQueue[iQHead++];	// pop out from queue
        r = pCurN->pos[0];
        c = pCurN->pos[1];
        z = pCurN->pos[2];

        // check connection from itself
        pCurE = pCurN->pChild;	// the initial child from initial solution
        if(pCurE)
        {
            pNxtN = pCurE->pChild;
            pNxtN->pParent = pCurN;
            pNxtN->pPEdge = pCurE;
            m_auxQueue[nQueue++] = pNxtN;
        }

        // check four neighbor nodes
        int	nNB	= dimension==2?4:6;
        for(int k=0;k<nNB;k++)
        {
            if(dimension==2)
            {
                if(k==0 && c>0) pNxtN = &(m_Nodes[r][c-1]);		// left
                else if(k==1 && r>0) pNxtN	= &(m_Nodes[r-1][c]);		// down
                else if(k==2 && c<binsDim2-1) pNxtN	= &(m_Nodes[r][c+1]);		// right
                else if(k==3 && r<binsDim1-1) pNxtN	= &(m_Nodes[r+1][c]);		// up
                else continue;
            }
            else if(dimension==3)
            {
                if(k==0 && c>0) pNxtN = &(m_3dNodes[r][c-1][z]); // left
                else if(k==1 && c<binsDim2-1) pNxtN	= &(m_3dNodes[r][c+1][z]); // right
                else if(k==2 && r>0) pNxtN	= &(m_3dNodes[r-1][c][z]); // down
                else if(k==3 && r<binsDim1-1) pNxtN	= &(m_3dNodes[r+1][c][z]); // up
                else if(k==4 && z>0) pNxtN = &(m_3dNodes[r][c][z-1]); // shallow
                else if(k==5 && z<binsDim3-1) pNxtN	= &(m_3dNodes[r][c][z+1]); // deep
                else continue;
            }
            if(pNxtN != pCurN->pParent)
            {
                pNxtE = pNxtN->pChild;
                if(pNxtE && pNxtE->pChild==pCurN) // has connection
                {
                    pNxtN->pParent = pCurN;
                    pNxtN->pPEdge = pNxtE;
                    pNxtN->pChild = NULL;
                    m_auxQueue[nQueue++] = pNxtN;

                    pNxtE->pParent = pCurN; // reverse direction
                    pNxtE->pChild = pNxtN;
                    pNxtE->iDir = !pNxtE->iDir;

                    if(pCurE) pCurE->pNxt = pNxtE;	// add to edge list
                    else pCurN->pChild = pNxtE;
                    pCurE = pNxtE;
                }
            }
        }
    }
}

void EmdL1::updateSubtree(cvPEmdNode pRoot)
{
    // Initialize auxiliary queue
    m_auxQueue[0] = pRoot;
    int nQueue = 1; // queue length
    int iQHead = 0; // head of queue

    // BFS browing
    cvPEmdNode pCurN=NULL,pNxtN=NULL;
    cvPEmdEdge pCurE=NULL;
    while(iQHead<nQueue)
    {
        pCurN = m_auxQueue[iQHead++];	// pop out from queue
        pCurE = pCurN->pChild;

        // browsing all children
        while(pCurE)
        {
            pNxtN = pCurE->pChild;
            pNxtN->iLevel = pCurN->iLevel+1;
            pNxtN->u = pCurE->iDir ? (pCurN->u - 1) : (pCurN->u + 1);
            pCurE = pCurE->pNxt;
            m_auxQueue[nQueue++] = pNxtN;
        }
    }
}

bool EmdL1::isOptimal()
{
    int iC, iMinC = 0;
    cvPEmdEdge pE;
    m_pEnter = NULL;
    m_iEnter = -1;

    // test each NON-BV edges
    for(int k=0; k<nNBV; ++k)
    {
        pE = m_NBVEdges[k];
        iC = 1 - pE->pParent->u + pE->pChild->u;
        if(iC<iMinC)
        {
            iMinC = iC;
            m_iEnter= k;
        }
        else
        {
            // Try reversing the direction
            iC	= 1 + pE->pParent->u - pE->pChild->u;
            if(iC<iMinC)
            {
                iMinC = iC;
                m_iEnter= k;
            }
        }
    }

    if(m_iEnter>=0)
    {
        m_pEnter = m_NBVEdges[m_iEnter];
        if(iMinC == (1 - m_pEnter->pChild->u + m_pEnter->pParent->u))	{
            // reverse direction
            cvPEmdNode pN = m_pEnter->pParent;
            m_pEnter->pParent = m_pEnter->pChild;
            m_pEnter->pChild = pN;
        }

        m_pEnter->iDir = 1;
    }
    return m_iEnter==-1;
}

void EmdL1::findNewSolution()
{
    // Find loop formed by adding the Enter BV edge.
    findLoopFromEnterBV();
    // Modify flow values along the loop
    cvPEmdEdge pE = NULL;
    float	minFlow = m_pLeave->flow;
    int k;
    for(k=0; k<m_iFrom; k++)
    {
        pE = m_fromLoop[k];
        if(pE->iDir) pE->flow += minFlow; // outward
        else pE->flow -= minFlow; // inward
    }
    for(k=0; k<m_iTo; k++)
    {
        pE = m_toLoop[k];
        if(pE->iDir) pE->flow -= minFlow; // outward
        else pE->flow += minFlow; // inward
    }

    // Update BV Tree, removing the Leaving-BV edge
    cvPEmdNode pLParentN = m_pLeave->pParent;
    cvPEmdNode pLChildN = m_pLeave->pChild;
    cvPEmdEdge pPreE = pLParentN->pChild;
    if(pPreE==m_pLeave)
    {
        pLParentN->pChild = m_pLeave->pNxt; // Leaving-BV is the first child
    }
    else
    {
        while(pPreE->pNxt != m_pLeave)
            pPreE	= pPreE->pNxt;
        pPreE->pNxt	= m_pLeave->pNxt; // remove Leaving-BV from child list
    }
    pLChildN->pParent = NULL;
    pLChildN->pPEdge = NULL;

    m_NBVEdges[m_iEnter]= m_pLeave; // put the leaving-BV into the NBV array

    // Add the Enter BV edge
    cvPEmdNode pEParentN = m_pEnter->pParent;
    cvPEmdNode pEChildN = m_pEnter->pChild;
    m_pEnter->flow = minFlow;
    m_pEnter->pNxt = pEParentN->pChild;		// insert the Enter BV as the first child
    pEParentN->pChild = m_pEnter;					//		of its parent

    // Recursively update the tree start from pEChildN
    cvPEmdNode pPreN = pEParentN;
    cvPEmdNode pCurN = pEChildN;
    cvPEmdNode pNxtN;
    cvPEmdEdge pNxtE, pPreE0;
    pPreE = m_pEnter;
    while(pCurN)
    {
        pNxtN = pCurN->pParent;
        pNxtE = pCurN->pPEdge;
        pCurN->pParent = pPreN;
        pCurN->pPEdge = pPreE;
        if(pNxtN)
        {
            // remove the edge from pNxtN's child list
            if(pNxtN->pChild==pNxtE)
            {
                pNxtN->pChild	= pNxtE->pNxt;			// first child
            }
            else
            {
                pPreE0	= pNxtN->pChild;
                while(pPreE0->pNxt != pNxtE)
                    pPreE0	= pPreE0->pNxt;
                pPreE0->pNxt	= pNxtE->pNxt;			// remove Leaving-BV from child list
            }
            // reverse the parent-child direction
            pNxtE->pParent = pCurN;
            pNxtE->pChild = pNxtN;
            pNxtE->iDir = !pNxtE->iDir;
            pNxtE->pNxt = pCurN->pChild;
            pCurN->pChild = pNxtE;
            pPreE = pNxtE;
            pPreN = pCurN;
        }
        pCurN = pNxtN;
    }

    // Update U at the child of the Enter BV
    pEChildN->u = m_pEnter->iDir?(pEParentN->u-1):(pEParentN->u + 1);
    pEChildN->iLevel = pEParentN->iLevel+1;
}

void EmdL1::findLoopFromEnterBV()
{
    // Initialize Leaving-BV edge
    float minFlow	= std::numeric_limits<float>::max();
    cvPEmdEdge pE = NULL;
    int iLFlag = 0;	// 0: in the FROM list, 1: in the TO list

    // Using two loop list to store the loop nodes
    cvPEmdNode pFrom = m_pEnter->pParent;
    cvPEmdNode pTo = m_pEnter->pChild;
    m_iFrom	= 0;
    m_iTo = 0;
    m_pLeave = NULL;

    // Trace back to make pFrom and pTo at the same level
    while(pFrom->iLevel > pTo->iLevel)
    {
        pE = pFrom->pPEdge;
        m_fromLoop[m_iFrom++] = pE;
        if(!pE->iDir && pE->flow<minFlow)
        {
            minFlow = pE->flow;
            m_pLeave = pE;
            iLFlag = 0;	// 0: in the FROM list
        }
        pFrom = pFrom->pParent;
    }

    while(pTo->iLevel > pFrom->iLevel)
    {
        pE = pTo->pPEdge;
        m_toLoop[m_iTo++] = pE;
        if(pE->iDir && pE->flow<minFlow)
        {
            minFlow = pE->flow;
            m_pLeave = pE;
            iLFlag = 1;	// 1: in the TO list
        }
        pTo	= pTo->pParent;
    }

    // Trace pTo and pFrom simultaneously till find their common ancester
    while(pTo!=pFrom)
    {
        pE = pFrom->pPEdge;
        m_fromLoop[m_iFrom++] = pE;
        if(!pE->iDir && pE->flow<minFlow)
        {
            minFlow = pE->flow;
            m_pLeave = pE;
            iLFlag = 0;	// 0: in the FROM list, 1: in the TO list
        }
        pFrom = pFrom->pParent;

        pE = pTo->pPEdge;
        m_toLoop[m_iTo++] = pE;
        if(pE->iDir && pE->flow<minFlow)
        {
            minFlow = pE->flow;
            m_pLeave = pE;
            iLFlag = 1;	// 0: in the FROM list, 1: in the TO list
        }
        pTo	= pTo->pParent;
    }

    // Reverse the direction of the Enter BV edge if necessary
    if(iLFlag==0)
    {
        cvPEmdNode pN = m_pEnter->pParent;
        m_pEnter->pParent = m_pEnter->pChild;
        m_pEnter->pChild = pN;
        m_pEnter->iDir = !m_pEnter->iDir;
    }
}

float EmdL1::compuTotalFlow()
{
    // Computing the total flow as the final distance
    float f = 0;

    // Initialize auxiliary queue
    m_auxQueue[0] = m_pRoot;
    int nQueue = 1; // length of queue
    int iQHead = 0; // head of queue

    // BFS browing the tree
    cvPEmdNode pCurN=NULL,pNxtN=NULL;
    cvPEmdEdge pCurE=NULL;
    while(iQHead<nQueue)
    {
        pCurN = m_auxQueue[iQHead++];	// pop out from queue
        pCurE = pCurN->pChild;

        // browsing all children
        while(pCurE)
        {
            f += pCurE->flow;
            pNxtN = pCurE->pChild;
            pCurE = pCurE->pNxt;
            m_auxQueue[nQueue++] = pNxtN;
        }
    }
    return f;
}

/****************************************************************************************\
*                                   EMDL1 Function                                      *
\****************************************************************************************/

float cv::EMDL1(InputArray _signature1, InputArray _signature2)
{
    CV_INSTRUMENT_REGION()

    Mat signature1 = _signature1.getMat(), signature2 = _signature2.getMat();
    EmdL1 emdl1;
    return emdl1.getEMDL1(signature1, signature2);
}
