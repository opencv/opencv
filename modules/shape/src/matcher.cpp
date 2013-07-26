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
#include <limits>

/*
 * Implementation of the paper Shape Matching and Object Recognition Using Shape Contexts
 * Belongie et al., 2002 by Juan Manuel Perez for GSoC 2013. 
 */
namespace cv
{
/* Constructors */
SCDMatcher::SCDMatcher(float _outlierWeight, int _numExtraDummies, int _configFlags)
{
    outlierWeight=_outlierWeight;
    configFlags=_configFlags;
    numExtraDummies=_numExtraDummies;
}

/* Public methods */
void SCDMatcher::matchDescriptors(Mat& descriptors1,  Mat& descriptors2, std::vector<DMatch>& matches, std::vector<int>& inliers)
{
    CV_Assert(!descriptors1.empty() && !descriptors2.empty());
    matches.clear();

    /* Build the cost Matrix between descriptors*/
    Mat costMat;
    buildCostMatrix(descriptors1, descriptors2, costMat, configFlags);
    
    /* Solve the matching problem using the hungarian method */
    hungarian(costMat, matches, inliers, descriptors1.rows, descriptors2.rows);
}

/* Protected methods */
void SCDMatcher::buildCostMatrix(const Mat& descriptors1, const Mat& descriptors2,
                                 Mat& costMatrix, int flags) const
{
    switch (flags)
    {
        case DistanceSCDFlags::DIST_CHI:
            buildChiCostMatrix(descriptors1,  descriptors2, costMatrix);
            break;
        case DistanceSCDFlags::DIST_EMD:
            break;
        case DistanceSCDFlags::DIST_L2:
            break;
        default:
            CV_Error(-206, "The available flags are: DIST_CHI, DIST_EMD, and DIST_EUCLIDEAN");
    }
}

void SCDMatcher::buildChiCostMatrix(const Mat& descriptors1,  const Mat& descriptors2,
                                    Mat& costMatrix) const
{
    // size of the costMatrix with dummies //
    int costrows = std::max(descriptors1.rows, descriptors2.rows)+numExtraDummies;
    // Obtain copies of the descriptors //
    Mat scd1=descriptors1.clone();
    Mat scd2=descriptors2.clone();

    // row normalization //
    for(int i=0; i<scd1.rows; i++)
    {
        Mat row = scd1.row(i);
        scd1.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        Mat row = scd2.row(i);
        scd2.row(i)/=(sum(row)[0]+FLT_EPSILON);
    }   

    // filling costMatrix //
    costMatrix = Mat(costrows, costrows, CV_32F, Scalar(outlierWeight));
        
    // Compute the Cost Matrix //
    for(int i=0; i<scd1.rows; i++)
    {
        for(int j=0; j<scd2.rows; j++)
        {
            float csum = 0;
            for(int k=0; k<scd2.cols; k++)
            {
                float resta=scd1.at<float>(i,k)-scd2.at<float>(j,k);
                float suma=scd1.at<float>(i,k)+scd2.at<float>(j,k);
                csum += resta*resta/(FLT_EPSILON+suma);
            }
            costMatrix.at<float>(i,j)=csum/2;
        }
    }
    // normalizing cost //
    //normalize(costMatrix, costMatrix, 0,1, NORM_MINMAX);
}

void SCDMatcher::buildEMDCostMatrix(const Mat& descriptors1, const Mat& descriptors2, Mat& costMatrix) const
{
    // Obtain copies of the descriptors //
    Mat scd1 = descriptors1.clone();
    Mat scd2 = descriptors2.clone();

    // row normalization //
    for(int i=0; i<scd1.rows; i++)
    {
        scd1.row(i)=scd1.row(i)*1/(sum(scd1.row(i))[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        scd2.row(i)=scd2.row(i)*1/(sum(scd2.row(i))[0]+FLT_EPSILON);
    }

    // initializing costMatrix with oulier weight values //
    int costrows = std::max(scd1.rows, scd2.rows);
    costMatrix = Mat::zeros(costrows, costrows, CV_32F)+outlierWeight;
}

void SCDMatcher::buildL2CostMatrix(const Mat& descriptors1, const Mat& descriptors2, Mat& costMatrix) const
{
    /* Obtain copies of the descriptors */
    Mat scd1 = descriptors1.clone();
    Mat scd2 = descriptors2.clone();

    /* row normalization */
    for(int i=0; i<scd1.rows; i++)
    {
        scd1.row(i)=scd1.row(i)*1/(sum(scd1.row(i))[0]+FLT_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        scd2.row(i)=scd2.row(i)*1/(sum(scd2.row(i))[0]+FLT_EPSILON);
    }

    /* initializing costMatrix with oulier weight values */
    int costrows = std::max(scd1.rows, scd2.rows);
    costMatrix = Mat::zeros(costrows, costrows, CV_32F)+outlierWeight;

    /* Compute the Cost Matrix */
    for(int i=0; i<scd1.rows; i++)
    {
        for(int j=0; j<scd2.rows; j++)
        {
            float csum = 0;
            for(int k=0; k<scd2.cols; k++)
            {
                float num=scd1.at<float>(i,k)-scd2.at<float>(j,k);
                csum += num*num;
            }
            costMatrix.at<float>(i,j)=std::sqrt(csum);
        }
    }
}

void SCDMatcher::hungarian(Mat& costMatrix, std::vector<DMatch>& outMatches, std::vector<int> &inliers, int sizeScd1, int sizeScd2)
{
    std::vector<int> free(costMatrix.rows, 0), collist(costMatrix.rows, 0);
    std::vector<int> matches(costMatrix.rows, 0), colsol(costMatrix.rows), rowsol(costMatrix.rows);
    std::vector<float> d(costMatrix.rows), pred(costMatrix.rows), v(costMatrix.rows);

    const float LOWV=1e-8;
    bool unassignedfound;
    int  i=0, imin=0, numfree=0, prvnumfree=0, f=0, i0=0, k=0, freerow=0;
    int  j=0, j1=0, j2=0, endofpath=0, last=0, low=0, up=0;
    float min=0, h=0, umin=0, usubmin=0, v2=0;

    /* COLUMN REDUCTION */
    for (j = costMatrix.rows-1; j >= 0; j--)
    {
        // find minimum cost over rows.
        min = costMatrix.at<float>(0,j);
        imin = 0;
        for (i = 1; i < costMatrix.rows; i++)
        if (costMatrix.at<float>(i,j) < min)
        {
            min = costMatrix.at<float>(i,j);
            imin = i;
        }
        v[j] = min;

        if (++matches[imin] == 1)
        {
            rowsol[imin] = j;
            colsol[j] = imin;
        }
        else
        {
            colsol[j]=-1;
        }
    }

    /* REDUCTION TRANSFER */
    for (i=0; i<costMatrix.rows; i++)
    {
        if (matches[i] == 0)
        {
            free[numfree++] = i;
        }
        else
        {
            if (matches[i] == 1)
            {
                j1=rowsol[i];
                min=std::numeric_limits<float>::max();
                for (j=0; j<costMatrix.rows; j++)
                {
                    if (j!=j1)
                    {
                        if (costMatrix.at<float>(i,j)-v[j] < min)
                        {
                            min=costMatrix.at<float>(i,j)-v[j];
                        }
                    }
                }
                v[j1] = v[j1]-min;
            }
        }
    }
    /* AUGMENTING ROW REDUCTION */
    int loopcnt = 0;
    do
    {
        loopcnt++;
        k=0;
        prvnumfree=numfree;
        numfree=0;
        while (k < prvnumfree)
        {
            i=free[k];
            k++;
            umin = costMatrix.at<float>(i,0)-v[0];
            j1=0;
            usubmin = std::numeric_limits<float>::max();
            for (j=1; j<costMatrix.rows; j++)
            {
                h = costMatrix.at<float>(i,j)-v[j];
                if (h < usubmin)
                {
                    if (h >= umin)
                    {
                        usubmin = h;
                        j2 = j;
                    }
                    else
                    {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }
            i0 = colsol[j1];

            if (fabs(umin-usubmin) > LOWV) //if( umin < usubmin )
            {
                v[j1] = v[j1] - (usubmin - umin);
            }
            else                   // minimum and subminimum equal.
            {
                if (i0 >= 0)         // minimum column j1 is assigned.
                {
                    j1 = j2;
                    i0 = colsol[j2];
                }
            }
            // (re-)assign i to j1, possibly de-assigning an i0.
            rowsol[i]=j1;
            colsol[j1]=i;

            if (i0 >= 0)
            {
                //if( umin < usubmin )
                if (fabs(umin-usubmin) > LOWV)
                {
                    free[--k] = i0;
                }
                else
                {
                    free[numfree++] = i0;
                }
            }
        }
    }while (loopcnt<2);       // repeat once.

    /* AUGMENT SOLUTION for each free row */
    for (f = 0; f<numfree; f++)
    {
        freerow = free[f];       // start row of augmenting path.
        // Dijkstra shortest path algorithm.
        // runs until unassigned column added to shortest path tree.
        for (j = 0; j < costMatrix.rows; j++)
        {
            d[j] = costMatrix.at<float>(freerow,j) - v[j];
            pred[j] = freerow;
            collist[j] = j;        // init column list.
        }

        low=0; // columns in 0..low-1 are ready, now none.
        up=0;  // columns in low..up-1 are to be scanned for current minimum, now none.
        unassignedfound = false;
        do
        {
            if (up == low)         // no more columns to be scanned for current minimum.
            {
                last=low-1;
                // scan columns for up..costMatrix.rows-1 to find all indices for which new minimum occurs.
                // store these indices between low..up-1 (increasing up).
                min = d[collist[up++]];
                for (k = up; k < costMatrix.rows; k++)
                {
                    j = collist[k];
                    h = d[j];
                    if (h <= min)
                    {
                        if (h < min)     // new minimum.
                        {
                            up = low;      // restart list at index low.
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }
                // check if any of the minimum columns happens to be unassigned.
                // if so, we have an augmenting path right away.
                for (k=low; k<up; k++)
                {
                    if (colsol[collist[k]] < 0)
                    {
                        endofpath = collist[k];
                        unassignedfound = true;
                        break;
                    }
                }
            }

            if (!unassignedfound)
            {
                // update 'distances' between freerow and all unscanned columns, via next scanned column.
                j1 = collist[low];
                low++;
                i = colsol[j1];
                h = costMatrix.at<float>(i,j1)-v[j1]-min;

                for (k = up; k < costMatrix.rows; k++)
                {
                    j = collist[k];
                    v2 = costMatrix.at<float>(i,j) - v[j] - h;
                    if (v2 < d[j])
                    {
                        pred[j] = i;
                        if (v2 == min)
                        {
                            if (colsol[j] < 0)
                            {
                                // if unassigned, shortest augmenting path is complete.
                                endofpath = j;
                                unassignedfound = true;
                                break;
                            }
                            else
                            {
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        }
                        d[j] = v2;
                    }
                }
            }
        }while (!unassignedfound);

        // update column prices.
        for (k = 0; k <= last; k++)
        {
            j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        // reset row and column assignments along the alternating path.
        do
        {
            i = pred[endofpath];
            colsol[endofpath] = i;
            j1 = endofpath;
            endofpath = rowsol[i];
            rowsol[i] = j1;
        }while (i != freerow);
    }



    // calculate symmetric shape context cost
    float leftcost = 0;
    for (i = 0; i<costMatrix.rows; i++)
    {
        if (rowsol[i]<sizeScd1) // if a real match
        {
            j = rowsol[i];
            leftcost+=costMatrix.at<float>(i,j);
        }
    }

    leftcost/=costMatrix.rows;
    float rightcost = 0;
    for (i = 0; i<costMatrix.cols; i++)
    {
        if (colsol[i]<sizeScd2) // if a real match
        {
            j = colsol[i];
            rightcost+=costMatrix.at<float>(j,i);
        }
    }
    rightcost/=costMatrix.cols;

    minMatchCost = leftcost+rightcost;

    // Update outliers
    inliers.reserve(sizeScd1);
    for (size_t kc = 0; kc<inliers.size(); kc++)
    {
        if (rowsol[kc]<sizeScd1) // if a real match
            inliers[kc]=1;
        else
            inliers[kc]=0;
    }

    // Save in a DMatch vector
    for (i=0;i<costMatrix.cols;i++)
    {
        DMatch singleMatch(colsol[i],i,costMatrix.at<float>(colsol[i],i));//queryIdx,trainIdx,distance
        outMatches.push_back(singleMatch);
    }
}

/* Getters */
float SCDMatcher::getMatchingCost()
{
    return minMatchCost;
}

int SCDMatcher::getNumDummies()
{
    return numExtraDummies;
}

/* Setters */
void SCDMatcher::setNumDummies(int _numExtraDummies)
{
    numExtraDummies=_numExtraDummies;
}

}//cv




