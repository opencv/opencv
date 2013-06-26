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

/*
 * Implementation of the paper Shape Matching and Object Recognition Using Shape Contexts
 * Belongie et al., 2002 by Juan Manuel Perez for GSoC 2013. 
 */
namespace cv
{
/* Constructors */
SCDMatcher::SCDMatcher(float _outlierWeight, int _configFlags)
{
    outlierWeight=_outlierWeight;
    configFlags=_configFlags;
}

/* Public methods */
void SCDMatcher::matchDescriptors(Mat& descriptors1,  Mat& descriptors2, std::vector<DMatch>& matches) const
{
    CV_Assert((descriptors1.cols>0) & (descriptors1.rows>0) & (descriptors2.cols>0) & (descriptors2.rows>0));
    /* Build the cost Matrix between descriptors*/
    Mat costMat;
    buildCostMatrix(descriptors1, descriptors2, costMat, configFlags);
    
    /* Solve the matching problem using the hungarian method */
    hungarian(costMat, matches);
}

/* Protected methods */
void SCDMatcher::buildCostMatrix(Mat& descriptors1,  Mat& descriptors2, 
                                   Mat& costMatrix, int flags) const
{  
    switch (flags)
    {
        case DistanceSCDFlags::DIST_CHI:
            buildChiCostMatrix(descriptors1,  descriptors2, costMatrix);
            break;
        case DistanceSCDFlags::DIST_EMD:
            break;
        case DistanceSCDFlags::DIST_EUCLIDEAN:
            break;
        default:
            CV_Error(-206, "The available flags are: DIST_CHI, DIST_EMD, and DIST_EUCLIDEAN");
    }
}

void SCDMatcher::buildChiCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const
{
    /* Obtain copies of the descriptors */
    Mat scd1 = descriptors1.clone();
    Mat scd2 = descriptors2.clone();
    
    /* row normalization */
    for(int i=0; i<scd1.rows; i++)
    {
        scd1.row(i)=scd1.row(i)*1/(sum(scd1.row(i))[0]+DBL_EPSILON);
    }
    for(int i=0; i<scd2.rows; i++)
    {
        scd2.row(i)=scd2.row(i)*1/(sum(scd2.row(i))[0]+DBL_EPSILON);
    }   

    /* initializing costMatrix with oulier weight values*/
    int costrows = (scd1.rows<scd2.rows)?scd2.rows:scd1.rows;
    costMatrix = Mat::zeros(costrows, costrows, CV_32F)+outlierWeight;
        
    /* compute the Cost Matrix*/
    for(int i=0; i<scd1.rows; i++)
    {
        for(int j=0; j<scd2.rows; j++)
        {
            float csum = 0;
            for(int k=0; k<scd2.cols; k++)
            {
                csum += pow(scd1.at<float>(i,k)-scd2.at<float>(j,k),2)/
                        (DBL_EPSILON+scd1.at<float>(i,k)+scd2.at<float>(j,k));
            }
            costMatrix.at<float>(i,j)=csum/2;
        }
    }
}

void SCDMatcher::buildEMDCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const
{
}

void SCDMatcher::buildEucCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const
{
}

void SCDMatcher::hungarian(Mat& costMatrix, std::vector<DMatch>& outMatches) const
{
    bool unassignedfound;
    int  imin, numfree = 0, prvnumfree, f, i0, freerow;
    int  j1, j2, endofpath, last, low, up;
    float min, h, umin, usubmin, v2;
    std::vector<int> free, collist, matches, pred;
    std::vector<float> d, v;
    int dim = costMatrix.rows;
    
    free.resize(dim);
    collist.resize(dim);
    matches.resize(dim);
    pred.resize(dim);
    d.resize(dim);
    v.resize(dim);
    
    for (int i = 0; i < dim; i++)  
    {
        matches.push_back(0);
    }

    // COLUMN REDUCTION 
    for (int j=dim-1; j>=0; j--)    
    {
        min = costMatrix.at<float>(0,j); 
        imin = 0;
        for (int i = 1; i < dim; i++)  
        {
            if (costMatrix.at<float>(i,j) < min) 
            { 
                min = costMatrix.at<float>(i,j); 
                imin = i;
            }
        }
        v[j] = min; 

        if (++matches[imin] == 1) 
        { 
            //rowsol[imin] = j; 
            //colsol[j] = imin; 
        }
        else
        {
            //colsol[j] = -1;
        }
    }
    
    // REDUCTION TRANSFER
    for (int i = 0; i < dim; i++)
    { 
        if (matches[i] == 0)
        {
            free[numfree++] = i;
        }
        else
        {
            if (matches[i] == 1)
            {
                //j1 = rowsol[i]; 
                min = 1e10;
                for (int j = 0; j < dim; j++)
                {
                    if (j != j1)
                    {
                        if (costMatrix.at<float>(i,j) - v[j] < min) 
                        {
                            min = costMatrix.at<float>(i,j) - v[j];
                        }
                    }
                }
                v[j1] = v[j1] - min;
            }
        }
    }
}
}//cv




