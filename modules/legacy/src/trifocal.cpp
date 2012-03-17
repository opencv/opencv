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

//#include "cvtypes.h"
#include <float.h>
#include <limits.h>
//#include "cv.h"
//#include "windows.h"

#include <stdio.h>

/* Valery Mosyagin */

/* Function defenitions */

/* ----------------- */

void cvOptimizeLevenbergMarquardtBundle( CvMat** projMatrs, CvMat** observProjPoints,
                                       CvMat** pointsPres, int numImages,
                                       CvMat** resultProjMatrs, CvMat* resultPoints4D,int maxIter,double epsilon );

int icvComputeProjectMatrices6Points(  CvMat* points1,CvMat* points2,CvMat* points3,
                                        CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3);

void icvFindBaseTransform(CvMat* points,CvMat* resultT);

void GetGeneratorReduceFundSolution(CvMat* points1,CvMat* points2,CvMat* fundReduceCoef1,CvMat* fundReduceCoef2);

int GetGoodReduceFundamMatrFromTwo(CvMat* fundReduceCoef1,CvMat* fundReduceCoef2,CvMat* resFundReduceCoef);

void GetProjMatrFromReducedFundamental(CvMat* fundReduceCoefs,CvMat* projMatrCoefs);

void icvComputeProjectMatrix(CvMat* objPoints,CvMat* projPoints,CvMat* projMatr);

void icvComputeTransform4D(CvMat* points1,CvMat* points2,CvMat* transMatr);

int icvComputeProjectMatricesNPoints(  CvMat* points1,CvMat* points2,CvMat* points3,
                                       CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                       double threshold,/* Threshold for good point */
                                       double p,/* Probability of good result. */
                                       CvMat* status,
                                       CvMat* points4D);

int icvComputeProjectMatricesNPoints(  CvMat* points1,CvMat* points2,CvMat* points3,
                                       CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                       double threshold,/* Threshold for good point */
                                       double p,/* Probability of good result. */
                                       CvMat* status,
                                       CvMat* points4D);

void icvReconstructPointsFor3View( CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                CvMat* projPoints1,CvMat* projPoints2,CvMat* projPoints3,
                                CvMat* points4D);

void icvReconstructPointsFor3View( CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                CvMat* projPoints1,CvMat* projPoints2,CvMat* projPoints3,
                                CvMat* points4D);

/*==========================================================================================*/
/*                        Functions for calculation the tensor                              */
/*==========================================================================================*/
#if 1
void fprintMatrix(FILE* file,CvMat* matrix)
{
    int i,j;
    fprintf(file,"\n");
    for( i=0;i<matrix->rows;i++ )
    {
        for(j=0;j<matrix->cols;j++)
        {
            fprintf(file,"%10.7lf  ",cvmGet(matrix,i,j));
        }
        fprintf(file,"\n");
    }
}
#endif
/*==========================================================================================*/

void icvNormalizePoints( CvMat* points, CvMat* normPoints,CvMat* cameraMatr )
{
    /* Normalize image points using camera matrix */

    CV_FUNCNAME( "icvNormalizePoints" );
    __BEGIN__;

    /* Test for null pointers */
    if( points == 0 || normPoints == 0 || cameraMatr == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points) || !CV_IS_MAT(normPoints) || !CV_IS_MAT(cameraMatr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    int numPoints;
    numPoints = points->cols;
    if( numPoints <= 0 || numPoints != normPoints->cols )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be the same and more than 0" );
    }

    if( normPoints->rows != 2 || normPoints->rows != points->rows )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Points must have 2 coordinates" );
    }

    if(cameraMatr->rows != 3 || cameraMatr->cols != 3)
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of camera matrix must be 3x3" );
    }

    double fx,fy,cx,cy;

    fx = cvmGet(cameraMatr,0,0);
    fy = cvmGet(cameraMatr,1,1);
    cx = cvmGet(cameraMatr,0,2);
    cy = cvmGet(cameraMatr,1,2);

    int i;
    for( i = 0; i < numPoints; i++ )
    {
        cvmSet(normPoints, 0, i, (cvmGet(points,0,i) - cx) / fx );
        cvmSet(normPoints, 1, i, (cvmGet(points,1,i) - cy) / fy );
    }

    __END__;

    return;
}


/*=====================================================================================*/
/*
Computes projection matrices for given 6 points on 3 images
May returns 3 results. */
int icvComputeProjectMatrices6Points( CvMat* points1,CvMat* points2,CvMat* points3,
                                      CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3/*,
                                      CvMat* points4D*/)
{
    /* Test input data correctness */

    int numSol = 0;

    CV_FUNCNAME( "icvComputeProjectMatrices6Points" );
    __BEGIN__;

    /* Test for null pointers */
    if( points1   == 0 || points2   == 0 || points3   == 0 ||
        projMatr1 == 0 || projMatr2 == 0 || projMatr3 == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points1)   || !CV_IS_MAT(points2)   || !CV_IS_MAT(points3)   ||
        !CV_IS_MAT(projMatr1) || !CV_IS_MAT(projMatr2) || !CV_IS_MAT(projMatr3)  )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    if( (points1->cols != points2->cols) || (points1->cols != points3->cols) || (points1->cols != 6) /* || (points4D->cols !=6) */)
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be same and == 6" );
    }

    if( points1->rows != 2 || points2->rows != 2 || points3->rows != 2 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points coordinates must be 2" );
    }

    if( projMatr1->cols != 4 || projMatr2->cols != 4 || projMatr3->cols != 4 ||
        (!(projMatr1->rows == 3 && projMatr2->rows == 3 && projMatr3->rows == 3) &&
        !(projMatr1->rows == 9 && projMatr2->rows == 9 && projMatr3->rows == 9)) )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of project matrix must be 3x4 or 9x4 (for 3 matrices)" );
    }

#if 0
    if( points4D->row != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of coordinates of points4D  must be 4" );
    }
#endif

    /* Find transform matrix for each camera */
    int i;
    CvMat* points[3];
    points[0] = points1;
    points[1] = points2;
    points[2] = points3;

    CvMat* projMatrs[3];
    projMatrs[0] = projMatr1;
    projMatrs[1] = projMatr2;
    projMatrs[2] = projMatr3;

    CvMat transMatr;
    double transMatr_dat[9];
    transMatr = cvMat(3,3,CV_64F,transMatr_dat);

    CvMat corrPoints1;
    CvMat corrPoints2;

    double corrPoints_dat[3*3*2];/* 3-point(images) by 3-coordinates by 2-correspondence*/

    corrPoints1 = cvMat(3,3,CV_64F,corrPoints_dat);  /* 3-coordinates for each of 3-points(3-image) */
    corrPoints2 = cvMat(3,3,CV_64F,corrPoints_dat+9);/* 3-coordinates for each of 3-points(3-image) */

    for( i = 0; i < 3; i++ )/* for each image */
    {
        /* Get last 4 points for computing transformation */
        CvMat tmpPoints;
        /* find base points transform for last four points on i-th image */
        cvGetSubRect(points[i],&tmpPoints,cvRect(2,0,4,2));
        icvFindBaseTransform(&tmpPoints,&transMatr);

        {/* We have base transform. Compute error scales for three first points */
            CvMat trPoint;
            double trPoint_dat[3*3];
            trPoint = cvMat(3,3,CV_64F,trPoint_dat);
            /* fill points */
            for( int kk = 0; kk < 3; kk++ )
            {
                cvmSet(&trPoint,0,kk,cvmGet(points[i],0,kk+2));
                cvmSet(&trPoint,1,kk,cvmGet(points[i],1,kk+2));
                cvmSet(&trPoint,2,kk,1);
            }

            /* Transform points */
            CvMat resPnts;
            double resPnts_dat[9];
            resPnts = cvMat(3,3,CV_64F,resPnts_dat);
            cvmMul(&transMatr,&trPoint,&resPnts);
        }

        /* Transform two first points */
        for( int j = 0; j < 2; j++ )
        {
            CvMat pnt;
            double pnt_dat[3];
            pnt = cvMat(3,1,CV_64F,pnt_dat);
            pnt_dat[0] = cvmGet(points[i],0,j);
            pnt_dat[1] = cvmGet(points[i],1,j);
            pnt_dat[2] = 1.0;

            CvMat trPnt;
            double trPnt_dat[3];
            trPnt = cvMat(3,1,CV_64F,trPnt_dat);

            cvmMul(&transMatr,&pnt,&trPnt);

            /* Collect transformed points  */
            corrPoints_dat[j * 9 + 0 * 3 + i] = trPnt_dat[0];/* x */
            corrPoints_dat[j * 9 + 1 * 3 + i] = trPnt_dat[1];/* y */
            corrPoints_dat[j * 9 + 2 * 3 + i] = trPnt_dat[2];/* w */
        }
    }

    /* We have computed corr points. Now we can compute generators for reduced fundamental matrix */

    /* Compute generators for reduced fundamental matrix from 3 pair of collect points */
    CvMat fundReduceCoef1;
    CvMat fundReduceCoef2;
    double fundReduceCoef1_dat[5];
    double fundReduceCoef2_dat[5];

    fundReduceCoef1 = cvMat(1,5,CV_64F,fundReduceCoef1_dat);
    fundReduceCoef2 = cvMat(1,5,CV_64F,fundReduceCoef2_dat);

    GetGeneratorReduceFundSolution(&corrPoints1, &corrPoints2, &fundReduceCoef1, &fundReduceCoef2);

    /* Choose best solutions for two generators. We can get 3 solutions */
    CvMat resFundReduceCoef;
    double resFundReduceCoef_dat[3*5];

    resFundReduceCoef = cvMat(3,5,CV_64F,resFundReduceCoef_dat);

    numSol = GetGoodReduceFundamMatrFromTwo(&fundReduceCoef1, &fundReduceCoef2,&resFundReduceCoef);

    int maxSol;
    maxSol = projMatrs[0]->rows / 3;

    int currSol;
    for( currSol = 0; (currSol < numSol && currSol < maxSol); currSol++ )
    {
        /* For current solution compute projection matrix */
        CvMat fundCoefs;
        cvGetSubRect(&resFundReduceCoef, &fundCoefs, cvRect(0,currSol,5,1));

        CvMat projMatrCoefs;
        double projMatrCoefs_dat[4];
        projMatrCoefs = cvMat(1,4,CV_64F,projMatrCoefs_dat);

        GetProjMatrFromReducedFundamental(&fundCoefs,&projMatrCoefs);
        /* we have computed coeffs for reduced project matrix */

        CvMat objPoints;
        double objPoints_dat[4*6];
        objPoints  = cvMat(4,6,CV_64F,objPoints_dat);
        cvZero(&objPoints);

        /* fill object points */
        for( i =0; i < 4; i++ )
        {
            objPoints_dat[i*6]   = 1;
            objPoints_dat[i*6+1] = projMatrCoefs_dat[i];
            objPoints_dat[i*7+2] = 1;
        }

        int currCamera;
        for( currCamera = 0; currCamera < 3; currCamera++ )
        {

            CvMat projPoints;
            double projPoints_dat[3*6];
            projPoints = cvMat(3,6,CV_64F,projPoints_dat);

            /* fill projected points for current camera */
            for( i = 0; i < 6; i++ )/* for each points for current camera */
            {
                projPoints_dat[6*0+i] = cvmGet(points[currCamera],0,i);/* x */
                projPoints_dat[6*1+i] = cvmGet(points[currCamera],1,i);/* y */
                projPoints_dat[6*2+i] = 1;/* w */
            }

            /* compute project matrix for current camera */
            CvMat projMatrix;
            double projMatrix_dat[3*4];
            projMatrix = cvMat(3,4,CV_64F,projMatrix_dat);

            icvComputeProjectMatrix(&objPoints,&projPoints,&projMatrix);

            /* Add this matrix to result */
            CvMat tmpSubRes;
            cvGetSubRect(projMatrs[currCamera],&tmpSubRes,cvRect(0,currSol*3,4,3));
            cvConvert(&projMatrix,&tmpSubRes);
        }

        /* We know project matrices. And we can reconstruct 6 3D-points if need */
#if 0
        if( points4D )
        {
            if( currSol < points4D->rows / 4 )
            {
                CvMat tmpPoints4D;
                double tmpPoints4D_dat[4*6];
                tmpPoints4D = cvMat(4,6,CV_64F,tmpPoints4D_dat);

                icvReconstructPointsFor3View( &wProjMatr[0], &wProjMatr[1], &wProjMatr[2],
                                           points1, points2, points3,
                                           &tmpPoints4D);

                CvMat tmpSubRes;
                cvGetSubRect(points4D,tmpSubRes,cvRect(0,currSol*4,6,4));
                cvConvert(tmpPoints4D,points4D);
            }
        }
#endif

    }/* for all sollutions */

    __END__;
    return numSol;
}

/*==========================================================================================*/
int icvGetRandNumbers(int range,int count,int* arr)
{
    /* Generate random numbers [0,range-1] */

    CV_FUNCNAME( "icvGetRandNumbers" );
    __BEGIN__;

    /* Test input data */
    if( arr == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Parameter 'arr' is a NULL pointer" );
    }


    /* Test for errors input data  */
    if( range < count || range <= 0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Can't generate such numbers. Count must be <= range and range must be > 0" );
    }

    int i,j;
    int newRand;
    for( i = 0; i < count; i++ )
    {

        int haveRep = 0;/* firstly we have not repeats */
        do
        {
            /* generate new number */
            newRand = rand()%range;
            haveRep = 0;
            /* Test for repeats in previous numbers */
            for( j = 0; j < i; j++ )
            {
                if( arr[j] == newRand )
                {
                    haveRep = 1;
                    break;
                }
            }
        } while(haveRep);

        /* We have good random number */
        arr[i] = newRand;
    }
    __END__;
    return 1;
}
/*==========================================================================================*/
void icvSelectColsByNumbers(CvMat* srcMatr, CvMat* dstMatr, int* indexes,int number)
{

    CV_FUNCNAME( "icvSelectColsByNumbers" );
    __BEGIN__;

    /* Test input data */
    if( srcMatr == 0 || dstMatr == 0 || indexes == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(srcMatr) || !CV_IS_MAT(dstMatr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "srcMatr and dstMatr must be a matrices" );
    }

    int srcSize;
    int numRows;
    numRows = srcMatr->rows;
    srcSize = srcMatr->cols;

    if( numRows != dstMatr->rows )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of rows of matrices must be the same" );
    }

    int dst;
    for( dst = 0; dst < number; dst++ )
    {
        int src = indexes[dst];
        if( src >=0 && src < srcSize )
        {
            /* Copy each elements in column */
            int i;
            for( i = 0; i < numRows; i++ )
            {
                cvmSet(dstMatr,i,dst,cvmGet(srcMatr,i,src));
            }
        }
    }

    __END__;
    return;
}

/*==========================================================================================*/
void icvProject4DPoints(CvMat* points4D,CvMat* projMatr, CvMat* projPoints)
{

    CvMat* tmpProjPoints = 0;

    CV_FUNCNAME( "icvProject4DPoints" );

    __BEGIN__;

    if( points4D == 0 || projMatr == 0 || projPoints == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points4D) || !CV_IS_MAT(projMatr) || !CV_IS_MAT(projPoints) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    int numPoints;
    numPoints = points4D->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points4D must be more than zero" );
    }

    if( numPoints != projPoints->cols )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be the same");
    }

    if( projPoints->rows != 2 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of coordinates of projected points must be 2");
    }

    if( points4D->rows != 4 )
    {
        CV_ERROR(CV_StsUnmatchedSizes, "Number of coordinates of 4D points must be 4");
    }

    if( projMatr->cols != 4 || projMatr->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of projection matrix must be 3x4");
    }


    CV_CALL( tmpProjPoints = cvCreateMat(3,numPoints,CV_64F) );

    cvmMul(projMatr,points4D,tmpProjPoints);

    /* Scale points */
    int i;
    for( i = 0; i < numPoints; i++ )
    {
        double scale,x,y;

        scale = cvmGet(tmpProjPoints,2,i);
        x = cvmGet(tmpProjPoints,0,i);
        y = cvmGet(tmpProjPoints,1,i);

        if( fabs(scale) > 1e-7 )
        {
            x /= scale;
            y /= scale;
        }
        else
        {
            x = 1e8;
            y = 1e8;
        }

        cvmSet(projPoints,0,i,x);
        cvmSet(projPoints,1,i,y);
    }

    __END__;

    cvReleaseMat(&tmpProjPoints);

    return;
}
/*==========================================================================================*/
int icvCompute3ProjectMatricesNPointsStatus( CvMat** points,/* 3 arrays of points on image  */
                                             CvMat** projMatrs,/* array of 3 prejection matrices */
                                             CvMat** statuses,/* 3 arrays of status of points */
                                             double threshold,/* Threshold for good point */
                                             double p,/* Probability of good result. */
                                             CvMat* resStatus,
                                             CvMat* points4D)
{
    int numProjMatrs = 0;
    unsigned char *comStat = 0;
    CvMat *triPoints[3] = {0,0,0};
    CvMat *status = 0;
    CvMat *triPoints4D = 0;

    CV_FUNCNAME( "icvCompute3ProjectMatricesNPointsStatus" );
    __BEGIN__;

    /* Test for errors */
    if( points == 0 || projMatrs == 0 || statuses == 0 || resStatus == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    int currImage;
    for( currImage = 0; currImage < 3; currImage++ )
    {
        /* Test for null pointers */
        if( points[currImage] == 0 )
        {
            CV_ERROR( CV_StsNullPtr, "Some of points arrays is a NULL pointer" );
        }

        if( projMatrs[currImage] == 0 )
        {
            CV_ERROR( CV_StsNullPtr, "Some of projMatr is a NULL pointer" );
        }

        if( statuses[currImage] == 0 )
        {
            CV_ERROR( CV_StsNullPtr, "Some of status arrays is a NULL pointer" );
        }

        /* Test for matrices */
        if( !CV_IS_MAT(points[currImage]) )
        {
            CV_ERROR( CV_StsNullPtr, "Some of points arrays is not a matrix" );
        }

        if( !CV_IS_MAT(projMatrs[currImage]) )
        {
            CV_ERROR( CV_StsNullPtr, "Some of projMatr is not a matrix" );
        }

        if( !CV_IS_MASK_ARR(statuses[currImage]) )
        {
            CV_ERROR( CV_StsNullPtr, "Some of status arrays is not a mask array" );
        }
    }

    int numPoints;
    numPoints = points[0]->cols;
    if( numPoints < 6 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number points must be more than 6" );
    }

    for( currImage = 0; currImage < 3; currImage++ )
    {
        if( points[currImage]->cols != numPoints || statuses[currImage]->cols != numPoints )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Number of points and statuses must be the same" );
        }

        if( points[currImage]->rows != 2 )
        {
            CV_ERROR( CV_StsOutOfRange, "Number of points coordinates must be == 2" );
        }

        if( statuses[currImage]->rows != 1 )
        {
            CV_ERROR( CV_StsOutOfRange, "Each of status must be matrix 1xN" );
        }

        if( projMatrs[currImage]->rows != 3 || projMatrs[currImage]->cols != 4 )
        {
            CV_ERROR( CV_StsOutOfRange, "Each of projection matrix must be 3x4" );
        }
    }


    /* Create common status for all points */

    int i;

    CV_CALL( comStat = (unsigned char*)cvAlloc(sizeof(unsigned char)*numPoints) );

    unsigned char *stats[3];

    stats[0] = statuses[0]->data.ptr;
    stats[1] = statuses[1]->data.ptr;
    stats[2] = statuses[2]->data.ptr;

    int numTripl;
    numTripl = 0;
    for( i = 0; i < numPoints; i++ )
    {
        comStat[i] = (unsigned char)(stats[0][i] * stats[1][i] * stats[2][i]);
        numTripl += comStat[i];
    }

    if( numTripl > 0 )
    {
        /* Create new arrays with points */
        CV_CALL( triPoints[0] = cvCreateMat(2,numTripl,CV_64F) );
        CV_CALL( triPoints[1] = cvCreateMat(2,numTripl,CV_64F) );
        CV_CALL( triPoints[2] = cvCreateMat(2,numTripl,CV_64F) );
        if( points4D )
        {
            CV_CALL( triPoints4D  = cvCreateMat(4,numTripl,CV_64F) );
        }

        /* Create status array */
        CV_CALL( status = cvCreateMat(1,numTripl,CV_64F) );

        /* Copy points to new arrays */
        int currPnt = 0;
        for( i = 0; i < numPoints; i++ )
        {
            if( comStat[i] )
            {
                for( currImage = 0; currImage < 3; currImage++ )
                {
                    cvmSet(triPoints[currImage],0,currPnt,cvmGet(points[currImage],0,i));
                    cvmSet(triPoints[currImage],1,currPnt,cvmGet(points[currImage],1,i));
                }
                currPnt++;
            }
        }

        /* Call function */
        numProjMatrs = icvComputeProjectMatricesNPoints( triPoints[0],triPoints[1],triPoints[2],
                                                         projMatrs[0],projMatrs[1],projMatrs[2],
                                                         threshold,/* Threshold for good point */
                                                         p,/* Probability of good result. */
                                                         status,
                                                         triPoints4D);

        /* Get computed status and set to result */
        cvZero(resStatus);
        currPnt = 0;
        for( i = 0; i < numPoints; i++ )
        {
            if( comStat[i] )
            {
                if( cvmGet(status,0,currPnt) > 0 )
                {
                    resStatus->data.ptr[i] = 1;
                }
                currPnt++;
            }
        }

        if( triPoints4D )
        {
            /* Copy copmuted 4D points */
            cvZero(points4D);
            currPnt = 0;
            for( i = 0; i < numPoints; i++ )
            {
                if( comStat[i] )
                {
                    if( cvmGet(status,0,currPnt) > 0 )
                    {
                        cvmSet( points4D, 0, i, cvmGet( triPoints4D , 0, currPnt) );
                        cvmSet( points4D, 1, i, cvmGet( triPoints4D , 1, currPnt) );
                        cvmSet( points4D, 2, i, cvmGet( triPoints4D , 2, currPnt) );
                        cvmSet( points4D, 3, i, cvmGet( triPoints4D , 3, currPnt) );
                    }
                    currPnt++;
                }
            }
        }
    }

    __END__;

    /* Free allocated memory */
    cvReleaseMat(&status);
    cvFree( &comStat);
    cvReleaseMat(&status);

    cvReleaseMat(&triPoints[0]);
    cvReleaseMat(&triPoints[1]);
    cvReleaseMat(&triPoints[2]);
    cvReleaseMat(&triPoints4D);

    return numProjMatrs;

}

/*==========================================================================================*/
int icvComputeProjectMatricesNPoints(  CvMat* points1,CvMat* points2,CvMat* points3,
                                       CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                       double threshold,/* Threshold for good point */
                                       double p,/* Probability of good result. */
                                       CvMat* status,
                                       CvMat* points4D)
{
    /* Returns status for each point, Good or bad */

    /* Compute projection matrices using N points */

    char* flags = 0;
    char* bestFlags = 0;

    int numProjMatrs = 0;

    CvMat* tmpProjPoints[3]={0,0,0};
    CvMat* recPoints4D = 0;
    CvMat *reconPoints4D = 0;


    CV_FUNCNAME( "icvComputeProjectMatricesNPoints" );
    __BEGIN__;

    CvMat* points[3];
    points[0] = points1;
    points[1] = points2;
    points[2] = points3;

    /* Test for errors */
    if( points1   == 0 || points2   == 0 || points3   == 0 ||
        projMatr1 == 0 || projMatr2 == 0 || projMatr3 == 0 ||
        status == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points1)   || !CV_IS_MAT(points2)   || !CV_IS_MAT(points3)   ||
        !CV_IS_MAT(projMatr1) || !CV_IS_MAT(projMatr2) || !CV_IS_MAT(projMatr3)  ||
        !CV_IS_MAT(status) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    int numPoints;
    numPoints = points1->cols;

    if( numPoints < 6 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number points must be more than 6" );
    }

    if( numPoints != points2->cols || numPoints != points3->cols )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "number of points must be the same" );
    }

    if( p < 0 || p > 1.0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Probability must be >=0 and <=1" );
    }

    if( threshold < 0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Threshold for good points must be at least >= 0" );
    }

    CvMat* projMatrs[3];

    projMatrs[0] = projMatr1;
    projMatrs[1] = projMatr2;
    projMatrs[2] = projMatr3;

    int i;
    for( i = 0; i < 3; i++ )
    {
        if( projMatrs[i]->cols != 4 || projMatrs[i]->rows != 3 )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of projection matrices must be 3x4" );
        }
    }

    for( i = 0; i < 3; i++ )
    {
        if( points[i]->rows != 2)
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Number of coordinates of points must be 2" );
        }
    }

    /* use RANSAC algorithm to compute projection matrices */

    CV_CALL( recPoints4D = cvCreateMat(4,numPoints,CV_64F) );
    CV_CALL( tmpProjPoints[0] = cvCreateMat(2,numPoints,CV_64F) );
    CV_CALL( tmpProjPoints[1] = cvCreateMat(2,numPoints,CV_64F) );
    CV_CALL( tmpProjPoints[2] = cvCreateMat(2,numPoints,CV_64F) );

    CV_CALL( flags = (char*)cvAlloc(sizeof(char)*numPoints) );
    CV_CALL( bestFlags = (char*)cvAlloc(sizeof(char)*numPoints) );

    {
        int NumSamples = 500;/* just init number of samples */
        int wasCount = 0;  /* count of choosing samples */
        int maxGoodPoints = 0;
        int numGoodPoints = 0;

        double bestProjMatrs_dat[36];
        CvMat  bestProjMatrs[3];
        bestProjMatrs[0] = cvMat(3,4,CV_64F,bestProjMatrs_dat);
        bestProjMatrs[1] = cvMat(3,4,CV_64F,bestProjMatrs_dat+12);
        bestProjMatrs[2] = cvMat(3,4,CV_64F,bestProjMatrs_dat+24);

        double tmpProjMatr_dat[36*3];
        CvMat  tmpProjMatr[3];
        tmpProjMatr[0] = cvMat(9,4,CV_64F,tmpProjMatr_dat);
        tmpProjMatr[1] = cvMat(9,4,CV_64F,tmpProjMatr_dat+36);
        tmpProjMatr[2] = cvMat(9,4,CV_64F,tmpProjMatr_dat+72);

        /* choosen points */

        while( wasCount < NumSamples )
        {
            /* select samples */
            int randNumbs[6];
            icvGetRandNumbers(numPoints,6,randNumbs);

            /* random numbers of points was generated */
            /* select points */

            double selPoints_dat[2*6*3];
            CvMat selPoints[3];
            selPoints[0] = cvMat(2,6,CV_64F,selPoints_dat);
            selPoints[1] = cvMat(2,6,CV_64F,selPoints_dat+12);
            selPoints[2] = cvMat(2,6,CV_64F,selPoints_dat+24);

            /* Copy 6 point for random indexes */
            icvSelectColsByNumbers( points[0], &selPoints[0], randNumbs,6);
            icvSelectColsByNumbers( points[1], &selPoints[1], randNumbs,6);
            icvSelectColsByNumbers( points[2], &selPoints[2], randNumbs,6);

            /* Compute projection matrices for this points */
            int numProj = icvComputeProjectMatrices6Points( &selPoints[0],&selPoints[1],&selPoints[2],
                                                            &tmpProjMatr[0],&tmpProjMatr[1],&tmpProjMatr[2]);

            /* Compute number of good points for each matrix */
            CvMat proj6[3];
            for( int currProj = 0; currProj < numProj; currProj++ )
            {
                cvGetSubArr(&tmpProjMatr[0],&proj6[0],cvRect(0,currProj*3,4,3));
                cvGetSubArr(&tmpProjMatr[1],&proj6[1],cvRect(0,currProj*3,4,3));
                cvGetSubArr(&tmpProjMatr[2],&proj6[2],cvRect(0,currProj*3,4,3));

                /* Reconstruct points for projection matrices */
                icvReconstructPointsFor3View( &proj6[0],&proj6[1],&proj6[2],
                                           points[0], points[1], points[2],
                                           recPoints4D);

                /* Project points to images using projection matrices */
                icvProject4DPoints(recPoints4D,&proj6[0],tmpProjPoints[0]);
                icvProject4DPoints(recPoints4D,&proj6[1],tmpProjPoints[1]);
                icvProject4DPoints(recPoints4D,&proj6[2],tmpProjPoints[2]);

                /* Compute distances and number of good points (inliers) */
                int i;
                int currImage;
                numGoodPoints = 0;
                for( i = 0; i < numPoints; i++ )
                {
                    double dist=-1;
                    dist = 0;
                    /* Choose max distance for each of three points */
                    for( currImage = 0; currImage < 3; currImage++ )
                    {
                        double x1,y1,x2,y2;
                        x1 = cvmGet(tmpProjPoints[currImage],0,i);
                        y1 = cvmGet(tmpProjPoints[currImage],1,i);
                        x2 = cvmGet(points[currImage],0,i);
                        y2 = cvmGet(points[currImage],1,i);

                        double dx,dy;
                        dx = x1-x2;
                        dy = y1-y2;
#if 1
                        double newDist = dx*dx+dy*dy;
                        if( newDist > dist )
                        {
                            dist = newDist;
                        }
#else
                        dist += sqrt(dx*dx+dy*dy)/3.0;
#endif
                    }
                    dist = sqrt(dist);
                    flags[i] = (char)(dist > threshold ? 0 : 1);
                    numGoodPoints += flags[i];

                }


                if( numGoodPoints > maxGoodPoints )
                {/* Copy current projection matrices as best */

                    cvCopy(&proj6[0],&bestProjMatrs[0]);
                    cvCopy(&proj6[1],&bestProjMatrs[1]);
                    cvCopy(&proj6[2],&bestProjMatrs[2]);

                    maxGoodPoints = numGoodPoints;
                    /* copy best flags */
                    memcpy(bestFlags,flags,sizeof(flags[0])*numPoints);

                    /* Adaptive number of samples to count*/
			        double ep = 1 - (double)numGoodPoints / (double)numPoints;
                    if( ep == 1 )
                    {
                        ep = 0.5;/* if there is not good points set ration of outliers to 50% */
                    }

			        double newNumSamples = (log(1-p) / log(1-pow(1-ep,6)));
                    if(  newNumSamples < double(NumSamples) )
                    {
                        NumSamples = cvRound(newNumSamples);
                    }
                }
            }

            wasCount++;
        }
#if 0
        char str[300];
        sprintf(str,"Initial numPoints = %d\nmaxGoodPoints=%d\nRANSAC made %d steps",
                    numPoints,
                    maxGoodPoints,
                    cvRound(wasCount));
        MessageBox(0,str,"Info",MB_OK|MB_TASKMODAL);
#endif

        /* we may have best 6-point projection matrices. */
        /* and best points */
        /* use these points to improve matrices */

        if( maxGoodPoints < 6 )
        {
            /*  matrix not found */
            numProjMatrs = 0;
        }
        else
        {
            /* We may Improove matrices using ---- method */
            /* We may try to use Levenberg-Marquardt optimization */
            //int currIter = 0;
            int finalGoodPoints = 0;
            char *goodFlags = 0;
            goodFlags = (char*)cvAlloc(numPoints*sizeof(char));

            int needRepeat;
            do
            {
#if 0
/* Version without using status for Levenberg-Marquardt minimization */

                CvMat *optStatus;
                optStatus = cvCreateMat(1,numPoints,CV_64F);
                int testNumber = 0;
                for( i=0;i<numPoints;i++ )
                {
                    cvmSet(optStatus,0,i,(double)bestFlags[i]);
                    testNumber += bestFlags[i];
                }

                char str2[200];
                sprintf(str2,"test good num=%d\nmaxGoodPoints=%d",testNumber,maxGoodPoints);
                MessageBox(0,str2,"Info",MB_OK|MB_TASKMODAL);

                CvMat *gPresPoints;
                gPresPoints = cvCreateMat(1,maxGoodPoints,CV_64F);
                for( i = 0; i < maxGoodPoints; i++)
                {
                    cvmSet(gPresPoints,0,i,1.0);
                }

                /* Create array of points pres */
                CvMat *pointsPres[3];
                pointsPres[0] = gPresPoints;
                pointsPres[1] = gPresPoints;
                pointsPres[2] = gPresPoints;

                /* Create just good points 2D */
                CvMat *gPoints[3];
                icvCreateGoodPoints(points[0],&gPoints[0],optStatus);
                icvCreateGoodPoints(points[1],&gPoints[1],optStatus);
                icvCreateGoodPoints(points[2],&gPoints[2],optStatus);

                /* Create 4D points array for good points */
                CvMat *resPoints4D;
                resPoints4D = cvCreateMat(4,maxGoodPoints,CV_64F);

                CvMat* projMs[3];

                projMs[0] = &bestProjMatrs[0];
                projMs[1] = &bestProjMatrs[1];
                projMs[2] = &bestProjMatrs[2];


                CvMat resProjMatrs[3];
                double resProjMatrs_dat[36];
                resProjMatrs[0] = cvMat(3,4,CV_64F,resProjMatrs_dat);
                resProjMatrs[1] = cvMat(3,4,CV_64F,resProjMatrs_dat+12);
                resProjMatrs[2] = cvMat(3,4,CV_64F,resProjMatrs_dat+24);

                CvMat* resMatrs[3];
                resMatrs[0] = &resProjMatrs[0];
                resMatrs[1] = &resProjMatrs[1];
                resMatrs[2] = &resProjMatrs[2];

                cvOptimizeLevenbergMarquardtBundle( projMs,//projMs,
                                                    gPoints,//points,//points2D,
                                                    pointsPres,//pointsPres,
                                                    3,
                                                    resMatrs,//resProjMatrs,
                                                    resPoints4D,//resPoints4D,
                                                    100, 1e-9 );

                /* We found optimized projection matrices */

                CvMat *reconPoints4D;
                reconPoints4D = cvCreateMat(4,numPoints,CV_64F);

                /* Reconstruct all points using found projection matrices */
                icvReconstructPointsFor3View( &resProjMatrs[0],&resProjMatrs[1],&resProjMatrs[2],
                                              points[0], points[1], points[2],
                                              reconPoints4D);

                /* Project points to images using projection matrices */
                icvProject4DPoints(reconPoints4D,&resProjMatrs[0],tmpProjPoints[0]);
                icvProject4DPoints(reconPoints4D,&resProjMatrs[1],tmpProjPoints[1]);
                icvProject4DPoints(reconPoints4D,&resProjMatrs[2],tmpProjPoints[2]);


                /* Compute error for each point and select good */

                int currImage;
                finalGoodPoints = 0;
                for( i = 0; i < numPoints; i++ )
                {
                    double dist=-1;
                    /* Choose max distance for each of three points */
                    for( currImage = 0; currImage < 3; currImage++ )
                    {
                        double x1,y1,x2,y2;
                        x1 = cvmGet(tmpProjPoints[currImage],0,i);
                        y1 = cvmGet(tmpProjPoints[currImage],1,i);
                        x2 = cvmGet(points[currImage],0,i);
                        y2 = cvmGet(points[currImage],1,i);

                        double dx,dy;
                        dx = x1-x2;
                        dy = y1-y2;

                        double newDist = dx*dx+dy*dy;
                        if( newDist > dist )
                        {
                            dist = newDist;
                        }
                    }
                    dist = sqrt(dist);
                    goodFlags[i] = (char)(dist > threshold ? 0 : 1);
                    finalGoodPoints += goodFlags[i];
                }

                char str[200];
                sprintf(str,"Was num = %d\nNew num=%d",maxGoodPoints,finalGoodPoints);
                MessageBox(0,str,"Info",MB_OK|MB_TASKMODAL);
                if( finalGoodPoints > maxGoodPoints )
                {
                    /* Copy new version of projection matrices */
                    cvCopy(&resProjMatrs[0],&bestProjMatrs[0]);
                    cvCopy(&resProjMatrs[1],&bestProjMatrs[1]);
                    cvCopy(&resProjMatrs[2],&bestProjMatrs[2]);
                    memcpy(bestFlags,goodFlags,numPoints*sizeof(char));
                    maxGoodPoints = finalGoodPoints;
                }

                cvReleaseMat(&optStatus);
                cvReleaseMat(&resPoints4D);
#else
/* Version with using status for Levenberd-Marquardt minimization */

                /* Create status */
                CvMat *optStatus;
                optStatus = cvCreateMat(1,numPoints,CV_64F);
                for( i=0;i<numPoints;i++ )
                {
                    cvmSet(optStatus,0,i,(double)bestFlags[i]);
                }

                CvMat *pointsPres[3];
                pointsPres[0] = optStatus;
                pointsPres[1] = optStatus;
                pointsPres[2] = optStatus;

                /* Create 4D points array for good points */
                CvMat *resPoints4D;
                resPoints4D = cvCreateMat(4,numPoints,CV_64F);

                CvMat* projMs[3];

                projMs[0] = &bestProjMatrs[0];
                projMs[1] = &bestProjMatrs[1];
                projMs[2] = &bestProjMatrs[2];

                CvMat resProjMatrs[3];
                double resProjMatrs_dat[36];
                resProjMatrs[0] = cvMat(3,4,CV_64F,resProjMatrs_dat);
                resProjMatrs[1] = cvMat(3,4,CV_64F,resProjMatrs_dat+12);
                resProjMatrs[2] = cvMat(3,4,CV_64F,resProjMatrs_dat+24);

                CvMat* resMatrs[3];
                resMatrs[0] = &resProjMatrs[0];
                resMatrs[1] = &resProjMatrs[1];
                resMatrs[2] = &resProjMatrs[2];

                cvOptimizeLevenbergMarquardtBundle( projMs,//projMs,
                                                    points,//points2D,
                                                    pointsPres,//pointsPres,
                                                    3,
                                                    resMatrs,//resProjMatrs,
                                                    resPoints4D,//resPoints4D,
                                                    100, 1e-9 );

                /* We found optimized projection matrices */

                reconPoints4D = cvCreateMat(4,numPoints,CV_64F);

                /* Reconstruct all points using found projection matrices */
                icvReconstructPointsFor3View( &resProjMatrs[0],&resProjMatrs[1],&resProjMatrs[2],
                                              points[0], points[1], points[2],
                                              reconPoints4D);

                /* Project points to images using projection matrices */
                icvProject4DPoints(reconPoints4D,&resProjMatrs[0],tmpProjPoints[0]);
                icvProject4DPoints(reconPoints4D,&resProjMatrs[1],tmpProjPoints[1]);
                icvProject4DPoints(reconPoints4D,&resProjMatrs[2],tmpProjPoints[2]);


                /* Compute error for each point and select good */

                int currImage;
                finalGoodPoints = 0;
                for( i = 0; i < numPoints; i++ )
                {
                    double dist=-1;
                    /* Choose max distance for each of three points */
                    for( currImage = 0; currImage < 3; currImage++ )
                    {
                        double x1,y1,x2,y2;
                        x1 = cvmGet(tmpProjPoints[currImage],0,i);
                        y1 = cvmGet(tmpProjPoints[currImage],1,i);
                        x2 = cvmGet(points[currImage],0,i);
                        y2 = cvmGet(points[currImage],1,i);

                        double dx,dy;
                        dx = x1-x2;
                        dy = y1-y2;

                        double newDist = dx*dx+dy*dy;
                        if( newDist > dist )
                        {
                            dist = newDist;
                        }
                    }
                    dist = sqrt(dist);
                    goodFlags[i] = (char)(dist > threshold ? 0 : 1);
                    finalGoodPoints += goodFlags[i];
                }

                /*char str[200];
                sprintf(str,"Was num = %d\nNew num=%d",maxGoodPoints,finalGoodPoints);
                MessageBox(0,str,"Info",MB_OK|MB_TASKMODAL);*/

                needRepeat = 0;
                if( finalGoodPoints > maxGoodPoints )
                {
                    /* Copy new version of projection matrices */
                    cvCopy(&resProjMatrs[0],&bestProjMatrs[0]);
                    cvCopy(&resProjMatrs[1],&bestProjMatrs[1]);
                    cvCopy(&resProjMatrs[2],&bestProjMatrs[2]);
                    memcpy(bestFlags,goodFlags,numPoints*sizeof(char));
                    maxGoodPoints = finalGoodPoints;
                    needRepeat = 1;
                }

                cvReleaseMat(&optStatus);
                cvReleaseMat(&resPoints4D);


#endif
            } while ( needRepeat );

            cvFree( &goodFlags);




            numProjMatrs = 1;

            /* Copy projection matrices */
            cvConvert(&bestProjMatrs[0],projMatr1);
            cvConvert(&bestProjMatrs[1],projMatr2);
            cvConvert(&bestProjMatrs[2],projMatr3);

            if( status )
            {
                /* copy status for each points if need */
                for( int i = 0; i < numPoints; i++)
                {
                    cvmSet(status,0,i,(double)bestFlags[i]);
                }
            }
        }
    }

    if( points4D )
    {/* Fill reconstructed points */

        cvZero(points4D);
        icvReconstructPointsFor3View( projMatr1,projMatr2,projMatr3,
                                      points[0], points[1], points[2],
                                      points4D);
    }



    __END__;

    cvFree( &flags);
    cvFree( &bestFlags);

    cvReleaseMat(&recPoints4D);
    cvReleaseMat(&tmpProjPoints[0]);
    cvReleaseMat(&tmpProjPoints[1]);
    cvReleaseMat(&tmpProjPoints[2]);

    return numProjMatrs;
}

/*==========================================================================================*/

void icvFindBaseTransform(CvMat* points,CvMat* resultT)
{

    CV_FUNCNAME( "icvFindBaseTransform" );
    __BEGIN__;

    if( points == 0 || resultT == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points) || !CV_IS_MAT(resultT) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "points and resultT must be a matrices" );
    }

    if( points->rows != 2 || points->cols != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be 4. And they must have 2 coordinates" );
    }

    if( resultT->rows != 3 || resultT->cols != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "size of matrix resultT must be 3x3" );
    }

    /* Function gets four points and compute transformation to e1=(100) e2=(010) e3=(001) e4=(111) */

    /* !!! test each three points not collinear. Need to test */

    /* Create matrices */
    CvMat matrA;
    CvMat vectB;
    double matrA_dat[3*3];
    double vectB_dat[3];
    matrA = cvMat(3,3,CV_64F,matrA_dat);
    vectB = cvMat(3,1,CV_64F,vectB_dat);

    /* fill matrices */
    int i;
    for( i = 0; i < 3; i++ )
    {
        cvmSet(&matrA,0,i,cvmGet(points,0,i));
        cvmSet(&matrA,1,i,cvmGet(points,1,i));
        cvmSet(&matrA,2,i,1);
    }

    /* Fill vector B */
    cvmSet(&vectB,0,0,cvmGet(points,0,3));
    cvmSet(&vectB,1,0,cvmGet(points,1,3));
    cvmSet(&vectB,2,0,1);

    /* result scale */
    CvMat scale;
    double scale_dat[3];
    scale = cvMat(3,1,CV_64F,scale_dat);

    cvSolve(&matrA,&vectB,&scale,CV_SVD);

    /* multiply by scale */
    int j;
    for( j = 0; j < 3; j++ )
    {
        double sc = scale_dat[j];
        for( i = 0; i < 3; i++ )
        {
            matrA_dat[i*3+j] *= sc;
        }
    }

    /* Convert inverse matrix */
    CvMat tmpRes;
    double tmpRes_dat[9];
    tmpRes = cvMat(3,3,CV_64F,tmpRes_dat);
    cvInvert(&matrA,&tmpRes);

    cvConvert(&tmpRes,resultT);

    __END__;

    return;
}


/*==========================================================================================*/
void GetGeneratorReduceFundSolution(CvMat* points1,CvMat* points2,CvMat* fundReduceCoef1,CvMat* fundReduceCoef2)
{

    CV_FUNCNAME( "GetGeneratorReduceFundSolution" );
    __BEGIN__;

    /* Test input data for errors */

    if( points1 == 0 || points2 == 0 || fundReduceCoef1 == 0 || fundReduceCoef2 == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points1) || !CV_IS_MAT(points2) || !CV_IS_MAT(fundReduceCoef1) || !CV_IS_MAT(fundReduceCoef2) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }



    if( points1->rows != 3 || points1->cols != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points1 must be 3 and and have 3 coordinates" );
    }

    if( points2->rows != 3 || points2->cols != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points2 must be 3 and and have 3 coordinates" );
    }

    if( fundReduceCoef1->rows != 1 || fundReduceCoef1->cols != 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of fundReduceCoef1 must be 1x5" );
    }

    if( fundReduceCoef2->rows != 1 || fundReduceCoef2->cols != 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of fundReduceCoef2 must be 1x5" );
    }

    /* Using 3 corr. points compute reduce */

    /* Create matrix */
    CvMat matrA;
    double matrA_dat[3*5];
    matrA = cvMat(3,5,CV_64F,matrA_dat);
    int i;
    for( i = 0; i < 3; i++ )
    {
        double x1,y1,w1,x2,y2,w2;
        x1 = cvmGet(points1,0,i);
        y1 = cvmGet(points1,1,i);
        w1 = cvmGet(points1,2,i);

        x2 = cvmGet(points2,0,i);
        y2 = cvmGet(points2,1,i);
        w2 = cvmGet(points2,2,i);

        cvmSet(&matrA,i,0,y1*x2-y1*w2);
        cvmSet(&matrA,i,1,w1*x2-y1*w2);
        cvmSet(&matrA,i,2,x1*y2-y1*w2);
        cvmSet(&matrA,i,3,w1*y2-y1*w2);
        cvmSet(&matrA,i,4,x1*w2-y1*w2);
    }

    /* solve system using svd */
    CvMat matrU;
    CvMat matrW;
    CvMat matrV;

    double matrU_dat[3*3];
    double matrW_dat[3*5];
    double matrV_dat[5*5];

    matrU = cvMat(3,3,CV_64F,matrU_dat);
    matrW = cvMat(3,5,CV_64F,matrW_dat);
    matrV = cvMat(5,5,CV_64F,matrV_dat);

    /* From svd we need just two last vectors of V or two last row V' */
    /* We get transposed matrixes U and V */

    cvSVD(&matrA,&matrW,0,&matrV,CV_SVD_V_T);

    /* copy results to fundamental matrices */
    for(i=0;i<5;i++)
    {
        cvmSet(fundReduceCoef1,0,i,cvmGet(&matrV,3,i));
        cvmSet(fundReduceCoef2,0,i,cvmGet(&matrV,4,i));
    }

    __END__;
    return;

}

/*==========================================================================================*/

int GetGoodReduceFundamMatrFromTwo(CvMat* fundReduceCoef1,CvMat* fundReduceCoef2,CvMat* resFundReduceCoef)
{
    int numRoots = 0;

    CV_FUNCNAME( "GetGoodReduceFundamMatrFromTwo" );
    __BEGIN__;

    if( fundReduceCoef1 == 0 || fundReduceCoef2 == 0 || resFundReduceCoef == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(fundReduceCoef1) || !CV_IS_MAT(fundReduceCoef2) || !CV_IS_MAT(resFundReduceCoef) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    /* using two fundamental matrix comute matrixes for det(F)=0 */
    /* May compute 1 or 3 matrices. Returns number of solutions */
    /* Here we will use case F=a*F1+(1-a)*F2  instead of F=m*F1+l*F2 */

    /* Test for errors */
    if( fundReduceCoef1->rows != 1 || fundReduceCoef1->cols != 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of fundReduceCoef1 must be 1x5" );
    }

    if( fundReduceCoef2->rows != 1 || fundReduceCoef2->cols != 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of fundReduceCoef2 must be 1x5" );
    }

    if( (resFundReduceCoef->rows != 1 && resFundReduceCoef->rows != 3)  || resFundReduceCoef->cols != 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of resFundReduceCoef must be 1x5" );
    }

    double p1,q1,r1,s1,t1;
    double p2,q2,r2,s2,t2;
    p1 = cvmGet(fundReduceCoef1,0,0);
    q1 = cvmGet(fundReduceCoef1,0,1);
    r1 = cvmGet(fundReduceCoef1,0,2);
    s1 = cvmGet(fundReduceCoef1,0,3);
    t1 = cvmGet(fundReduceCoef1,0,4);

    p2 = cvmGet(fundReduceCoef2,0,0);
    q2 = cvmGet(fundReduceCoef2,0,1);
    r2 = cvmGet(fundReduceCoef2,0,2);
    s2 = cvmGet(fundReduceCoef2,0,3);
    t2 = cvmGet(fundReduceCoef2,0,4);

    /* solve equation */
    CvMat result;
    CvMat coeffs;
    double result_dat[2*3];
    double coeffs_dat[4];
    result = cvMat(2,3,CV_64F,result_dat);
    coeffs = cvMat(1,4,CV_64F,coeffs_dat);

    coeffs_dat[0] = ((r1-r2)*(-p1-q1-r1-s1-t1+p2+q2+r2+s2+t2)*(q1-q2)+(p1-p2)*(s1-s2)*(t1-t2));/* *a^3 */
    coeffs_dat[1] = ((r2*(-p1-q1-r1-s1-t1+p2+q2+r2+s2+t2)+(r1-r2)*(-p2-q2-r2-s2-t2))*(q1-q2)+(r1-r2)*(-p1-q1-r1-s1-t1+p2+q2+r2+s2+t2)*q2+(p2*(s1-s2)+(p1-p2)*s2)*(t1-t2)+(p1-p2)*(s1-s2)*t2);/* *a^2 */
    coeffs_dat[2] = (r2*(-p2-q2-r2-s2-t2)*(q1-q2)+(r2*(-p1-q1-r1-s1-t1+p2+q2+r2+s2+t2)+(r1-r2)*(-p2-q2-r2-s2-t2))*q2+p2*s2*(t1-t2)+(p2*(s1-s2)+(p1-p2)*s2)*t2);/* *a */
    coeffs_dat[3] = r2*(-p2-q2-r2-s2-t2)*q2+p2*s2*t2;/* 1 */

    int num;
    num = cvSolveCubic(&coeffs,&result);


    /* test number of solutions and test for real solutions */
    int i;
    for( i = 0; i < num; i++ )
    {
        if( fabs(cvmGet(&result,1,i)) < 1e-8 )
        {
            double alpha = cvmGet(&result,0,i);
            int j;
            for( j = 0; j < 5; j++ )
            {
                cvmSet(resFundReduceCoef,numRoots,j,
                    alpha * cvmGet(fundReduceCoef1,0,j) + (1-alpha) * cvmGet(fundReduceCoef2,0,j) );
            }
            numRoots++;
        }
    }

    __END__;
    return numRoots;
}

/*==========================================================================================*/

void GetProjMatrFromReducedFundamental(CvMat* fundReduceCoefs,CvMat* projMatrCoefs)
{
    CV_FUNCNAME( "GetProjMatrFromReducedFundamental" );
    __BEGIN__;

    /* Test for errors */
    if( fundReduceCoefs == 0 || projMatrCoefs == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(fundReduceCoefs) || !CV_IS_MAT(projMatrCoefs) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }


    if( fundReduceCoefs->rows != 1 || fundReduceCoefs->cols != 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of fundReduceCoefs must be 1x5" );
    }

    if( projMatrCoefs->rows != 1 || projMatrCoefs->cols != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of projMatrCoefs must be 1x4" );
    }

    /* Computes project matrix from given reduced matrix */
    /* we have p,q,r,s,t and need get a,b,c,d */
    /* Fill matrix to compute ratio a:b:c as A:B:C */

    CvMat matrA;
    double matrA_dat[3*3];
    matrA = cvMat(3,3,CV_64F,matrA_dat);

    double p,q,r,s,t;
    p = cvmGet(fundReduceCoefs,0,0);
    q = cvmGet(fundReduceCoefs,0,1);
    r = cvmGet(fundReduceCoefs,0,2);
    s = cvmGet(fundReduceCoefs,0,3);
    t = cvmGet(fundReduceCoefs,0,4);

    matrA_dat[0] = p;
    matrA_dat[1] = r;
    matrA_dat[2] = 0;

    matrA_dat[3] = q;
    matrA_dat[4] = 0;
    matrA_dat[5] = t;

    matrA_dat[6] = 0;
    matrA_dat[7] = s;
    matrA_dat[8] = -(p+q+r+s+t);

    CvMat matrU;
    CvMat matrW;
    CvMat matrV;

    double matrU_dat[3*3];
    double matrW_dat[3*3];
    double matrV_dat[3*3];

    matrU = cvMat(3,3,CV_64F,matrU_dat);
    matrW = cvMat(3,3,CV_64F,matrW_dat);
    matrV = cvMat(3,3,CV_64F,matrV_dat);

    /* From svd we need just last vector of V or last row V' */
    /* We get transposed matrixes U and V */

    cvSVD(&matrA,&matrW,0,&matrV,CV_SVD_V_T);

    double A1,B1,C1;
    A1 = matrV_dat[6];
    B1 = matrV_dat[7];
    C1 = matrV_dat[8];

    /* Get second coeffs */
    matrA_dat[0] = 0;
    matrA_dat[1] = r;
    matrA_dat[2] = t;

    matrA_dat[3] = p;
    matrA_dat[4] = 0;
    matrA_dat[5] = -(p+q+r+s+t);

    matrA_dat[6] = q;
    matrA_dat[7] = s;
    matrA_dat[8] = 0;

    cvSVD(&matrA,&matrW,0,&matrV,CV_SVD_V_T);

    double A2,B2,C2;
    A2 = matrV_dat[6];
    B2 = matrV_dat[7];
    C2 = matrV_dat[8];

    double a,b,c,d;
    {
        CvMat matrK;
        double matrK_dat[36];
        matrK = cvMat(6,6,CV_64F,matrK_dat);
        cvZero(&matrK);

        matrK_dat[0]  = 1;
        matrK_dat[7]  = 1;
        matrK_dat[14] = 1;

        matrK_dat[18] = -1;
        matrK_dat[25] = -1;
        matrK_dat[32] = -1;

        matrK_dat[21] = 1;
        matrK_dat[27] = 1;
        matrK_dat[33] = 1;

        matrK_dat[0*6+4] = -A1;
        matrK_dat[1*6+4] = -B1;
        matrK_dat[2*6+4] = -C1;

        matrK_dat[3*6+5] = -A2;
        matrK_dat[4*6+5] = -B2;
        matrK_dat[5*6+5] = -C2;

        CvMat matrU;
        CvMat matrW;
        CvMat matrV;

        double matrU_dat[36];
        double matrW_dat[36];
        double matrV_dat[36];

        matrU = cvMat(6,6,CV_64F,matrU_dat);
        matrW = cvMat(6,6,CV_64F,matrW_dat);
        matrV = cvMat(6,6,CV_64F,matrV_dat);

        /* From svd we need just last vector of V or last row V' */
        /* We get transposed matrixes U and V */

        cvSVD(&matrK,&matrW,0,&matrV,CV_SVD_V_T);

        a = matrV_dat[6*5+0];
        b = matrV_dat[6*5+1];
        c = matrV_dat[6*5+2];
        d = matrV_dat[6*5+3];
        /* we don't need last two coefficients. Because it just a k1,k2 */

        cvmSet(projMatrCoefs,0,0,a);
        cvmSet(projMatrCoefs,0,1,b);
        cvmSet(projMatrCoefs,0,2,c);
        cvmSet(projMatrCoefs,0,3,d);

    }

    __END__;
    return;
}

/*==========================================================================================*/

void icvComputeProjectMatrix(CvMat* objPoints,CvMat* projPoints,CvMat* projMatr)
{/* Using SVD method */

    /* Reconstruct points using object points and projected points */
    /* Number of points must be >=6 */

    CvMat matrV;
    CvMat* matrA = 0;
    CvMat* matrW = 0;
    CvMat* workProjPoints = 0;
    CvMat* tmpProjPoints = 0;

    CV_FUNCNAME( "icvComputeProjectMatrix" );
    __BEGIN__;

    /* Test for errors */
    if( objPoints == 0 || projPoints == 0 || projMatr == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(objPoints) || !CV_IS_MAT(projPoints) || !CV_IS_MAT(projMatr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    if( projMatr->rows != 3 || projMatr->cols != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of projMatr must be 3x4" );
    }

    int numPoints;
    numPoints = projPoints->cols;
    if( numPoints < 6 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must be at least 6" );
    }

    if( numPoints != objPoints->cols )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be same" );
    }

    if( objPoints->rows != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Object points must have 4 coordinates" );
    }

    if( projPoints->rows != 3 &&  projPoints->rows != 2 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Projected points must have 2 or 3 coordinates" );
    }

    /* Create and fill matrix A */
    CV_CALL( matrA = cvCreateMat(numPoints*3, 12, CV_64F) );
    CV_CALL( matrW = cvCreateMat(numPoints*3, 12, CV_64F) );

    if( projPoints->rows == 2 )
    {
        CV_CALL( tmpProjPoints = cvCreateMat(3,numPoints,CV_64F) );
        cvMake3DPoints(projPoints,tmpProjPoints);
        workProjPoints = tmpProjPoints;
    }
    else
    {
        workProjPoints = projPoints;
    }

    double matrV_dat[144];
    matrV = cvMat(12,12,CV_64F,matrV_dat);
    int i;

    char* dat;
    dat = (char*)(matrA->data.db);

#if 1
    FILE *file;
    file = fopen("d:\\test\\recProjMatr.txt","w");

#endif
    for( i = 0;i < numPoints; i++ )
    {
        double x,y,w;
        double X,Y,Z,W;
        double*  matrDat = (double*)dat;

        x = cvmGet(workProjPoints,0,i);
        y = cvmGet(workProjPoints,1,i);
        w = cvmGet(workProjPoints,2,i);


        X = cvmGet(objPoints,0,i);
        Y = cvmGet(objPoints,1,i);
        Z = cvmGet(objPoints,2,i);
        W = cvmGet(objPoints,3,i);

#if 1
        fprintf(file,"%d (%lf %lf %lf %lf) - (%lf %lf %lf)\n",i,X,Y,Z,W,x,y,w );
#endif

/*---*/
        matrDat[ 0] = 0;
        matrDat[ 1] = 0;
        matrDat[ 2] = 0;
        matrDat[ 3] = 0;

        matrDat[ 4] = -w*X;
        matrDat[ 5] = -w*Y;
        matrDat[ 6] = -w*Z;
        matrDat[ 7] = -w*W;

        matrDat[ 8] = y*X;
        matrDat[ 9] = y*Y;
        matrDat[10] = y*Z;
        matrDat[11] = y*W;
/*---*/
        matrDat[12] = w*X;
        matrDat[13] = w*Y;
        matrDat[14] = w*Z;
        matrDat[15] = w*W;

        matrDat[16] = 0;
        matrDat[17] = 0;
        matrDat[18] = 0;
        matrDat[19] = 0;

        matrDat[20] = -x*X;
        matrDat[21] = -x*Y;
        matrDat[22] = -x*Z;
        matrDat[23] = -x*W;
/*---*/
        matrDat[24] = -y*X;
        matrDat[25] = -y*Y;
        matrDat[26] = -y*Z;
        matrDat[27] = -y*W;

        matrDat[28] = x*X;
        matrDat[29] = x*Y;
        matrDat[30] = x*Z;
        matrDat[31] = x*W;

        matrDat[32] = 0;
        matrDat[33] = 0;
        matrDat[34] = 0;
        matrDat[35] = 0;
/*---*/
        dat += (matrA->step)*3;
    }
#if 1
    fclose(file);

#endif

    /* Solve this system */

    /* From svd we need just last vector of V or last row V' */
    /* We get transposed matrix V */

    cvSVD(matrA,matrW,0,&matrV,CV_SVD_V_T);

    /* projected matrix was computed */
    for( i = 0; i < 12; i++ )
    {
        cvmSet(projMatr,i/4,i%4,cvmGet(&matrV,11,i));
    }

    cvReleaseMat(&matrA);
    cvReleaseMat(&matrW);
    cvReleaseMat(&tmpProjPoints);
    __END__;
}


/*==========================================================================================*/
/*  May be useless function */
void icvComputeTransform4D(CvMat* points1,CvMat* points2,CvMat* transMatr)
{
    CvMat* matrA = 0;
    CvMat* matrW = 0;

    double matrV_dat[256];
    CvMat  matrV = cvMat(16,16,CV_64F,matrV_dat);

    CV_FUNCNAME( "icvComputeTransform4D" );
    __BEGIN__;

    if( points1 == 0 || points2 == 0 || transMatr == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points1) || !CV_IS_MAT(points2) || !CV_IS_MAT(transMatr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    /* Computes transformation matrix (4x4) for points1 -> points2 */
    /* p2=H*p1 */

    /* Test for errors */
    int numPoints;
    numPoints = points1->cols;

    /* we must have at least 5 points */
    if( numPoints < 5 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be at least 5" );
    }

    if( numPoints != points2->cols )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be the same" );
    }

    if( transMatr->rows != 4 || transMatr->cols != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of transMatr must be 4x4" );
    }

    if( points1->rows != 4 || points2->rows != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of coordinates of points must be 4" );
    }

    /* Create matrix */
    CV_CALL( matrA = cvCreateMat(6*numPoints,16,CV_64F) );
    CV_CALL( matrW = cvCreateMat(6*numPoints,16,CV_64F) );

    cvZero(matrA);

    /* Fill matrices */
    int i;
    for( i = 0; i < numPoints; i++ )/* For each point */
    {
        double X1,Y1,Z1,W1;
        double P[4];

        P[0] = cvmGet(points1,0,i);
        P[1] = cvmGet(points1,1,i);
        P[2] = cvmGet(points1,2,i);
        P[3] = cvmGet(points1,3,i);

        X1 = cvmGet(points2,0,i);
        Y1 = cvmGet(points2,1,i);
        Z1 = cvmGet(points2,2,i);
        W1 = cvmGet(points2,3,i);

        /* Fill matrA */
        for( int j = 0; j < 4; j++ )/* For each coordinate */
        {
            double x,y,z,w;

            x = X1*P[j];
            y = Y1*P[j];
            z = Z1*P[j];
            w = W1*P[j];

            cvmSet(matrA,6*i+0,4*0+j,y);
            cvmSet(matrA,6*i+0,4*1+j,-x);

            cvmSet(matrA,6*i+1,4*0+j,z);
            cvmSet(matrA,6*i+1,4*2+j,-x);

            cvmSet(matrA,6*i+2,4*0+j,w);
            cvmSet(matrA,6*i+2,4*3+j,-x);

            cvmSet(matrA,6*i+3,4*1+j,-z);
            cvmSet(matrA,6*i+3,4*2+j,y);

            cvmSet(matrA,6*i+4,4*1+j,-w);
            cvmSet(matrA,6*i+4,4*3+j,y);

            cvmSet(matrA,6*i+5,4*2+j,-w);
            cvmSet(matrA,6*i+5,4*3+j,z);
        }
    }

    /* From svd we need just two last vectors of V or two last row V' */
    /* We get transposed matrixes U and V */

    cvSVD(matrA,matrW,0,&matrV,CV_SVD_V_T);

    /* Copy result to result matrix */
    for( i = 0; i < 16; i++ )
    {
        cvmSet(transMatr,i/4,i%4,cvmGet(&matrV,15,i));
    }

    cvReleaseMat(&matrA);
    cvReleaseMat(&matrW);

    __END__;
    return;
}

/*==========================================================================================*/

void icvReconstructPointsFor3View( CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                CvMat* projPoints1,CvMat* projPoints2,CvMat* projPoints3,
                                CvMat* points4D)
{
    CV_FUNCNAME( "icvReconstructPointsFor3View" );
    __BEGIN__;

    if( projMatr1 == 0 || projMatr2 == 0 || projMatr3 == 0 ||
        projPoints1 == 0 || projPoints2 == 0 || projPoints3 == 0 ||
        points4D == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(projMatr1) || !CV_IS_MAT(projMatr2) || !CV_IS_MAT(projMatr3) ||
        !CV_IS_MAT(projPoints1) || !CV_IS_MAT(projPoints2) || !CV_IS_MAT(projPoints3)  ||
        !CV_IS_MAT(points4D) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    int numPoints;
    numPoints = projPoints1->cols;

    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must be more than zero" );
    }

    if( projPoints2->cols != numPoints || projPoints3->cols != numPoints || points4D->cols != numPoints )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be the same" );
    }

    if( projPoints1->rows != 2 || projPoints2->rows != 2 || projPoints3->rows != 2)
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of proj points coordinates must be == 2" );
    }

    if( points4D->rows != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of world points coordinates must be == 4" );
    }

    if( projMatr1->cols != 4 || projMatr1->rows != 3 ||
        projMatr2->cols != 4 || projMatr2->rows != 3 ||
        projMatr3->cols != 4 || projMatr3->rows != 3)
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of projection matrices must be 3x4" );
    }

    CvMat matrA;
    double matrA_dat[36];
    matrA = cvMat(9,4,CV_64F,matrA_dat);

    //CvMat matrU;
    CvMat matrW;
    CvMat matrV;
    //double matrU_dat[9*9];
    double matrW_dat[9*4];
    double matrV_dat[4*4];

    //matrU = cvMat(9,9,CV_64F,matrU_dat);
    matrW = cvMat(9,4,CV_64F,matrW_dat);
    matrV = cvMat(4,4,CV_64F,matrV_dat);

    CvMat* projPoints[3];
    CvMat* projMatrs[3];

    projPoints[0] = projPoints1;
    projPoints[1] = projPoints2;
    projPoints[2] = projPoints3;

    projMatrs[0] = projMatr1;
    projMatrs[1] = projMatr2;
    projMatrs[2] = projMatr3;

    /* Solve system for each point */
    int i,j;
    for( i = 0; i < numPoints; i++ )/* For each point */
    {
        /* Fill matrix for current point */
        for( j = 0; j < 3; j++ )/* For each view */
        {
            double x,y;
            x = cvmGet(projPoints[j],0,i);
            y = cvmGet(projPoints[j],1,i);
            for( int k = 0; k < 4; k++ )
            {
                cvmSet(&matrA, j*3+0, k, x * cvmGet(projMatrs[j],2,k) -     cvmGet(projMatrs[j],0,k) );
                cvmSet(&matrA, j*3+1, k, y * cvmGet(projMatrs[j],2,k) -     cvmGet(projMatrs[j],1,k) );
                cvmSet(&matrA, j*3+2, k, x * cvmGet(projMatrs[j],1,k) - y * cvmGet(projMatrs[j],0,k) );
            }
        }
        /* Solve system for current point */
        {
            cvSVD(&matrA,&matrW,0,&matrV,CV_SVD_V_T);

            /* Copy computed point */
            cvmSet(points4D,0,i,cvmGet(&matrV,3,0));/* X */
            cvmSet(points4D,1,i,cvmGet(&matrV,3,1));/* Y */
            cvmSet(points4D,2,i,cvmGet(&matrV,3,2));/* Z */
            cvmSet(points4D,3,i,cvmGet(&matrV,3,3));/* W */
        }
    }

    /* Points was reconstructed. Try to reproject points */
    /* We can compute reprojection error if need */
    /*{
        int i;
        CvMat point3D;
        double point3D_dat[4];
        point3D = cvMat(4,1,CV_64F,point3D_dat);

        CvMat point2D;
        double point2D_dat[3];
        point2D = cvMat(3,1,CV_64F,point2D_dat);

        for( i = 0; i < numPoints; i++ )
        {
            double W = cvmGet(points4D,3,i);

            point3D_dat[0] = cvmGet(points4D,0,i)/W;
            point3D_dat[1] = cvmGet(points4D,1,i)/W;
            point3D_dat[2] = cvmGet(points4D,2,i)/W;
            point3D_dat[3] = 1;

                // !!! Project this point for each camera
                for( int currCamera = 0; currCamera < 3; currCamera++ )
                {
                    cvmMul(projMatrs[currCamera], &point3D, &point2D);

                    float x,y;
                    float xr,yr,wr;
                    x = (float)cvmGet(projPoints[currCamera],0,i);
                    y = (float)cvmGet(projPoints[currCamera],1,i);

                    wr = (float)point2D_dat[2];
                    xr = (float)(point2D_dat[0]/wr);
                    yr = (float)(point2D_dat[1]/wr);

                    float deltaX,deltaY;
                    deltaX = (float)fabs(x-xr);
                    deltaY = (float)fabs(y-yr);
                }
        }
    }*/

    __END__;
    return;
}




#if 0
void ReconstructPointsFor3View_bySolve( CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                CvMat* projPoints1,CvMat* projPoints2,CvMat* projPoints3,
                                CvMat* points3D)
{
    CV_FUNCNAME( "ReconstructPointsFor3View" );
    __BEGIN__;


    int numPoints;
    numPoints = projPoints1->cols;
    if( projPoints2->cols != numPoints || projPoints3->cols != numPoints || points3D->cols != numPoints )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points must be the same" );
    }

    if( projPoints1->rows != 2 || projPoints2->rows != 2 || projPoints3->rows != 2)
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of proj points coordinates must be == 2" );
    }

    if( points3D->rows != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of world points coordinates must be == 4" );
    }

    if( projMatr1->cols != 4 || projMatr1->rows != 3 ||
        projMatr2->cols != 4 || projMatr2->rows != 3 ||
        projMatr3->cols != 4 || projMatr3->rows != 3)
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of proj matrix must be 3x4" );
    }

    CvMat matrA;
    double matrA_dat[3*3*3];
    matrA = cvMat(3*3,3,CV_64F,matrA_dat);

    CvMat vectB;
    double vectB_dat[9];
    vectB = cvMat(9,1,CV_64F,vectB_dat);

    CvMat result;
    double result_dat[3];
    result = cvMat(3,1,CV_64F,result_dat);

    CvMat* projPoints[3];
    CvMat* projMatrs[3];

    projPoints[0] = projPoints1;
    projPoints[1] = projPoints2;
    projPoints[2] = projPoints3;

    projMatrs[0] = projMatr1;
    projMatrs[1] = projMatr2;
    projMatrs[2] = projMatr3;

    /* Solve system for each point */
    int i,j;
    for( i = 0; i < numPoints; i++ )/* For each point */
    {
        /* Fill matrix for current point */
        for( j = 0; j < 3; j++ )/* For each view */
        {
            double x,y;
            x = cvmGet(projPoints[j],0,i);
            y = cvmGet(projPoints[j],1,i);

            cvmSet(&vectB,j*3+0,0,x-cvmGet(projMatrs[j],0,3));
            cvmSet(&vectB,j*3+1,0,y-cvmGet(projMatrs[j],1,3));
            cvmSet(&vectB,j*3+2,0,1-cvmGet(projMatrs[j],2,3));

            for( int t = 0; t < 3; t++ )
            {
                for( int k = 0; k < 3; k++ )
                {
                    cvmSet(&matrA, j*3+t, k, cvmGet(projMatrs[j],t,k) );
                }
            }
        }


        /* Solve system for current point */
        cvSolve(&matrA,&vectB,&result,CV_SVD);

        cvmSet(points3D,0,i,result_dat[0]);/* X */
        cvmSet(points3D,1,i,result_dat[1]);/* Y */
        cvmSet(points3D,2,i,result_dat[2]);/* Z */
        cvmSet(points3D,3,i,1);/* W */

    }

    /* Points was reconstructed. Try to reproject points */
    {
        int i;
        CvMat point3D;
        double point3D_dat[4];
        point3D = cvMat(4,1,CV_64F,point3D_dat);

        CvMat point2D;
        double point2D_dat[3];
        point2D = cvMat(3,1,CV_64F,point2D_dat);

        for( i = 0; i < numPoints; i++ )
        {
            double W = cvmGet(points3D,3,i);

            point3D_dat[0] = cvmGet(points3D,0,i)/W;
            point3D_dat[1] = cvmGet(points3D,1,i)/W;
            point3D_dat[2] = cvmGet(points3D,2,i)/W;
            point3D_dat[3] = 1;

                /* Project this point for each camera */
                for( int currCamera = 0; currCamera < 3; currCamera++ )
                {
                    cvmMul(projMatrs[currCamera], &point3D, &point2D);
                    float x,y;
                    float xr,yr,wr;
                    x = (float)cvmGet(projPoints[currCamera],0,i);
                    y = (float)cvmGet(projPoints[currCamera],1,i);

                    wr = (float)point2D_dat[2];
                    xr = (float)(point2D_dat[0]/wr);
                    yr = (float)(point2D_dat[1]/wr);

                }
        }
    }

    __END__;
    return;
}
#endif

/*==========================================================================================*/

void icvComputeCameraExrinnsicByPosition(CvMat* camPos, CvMat* rotMatr, CvMat* transVect)
{
    /* We know position of camera. we must to compute rotate matrix and translate vector */

    CV_FUNCNAME( "icvComputeCameraExrinnsicByPosition" );
    __BEGIN__;

    /* Test input paramaters */
    if( camPos == 0 || rotMatr == 0 || transVect == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(camPos) || !CV_IS_MAT(rotMatr) || !CV_IS_MAT(transVect) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    if( camPos->cols != 1 || camPos->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of coordinates of camera position must be 3x1 vector" );
    }

    if( rotMatr->cols != 3 || rotMatr->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Rotate matrix must be 3x3" );
    }

    if( transVect->cols != 1 || transVect->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Translate vector must be 3x1" );
    }

    double x,y,z;
    x = cvmGet(camPos,0,0);
    y = cvmGet(camPos,1,0);
    z = cvmGet(camPos,2,0);

    /* Set translate vector. It same as camea position */
    cvmSet(transVect,0,0,x);
    cvmSet(transVect,1,0,y);
    cvmSet(transVect,2,0,z);

    /* Compute rotate matrix. Compute each unit transformed vector */

    /* normalize flat direction x,y */
    double vectorX[3];
    double vectorY[3];
    double vectorZ[3];

    vectorX[0] = -z;
    vectorX[1] =  0;
    vectorX[2] =  x;

    vectorY[0] =  x*y;
    vectorY[1] =  x*x+z*z;
    vectorY[2] =  z*y;

    vectorZ[0] = -x;
    vectorZ[1] = -y;
    vectorZ[2] = -z;

    /* normaize vectors */
    double norm;
    int i;

    /* Norm X */
    norm = 0;
    for( i = 0; i < 3; i++ )
        norm += vectorX[i]*vectorX[i];
    norm = sqrt(norm);
    for( i = 0; i < 3; i++ )
        vectorX[i] /= norm;

    /* Norm Y */
    norm = 0;
    for( i = 0; i < 3; i++ )
        norm += vectorY[i]*vectorY[i];
    norm = sqrt(norm);
    for( i = 0; i < 3; i++ )
        vectorY[i] /= norm;

    /* Norm Z */
    norm = 0;
    for( i = 0; i < 3; i++ )
        norm += vectorZ[i]*vectorZ[i];
    norm = sqrt(norm);
    for( i = 0; i < 3; i++ )
        vectorZ[i] /= norm;

    /* Set output results */

    for( i = 0; i < 3; i++ )
    {
        cvmSet(rotMatr,i,0,vectorX[i]);
        cvmSet(rotMatr,i,1,vectorY[i]);
        cvmSet(rotMatr,i,2,vectorZ[i]);
    }

    {/* Try to inverse rotate matrix */
        CvMat tmpInvRot;
        double tmpInvRot_dat[9];
        tmpInvRot = cvMat(3,3,CV_64F,tmpInvRot_dat);
        cvInvert(rotMatr,&tmpInvRot,CV_SVD);
        cvConvert(&tmpInvRot,rotMatr);



    }

    __END__;

    return;
}

/*==========================================================================================*/

void FindTransformForProjectMatrices(CvMat* projMatr1,CvMat* projMatr2,CvMat* rotMatr,CvMat* transVect)
{
    /* Computes homography for project matrix be "canonical" form */
    CV_FUNCNAME( "computeProjMatrHomography" );
    __BEGIN__;

    /* Test input paramaters */
    if( projMatr1 == 0 || projMatr2 == 0 || rotMatr == 0 || transVect == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(projMatr1) || !CV_IS_MAT(projMatr2) || !CV_IS_MAT(rotMatr) || !CV_IS_MAT(transVect) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    if( projMatr1->cols != 4 || projMatr1->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of project matrix 1 must be 3x4" );
    }

    if( projMatr2->cols != 4 || projMatr2->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of project matrix 2 must be 3x4" );
    }

    if( rotMatr->cols != 3 || rotMatr->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of rotation matrix must be 3x3" );
    }

    if( transVect->cols != 1 || transVect->rows != 3 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of translation vector must be 3x1" );
    }

    CvMat matrA;
    double matrA_dat[12*12];
    matrA = cvMat(12,12,CV_64F,matrA_dat);
    CvMat vectB;
    double vectB_dat[12];
    vectB = cvMat(12,1,CV_64F,vectB_dat);

    cvZero(&matrA);
    cvZero(&vectB);
    int i,j;
    for( i = 0; i < 12; i++ )
    {
        for( j = 0; j < 12; j++ )
        {
            cvmSet(&matrA,i,j,cvmGet(projMatr1,i/4,j%4));
        }
        /* Fill vector B */

        double val = cvmGet(projMatr2,i/4,i%4);
        if( (i+1)%4 == 0 )
        {
            val -= cvmGet(projMatr1,i/4,3);

        }
        cvmSet(&vectB,i,0,val);
    }

    /* Solve system */
    CvMat resVect;
    double resVect_dat[12];
    resVect = cvMat(12,1,CV_64F,resVect_dat);

    cvSolve(&matrA,&vectB,&resVect);

    /* Fill rotation matrix */
    for( i = 0; i < 12; i++ )
    {
        double val = cvmGet(&resVect,i,0);
        if( i < 9 )
            cvmSet(rotMatr,i%3,i/3,val);
        else
            cvmSet(transVect,i-9,0,val);
    }

    __END__;

    return;
}

/*==========================================================================================*/
#if 0
void icvComputeQknowPrincipalPoint(int numImages, CvMat **projMatrs,CvMat *matrQ, double cx,double cy)
{
    /* Computes matrix Q */
    /* focal x and y eqauls () */
    /* we know principal point for camera */
    /* focal may differ from image to image */
    /* image skew is 0 */

    if( numImages < 10 )
    {
        return;
        //Error. Number of images too few
    }

    /* Create  */


    return;
}
#endif

/*==========================================================================================*/

/*==========================================================================================*/
/*==========================================================================================*/
/*==========================================================================================*/
/*==========================================================================================*/
/* Part with metric reconstruction */

#if 1
void icvComputeQ(int numMatr, CvMat** projMatr, CvMat** cameraMatr, CvMat* matrQ)
{
    /* K*K' = P*Q*P' */
    /* try to solve Q by linear method */

    CvMat* matrA = 0;
    CvMat* vectB = 0;

    CV_FUNCNAME( "ComputeQ" );
    __BEGIN__;

    /* Define number of projection matrices */
    if( numMatr < 2 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of projection matrices must be at least 2" );
    }


    /* test matrices sizes */
    if( matrQ->cols != 4 || matrQ->rows != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of matrix Q must be 3x3" );
    }

    int currMatr;
    for( currMatr = 0; currMatr < numMatr; currMatr++ )
    {

        if( cameraMatr[currMatr]->cols != 3 || cameraMatr[currMatr]->rows != 3 )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of each camera matrix must be 3x3" );
        }

        if( projMatr[currMatr]->cols != 4 || projMatr[currMatr]->rows != 3 )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of each camera matrix must be 3x3" );
        }
    }

    CvMat matrw;
    double matrw_dat[9];
    matrw = cvMat(3,3,CV_64F,matrw_dat);

    CvMat matrKt;
    double matrKt_dat[9];
    matrKt = cvMat(3,3,CV_64F,matrKt_dat);


    /* Create matrix A and vector B */
    CV_CALL( matrA = cvCreateMat(9*numMatr,10,CV_64F) );
    CV_CALL( vectB = cvCreateMat(9*numMatr,1,CV_64F) );

    double dataQ[16];

    for( currMatr = 0; currMatr < numMatr; currMatr++ )
    {
        int ord10[10] = {0,1,2,3,5,6,7,10,11,15};
        /* Fill atrix A by data from matrices  */

        /* Compute matrix w for current camera matrix */
        cvTranspose(cameraMatr[currMatr],&matrKt);
        cvmMul(cameraMatr[currMatr],&matrKt,&matrw);

        /* Fill matrix A and vector B */

        int currWi,currWj;
        int currMatr;
        for( currMatr = 0; currMatr < numMatr; currMatr++ )
        {
            for( currWi = 0; currWi < 3; currWi++ )
            {
                for( currWj = 0; currWj < 3; currWj++ )
                {
                    int i,j;
                    for( i = 0; i < 4; i++ )
                    {
                        for( j = 0; j < 4; j++ )
                        {
                            /* get elements from current projection matrix */
                            dataQ[i*4+j] = cvmGet(projMatr[currMatr],currWi,j) *
                                           cvmGet(projMatr[currMatr],currWj,i);
                        }
                    }

                    /* we know 16 elements in dataQ move them to matrQ 10 */
                    dataQ[1]  += dataQ[4];
                    dataQ[2]  += dataQ[8];
                    dataQ[3]  += dataQ[12];
                    dataQ[6]  += dataQ[9];
                    dataQ[7]  += dataQ[13];
                    dataQ[11] += dataQ[14];
                    /* Now first 10 elements has coeffs */

                    /* copy to matrix A */
                    for( i = 0; i < 10; i++ )
                    {
                        cvmSet(matrA,currMatr*9 + currWi*3+currWj,i,dataQ[ord10[i]]);
                    }
                }
            }

            /* Fill vector B */
            for( int i = 0; i < 9; i++ )
            {
                cvmSet(vectB,currMatr*9+i,0,matrw_dat[i]);
            }
        }
    }

    /* Matrix A and vector B filled and we can solve system */

    /* Solve system */
    CvMat resQ;
    double resQ_dat[10];
    resQ = cvMat(10,1,CV_64F,resQ_dat);

    cvSolve(matrA,vectB,&resQ,CV_SVD);

    /* System was solved. We know matrix Q. But we must have condition det Q=0 */
    /* Just copy result matrix Q */
    {
        int curr = 0;
        int ord16[16] = {0,1,2,3,1,4,5,6,2,5,7,8,3,6,8,9};

        for( int i = 0; i < 4; i++ )
        {
            for( int j = 0; j < 4; j++ )
            {
                cvmSet(matrQ,i,j,resQ_dat[ord16[curr++]]);
            }
        }
    }


    __END__;

    /* Free allocated memory */
    cvReleaseMat(&matrA);
    cvReleaseMat(&vectB);

    return;
}
#endif
/*-----------------------------------------------------------------------------------------------------*/

void icvDecomposeQ(CvMat* /*matrQ*/,CvMat* /*matrH*/)
{
#if 0
    /* Use SVD to decompose matrix Q=H*I*H' */
    /* test input data */

    CvMat matrW;
    CvMat matrU;
//    CvMat matrV;
    double matrW_dat[16];
    double matrU_dat[16];
//    double matrV_dat[16];

    matrW = cvMat(4,4,CV_64F,matrW_dat);
    matrU = cvMat(4,4,CV_64F,matrU_dat);
//    matrV = cvMat(4,4,CV_64F,matrV_dat);

    cvSVD(matrQ,&matrW,&matrU,0);

    double eig[3];
    eig[0] = fsqrt(cvmGet(&matrW,0,0));
    eig[1] = fsqrt(cvmGet(&matrW,1,1));
    eig[2] = fsqrt(cvmGet(&matrW,2,2));

    CvMat matrIS;
    double matrIS_dat[16];
    matrIS =




/* det for matrix Q with q1-q10 */
/*
+ q1*q5*q8*q10
- q1*q5*q9*q9
- q1*q6*q6*q10
+ 2*q1*q6*q7*q9
- q1*q7*q7*q8
- q2*q2*q8*q10
+ q2*q2*q9*q9
+ 2*q2*q6*q3*q10
- 2*q2*q6*q4*q9
- 2*q2*q7*q3*q9
+ 2*q2*q7*q4*q8
- q5*q3*q3*q10
+ 2*q3*q5*q4*q9
+ q3*q3*q7*q7
- 2*q3*q7*q4*q6
- q5*q4*q4*q8
+ q4*q4*q6*q6
*/

//  (1-a)^4 = 1  -  4 * a  +  6 * a * a  -  4 * a * a * a  +  a * a * a * a;


#endif
}

