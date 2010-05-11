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

#include <float.h>
#include <limits.h>

/* Valery Mosyagin */

typedef void (*pointer_LMJac)( const CvMat* src, CvMat* dst );
typedef void (*pointer_LMFunc)( const CvMat* src, CvMat* dst );

void cvLevenbergMarquardtOptimization(pointer_LMJac JacobianFunction,
                                    pointer_LMFunc function,
                                    /*pointer_Err error_function,*/
                                    CvMat *X0,CvMat *observRes,CvMat *resultX,
                                    int maxIter,double epsilon);

void icvReconstructPointsFor3View( CvMat* projMatr1,CvMat* projMatr2,CvMat* projMatr3,
                                CvMat* projPoints1,CvMat* projPoints2,CvMat* projPoints3,
                                CvMat* points4D);


/* Jacobian computation for trifocal case */
void icvJacobianFunction_ProjTrifocal(const CvMat *vectX,CvMat *Jacobian)
{
    CV_FUNCNAME( "icvJacobianFunction_ProjTrifocal" );
    __BEGIN__;

    /* Test data for errors */
    if( vectX == 0 || Jacobian == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(vectX) || !CV_IS_MAT(Jacobian) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    int numPoints;
    numPoints = (vectX->rows - 36)/4;

    if( numPoints < 1 )//!!! Need to correct this minimal number of points
    {
        CV_ERROR( CV_StsUnmatchedSizes, "number of points must be more than 0" );
    }

    if( Jacobian->rows == numPoints*6 || Jacobian->cols != 36+numPoints*4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of Jacobian is not correct it must be 6*numPoints x (36+numPoints*4)" );
    }

    /* Computed Jacobian in a given point */
    /* This is for function with 3 projection matrices */
    /* vector X consists of projection matrices and points3D */
    /* each 3D points has X,Y,Z,W */
    /* each projection matrices has 3x4 coeffs */
    /* For N points 4D we have Jacobian 2N x (12*3+4N) */

    /* Will store derivates as  */
    /* Fill Jacobian matrix */
    int currProjPoint;
    int currMatr;
    
    cvZero(Jacobian);
    for( currMatr = 0; currMatr < 3; currMatr++ )
    {
        double p[12];
        for( int i=0;i<12;i++ )
        {
            p[i] = cvmGet(vectX,currMatr*12+i,0);
        }

        int currVal = 36;
        for( currProjPoint = 0; currProjPoint < numPoints; currProjPoint++ )
        {
            /* Compute */
            double X[4];
            X[0] = cvmGet(vectX,currVal++,0);
            X[1] = cvmGet(vectX,currVal++,0);
            X[2] = cvmGet(vectX,currVal++,0);
            X[3] = cvmGet(vectX,currVal++,0);

            double piX[3];
            piX[0] = X[0]*p[0] + X[1]*p[1] + X[2]*p[2]  + X[3]*p[3];
            piX[1] = X[0]*p[4] + X[1]*p[5] + X[2]*p[6]  + X[3]*p[7];
            piX[2] = X[0]*p[8] + X[1]*p[9] + X[2]*p[10] + X[3]*p[11];

            int i,j;
            /* fill derivate by point */

            double tmp3 = 1/(piX[2]*piX[2]);

            double tmp1 = -piX[0]*tmp3;
            double tmp2 = -piX[1]*tmp3;
            for( j = 0; j < 2; j++ )//for x and y
            {
                for( i = 0; i < 4; i++ )// for X,Y,Z,W
                {
                    cvmSet( Jacobian, 
                            currMatr*numPoints*2+currProjPoint*2+j, 36+currProjPoint*4+i,
                            (p[j*4+i]*piX[2]-p[8+i]*piX[j]) * tmp3  );
                }
            }
                /* fill derivate by projection matrix */
            for( i = 0; i < 4; i++ )
            {
                /* derivate for x */
                cvmSet(Jacobian,currMatr*numPoints*2+currProjPoint*2,currMatr*12+i,X[i]/piX[2]);//x' p1i
                cvmSet(Jacobian,currMatr*numPoints*2+currProjPoint*2,currMatr*12+8+i,X[i]*tmp1);//x' p3i

                /* derivate for y */
                cvmSet(Jacobian,currMatr*numPoints*2+currProjPoint*2+1,currMatr*12+4+i,X[i]/piX[2]);//y' p2i
                cvmSet(Jacobian,currMatr*numPoints*2+currProjPoint*2+1,currMatr*12+8+i,X[i]*tmp2);//y' p3i
            }

        }
    }

    __END__;
    return;
}

void icvFunc_ProjTrifocal(const CvMat *vectX, CvMat *resFunc)
{
    /* Computes function in a given point */
    /* Computers project points using 3 projection matrices and points 3D */

    /* vector X consists of projection matrices and points3D */
    /* each projection matrices has 3x4 coeffs */
    /* each 3D points has X,Y,Z,W(?) */

    /* result of function is projection of N 3D points using 3 projection matrices */
    /* projected points store as (projection by matrix P1),(projection by matrix P2),(projection by matrix P3) */
    /* each projection is x1,y1,x2,y2,x3,y3,x4,y4 */

    /* Compute projection of points */

    /* Fill projection matrices */

    CV_FUNCNAME( "icvFunc_ProjTrifocal" );
    __BEGIN__;

    /* Test data for errors */
    if( vectX == 0 || resFunc == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(vectX) || !CV_IS_MAT(resFunc) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters must be a matrices" );
    }

    int numPoints;
    numPoints = (vectX->rows - 36)/4;

    if( numPoints < 1 )//!!! Need to correct this minimal number of points
    {
        CV_ERROR( CV_StsUnmatchedSizes, "number of points must be more than 0" );
    }

    if( resFunc->rows == 2*numPoints*3 || resFunc->cols != 1 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of resFunc is not correct it must be 2*numPoints*3 x 1");
    }


    CvMat projMatrs[3];
    double projMatrs_dat[36];
    projMatrs[0] = cvMat(3,4,CV_64F,projMatrs_dat);
    projMatrs[1] = cvMat(3,4,CV_64F,projMatrs_dat+12);
    projMatrs[2] = cvMat(3,4,CV_64F,projMatrs_dat+24);

    CvMat point3D;
    double point3D_dat[3];
    point3D = cvMat(3,1,CV_64F,point3D_dat);

    int currMatr;
    int currV;
    int i,j;

    currV=0;
    for( currMatr = 0; currMatr < 3; currMatr++ )
    {
        for( i = 0; i < 3; i++ )
        {
            for( j = 0;j < 4; j++ )
            {
                double val = cvmGet(vectX,currV,0);
                cvmSet(&projMatrs[currMatr],i,j,val);
                currV++;
            }
        }
    }

    /* Project points */
    int currPoint;
    CvMat point4D;
    double point4D_dat[4];
    point4D = cvMat(4,1,CV_64F,point4D_dat);
    for( currPoint = 0; currPoint < numPoints; currPoint++ )
    {
        /* get curr point */
        point4D_dat[0] = cvmGet(vectX,currV++,0);
        point4D_dat[1] = cvmGet(vectX,currV++,0);
        point4D_dat[2] = cvmGet(vectX,currV++,0);
        point4D_dat[3] = cvmGet(vectX,currV++,0);

        for( currMatr = 0; currMatr < 3; currMatr++ )
        {
            /* Compute projection for current point */
            cvmMul(&projMatrs[currMatr],&point4D,&point3D);
            double z = point3D_dat[2];
            cvmSet(resFunc,currMatr*numPoints*2 + currPoint*2,  0,point3D_dat[0]/z);
            cvmSet(resFunc,currMatr*numPoints*2 + currPoint*2+1,0,point3D_dat[1]/z);
        }
    }

    __END__;
    return;
}


/*----------------------------------------------------------------------------------------*/

void icvOptimizeProjectionTrifocal(CvMat **projMatrs,CvMat **projPoints,
                                CvMat **resultProjMatrs, CvMat *resultPoints4D)
{

    CvMat *optimX    = 0;
    CvMat *points4D  = 0;
    CvMat *vectorX0  = 0;
    CvMat *observRes = 0;
    //CvMat *error     = 0;

    CV_FUNCNAME( "icvOptimizeProjectionTrifocal" );
    __BEGIN__;

    /* Test data for errors */
    if( projMatrs == 0 || projPoints == 0 || resultProjMatrs == 0 || resultPoints4D == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(resultPoints4D) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "resultPoints4D must be a matrix" );
    }

    int numPoints;
    numPoints = resultPoints4D->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number points of resultPoints4D must be more than 0" );
    }

    if( resultPoints4D->rows != 4 )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of coordinates of points4D must be 4" );
    }

    int i;
    for( i = 0; i < 3; i++ )
    {
        if( projMatrs[i] == 0 )
        {
            CV_ERROR( CV_StsNullPtr, "Some of projMatrs is a NULL pointer" );
        }

        if( projPoints[i] == 0 )
        {
            CV_ERROR( CV_StsNullPtr, "Some of projPoints is a NULL pointer" );
        }
    
        if( resultProjMatrs[i] == 0 )
        {
            CV_ERROR( CV_StsNullPtr, "Some of resultProjMatrs is a NULL pointer" );
        }

        /* ----------- test for matrix ------------- */
        if( !CV_IS_MAT(projMatrs[i]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "Each of projMatrs must be a matrix" );
        }

        if( !CV_IS_MAT(projPoints[i]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "Each of projPoints must be a matrix" );
        }

        if( !CV_IS_MAT(resultProjMatrs[i]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "Each of resultProjMatrs must be a matrix" );
        }

        /* ------------- Test sizes --------------- */
        if( projMatrs[i]->rows != 3 || projMatrs[i]->cols != 4 )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of projMatr must be 3x4" );
        }

        if( projPoints[i]->rows != 2 || projPoints[i]->cols != numPoints )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of resultProjMatrs must be 3x4" );
        }

        if( resultProjMatrs[i]->rows != 3 || resultProjMatrs[i]->cols != 4 )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of resultProjMatrs must be 3x4" );
        }
    }


    /* Allocate memory for points 4D */
    CV_CALL( points4D  = cvCreateMat(4,numPoints,CV_64F) );
    CV_CALL( vectorX0  = cvCreateMat(36 + numPoints*4,1,CV_64F) );
    CV_CALL( observRes = cvCreateMat(2*numPoints*3,1,CV_64F) );
    CV_CALL( optimX    = cvCreateMat(36+numPoints*4,1,CV_64F) );
    //CV_CALL( error     = cvCreateMat(numPoints*2*3,1,CV_64F) );


    /* Reconstruct points 4D using projected points and projection matrices */
    icvReconstructPointsFor3View( projMatrs[0],projMatrs[1],projMatrs[2],
                                  projPoints[0],projPoints[1],projPoints[2],
                                  points4D);



    /* Fill observed points on images */
    /* result of function is projection of N 3D points using 3 projection matrices */
    /* projected points store as (projection by matrix P1),(projection by matrix P2),(projection by matrix P3) */
    /* each projection is x1,y1,x2,y2,x3,y3,x4,y4 */
    int currMatr;
    for( currMatr = 0; currMatr < 3; currMatr++ )
    {
        for( i = 0; i < numPoints; i++ )
        {
            cvmSet(observRes,currMatr*numPoints*2+i*2  ,0,cvmGet(projPoints[currMatr],0,i) );/* x */
            cvmSet(observRes,currMatr*numPoints*2+i*2+1,0,cvmGet(projPoints[currMatr],1,i) );/* y */
        }
    }

    /* Fill with projection matrices */
    for( currMatr = 0; currMatr < 3; currMatr++ )
    {
        int i;
        for( i = 0; i < 12; i++ )
        {
            cvmSet(vectorX0,currMatr*12+i,0,cvmGet(projMatrs[currMatr],i/4,i%4));
        }
    }

    /* Fill with 4D points */

    int currPoint;
    for( currPoint = 0; currPoint < numPoints; currPoint++ )
    {
        cvmSet(vectorX0,36 + currPoint*4 + 0,0,cvmGet(points4D,0,currPoint));
        cvmSet(vectorX0,36 + currPoint*4 + 1,0,cvmGet(points4D,1,currPoint));
        cvmSet(vectorX0,36 + currPoint*4 + 2,0,cvmGet(points4D,2,currPoint));
        cvmSet(vectorX0,36 + currPoint*4 + 3,0,cvmGet(points4D,3,currPoint));
    }

    
    /* Allocate memory for result */
    cvLevenbergMarquardtOptimization( icvJacobianFunction_ProjTrifocal, icvFunc_ProjTrifocal,
                                      vectorX0,observRes,optimX,100,1e-6);

    /* Copy results */
    for( currMatr = 0; currMatr < 3; currMatr++ )
    {
        /* Copy projection matrices */
        for(int i=0;i<12;i++)
        {
            cvmSet(resultProjMatrs[currMatr],i/4,i%4,cvmGet(optimX,currMatr*12+i,0));
        }
    }

    /* Copy 4D points */
    for( currPoint = 0; currPoint < numPoints; currPoint++ )
    {
        cvmSet(resultPoints4D,0,currPoint,cvmGet(optimX,36 + currPoint*4,0));
        cvmSet(resultPoints4D,1,currPoint,cvmGet(optimX,36 + currPoint*4+1,0));
        cvmSet(resultPoints4D,2,currPoint,cvmGet(optimX,36 + currPoint*4+2,0));
        cvmSet(resultPoints4D,3,currPoint,cvmGet(optimX,36 + currPoint*4+3,0));
    }

    __END__;

    /* Free allocated memory */
    cvReleaseMat(&optimX);
    cvReleaseMat(&points4D);
    cvReleaseMat(&vectorX0);
    cvReleaseMat(&observRes);

    return;


}

/*------------------------------------------------------------------------------*/
/* Create good points using status information */
void icvCreateGoodPoints(CvMat *points,CvMat **goodPoints, CvMat *status)
{
    *goodPoints = 0;

    CV_FUNCNAME( "icvCreateGoodPoints" );
    __BEGIN__;

    int numPoints;
    numPoints = points->cols;

    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must be more than 0" );
    }

    int numCoord;
    numCoord = points->rows;
    if( numCoord < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points coordinates must be more than 0" );
    }

    /* Define number of good points */
    int goodNum;
    int i,j;

    goodNum = 0;
    for( i = 0; i < numPoints; i++)
    {
        if( cvmGet(status,0,i) > 0 )
            goodNum++;
    }

    /* Allocate memory for good points */
    CV_CALL( *goodPoints = cvCreateMat(numCoord,goodNum,CV_64F) );

    for( i = 0; i < numCoord; i++ )
    {
        int currPoint = 0;
        for( j = 0; j < numPoints; j++)
        {
            if( cvmGet(status,0,j) > 0 )
            {
                cvmSet(*goodPoints,i,currPoint,cvmGet(points,i,j));
                currPoint++;
            }
        }
    }
    __END__;
    return;
}

