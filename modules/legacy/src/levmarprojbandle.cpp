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

#include <stdio.h>

void icvReconstructPoints4DStatus(CvMat** projPoints, CvMat **projMatrs, CvMat** presPoints, CvMat *points4D,int numImages,CvMat **projError=0);

/* Valery Mosyagin */

/* If you want to save internal debug info to files uncomment next lines and set paths to files if need */
/* Note these file may be very large */
/*
#define TRACK_BUNDLE
#define TRACK_BUNDLE_FILE            "d:\\test\\bundle.txt"
#define TRACK_BUNDLE_FILE_JAC        "d:\\test\\bundle.txt"
#define TRACK_BUNDLE_FILE_JACERRPROJ "d:\\test\\JacErrProj.txt"
#define TRACK_BUNDLE_FILE_JACERRPNT  "d:\\test\\JacErrPoint.txt"
#define TRACK_BUNDLE_FILE_MATRW      "d:\\test\\matrWt.txt"
#define TRACK_BUNDLE_FILE_DELTAP     "d:\\test\\deltaP.txt"
*/
#define TRACK_BUNDLE_FILE            "d:\\test\\bundle.txt"

void cvOptimizeLevenbergMarquardtBundle( CvMat** projMatrs, CvMat** observProjPoints,
                                       CvMat** pointsPres, int numImages,
                                       CvMat** resultProjMatrs, CvMat* resultPoints4D,int maxIter,double epsilon );


/* ============== Bundle adjustment optimization ================= */
static void icvComputeDerivateProj(CvMat *points4D,CvMat *projMatr, CvMat *status, CvMat *derivProj)
{
    /* Compute derivate for given projection matrix points and status of points */

    CV_FUNCNAME( "icvComputeDerivateProj" );
    __BEGIN__;


    /* ----- Test input params for errors ----- */
    if( points4D == 0 || projMatr == 0 || status == 0 || derivProj == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points4D) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "points4D must be a matrix 4xN" );
    }

    /* Compute number of points */
    int numPoints;
    numPoints = points4D->cols;

    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points4D must be more than zero" );
    }

    if( points4D->rows != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of coordinates of points4D must be 4" );
    }

    if( !CV_IS_MAT(projMatr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "projMatr must be a matrix 3x4" );
    }

    if( projMatr->rows != 3 || projMatr->cols != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "Size of projection matrix (projMatr) must be 3x4" );
    }

    if( !CV_IS_MAT(status) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Status must be a matrix 1xN" );
    }

    if( status->rows != 1 || status->cols != numPoints )
    {
        CV_ERROR( CV_StsOutOfRange, "Size of status of points must be 1xN" );
    }

    if( !CV_IS_MAT(derivProj) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "derivProj must be a matrix VisN x 12" );
    }

    if( derivProj->cols != 12 )
    {
        CV_ERROR( CV_StsOutOfRange, "derivProj must be a matrix VisN x 12" );
    }
    /* ----- End test ----- */

    int i;

    /* Allocate memory for derivates */

    double p[12];
    /* Copy projection matrix */
    for( i = 0; i < 12; i++ )
    {
        p[i] = cvmGet(projMatr,i/4,i%4);
    }

    /* Fill deriv matrix */
    int currVisPoint;
    int currPoint;

    currVisPoint = 0;
    for( currPoint = 0; currPoint < numPoints; currPoint++ )
    {
        if( cvmGet(status,0,currPoint) > 0 )
        {
            double X[4];
            X[0] = cvmGet(points4D,0,currVisPoint);
            X[1] = cvmGet(points4D,1,currVisPoint);
            X[2] = cvmGet(points4D,2,currVisPoint);
            X[3] = cvmGet(points4D,3,currVisPoint);

            /* Compute derivate for this point */

            double piX[3];
            piX[0] = X[0]*p[0] + X[1]*p[1] + X[2]*p[2]  + X[3]*p[3];
            piX[1] = X[0]*p[4] + X[1]*p[5] + X[2]*p[6]  + X[3]*p[7];
            piX[2] = X[0]*p[8] + X[1]*p[9] + X[2]*p[10] + X[3]*p[11];

            int i;
            /* fill derivate by point */

            double tmp3 = 1/(piX[2]*piX[2]);

            double tmp1 = -piX[0]*tmp3;
            double tmp2 = -piX[1]*tmp3;

            /* fill derivate by projection matrix */
            for( i = 0; i < 4; i++ )
            {
                /* derivate for x */
                cvmSet(derivProj,currVisPoint*2,i,X[i]/piX[2]);//x' p1i
                cvmSet(derivProj,currVisPoint*2,4+i,0);//x' p1i
                cvmSet(derivProj,currVisPoint*2,8+i,X[i]*tmp1);//x' p3i

                /* derivate for y */
                cvmSet(derivProj,currVisPoint*2+1,i,0);//y' p2i
                cvmSet(derivProj,currVisPoint*2+1,4+i,X[i]/piX[2]);//y' p2i
                cvmSet(derivProj,currVisPoint*2+1,8+i,X[i]*tmp2);//y' p3i
            }

            currVisPoint++;
        }
    }

    if( derivProj->rows != currVisPoint * 2 )
    {
        CV_ERROR( CV_StsOutOfRange, "derivProj must be a matrix 2VisN x 12" );
    }


    __END__;
    return;
}
/*======================================================================================*/

static void icvComputeDerivateProjAll(CvMat *points4D, CvMat **projMatrs, CvMat **pointPres, int numImages,CvMat **projDerives)
{
    CV_FUNCNAME( "icvComputeDerivateProjAll" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }
    if( projMatrs == 0 || pointPres == 0 || projDerives == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    /* ----- End test ----- */

    int currImage;
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        icvComputeDerivateProj(points4D,projMatrs[currImage], pointPres[currImage], projDerives[currImage]);
    }

    __END__;
    return;
}
/*======================================================================================*/

static void icvComputeDerivatePoints(CvMat *points4D,CvMat *projMatr, CvMat *presPoints, CvMat *derivPoint)
{

    CV_FUNCNAME( "icvComputeDerivatePoints" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( points4D == 0 || projMatr == 0 || presPoints == 0 || derivPoint == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(points4D) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "points4D must be a matrix N x 4" );
    }

    int numPoints;
    numPoints = presPoints->cols;

    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must be more than zero" );
    }

    if( points4D->rows != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "points4D must be a matrix N x 4" );
    }

    if( !CV_IS_MAT(projMatr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "projMatr must be a matrix 3x4" );
    }

    if( projMatr->rows != 3 || projMatr->cols != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "Size of projection matrix (projMatr) must be 3x4" );
    }

    if( !CV_IS_MAT(presPoints) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Status must be a matrix 1xN" );
    }

    if( presPoints->rows != 1 || presPoints->cols != numPoints )
    {
        CV_ERROR( CV_StsOutOfRange, "Size of presPoints status must be 1xN" );
    }

    if( !CV_IS_MAT(derivPoint) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "derivPoint must be a matrix 2 x 4VisNum" );
    }
    /* ----- End test ----- */

    /* Compute derivates by points */

    double p[12];
    int i;
    for( i = 0; i < 12; i++ )
    {
        p[i] = cvmGet(projMatr,i/4,i%4);
    }

    int currVisPoint;
    int currProjPoint;

    currVisPoint = 0;
    for( currProjPoint = 0; currProjPoint < numPoints; currProjPoint++ )
    {
        if( cvmGet(presPoints,0,currProjPoint) > 0 )
        {
            double X[4];
            X[0] = cvmGet(points4D,0,currProjPoint);
            X[1] = cvmGet(points4D,1,currProjPoint);
            X[2] = cvmGet(points4D,2,currProjPoint);
            X[3] = cvmGet(points4D,3,currProjPoint);

            double piX[3];
            piX[0] = X[0]*p[0] + X[1]*p[1] + X[2]*p[2]  + X[3]*p[3];
            piX[1] = X[0]*p[4] + X[1]*p[5] + X[2]*p[6]  + X[3]*p[7];
            piX[2] = X[0]*p[8] + X[1]*p[9] + X[2]*p[10] + X[3]*p[11];

            int i,j;

            double tmp3 = 1/(piX[2]*piX[2]);

            for( j = 0; j < 2; j++ )//for x and y
            {
                for( i = 0; i < 4; i++ )// for X,Y,Z,W
                {
                    cvmSet( derivPoint,
                            j, currVisPoint*4+i,
                            (p[j*4+i]*piX[2]-p[8+i]*piX[j]) * tmp3  );
                }
            }
            currVisPoint++;
        }
    }

    if( derivPoint->rows != 2 || derivPoint->cols != currVisPoint*4 )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "derivPoint must be a matrix 2 x 4VisNum" );
    }

    __END__;
    return;
}

/*======================================================================================*/
static void icvComputeDerivatePointsAll(CvMat *points4D, CvMat **projMatrs, CvMat **pointPres, int numImages,CvMat **pointDerives)
{
    CV_FUNCNAME( "icvComputeDerivatePointsAll" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }
    if( projMatrs == 0 || pointPres == 0 || pointDerives == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    /* ----- End test ----- */

    int currImage;
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        icvComputeDerivatePoints(points4D, projMatrs[currImage], pointPres[currImage], pointDerives[currImage]);
    }

    __END__;
    return;
}
/*======================================================================================*/
static void icvComputeMatrixVAll(int numImages,CvMat **pointDeriv,CvMat **presPoints, CvMat **matrV)
{
    int *shifts = 0;

    CV_FUNCNAME( "icvComputeMatrixVAll" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }
    if( pointDeriv == 0 || presPoints == 0 || matrV == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    /*  !!! not tested all parameters */
    /* ----- End test ----- */

    /* Compute all matrices U */
    int currImage;
    int currPoint;
    int numPoints;
    numPoints = presPoints[0]->cols;
    CV_CALL(shifts = (int*)cvAlloc(sizeof(int)*numImages));
    memset(shifts,0,sizeof(int)*numImages);

    for( currPoint = 0; currPoint < numPoints; currPoint++ )//For each point (matrix V)
    {
        int i,j;

        for( i = 0; i < 4; i++ )
        {
            for( j = 0; j < 4; j++ )
            {
                double sum = 0;
                for( currImage = 0; currImage < numImages; currImage++ )
                {
                    if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
                    {
                        sum += cvmGet(pointDeriv[currImage],0,shifts[currImage]*4+i) *
                               cvmGet(pointDeriv[currImage],0,shifts[currImage]*4+j);

                        sum += cvmGet(pointDeriv[currImage],1,shifts[currImage]*4+i) *
                               cvmGet(pointDeriv[currImage],1,shifts[currImage]*4+j);
                    }
                }

                cvmSet(matrV[currPoint],i,j,sum);
            }
        }


        /* shift position of visible points */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
            {
                shifts[currImage]++;
            }
        }
    }

    __END__;
    cvFree( &shifts);

    return;
}
/*======================================================================================*/
static void icvComputeMatrixUAll(int numImages,CvMat **projDeriv,CvMat** matrU)
{
    CV_FUNCNAME( "icvComputeMatrixVAll" );
    __BEGIN__;
    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }
    if( projDeriv == 0 || matrU == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    /* !!! Not tested all input parameters */
    /* ----- End test ----- */

    /* Compute matrices V */
    int currImage;
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        cvMulTransposed(projDeriv[currImage],matrU[currImage],1);
    }

    __END__;
    return;
}
/*======================================================================================*/
static void icvComputeMatrixW(int numImages, CvMat **projDeriv, CvMat **pointDeriv, CvMat **presPoints, CvMat *matrW)
{
    CV_FUNCNAME( "icvComputeMatrixW" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }
    if( projDeriv == 0 || pointDeriv == 0 || presPoints == 0 || matrW == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    int numPoints;
    numPoints = presPoints[0]->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must more than zero" );
    }
    if( !CV_IS_MAT(matrW) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "matrW must be a matrix 12NumIm x 4NumPnt" );
    }
    if( matrW->rows != numImages*12 || matrW->cols != numPoints*4 )
    {
        CV_ERROR( CV_StsOutOfRange, "matrW must be a matrix 12NumIm x 4NumPnt" );
    }
    /* !!! Not tested all input parameters */
    /* ----- End test ----- */

    /* Compute number of points */
    /* Compute matrix W using derivate proj and points */

    int currImage;

    for( currImage = 0; currImage < numImages; currImage++ )
    {
        for( int currLine = 0; currLine < 12; currLine++ )
        {
            int currVis = 0;
            for( int currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
                {

                    for( int currCol = 0; currCol < 4; currCol++ )
                    {
                        double sum;
                        sum = cvmGet(projDeriv[currImage],currVis*2+0,currLine) *
                              cvmGet(pointDeriv[currImage],0,currVis*4+currCol);

                        sum += cvmGet(projDeriv[currImage],currVis*2+1,currLine) *
                              cvmGet(pointDeriv[currImage],1,currVis*4+currCol);

                        cvmSet(matrW,currImage*12+currLine,currPoint*4+currCol,sum);
                    }
                    currVis++;
                }
                else
                {/* set all sub elements to zero */
                    for( int currCol = 0; currCol < 4; currCol++ )
                    {
                        cvmSet(matrW,currImage*12+currLine,currPoint*4+currCol,0);
                    }
                }
            }
        }
    }

#ifdef TRACK_BUNDLE
    {
        FILE *file;
        file = fopen( TRACK_BUNDLE_FILE_MATRW ,"w");
        int currPoint,currImage;
        for( currPoint = 0; currPoint < numPoints; currPoint++ )
        {
            fprintf(file,"\nPoint=%d\n",currPoint);
            int currRow;
            for( currRow = 0; currRow < 4; currRow++  )
            {
                for( currImage = 0; currImage< numImages; currImage++ )
                {
                    int i;
                    for( i = 0; i < 12; i++ )
                    {
                        double val = cvmGet(matrW, currImage * 12 + i, currPoint * 4 + currRow);
                        fprintf(file,"%lf ",val);
                    }
                }
                fprintf(file,"\n");
            }
        }
        fclose(file);
    }
#endif

    __END__;
    return;
}

/*======================================================================================*/
/* Compute jacobian mult projection matrices error */
static void icvComputeJacErrorProj(int numImages,CvMat **projDeriv,CvMat **projErrors,CvMat *jacProjErr )
{
    CV_FUNCNAME( "icvComputeJacErrorProj" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }
    if( projDeriv == 0 || projErrors == 0 || jacProjErr == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    if( !CV_IS_MAT(jacProjErr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "jacProjErr must be a matrix 12NumIm x 1" );
    }
    if( jacProjErr->rows != numImages*12 || jacProjErr->cols != 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "jacProjErr must be a matrix 12NumIm x 1" );
    }
    /* !!! Not tested all input parameters */
    /* ----- End test ----- */

    int currImage;
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        for( int currCol = 0; currCol < 12; currCol++ )
        {
            int num = projDeriv[currImage]->rows;
            double sum = 0;
            for( int i = 0; i < num; i++ )
            {
                sum += cvmGet(projDeriv[currImage],i,currCol) *
                       cvmGet(projErrors[currImage],i%2,i/2);
            }
            cvmSet(jacProjErr,currImage*12+currCol,0,sum);
        }
    }

#ifdef TRACK_BUNDLE
    {
        FILE *file;
        file = fopen( TRACK_BUNDLE_FILE_JACERRPROJ ,"w");
        int currImage;
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            fprintf(file,"\nImage=%d\n",currImage);
            int currRow;
            for( currRow = 0; currRow < 12; currRow++  )
            {
                double val = cvmGet(jacProjErr, currImage * 12 + currRow, 0);
                fprintf(file,"%lf\n",val);
            }
            fprintf(file,"\n");
        }
        fclose(file);
    }
#endif


    __END__;
    return;
}

/*======================================================================================*/
/* Compute jacobian mult points error */
static void icvComputeJacErrorPoint(int numImages,CvMat **pointDeriv,CvMat **projErrors, CvMat **presPoints,CvMat *jacPointErr )
{
    int *shifts = 0;

    CV_FUNCNAME( "icvComputeJacErrorPoint" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must more than zero" );
    }

    if( pointDeriv == 0 || projErrors == 0 || presPoints == 0 || jacPointErr == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    int numPoints;
    numPoints = presPoints[0]->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must more than zero" );
    }

    if( !CV_IS_MAT(jacPointErr) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "jacPointErr must be a matrix 4NumPnt x 1" );
    }

    if( jacPointErr->rows != numPoints*4 || jacPointErr->cols != 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "jacPointErr must be a matrix 4NumPnt x 1" );
    }
    /* !!! Not tested all input parameters */
    /* ----- End test ----- */

    int currImage;
    int currPoint;
    CV_CALL(shifts = (int*)cvAlloc(sizeof(int)*numImages));
    memset(shifts,0,sizeof(int)*numImages);
    for( currPoint = 0; currPoint < numPoints; currPoint++ )
    {
        for( int currCoord = 0; currCoord < 4; currCoord++ )
        {
            double sum = 0;
            {
                int currVis = 0;
                for( currImage = 0; currImage < numImages; currImage++ )
                {
                    if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
                    {
                        sum += cvmGet(pointDeriv[currImage],0,shifts[currImage]*4+currCoord) *
                               cvmGet(projErrors[currImage],0,shifts[currImage]);//currVis);

                        sum += cvmGet(pointDeriv[currImage],1,shifts[currImage]*4+currCoord) *
                               cvmGet(projErrors[currImage],1,shifts[currImage]);//currVis);

                        currVis++;
                    }
                }
            }

            cvmSet(jacPointErr,currPoint*4+currCoord,0,sum);
        }

        /* Increase shifts */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
            {
                shifts[currImage]++;
            }
        }
    }


#ifdef TRACK_BUNDLE
    {
        FILE *file;
        file = fopen(TRACK_BUNDLE_FILE_JACERRPNT,"w");
        int currPoint;
        for( currPoint = 0; currPoint < numPoints; currPoint++ )
        {
            fprintf(file,"\nPoint=%d\n",currPoint);
            int currRow;
            for( currRow = 0; currRow < 4; currRow++  )
            {
                double val = cvmGet(jacPointErr, currPoint * 4 + currRow, 0);
                fprintf(file,"%lf\n",val);
            }
            fprintf(file,"\n");
        }
        fclose(file);
    }
#endif



    __END__;
    cvFree( &shifts);

}
/*======================================================================================*/


/* Reconstruct 4D points using status */
void icvReconstructPoints4DStatus(CvMat** projPoints, CvMat **projMatrs, CvMat** presPoints,
                                  CvMat *points4D,int numImages,CvMat **projError)
{

    double* matrA_dat = 0;
    double* matrW_dat = 0;

    CV_FUNCNAME( "icvReconstructPoints4DStatus" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 2 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must be more than one" );
    }

    if( projPoints == 0 || projMatrs == 0 || presPoints == 0 || points4D == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    int numPoints;
    numPoints = points4D->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points4D must be more than zero" );
    }

    if( points4D->rows != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "Points must have 4 cordinates" );
    }

    /* !!! Not tested all input parameters */
    /* ----- End test ----- */

    int currImage;
    int currPoint;

    /* Allocate maximum data */


    CvMat matrV;
    double matrV_dat[4*4];
    matrV = cvMat(4,4,CV_64F,matrV_dat);

    CV_CALL(matrA_dat = (double*)cvAlloc(3*numImages * 4 * sizeof(double)));
    CV_CALL(matrW_dat = (double*)cvAlloc(3*numImages * 4 * sizeof(double)));

    /* reconstruct each point */
    for( currPoint = 0; currPoint < numPoints; currPoint++ )
    {
        /* Reconstruct current point */
        /* Define number of visible projections */
        int numVisProj = 0;
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
            {
                numVisProj++;
            }
        }

        if( numVisProj < 2 )
        {
            /* This point can't be reconstructed */
            continue;
        }

        /* Allocate memory and create matrices */
        CvMat matrA;
        matrA = cvMat(3*numVisProj,4,CV_64F,matrA_dat);

        CvMat matrW;
        matrW = cvMat(3*numVisProj,4,CV_64F,matrW_dat);

        int currVisProj = 0;
        for( currImage = 0; currImage < numImages; currImage++ )/* For each view */
        {
            if( cvmGet(presPoints[currImage],0,currPoint) > 0 )
            {
                double x,y;
                x = cvmGet(projPoints[currImage],0,currPoint);
                y = cvmGet(projPoints[currImage],1,currPoint);
                for( int k = 0; k < 4; k++ )
                {
                    matrA_dat[currVisProj*12   + k] =
                           x * cvmGet(projMatrs[currImage],2,k) -     cvmGet(projMatrs[currImage],0,k);

                    matrA_dat[currVisProj*12+4 + k] =
                           y * cvmGet(projMatrs[currImage],2,k) -     cvmGet(projMatrs[currImage],1,k);

                    matrA_dat[currVisProj*12+8 + k] =
                           x * cvmGet(projMatrs[currImage],1,k) - y * cvmGet(projMatrs[currImage],0,k);
                }
                currVisProj++;
            }
        }

        /* Solve system for current point */
        {
            cvSVD(&matrA,&matrW,0,&matrV,CV_SVD_V_T);

            /* Copy computed point */
            cvmSet(points4D,0,currPoint,cvmGet(&matrV,3,0));//X
            cvmSet(points4D,1,currPoint,cvmGet(&matrV,3,1));//Y
            cvmSet(points4D,2,currPoint,cvmGet(&matrV,3,2));//Z
            cvmSet(points4D,3,currPoint,cvmGet(&matrV,3,3));//W
        }

    }

    {/* Compute projection error */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            CvMat point4D;
            CvMat point3D;
            double point3D_dat[3];
            point3D = cvMat(3,1,CV_64F,point3D_dat);

            int currPoint;
            int numVis = 0;
            double totalError = 0;
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                if( cvmGet(presPoints[currImage],0,currPoint) > 0)
                {
                    double  dx,dy;
                    cvGetCol(points4D,&point4D,currPoint);
                    cvmMul(projMatrs[currImage],&point4D,&point3D);
                    double w = point3D_dat[2];
                    double x = point3D_dat[0] / w;
                    double y = point3D_dat[1] / w;

                    dx = cvmGet(projPoints[currImage],0,currPoint) - x;
                    dy = cvmGet(projPoints[currImage],1,currPoint) - y;
                    if( projError )
                    {
                        cvmSet(projError[currImage],0,currPoint,dx);
                        cvmSet(projError[currImage],1,currPoint,dy);
                    }
                    totalError += sqrt(dx*dx+dy*dy);
                    numVis++;
                }
            }

            //double meanError = totalError / (double)numVis;

        }

    }

    __END__;

    cvFree( &matrA_dat);
    cvFree( &matrW_dat);

    return;
}

/*======================================================================================*/

static void icvProjPointsStatusFunc( int numImages, CvMat *points4D, CvMat **projMatrs, CvMat **pointsPres, CvMat **projPoints)
{
    CV_FUNCNAME( "icvProjPointsStatusFunc" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must be more than zero" );
    }

    if( points4D == 0 || projMatrs == 0 || pointsPres == 0 || projPoints == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    int numPoints;
    numPoints = points4D->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points4D must be more than zero" );
    }

    if( points4D->rows != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "Points must have 4 cordinates" );
    }

    /* !!! Not tested all input parameters */
    /* ----- End test ----- */

    CvMat point4D;
    CvMat point3D;
    double point4D_dat[4];
    double point3D_dat[3];
    point4D = cvMat(4,1,CV_64F,point4D_dat);
    point3D = cvMat(3,1,CV_64F,point3D_dat);

#ifdef TRACK_BUNDLE
        {
            FILE *file;
            file = fopen( TRACK_BUNDLE_FILE ,"a");
            fprintf(file,"\n----- test 14.01 icvProjPointsStatusFunc -----\n");
            fclose(file);
        }
#endif

    int currImage;
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        /* Get project matrix */
        /* project visible points using current projection matrix */
        int currVisPoint = 0;
        for( int currPoint = 0; currPoint < numPoints; currPoint++ )
        {
            if( cvmGet(pointsPres[currImage],0,currPoint) > 0 )
            {
                /* project current point */
                cvGetSubRect(points4D,&point4D,cvRect(currPoint,0,1,4));

#ifdef TRACK_BUNDLE
                {
                    FILE *file;
                    file = fopen( TRACK_BUNDLE_FILE ,"a");
                    fprintf(file,"\n----- test 14.02 point4D (%lf, %lf, %lf, %lf) -----\n",
                                 cvmGet(&point4D,0,0),
                                 cvmGet(&point4D,1,0),
                                 cvmGet(&point4D,2,0),
                                 cvmGet(&point4D,3,0));
                    fclose(file);
                }
#endif

                cvmMul(projMatrs[currImage],&point4D,&point3D);
                double w = point3D_dat[2];
                cvmSet(projPoints[currImage],0,currVisPoint,point3D_dat[0]/w);
                cvmSet(projPoints[currImage],1,currVisPoint,point3D_dat[1]/w);

#ifdef TRACK_BUNDLE
                {
                    FILE *file;
                    file = fopen( TRACK_BUNDLE_FILE ,"a");
                    fprintf(file,"\n----- test 14.03 (%lf, %lf, %lf) -> (%lf, %lf)-----\n",
                                 point3D_dat[0],
                                 point3D_dat[1],
                                 point3D_dat[2],
                                 point3D_dat[0]/w,
                                 point3D_dat[1]/w
                                 );
                    fclose(file);
                }
#endif
                currVisPoint++;
            }
        }
    }

    __END__;
}

/*======================================================================================*/
static void icvFreeMatrixArray(CvMat ***matrArray,int numMatr)
{
    /* Free each matrix */
    int currMatr;

    if( *matrArray != 0 )
    {/* Need delete */
        for( currMatr = 0; currMatr < numMatr; currMatr++ )
        {
            cvReleaseMat((*matrArray)+currMatr);
        }
        cvFree( matrArray);
    }
    return;
}

/*======================================================================================*/
static void *icvClearAlloc(int size)
{
    void *ptr = 0;

    CV_FUNCNAME( "icvClearAlloc" );
    __BEGIN__;

    if( size > 0 )
    {
        CV_CALL(ptr = cvAlloc(size));
        memset(ptr,0,size);
    }

    __END__;
    return ptr;
}

/*======================================================================================*/
#if 0
void cvOptimizeLevenbergMarquardtBundleWraper( CvMat** projMatrs, CvMat** observProjPoints,
                                       CvMat** pointsPres, int numImages,
                                       CvMat** resultProjMatrs, CvMat* resultPoints4D,int maxIter,double epsilon )
{
    /* Delete al sparse points */

int icvDeleteSparsInPoints(  int numImages,
                             CvMat **points,
                             CvMat **status,
                             CvMat *wasStatus)/* status of previous configuration */

}
#endif

/*======================================================================================*/
/* !!! may be useful to return norm of error */
/* !!! may be does not work correct with not all visible 4D points */
void cvOptimizeLevenbergMarquardtBundle( CvMat** projMatrs, CvMat** observProjPoints,
                                       CvMat** pointsPres, int numImages,
                                       CvMat** resultProjMatrs, CvMat* resultPoints4D,int maxIter,double epsilon )
{

    CvMat  *vectorX_points4D = 0;
    CvMat **vectorX_projMatrs = 0;

    CvMat  *newVectorX_points4D = 0;
    CvMat **newVectorX_projMatrs = 0;

    CvMat  *changeVectorX_points4D = 0;
    CvMat  *changeVectorX_projMatrs = 0;

    CvMat **observVisPoints = 0;
    CvMat **projVisPoints = 0;
    CvMat **errorProjPoints = 0;
    CvMat **DerivProj = 0;
    CvMat **DerivPoint = 0;
    CvMat *matrW = 0;
    CvMat **matrsUk = 0;
    CvMat **workMatrsUk = 0;
    CvMat **matrsVi = 0;
    CvMat *workMatrVi = 0;
    CvMat **workMatrsInvVi = 0;
    CvMat *jacProjErr = 0;
    CvMat *jacPointErr = 0;

    CvMat *matrTmpSys1 = 0;
    CvMat *matrSysDeltaP = 0;
    CvMat *vectTmpSys3 = 0;
    CvMat *vectSysDeltaP = 0;
    CvMat *deltaP = 0;
    CvMat *deltaM = 0;
    CvMat *vectTmpSysM = 0;

    int numPoints = 0;


    CV_FUNCNAME( "cvOptimizeLevenbergMarquardtBundle" );
    __BEGIN__;

    /* ----- Test input params for errors ----- */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must be more than zero" );
    }

    if( maxIter < 1 || maxIter > 2000 )
    {
        CV_ERROR( CV_StsOutOfRange, "Maximum number of iteration must be in [1..1000]" );
    }

    if( epsilon < 0  )
    {
        CV_ERROR( CV_StsOutOfRange, "Epsilon parameter must be >= 0" );
    }

    if( !CV_IS_MAT(resultPoints4D) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "resultPoints4D must be a matrix 4 x NumPnt" );
    }

    numPoints = resultPoints4D->cols;
    if( numPoints < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points must be more than zero" );//!!!
    }

    if( resultPoints4D->rows != 4 )
    {
        CV_ERROR( CV_StsOutOfRange, "resultPoints4D must have 4 cordinates" );
    }

    /* ----- End test ----- */

    /* Optimization using bundle adjustment */
    /* work with non visible points */

    CV_CALL( vectorX_points4D  = cvCreateMat(4,numPoints,CV_64F));
    CV_CALL( vectorX_projMatrs = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages));

    CV_CALL( newVectorX_points4D  = cvCreateMat(4,numPoints,CV_64F));
    CV_CALL( newVectorX_projMatrs = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages));

    CV_CALL( changeVectorX_points4D  = cvCreateMat(4,numPoints,CV_64F));
    CV_CALL( changeVectorX_projMatrs = cvCreateMat(3,4,CV_64F));

    int currImage;

    /* ----- Test input params ----- */
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        /* Test size of input initial and result projection matrices */
        if( !CV_IS_MAT(projMatrs[currImage]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "each of initial projMatrs must be a matrix 3 x 4" );
        }
        if( projMatrs[currImage]->rows != 3 || projMatrs[currImage]->cols != 4 )
        {
            CV_ERROR( CV_StsOutOfRange, "each of initial projMatrs must be a matrix 3 x 4" );
        }


        if( !CV_IS_MAT(observProjPoints[currImage]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "each of observProjPoints must be a matrix 2 x NumPnts" );
        }
        if( observProjPoints[currImage]->rows != 2 || observProjPoints[currImage]->cols != numPoints )
        {
            CV_ERROR( CV_StsOutOfRange, "each of observProjPoints must be a matrix 2 x NumPnts" );
        }

        if( !CV_IS_MAT(pointsPres[currImage]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "each of pointsPres must be a matrix 1 x NumPnt" );
        }
        if( pointsPres[currImage]->rows != 1 || pointsPres[currImage]->cols != numPoints )
        {
            CV_ERROR( CV_StsOutOfRange, "each of pointsPres must be a matrix 1 x NumPnt" );
        }

        if( !CV_IS_MAT(resultProjMatrs[currImage]) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "each of resultProjMatrs must be a matrix 3 x 4" );
        }
        if( resultProjMatrs[currImage]->rows != 3 || resultProjMatrs[currImage]->cols != 4 )
        {
            CV_ERROR( CV_StsOutOfRange, "each of resultProjMatrs must be a matrix 3 x 4" );
        }

    }
    /* ----- End test ----- */

    /* Copy projection matrices to vectorX0 */
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        CV_CALL( vectorX_projMatrs[currImage] = cvCreateMat(3,4,CV_64F));
        CV_CALL( newVectorX_projMatrs[currImage] = cvCreateMat(3,4,CV_64F));
        cvCopy(projMatrs[currImage],vectorX_projMatrs[currImage]);
    }

    /* Reconstruct points4D using projection matrices and status information */
    icvReconstructPoints4DStatus(observProjPoints, projMatrs, pointsPres, vectorX_points4D, numImages);

    /* ----- Allocate memory for work matrices ----- */
    /* Compute number of good points on each images */

    CV_CALL( observVisPoints = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( projVisPoints   = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( errorProjPoints = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( DerivProj       = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( DerivPoint      = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( matrW           = cvCreateMat(12*numImages,4*numPoints,CV_64F) );
    CV_CALL( matrsUk         = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( workMatrsUk     = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numImages) );
    CV_CALL( matrsVi         = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numPoints) );
    CV_CALL( workMatrVi      = cvCreateMat(4,4,CV_64F) );
    CV_CALL( workMatrsInvVi  = (CvMat**)icvClearAlloc(sizeof(CvMat*)*numPoints) );

    CV_CALL( jacProjErr      = cvCreateMat(12*numImages,1,CV_64F) );
    CV_CALL( jacPointErr     = cvCreateMat(4*numPoints,1,CV_64F) );


    int i;
    for( i = 0; i < numPoints; i++ )
    {
        CV_CALL( matrsVi[i]        = cvCreateMat(4,4,CV_64F) );
        CV_CALL( workMatrsInvVi[i] = cvCreateMat(4,4,CV_64F) );
    }

    for( currImage = 0; currImage < numImages; currImage++ )
    {
        CV_CALL( matrsUk[currImage]     = cvCreateMat(12,12,CV_64F) );
        CV_CALL( workMatrsUk[currImage] = cvCreateMat(12,12,CV_64F) );

        int currVisPoint = 0, currPoint, numVisPoint;
        for( currPoint = 0; currPoint < numPoints; currPoint++ )
        {
            if( cvmGet(pointsPres[currImage],0,currPoint) > 0 )
            {
                currVisPoint++;
            }
        }

        numVisPoint = currVisPoint;

        /* Allocate memory for current image data */
        CV_CALL( observVisPoints[currImage]  = cvCreateMat(2,numVisPoint,CV_64F) );
        CV_CALL( projVisPoints[currImage]    = cvCreateMat(2,numVisPoint,CV_64F) );

        /* create error matrix */
        CV_CALL( errorProjPoints[currImage] = cvCreateMat(2,numVisPoint,CV_64F) );

        /* Create derivate matrices */
        CV_CALL( DerivProj[currImage]  = cvCreateMat(2*numVisPoint,12,CV_64F) );
        CV_CALL( DerivPoint[currImage] = cvCreateMat(2,numVisPoint*4,CV_64F) );

        /* Copy observed projected visible points */
        currVisPoint = 0;
        for( currPoint = 0; currPoint < numPoints; currPoint++ )
        {
            if( cvmGet(pointsPres[currImage],0,currPoint) > 0 )
            {
                cvmSet(observVisPoints[currImage],0,currVisPoint,cvmGet(observProjPoints[currImage],0,currPoint));
                cvmSet(observVisPoints[currImage],1,currVisPoint,cvmGet(observProjPoints[currImage],1,currPoint));
                currVisPoint++;
            }
        }
    }
    /*---------------------------------------------------*/

    CV_CALL( matrTmpSys1   = cvCreateMat(numPoints*4, numImages*12, CV_64F) );
    CV_CALL( matrSysDeltaP = cvCreateMat(numImages*12, numImages*12, CV_64F) );
    CV_CALL( vectTmpSys3   = cvCreateMat(numPoints*4,1,CV_64F) );
    CV_CALL( vectSysDeltaP = cvCreateMat(numImages*12,1,CV_64F) );
    CV_CALL( deltaP        = cvCreateMat(numImages*12,1,CV_64F) );
    CV_CALL( deltaM        = cvCreateMat(numPoints*4,1,CV_64F) );
    CV_CALL( vectTmpSysM   = cvCreateMat(numPoints*4,1,CV_64F) );


//#ifdef TRACK_BUNDLE
#if 1
    {
        /* Create file to track */
        FILE* file;
        file = fopen( TRACK_BUNDLE_FILE ,"w");
        fprintf(file,"begin\n");
        fclose(file);
    }
#endif

    /* ============= main optimization loop ============== */

    /* project all points using current vector X */
    icvProjPointsStatusFunc(numImages, vectorX_points4D, vectorX_projMatrs, pointsPres, projVisPoints);

    /* Compute error with observed value and computed projection */
    double prevError;
    prevError = 0;
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        cvSub(observVisPoints[currImage],projVisPoints[currImage],errorProjPoints[currImage]);
        double currNorm = cvNorm(errorProjPoints[currImage]);
        prevError += currNorm * currNorm;
    }
    prevError = sqrt(prevError);

    int currIter;
    double change;
    double alpha;

//#ifdef TRACK_BUNDLE
#if 1
    {
        /* Create file to track */
        FILE* file;
        file = fopen( TRACK_BUNDLE_FILE ,"a");
        fprintf(file,"\n========================================\n");;
        fprintf(file,"Iter=0\n");
        fprintf(file,"Error = %20.15lf\n",prevError);
        fprintf(file,"Change = %20.15lf\n",1.0);

        fprintf(file,"projection errors\n");

        /* Print all proejction errors */
        int currImage;
        for( currImage = 0; currImage < numImages; currImage++)
        {
            fprintf(file,"\nImage=%d\n",currImage);
            int numPn = errorProjPoints[currImage]->cols;
            for( int currPoint = 0; currPoint < numPn; currPoint++ )
            {
                double ex,ey;
                ex = cvmGet(errorProjPoints[currImage],0,currPoint);
                ey = cvmGet(errorProjPoints[currImage],1,currPoint);
                fprintf(file,"%40.35lf, %40.35lf\n",ex,ey);
            }
        }
        fclose(file);
    }
#endif

    currIter = 0;
    change = 1;
    alpha = 0.001;


    do
    {

#ifdef TRACK_BUNDLE
        {
            FILE *file;
            file = fopen( TRACK_BUNDLE_FILE ,"a");
            fprintf(file,"\n----- test 6 do main -----\n");

            double norm = cvNorm(vectorX_points4D);
            fprintf(file,"        test 6.01 prev normPnts=%lf\n",norm);

            for( int i=0;i<numImages;i++ )
            {
                double norm = cvNorm(vectorX_projMatrs[i]);
                fprintf(file,"        test 6.01 prev normProj=%lf\n",norm);
            }

            fclose(file);
        }
#endif

        /* Compute derivates by projectinon matrices */
        icvComputeDerivateProjAll(vectorX_points4D,vectorX_projMatrs,pointsPres,numImages,DerivProj);

        /* Compute derivates by 4D points */
        icvComputeDerivatePointsAll(vectorX_points4D,vectorX_projMatrs,pointsPres,numImages,DerivPoint);

        /* Compute matrces Uk */
        icvComputeMatrixUAll(numImages,DerivProj,matrsUk);
        icvComputeMatrixVAll(numImages,DerivPoint,pointsPres,matrsVi);
        icvComputeMatrixW(numImages,DerivProj,DerivPoint,pointsPres,matrW);


#ifdef TRACK_BUNDLE
        {
            FILE *file;
            file = fopen( TRACK_BUNDLE_FILE ,"a");
            fprintf(file,"\n----- test 6.03 do matrs U V -----\n");

            int i;
            for( i = 0; i < numImages; i++ )
            {
                double norm = cvNorm(matrsUk[i]);
                fprintf(file,"        test 6.01 prev matrsUk=%lf\n",norm);
            }

            for( i = 0; i < numPoints; i++ )
            {
                double norm = cvNorm(matrsVi[i]);
                fprintf(file,"        test 6.01 prev matrsVi=%lf\n",norm);
            }

            fclose(file);
        }
#endif

        /* Compute jac errors */
        icvComputeJacErrorProj(numImages, DerivProj, errorProjPoints, jacProjErr);
        icvComputeJacErrorPoint(numImages, DerivPoint, errorProjPoints, pointsPres, jacPointErr);

#ifdef TRACK_BUNDLE
        {
            FILE *file;
            file = fopen( TRACK_BUNDLE_FILE ,"a");
            fprintf(file,"\n----- test 6 do main -----\n");
            double norm1 = cvNorm(vectorX_points4D);
            fprintf(file,"        test 6.02 post normPnts=%lf\n",norm1);
            fclose(file);
        }
#endif
        /* Copy matrices Uk to work matrices Uk */
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            cvCopy(matrsUk[currImage],workMatrsUk[currImage]);
        }

#ifdef TRACK_BUNDLE
        {
            FILE *file;
            file = fopen( TRACK_BUNDLE_FILE ,"a");
            fprintf(file,"\n----- test 60.3 do matrs U V -----\n");

            int i;
            for( i = 0; i < numImages; i++ )
            {
                double norm = cvNorm(matrsUk[i]);
                fprintf(file,"        test 6.01 post1 matrsUk=%lf\n",norm);
            }

            for( i = 0; i < numPoints; i++ )
            {
                double norm = cvNorm(matrsVi[i]);
                fprintf(file,"        test 6.01 post1 matrsVi=%lf\n",norm);
            }

            fclose(file);
        }
#endif

        /* ========== Solve normal equation for given alpha and Jacobian ============ */

        do
        {
            /* ---- Add alpha to matrices --- */
            /* Add alpha to matrInvVi and make workMatrsInvVi */

            int currV;
            for( currV = 0; currV < numPoints; currV++ )
            {
                cvCopy(matrsVi[currV],workMatrVi);

                for( int i = 0; i < 4; i++ )
                {
                    cvmSet(workMatrVi,i,i,cvmGet(matrsVi[currV],i,i)*(1+alpha) );
                }

                cvInvert(workMatrVi,workMatrsInvVi[currV],CV_LU/*,&currV*/);
            }

            /* Add alpha to matrUk and make matrix workMatrsUk */
            for( currImage = 0; currImage< numImages; currImage++ )
            {

                for( i = 0; i < 12; i++ )
                {
                    cvmSet(workMatrsUk[currImage],i,i,cvmGet(matrsUk[currImage],i,i)*(1+alpha));
                }


            }

            /* Fill matrix to make system for computing delta P (matrTmpSys1 = inv(V)*tr(W) )*/
            for( currV = 0; currV < numPoints; currV++ )
            {
                int currRowV;
                for( currRowV = 0; currRowV < 4; currRowV++ )
                {
                    for( currImage = 0; currImage < numImages; currImage++ )
                    {
                        for( int currCol = 0; currCol < 12; currCol++ )/* For each column of transposed matrix W */
                        {
                            double sum = 0;
                            for( i = 0; i < 4; i++ )
                            {
                                sum += cvmGet(workMatrsInvVi[currV],currRowV,i) *
                                       cvmGet(matrW,currImage*12+currCol,currV*4+i);
                            }
                            cvmSet(matrTmpSys1,currV*4+currRowV,currImage*12+currCol,sum);
                        }
                    }
                }
            }


            /* Fill matrix to make system for computing delta P (matrTmpSys2 = W * matrTmpSys ) */
            cvmMul(matrW,matrTmpSys1,matrSysDeltaP);

            /* need to compute U-matrTmpSys2. But we compute matTmpSys2-U */
            for( currImage = 0; currImage < numImages; currImage++ )
            {
                CvMat subMatr;
                cvGetSubRect(matrSysDeltaP,&subMatr,cvRect(currImage*12,currImage*12,12,12));
                cvSub(&subMatr,workMatrsUk[currImage],&subMatr);
            }

            /* Compute right side of normal equation  */
            for( currV = 0; currV < numPoints; currV++ )
            {
                CvMat subMatrErPnts;
                CvMat subMatr;
                cvGetSubRect(jacPointErr,&subMatrErPnts,cvRect(0,currV*4,1,4));
                cvGetSubRect(vectTmpSys3,&subMatr,cvRect(0,currV*4,1,4));
                cvmMul(workMatrsInvVi[currV],&subMatrErPnts,&subMatr);
            }

            cvmMul(matrW,vectTmpSys3,vectSysDeltaP);
            cvSub(vectSysDeltaP,jacProjErr,vectSysDeltaP);

            /* Now we can compute part of normal system - deltaP */
            cvSolve(matrSysDeltaP ,vectSysDeltaP, deltaP, CV_SVD);

            /* Print deltaP to file */

#ifdef TRACK_BUNDLE
            {
                FILE* file;
                file = fopen( TRACK_BUNDLE_FILE_DELTAP ,"w");

                int currImage;
                for( currImage = 0; currImage < numImages; currImage++ )
                {
                    fprintf(file,"\nImage=%d\n",currImage);
                    int i;
                    for( i = 0; i < 12; i++ )
                    {
                        double val;
                        val = cvmGet(deltaP,currImage*12+i,0);
                        fprintf(file,"%lf\n",val);
                    }
                    fprintf(file,"\n");
                }
                fclose(file);
            }
#endif
            /* We know deltaP and now we can compute system for deltaM */
            for( i = 0; i < numPoints * 4; i++ )
            {
                double sum = 0;
                for( int j = 0; j < numImages * 12; j++ )
                {
                    sum += cvmGet(matrW,j,i) * cvmGet(deltaP,j,0);
                }
                cvmSet(vectTmpSysM,i,0,cvmGet(jacPointErr,i,0)-sum);
            }

            /* Compute deltaM */
            for( currV = 0; currV < numPoints; currV++ )
            {
                CvMat subMatr;
                CvMat subMatrM;
                cvGetSubRect(vectTmpSysM,&subMatr,cvRect(0,currV*4,1,4));
                cvGetSubRect(deltaM,&subMatrM,cvRect(0,currV*4,1,4));
                cvmMul(workMatrsInvVi[currV],&subMatr,&subMatrM);
            }

            /* We know delta and compute new value of vector X: nextVectX = vectX + deltas */

            /* Compute new P */
            for( currImage = 0; currImage < numImages; currImage++ )
            {
                for( i = 0; i < 3; i++ )
                {
                    for( int j = 0; j < 4; j++ )
                    {
                        cvmSet(newVectorX_projMatrs[currImage],i,j,
                                cvmGet(vectorX_projMatrs[currImage],i,j) + cvmGet(deltaP,currImage*12+i*4+j,0));
                    }
                }
            }

            /* Compute new M */
            int currPoint;
            for( currPoint = 0; currPoint < numPoints; currPoint++ )
            {
                for( i = 0; i < 4; i++ )
                {
                    cvmSet(newVectorX_points4D,i,currPoint,
                        cvmGet(vectorX_points4D,i,currPoint) + cvmGet(deltaM,currPoint*4+i,0));
                }
            }

            /* ----- Compute errors for new vectorX ----- */
            /* Project points using new vectorX and status of each point */
            icvProjPointsStatusFunc(numImages, newVectorX_points4D, newVectorX_projMatrs, pointsPres, projVisPoints);
            /* Compute error with observed value and computed projection */
            double newError = 0;
            for( currImage = 0; currImage < numImages; currImage++ )
            {
                cvSub(observVisPoints[currImage],projVisPoints[currImage],errorProjPoints[currImage]);
                double currNorm = cvNorm(errorProjPoints[currImage]);

//#ifdef TRACK_BUNDLE
#if 1
                {
                    FILE *file;
                    file = fopen( TRACK_BUNDLE_FILE ,"a");
                    fprintf(file,"\n----- test 13,01 currImage=%d currNorm=%lf -----\n",currImage,currNorm);
                    fclose(file);
                }
#endif
                newError += currNorm * currNorm;
            }
            newError = sqrt(newError);

            currIter++;




//#ifdef TRACK_BUNDLE
#if 1
            {
                /* Create file to track */
                FILE* file;
                file = fopen( TRACK_BUNDLE_FILE ,"a");
                fprintf(file,"\n========================================\n");
                fprintf(file,"numPoints=%d\n",numPoints);
                fprintf(file,"Iter=%d\n",currIter);
                fprintf(file,"Error = %20.15lf\n",newError);
                fprintf(file,"Change = %20.15lf\n",change);


                /* Print all projection errors */
#if 0
                fprintf(file,"projection errors\n");
                int currImage;
                for( currImage = 0; currImage < numImages; currImage++)
                {
                    fprintf(file,"\nImage=%d\n",currImage);
                    int numPn = errorProjPoints[currImage]->cols;
                    for( int currPoint = 0; currPoint < numPn; currPoint++ )
                    {
                        double ex,ey;
                        ex = cvmGet(errorProjPoints[currImage],0,currPoint);
                        ey = cvmGet(errorProjPoints[currImage],1,currPoint);
                        fprintf(file,"%lf,%lf\n",ex,ey);
                    }
                }
                fprintf(file,"\n---- test 0 -----\n");
#endif

                fclose(file);
            }
#endif



            /* Compare new error and last error */
            if( newError < prevError )
            {/* accept new value */
                prevError = newError;
                /* Compute relative change of required parameter vectorX. change = norm(curr-prev) / norm(curr) )  */
                {
                    double normAll1 = 0;
                    double normAll2 = 0;
                    double currNorm1 = 0;
                    double currNorm2 = 0;
                    /* compute norm for projection matrices */
                    for( currImage = 0; currImage < numImages; currImage++ )
                    {
                        currNorm1 = cvNorm(newVectorX_projMatrs[currImage],vectorX_projMatrs[currImage]);
                        currNorm2 = cvNorm(newVectorX_projMatrs[currImage]);

                        normAll1 += currNorm1 * currNorm1;
                        normAll2 += currNorm2 * currNorm2;
                    }

                    /* compute norm for points */
                    currNorm1 = cvNorm(newVectorX_points4D,vectorX_points4D);
                    currNorm2 = cvNorm(newVectorX_points4D);

                    normAll1 += currNorm1 * currNorm1;
                    normAll2 += currNorm2 * currNorm2;

                    /* compute change */
                    change = sqrt(normAll1) / sqrt(normAll2);


//#ifdef TRACK_BUNDLE
#if 1
                    {
                        /* Create file to track */
                        FILE* file;
                        file = fopen( TRACK_BUNDLE_FILE ,"a");
                        fprintf(file,"\nChange inside newVal change = %20.15lf\n",change);
                        fprintf(file,"   normAll1= %20.15lf\n",sqrt(normAll1));
                        fprintf(file,"   normAll2= %20.15lf\n",sqrt(normAll2));

                        fclose(file);
                    }
#endif

                }

                alpha /= 10;
                for( currImage = 0; currImage < numImages; currImage++ )
                {
                    cvCopy(newVectorX_projMatrs[currImage],vectorX_projMatrs[currImage]);
                }
                cvCopy(newVectorX_points4D,vectorX_points4D);

                break;
            }
            else
            {
                alpha *= 10;
            }

        } while( change > epsilon && currIter < maxIter );/* solve normal equation using current alpha */

//#ifdef TRACK_BUNDLE
#if 1
        {
            FILE* file;
            file = fopen( TRACK_BUNDLE_FILE ,"a");
            fprintf(file,"\nBest error = %40.35lf\n",prevError);
            fclose(file);
        }

#endif


    } while( change > epsilon && currIter < maxIter );

    /*--------------------------------------------*/
    /* Optimization complete copy computed params */
    /* Copy projection matrices */
    for( currImage = 0; currImage < numImages; currImage++ )
    {
        cvCopy(newVectorX_projMatrs[currImage],resultProjMatrs[currImage]);
    }
    /* Copy 4D points */
    cvCopy(newVectorX_points4D,resultPoints4D);

//    free(memory);

    __END__;

    /* Free allocated memory */

    /* Free simple matrices */
    cvFree(&vectorX_points4D);
    cvFree(&newVectorX_points4D);
    cvFree(&changeVectorX_points4D);
    cvFree(&changeVectorX_projMatrs);
    cvFree(&matrW);
    cvFree(&workMatrVi);
    cvFree(&jacProjErr);
    cvFree(&jacPointErr);
    cvFree(&matrTmpSys1);
    cvFree(&matrSysDeltaP);
    cvFree(&vectTmpSys3);
    cvFree(&vectSysDeltaP);
    cvFree(&deltaP);
    cvFree(&deltaM);
    cvFree(&vectTmpSysM);

    /* Free arrays of matrices */
    icvFreeMatrixArray(&vectorX_projMatrs,numImages);
    icvFreeMatrixArray(&newVectorX_projMatrs,numImages);
    icvFreeMatrixArray(&observVisPoints,numImages);
    icvFreeMatrixArray(&projVisPoints,numImages);
    icvFreeMatrixArray(&errorProjPoints,numImages);
    icvFreeMatrixArray(&DerivProj,numImages);
    icvFreeMatrixArray(&DerivPoint,numImages);
    icvFreeMatrixArray(&matrsUk,numImages);
    icvFreeMatrixArray(&workMatrsUk,numImages);
    icvFreeMatrixArray(&matrsVi,numPoints);
    icvFreeMatrixArray(&workMatrsInvVi,numPoints);

    return;
}
