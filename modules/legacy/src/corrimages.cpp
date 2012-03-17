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
//#include <float.h>
//#include <limits.h>
//#include "cv.h"
//#include "highgui.h"

#include <stdio.h>

/* Valery Mosyagin */

/* ===== Function for find corresponding between images ===== */

/* Create feature points on image and return number of them. Array points fills by found points */
int icvCreateFeaturePoints(IplImage *image, CvMat *points, CvMat *status)
{
    int foundFeaturePoints = 0;
    IplImage *grayImage = 0;
    IplImage *eigImage = 0;
    IplImage *tmpImage = 0;
    CvPoint2D32f *cornerPoints = 0;

    CV_FUNCNAME( "icvFeatureCreatePoints" );
    __BEGIN__;

    /* Test for errors */
    if( image == 0 || points == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    /* Test image size */
    int w,h;
    w = image->width;
    h = image->height;

    if( w <= 0 || h <= 0)
    {
        CV_ERROR( CV_StsOutOfRange, "Size of image must be > 0" );
    }

    /* Test for matrices */
    if( !CV_IS_MAT(points) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameter points must be a matrix" );
    }

    int needNumPoints;
    needNumPoints = points->cols;
    if( needNumPoints <= 0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of need points must be > 0" );
    }

    if( points->rows != 2 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of point coordinates must be == 2" );
    }

    if( status != 0 )
    {
        /* If status matrix exist test it for correct */
        if( !CV_IS_MASK_ARR(status) )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "Statuses must be a mask arrays" );
        }

        if( status->cols != needNumPoints )
        {
            CV_ERROR( CV_StsUnmatchedSizes, "Size of points and statuses must be the same" );
        }

        if( status->rows !=1 )
        {
            CV_ERROR( CV_StsUnsupportedFormat, "Number of rows of status must be 1" );
        }
    }

    /* Create temporary images */
    CV_CALL( grayImage = cvCreateImage(cvSize(w,h), 8,1) );
    CV_CALL( eigImage   = cvCreateImage(cvSize(w,h),32,1) );
    CV_CALL( tmpImage   = cvCreateImage(cvSize(w,h),32,1) );

    /* Create points */
    CV_CALL( cornerPoints = (CvPoint2D32f*)cvAlloc( sizeof(CvPoint2D32f) * needNumPoints) );

    int foundNum;
    double quality;
    double minDist;

    cvCvtColor(image,grayImage, CV_BGR2GRAY);

    foundNum = needNumPoints;
    quality = 0.01;
    minDist = 5;
    cvGoodFeaturesToTrack(grayImage, eigImage, tmpImage, cornerPoints, &foundNum, quality, minDist);

    /* Copy found points to result */
    int i;
    for( i = 0; i < foundNum; i++ )
    {
        cvmSet(points,0,i,cornerPoints[i].x);
        cvmSet(points,1,i,cornerPoints[i].y);
    }

    /* Set status if need */
    if( status )
    {
        for( i = 0; i < foundNum; i++ )
        {
            status->data.ptr[i] = 1;
        }

        for( i = foundNum; i < needNumPoints; i++ )
        {
            status->data.ptr[i] = 0;
        }
    }

    foundFeaturePoints = foundNum;

    __END__;

    /* Free allocated memory */
    cvReleaseImage(&grayImage);
    cvReleaseImage(&eigImage);
    cvReleaseImage(&tmpImage);
    cvFree(&cornerPoints);

    return foundFeaturePoints;
}

/*-------------------------------------------------------------------------------------*/

/* For given points1 (with pntStatus) on image1 finds corresponding points2 on image2 and set pntStatus2 for them */
/* Returns number of corresponding points */
int icvFindCorrForGivenPoints( IplImage *image1,/* Image 1 */
                                IplImage *image2,/* Image 2 */
                                CvMat *points1, 
                                CvMat *pntStatus1,
                                CvMat *points2,
                                CvMat *pntStatus2,
                                int useFilter,/*Use fundamental matrix to filter points */
                                double threshold)/* Threshold for good points in filter */
{
    int resNumCorrPoints = 0;
    CvPoint2D32f* cornerPoints1 = 0;
    CvPoint2D32f* cornerPoints2 = 0;
    char*  status = 0;
    float* errors = 0;
    CvMat* tmpPoints1 = 0;
    CvMat* tmpPoints2 = 0;
    CvMat* pStatus = 0;
    IplImage *grayImage1 = 0;
    IplImage *grayImage2 = 0;
    IplImage *pyrImage1 = 0;
    IplImage *pyrImage2 = 0;

    CV_FUNCNAME( "icvFindCorrForGivenPoints" );
    __BEGIN__;

    /* Test input data for errors */

    /* Test for null pointers */
    if( image1     == 0 || image2     == 0 || 
        points1    == 0 || points2    == 0 ||
        pntStatus1 == 0 || pntStatus2 == 0)
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    /* Test image size */
    int w,h;
    w = image1->width;
    h = image1->height;

    if( w <= 0 || h <= 0)
    {
        CV_ERROR( CV_StsOutOfRange, "Size of image1 must be > 0" );
    }

    if( image2->width != w || image2->height != h )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Size of images must be the same" );
    }

    /* Test for matrices */
    if( !CV_IS_MAT(points1)    || !CV_IS_MAT(points2) || 
        !CV_IS_MAT(pntStatus1) || !CV_IS_MAT(pntStatus2) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters (points and status) must be a matrices" );
    }

    /* Test type of status matrices */
    if( !CV_IS_MASK_ARR(pntStatus1) || !CV_IS_MASK_ARR(pntStatus2) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Statuses must be a mask arrays" );
    }

    /* Test number of points */
    int numPoints;
    numPoints = points1->cols;

    if( numPoints <= 0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points1 must be > 0" );
    }

    if( points2->cols != numPoints || pntStatus1->cols != numPoints || pntStatus2->cols != numPoints )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of points and statuses must be the same" );
    }

    if( points1->rows != 2 || points2->rows != 2 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of points coordinates must be 2" );
    }

    if( pntStatus1->rows != 1 || pntStatus2->rows != 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Status must be a matrix 1xN" );
    }
    /* ----- End test ----- */


    /* Compute number of visible points on image1 */
    int numVisPoints;
    numVisPoints = cvCountNonZero(pntStatus1);

    if( numVisPoints > 0 )
    {
        /* Create temporary images */
        /* We must use iplImage againts hughgui images */

/*
        CvvImage grayImage1;
        CvvImage grayImage2;
        CvvImage pyrImage1;
        CvvImage pyrImage2;
*/

        /* Create Ipl images */
        CV_CALL( grayImage1 = cvCreateImage(cvSize(w,h),8,1) );
        CV_CALL( grayImage2 = cvCreateImage(cvSize(w,h),8,1) );
        CV_CALL( pyrImage1  = cvCreateImage(cvSize(w,h),8,1) );
        CV_CALL( pyrImage2  = cvCreateImage(cvSize(w,h),8,1) );

        CV_CALL( cornerPoints1 = (CvPoint2D32f*)cvAlloc( sizeof(CvPoint2D32f)*numVisPoints) );
        CV_CALL( cornerPoints2 = (CvPoint2D32f*)cvAlloc( sizeof(CvPoint2D32f)*numVisPoints) );
        CV_CALL( status = (char*)cvAlloc( sizeof(char)*numVisPoints) );
        CV_CALL( errors = (float*)cvAlloc( 2 * sizeof(float)*numVisPoints) );

        int i;
        for( i = 0; i < numVisPoints; i++ )
        {
            status[i] = 1;
        }

        /* !!! Need test creation errors */
        /*
        if( !grayImage1.Create(w,h,8)) EXIT;
        if( !grayImage2.Create(w,h,8)) EXIT;
        if( !pyrImage1. Create(w,h,8)) EXIT;
        if( !pyrImage2. Create(w,h,8)) EXIT;
        */

        cvCvtColor(image1,grayImage1,CV_BGR2GRAY);
        cvCvtColor(image2,grayImage2,CV_BGR2GRAY);

        /*
        grayImage1.CopyOf(image1,0);
        grayImage2.CopyOf(image2,0);
        */

        /* Copy points good points from input data */
        uchar *stat1 = pntStatus1->data.ptr;
        uchar *stat2 = pntStatus2->data.ptr;

        int curr = 0;
        for( i = 0; i < numPoints; i++ )
        {
            if( stat1[i] )
            {
                cornerPoints1[curr].x = (float)cvmGet(points1,0,i);
                cornerPoints1[curr].y = (float)cvmGet(points1,1,i);
                curr++;
            }
        }

        /* Define number of levels of pyramid */
        cvCalcOpticalFlowPyrLK( grayImage1, grayImage2,
                                pyrImage1, pyrImage2,
                                cornerPoints1, cornerPoints2,
                                numVisPoints, cvSize(10,10), 3,
                                status, errors, 
                                cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03),
                                0/*CV_LKFLOW_PYR_A_READY*/ );

        
        memset(stat2,0,sizeof(uchar)*numPoints);

        int currVis = 0;
        int totalCorns = 0;

        /* Copy new points and set status */
        /* stat1 may not be the same as stat2 */
        for( i = 0; i < numPoints; i++ )
        {
            if( stat1[i] )
            {
                if( status[currVis] && errors[currVis] < 1000 )
                {
                    stat2[i] = 1;
                    cvmSet(points2,0,i,cornerPoints2[currVis].x);
                    cvmSet(points2,1,i,cornerPoints2[currVis].y);
                    totalCorns++;
                }
                currVis++;
            }
        }

        resNumCorrPoints = totalCorns;

        /* Filter points using RANSAC */
        if( useFilter )
        {
            resNumCorrPoints = 0;
            /* Use RANSAC filter for found points */
            if( totalCorns > 7 )
            {
                /* Create array with good points only */
                CV_CALL( tmpPoints1 = cvCreateMat(2,totalCorns,CV_64F) );
                CV_CALL( tmpPoints2 = cvCreateMat(2,totalCorns,CV_64F) );

                /* Copy just good points */
                int currPoint = 0;
                for( i = 0; i < numPoints; i++ )
                {
                    if( stat2[i] )
                    {
                        cvmSet(tmpPoints1,0,currPoint,cvmGet(points1,0,i));
                        cvmSet(tmpPoints1,1,currPoint,cvmGet(points1,1,i));

                        cvmSet(tmpPoints2,0,currPoint,cvmGet(points2,0,i));
                        cvmSet(tmpPoints2,1,currPoint,cvmGet(points2,1,i));

                        currPoint++;
                    }
                }

                /* Compute fundamental matrix */
                CvMat fundMatr;
                double fundMatr_dat[9];
                fundMatr = cvMat(3,3,CV_64F,fundMatr_dat);
        
                CV_CALL( pStatus = cvCreateMat(1,totalCorns,CV_32F) );

                int num = cvFindFundamentalMat(tmpPoints1,tmpPoints2,&fundMatr,CV_FM_RANSAC,threshold,0.99,pStatus);
                if( num > 0 )
                {
                    int curr = 0;
                    /* Set final status for points2 */
                    for( i = 0; i < numPoints; i++ )
                    {
                        if( stat2[i] )
                        {
                            if( cvmGet(pStatus,0,curr) == 0 )
                            {
                                stat2[i] = 0;
                            }
                            curr++;
                        }
                    }
                    resNumCorrPoints = curr;
                }
            }
        }
    }

    __END__;

    /* Free allocated memory */
    cvFree(&cornerPoints1);
    cvFree(&cornerPoints2);
    cvFree(&status);
    cvFree(&errors);
    cvFree(&tmpPoints1);
    cvFree(&tmpPoints2);
    cvReleaseMat( &pStatus );
    cvReleaseImage( &grayImage1 );
    cvReleaseImage( &grayImage2 );
    cvReleaseImage( &pyrImage1 );
    cvReleaseImage( &pyrImage2 );

    return resNumCorrPoints;
}
/*-------------------------------------------------------------------------------------*/
int icvGrowPointsAndStatus(CvMat **oldPoints,CvMat **oldStatus,CvMat *addPoints,CvMat *addStatus,int addCreateNum)
{
    /* Add to existing points and status arrays new points or just grow */
    CvMat *newOldPoint  = 0;
    CvMat *newOldStatus = 0;
    int newTotalNumber = 0;

    CV_FUNCNAME( "icvGrowPointsAndStatus" );
    __BEGIN__;
    
    /* Test for errors */
    if( oldPoints == 0 || oldStatus == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( *oldPoints == 0 || *oldStatus == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(*oldPoints))
    {
        CV_ERROR( CV_StsUnsupportedFormat, "oldPoints must be a pointer to a matrix" );
    }

    if( !CV_IS_MASK_ARR(*oldStatus))
    {
        CV_ERROR( CV_StsUnsupportedFormat, "oldStatus must be a pointer to a mask array" );
    }

    int oldNum;
    oldNum = (*oldPoints)->cols;
    if( oldNum < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of old points must be > 0" );
    }

    /* Define if need number of add points */
    int addNum;
    addNum = 0;
    if( addPoints != 0 && addStatus != 0 )
    {/* We have aditional points */
        if( CV_IS_MAT(addPoints) && CV_IS_MASK_ARR(addStatus) )
        {
            addNum = addPoints->cols;
            if( addStatus->cols != addNum )
            {
                CV_ERROR( CV_StsOutOfRange, "Number of add points and statuses must be the same" );
            }
        }
    }

    /*  */

    int numCoord;
    numCoord = (*oldPoints)->rows;
    newTotalNumber = oldNum + addNum + addCreateNum;

    if( newTotalNumber )
    {
        /* Free allocated memory */
        newOldPoint  = cvCreateMat(numCoord,newTotalNumber,CV_64F);
        newOldStatus = cvCreateMat(1,newTotalNumber,CV_8S);

        /* Copy old values to  */
        int i;

        /* Clear all values */
        cvZero(newOldPoint);
        cvZero(newOldStatus);

        for( i = 0; i < oldNum; i++ )
        {
            int currCoord;
            for( currCoord = 0; currCoord < numCoord; currCoord++ )
            {
                cvmSet(newOldPoint,currCoord,i,cvmGet(*oldPoints,currCoord,i));
            }
            newOldStatus->data.ptr[i] = (*oldStatus)->data.ptr[i];
        }

        /* Copy additional points and statuses */
        if( addNum )
        {
            for( i = 0; i < addNum; i++ )
            {
                int currCoord;
                for( currCoord = 0; currCoord < numCoord; currCoord++ )
                {
                    cvmSet(newOldPoint,currCoord,i+oldNum,cvmGet(addPoints,currCoord,i));
                }
                newOldStatus->data.ptr[i+oldNum] = addStatus->data.ptr[i];
                //cvmSet(newOldStatus,0,i,cvmGet(addStatus,0,i));
            }
        }

        /* Delete previous data */
        cvReleaseMat(oldPoints);
        cvReleaseMat(oldStatus);

        /* copy pointers */
        *oldPoints  = newOldPoint;
        *oldStatus = newOldStatus;

    }
    __END__;

    return newTotalNumber;
}
/*-------------------------------------------------------------------------------------*/
int icvRemoveDoublePoins(   CvMat *oldPoints,/* Points on prev image */
                            CvMat *newPoints,/* New points */
                            CvMat *oldStatus,/* Status for old points */
                            CvMat *newStatus,
                            CvMat *origStatus,
                            float threshold)/* Status for new points */
{

    CvMemStorage* storage = 0;
    CvSubdiv2D* subdiv = 0;
    CvSeq* seq = 0;

    int originalPoints = 0;
    
    CV_FUNCNAME( "icvRemoveDoublePoins" );
    __BEGIN__;

    /* Test input data */
    if( oldPoints == 0 || newPoints == 0 ||
        oldStatus == 0 || newStatus == 0 || origStatus == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }

    if( !CV_IS_MAT(oldPoints) || !CV_IS_MAT(newPoints) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters points must be a matrices" );
    }

    if( !CV_IS_MASK_ARR(oldStatus) || !CV_IS_MASK_ARR(newStatus) || !CV_IS_MASK_ARR(origStatus) )
    {
        CV_ERROR( CV_StsUnsupportedFormat, "Input parameters statuses must be a mask array" );
    }

    int oldNumPoints;
    oldNumPoints = oldPoints->cols;
    if( oldNumPoints < 0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of oldPoints must be >= 0" );
    }

    if( oldStatus->cols != oldNumPoints )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of old Points and old Statuses must be the same" );
    }

    int newNumPoints;
    newNumPoints = newPoints->cols;
    if( newNumPoints < 0 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of newPoints must be >= 0" );
    }

    if( newStatus->cols != newNumPoints )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of new Points and new Statuses must be the same" );
    }

    if( origStatus->cols != newNumPoints )
    {
        CV_ERROR( CV_StsUnmatchedSizes, "Number of new Points and new original Status must be the same" );
    }

    if( oldPoints->rows != 2)
    {
        CV_ERROR( CV_StsOutOfRange, "OldPoints must have 2 coordinates >= 0" );
    }

    if( newPoints->rows != 2)
    {
        CV_ERROR( CV_StsOutOfRange, "NewPoints must have 2 coordinates >= 0" );
    }

    if( oldStatus->rows != 1 || newStatus->rows != 1 || origStatus->rows != 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Statuses must have 1 row" );
    }
    
    /* we have points on image and wants add new points */
    /* use subdivision for find nearest points */

    /* Define maximum and minimum X and Y */
    float minX,minY;
    float maxX,maxY;

    minX = minY = FLT_MAX;
    maxX = maxY = FLT_MIN;

    int i;

    for( i = 0; i < oldNumPoints; i++ )
    {
        if( oldStatus->data.ptr[i] )
        {
            float x = (float)cvmGet(oldPoints,0,i);
            float y = (float)cvmGet(oldPoints,1,i);

            if( x < minX )
                minX = x;

            if( x > maxX )
                maxX = x;

            if( y < minY )
                minY = y;

            if( y > maxY )
                maxY = y;
        }
    }

    for( i = 0; i < newNumPoints; i++ )
    {
        if( newStatus->data.ptr[i] )
        {
            float x = (float)cvmGet(newPoints,0,i);
            float y = (float)cvmGet(newPoints,1,i);

            if( x < minX )
                minX = x;

            if( x > maxX )
                maxX = x;

            if( y < minY )
                minY = y;

            if( y > maxY )
                maxY = y;
        }
    }


    /* Creare subdivision for old image */
    storage = cvCreateMemStorage(0);
//    subdiv = cvCreateSubdivDelaunay2D( cvRect( 0, 0, size.width, size.height ), storage );
    subdiv = cvCreateSubdivDelaunay2D( cvRect( cvRound(minX)-5, cvRound(minY)-5, cvRound(maxX-minX)+10, cvRound(maxY-minY)+10 ), storage );
    seq = cvCreateSeq( 0, sizeof(*seq), sizeof(CvPoint2D32f), storage );

    /* Insert each point from first image */
    for( i = 0; i < oldNumPoints; i++ )
    {
        /* Add just exist points */
        if( oldStatus->data.ptr[i] )
        {
            CvPoint2D32f pt;
            pt.x = (float)cvmGet(oldPoints,0,i);
            pt.y = (float)cvmGet(oldPoints,1,i);

            cvSubdivDelaunay2DInsert( subdiv, pt );
        }
    }


    /* Find nearest points */
    /* for each new point */
    int flag;
    for( i = 0; i < newNumPoints; i++ )
    {
        flag = 0;
        /* Test just exist points */
        if( newStatus->data.ptr[i] )
        {
            flag = 1;
            /* Let this is a good point */
            //originalPoints++;

            CvPoint2D32f pt;

            pt.x = (float)cvmGet(newPoints,0,i);
            pt.y = (float)cvmGet(newPoints,1,i);

            CvSubdiv2DPoint* point = cvFindNearestPoint2D( subdiv, pt );

            if( point )
            {
                /* Test distance of found nearest point */
                double minDistance = icvSqDist2D32f( pt, point->pt );

                if( minDistance < threshold*threshold )
                {
                    /* Point is double. Turn it off */
                    /* Set status */
                    //newStatus->data.ptr[i] = 0;
                    
                    /* No this is a double point */
                    //originalPoints--;
                    flag = 0;
                }
            }
        }
        originalPoints += flag;
        origStatus->data .ptr[i] = (uchar)flag;
    }

    __END__;

    cvReleaseMemStorage( &storage );
    

    return originalPoints;


}

void icvComputeProjectMatrix(CvMat* objPoints,CvMat* projPoints,CvMat* projMatr);

/*-------------------------------------------------------------------------------------*/
void icvComputeProjectMatrixStatus(CvMat *objPoints4D,CvMat *points2,CvMat *status, CvMat *projMatr)
{
    /* Compute number of good points */
    int num = cvCountNonZero(status);
    
    /* Create arrays */
    CvMat *objPoints = 0;
    objPoints = cvCreateMat(4,num,CV_64F);

    CvMat *points2D = 0;
    points2D = cvCreateMat(2,num,CV_64F);

    int currVis = 0;
    int i;
#if 1
    FILE *file;
    file = fopen("d:\\test\\projStatus.txt","w");
#endif
    int totalNum = objPoints4D->cols;
    for( i = 0; i < totalNum; i++ )
    {
        fprintf(file,"%d (%d) ",i,status->data.ptr[i]);
        if( status->data.ptr[i] )
        {

#if 1
            double X,Y,Z,W;
            double x,y;
            X = cvmGet(objPoints4D,0,i);
            Y = cvmGet(objPoints4D,1,i);
            Z = cvmGet(objPoints4D,2,i);
            W = cvmGet(objPoints4D,3,i);

            x = cvmGet(points2,0,i);
            y = cvmGet(points2,1,i);
            fprintf(file,"%d (%lf %lf %lf %lf) - (%lf %lf)",i,X,Y,Z,W,x,y );
#endif
            cvmSet(objPoints,0,currVis,cvmGet(objPoints4D,0,i));
            cvmSet(objPoints,1,currVis,cvmGet(objPoints4D,1,i));
            cvmSet(objPoints,2,currVis,cvmGet(objPoints4D,2,i));
            cvmSet(objPoints,3,currVis,cvmGet(objPoints4D,3,i));

            cvmSet(points2D,0,currVis,cvmGet(points2,0,i));
            cvmSet(points2D,1,currVis,cvmGet(points2,1,i));

            currVis++;
        }
        
        fprintf(file,"\n");
    }

#if 1
    fclose(file);
#endif

    icvComputeProjectMatrix(objPoints,points2D,projMatr);

    /* Free allocated memory */
    cvReleaseMat(&objPoints);
    cvReleaseMat(&points2D);
}



/*-------------------------------------------------------------------------------------*/
/* For given N images 
 we have corresponding points on N images
 computed projection matrices
 reconstructed 4D points

  we must to compute 
  

*/

void icvAddNewImageToPrevious____(
                                    IplImage *newImage,//Image to add
                                    IplImage *oldImage,//Previous image
                                    CvMat *oldPoints,// previous 2D points on prev image (some points may be not visible)
                                    CvMat *oldPntStatus,//Status for each point on prev image
                                    CvMat *objPoints4D,//prev 4D points
                                    CvMat *newPoints,  //Points on new image corr for prev
                                    CvMat *newPntStatus,// New point status for new image
                                    CvMat *newFPoints2D1,//new feature points on prev image
                                    CvMat *newFPoints2D2,//new feature points on new image
                                    CvMat *newFPointsStatus,
                                    CvMat *newProjMatr,
                                    int useFilter,
                                    double threshold)//New projection matrix
{
    CvMat *points2 = 0;
    CvMat *status = 0;
    CvMat *newFPointsStatusTmp = 0;

    //CV_FUNCNAME( "icvAddNewImageToPrevious____" );
    __BEGIN__;

    /* First found correspondence points for images */

    /* Test input params */

    int numPoints;
    numPoints = oldPoints->cols;

    /* Allocate memory */

    points2 = cvCreateMat(2,numPoints,CV_64F);
    status = cvCreateMat(1,numPoints,CV_8S);
    newFPointsStatusTmp = cvCreateMat(1, newFPoints2D1->cols,CV_8S);

    int corrNum;
    corrNum = icvFindCorrForGivenPoints(    oldImage,/* Image 1 */
                                            newImage,/* Image 2 */
                                            oldPoints, 
                                            oldPntStatus,
                                            points2,
                                            status,
                                            useFilter,/*Use fundamental matrix to filter points */
                                            threshold);/* Threshold for good points in filter */

    cvCopy(status,newPntStatus);
    cvCopy(points2,newPoints);

    CvMat projMatr;
    double projMatr_dat[12];
    projMatr = cvMat(3,4,CV_64F,projMatr_dat);

    if( corrNum >= 6 )
    {/* We can compute projection matrix */
//        icvComputeProjectMatrix(objPoints4D,points2,&projMatr);
        icvComputeProjectMatrixStatus(objPoints4D,points2,status,&projMatr);
        cvCopy(&projMatr,newProjMatr);
        
        /* Create new points and find correspondence */
        icvCreateFeaturePoints(newImage, newFPoints2D2,newFPointsStatus);
        
        /* Good if we test new points before find corr points */

        /* Find correspondence for new found points */
        icvFindCorrForGivenPoints( newImage,/* Image 1 */
                                   oldImage,/* Image 2 */
                                   newFPoints2D2,
                                   newFPointsStatus,//prev status
                                   newFPoints2D1,
                                   newFPointsStatusTmp,//new status
                                   useFilter,/*Use fundamental matrix to filter points */
                                   threshold);/* Threshold for good points in filter */

        /* We generated new points on image test for exist points */

        /* Remove all new double points */

        /* Find point of old image */
        icvRemoveDoublePoins( oldPoints,/* Points on prev image */
                              newFPoints2D1,/* New points */
                              oldPntStatus,/* Status for old points */
                              newFPointsStatusTmp,
                              newFPointsStatusTmp,//orig status
                              20);/* Status for new points */

        /* Find double points on new image */
        icvRemoveDoublePoins( newPoints,/* Points on prev image */
                              newFPoints2D2,/* New points */
                              newPntStatus,/* Status for old points */
                              newFPointsStatusTmp,
                              newFPointsStatusTmp,//orig status
                              20);/* Status for new points */



        /* Add all new good points to result */


        /* Copy new status to old */
        cvCopy(newFPointsStatusTmp,newFPointsStatus);


    }



    __END__;

    /* Free allocated memory */

    return;
}
/*-------------------------------------------------------------------------------------*/
//int icvDelete//
//CreateGood

/*-------------------------------------------------------------------------------------*/
int icvDeleteSparsInPoints(  int numImages,
                             CvMat **points,
                             CvMat **status,
                             CvMat *wasStatus)/* status of previous configuration */
{
    /* Delete points which no exist on any of images */
    /* numImages - number of images */
    /* points - arrays of points for each image. Changing */
    /* status - arrays of status for each image. Changing */
    /* Function returns number of common points */

    int comNumber = 0;
    CV_FUNCNAME( "icvDeleteSparsInPoints" );
    __BEGIN__;

    /* Test for errors */
    if( numImages < 1 )
    {
        CV_ERROR( CV_StsOutOfRange, "Number of images must be more than 0" );
    }

    if( points == 0 || status == 0 )
    {
        CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
    }
    int numPoints;

    numPoints = points[0]->cols;
    ////////// TESTS //////////

    int numCoord;
    numCoord = points[0]->rows;// !!! may be number of coordinates is not correct !!!
    
    int i;
    int currExistPoint;
    currExistPoint = 0;

    if( wasStatus )
    {
        cvZero(wasStatus);
    }

    int currImage;
    for( i = 0; i < numPoints; i++ )
    {
        int flag = 0;
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            flag |= status[currImage]->data.ptr[i];
        }

        if( flag )
        {
            /* Current point exists */
            /* Copy points and status */
            if( currExistPoint != i )/* Copy just if different */
            {
                for( currImage = 0; currImage < numImages; currImage++ )
                {
                    /* Copy points */
                    for( int currCoord = 0; currCoord < numCoord; currCoord++ )
                    {
                        cvmSet(points[currImage],currCoord,currExistPoint, cvmGet(points[currImage],currCoord,i) );
                    }

                    /* Copy status */
                    status[currImage]->data.ptr[currExistPoint] = status[currImage]->data.ptr[i];
                }
            }
            if( wasStatus )
            {
                wasStatus->data.ptr[i] = 1;
            }

            currExistPoint++;

        }
    }

    /* Rest of final status of points must be set to 0  */
    for( i = currExistPoint; i < numPoints; i++ )
    {
        for( currImage = 0; currImage < numImages; currImage++ )
        {
            status[currImage]->data.ptr[i] = 0;
        }
    }

    comNumber = currExistPoint;

    __END__;
    return comNumber;
}

#if 0
/*-------------------------------------------------------------------------------------*/
void icvGrowPointsArray(CvMat **points)
{


}

/*-------------------------------------------------------------------------------------*/
void icvAddNewArrayPoints()
{

}

/*-------------------------------------------------------------------------------------*/
#endif

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

/* Add image to existing images and corr points */
#if 0
/* Returns: 1 if new image was added good */
/*          0 image was not added. Not enought corr points */
int AddImageToStruct(  IplImage *newImage,//Image to add
                        IplImage *oldImage,//Previous image
                        CvMat *oldPoints,// previous 2D points on prev image (some points may be not visible)
                        CvMat *oldPntStatus,//Status for each point on prev image
                        CvMat *objPoints4D,//prev 4D points
                        CvMat *newPntStatus,// New point status for new image
                        CvMat *newPoints,//New corresponding points on new image
                        CvMat *newPoints2D1,//new points on prev image
                        CvMat *newPoints2D2,//new points on new image
                        CvMat *newProjMatr);//New projection matrix
{

    /* Add new image. Create new corr points */
    /* Track exist points from oldImage to newImage */
    /* Create new vector status */
    CvMat *status;
    int numPoints = oldPoints->cols;
    status = cvCreateMat(1,numPoints,CV_64F);
    /* Copy status */
    cvConvert(pntStatus,status);

    int corrNum = FindCorrForGivenPoints(oldImage,newImage,oldPoints,newPoints,status);
    
    /* Status has new status of points */

    CvMat projMatr;
    double projMatr_dat[12];
    projMatr = cvMat(3,4,CV_64F,projMatr_dat);

    /* If number of corr points is 6 or more can compute projection matrix */
    if( corrNum >= 6)
    {
        /* Compute projection matrix for new image using corresponding points */
        icvComputeProjectMatrix(objPoints4D,newPoints,&projMatr);

        CvMat *tmpPoints;
        /* Create new points and find correspondence */
        int num = FindFeaturePoints(newImage, &tmpPoints);
        if( num > 0 )
        {
            CvMat *newPoints;
            newPoints = cvCreateMat(2,num,CV_64F);
            CvMat *status;
            status = cvCreateMat(1,num,CV_64F);
            /* Set status for all points */
            int i;
            for( i = 0; i < num; i++ )
            {
                cvmSet(status,0,i,1.0);
            }

            int corrNum2 = FindCorrForGivenPoints(oldImage,newImage,tmpPoints,newPoints,status);

            /* !!! Filter points using projection matrices or not ??? */

            /* !!! Need to filter nearest points */

            /* Add new found points to exist points and optimize again */
            CvMat *new2DPoints;
            CvMat *newStatus;

            /* add new status to old status */





        }
        else
        {
            /* No new points were found */
        }
    }
    else
    {
        /* We can't compute projection matrix for new image */
        return 0;
    }

}
#endif
