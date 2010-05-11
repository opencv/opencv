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

#include "cvtest.h"

#if 0

/* Testing parameters */
static char FuncName[] = "cvCalcOpticalFlowHS";
static char TestName[] = "Optical flow (Horn & Schunck)";
static char TestClass[] = "Algorithm";

static long lImageWidth;
static long lImageHeight;
static float lambda;

#define EPSILON 0.0001f

static int fmaCalcOpticalFlowHS( void )
{
    /* Some Variables */
    int            i,j,k;
    
    uchar*         roiA;
    uchar*         roiB;

    float*         VelocityX;
    float*         VelocityY;

    float*         auxVelocityX;
    float*         auxVelocityY;

    float*         DerX;
    float*         DerY;
    float*         DerT;

    long            lErrors = 0;

    CvTermCriteria criteria;

    int usePrevious;

    int Stop = 0;
    int iteration = 0;
    float epsilon = 0;

    static int  read_param = 0;

    /* Initialization global parameters */
    if( !read_param )
    {
        read_param = 1;
        /* Reading test-parameters */
        trslRead( &lImageHeight, "300", "Image height" );
        trslRead( &lImageWidth, "300", "Image width" );
        trssRead( &lambda, "20", "lambda" );
    }

    /* initialization - for warning disable */
    criteria.epsilon = 0;
    criteria.max_iter = 0;
    criteria.type  = 1;

    /* Allocating memory for all frames */
    IplImage* imgA = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_8U, 1 );
    IplImage* imgB = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_8U, 1 );

    IplImage* testVelocityX = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_32F, 1 );
    IplImage* testVelocityY = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_32F, 1 );

    VelocityX = (float*)cvAlloc(  lImageWidth*lImageHeight * sizeof(float) );
    VelocityY = (float*)cvAlloc(  lImageWidth*lImageHeight * sizeof(float) );

    auxVelocityX = (float*)cvAlloc(  lImageWidth*lImageHeight * sizeof(float) );
    auxVelocityY = (float*)cvAlloc(  lImageWidth*lImageHeight * sizeof(float) );

    DerX = (float*)cvAlloc( lImageWidth*lImageHeight * sizeof(float) );
    DerY = (float*)cvAlloc( lImageWidth*lImageHeight * sizeof(float) );
    DerT = (float*)cvAlloc( lImageWidth*lImageHeight * sizeof(float) );

    /* Filling images */
    ats1bInitRandom( 0, 255, (uchar*)imgA->imageData, lImageWidth * lImageHeight );
    ats1bInitRandom( 0, 255, (uchar*)imgB->imageData, lImageWidth * lImageHeight );

    /* set ROI of images */
    roiA = (uchar*)imgA->imageData;
    roiB = (uchar*)imgB->imageData;

    /* example of 3*3 ROI*/
    /*roiA[0] = 0;
    roiA[1] = 1;
    roiA[2] = 2;
    roiA[lImageWidth] = 0;
    roiA[lImageWidth+1] = 1;
    roiA[lImageWidth+2] = 2;
    roiA[2*lImageWidth] = 0;
    roiA[2*lImageWidth+1] = 1;
    roiA[2*lImageWidth+2] = 2;

    roiB[0] = 1;
    roiB[1] = 2;
    roiB[2] = 3;
    roiB[lImageWidth] = 1;
    roiB[lImageWidth+1] = 2;
    roiB[lImageWidth+2] = 3;
    roiB[2*lImageWidth] = 1;
    roiB[2*lImageWidth+1] = 2;
    roiB[2*lImageWidth+2] = 3;*/
/****************************************************************************************\
*                  Calculate derivatives                                                 *
\****************************************************************************************/
    for (i=0; i<lImageHeight; i++)
    {
        for(j=0; j<lImageWidth; j++)
        {
            int jr,jl,it,ib;

            if ( j==lImageWidth-1 )
                jr = lImageWidth-1;
            else jr = j + 1;

            if ( j==0 )
                jl = 0;
            else jl = j - 1;

            if ( i==(lImageHeight - 1) )
                ib = lImageHeight - 1;
            else ib = i + 1;

            if ( i==0 )
                it = 0;
            else it = i - 1;

            DerX[ i*lImageWidth + j ] = (float)
                (roiA[ (it)*imgA->widthStep + jr ]
                - roiA[ (it)*imgA->widthStep + jl ]
                + 2*roiA[ (i)*imgA->widthStep + jr ]
                - 2*roiA[ (i)*imgA->widthStep + jl ]
                + roiA[ (ib)*imgA->widthStep + jr ]
                - roiA[ (ib)*imgA->widthStep + jl ])/8 ;

            DerY[ i*lImageWidth + j ] = (float)
                ( roiA[ (ib)*imgA->widthStep + jl ]
                + 2*roiA[ (ib)*imgA->widthStep + j  ]
                + roiA[ (ib)*imgA->widthStep + jr ]
                - roiA[ (it)*imgA->widthStep + jl ]
                - 2*roiA[ (it)*imgA->widthStep + j  ]
                - roiA[ (it)*imgA->widthStep + jr ])/8  ;

            DerT[ i*lImageWidth + j ] = (float)
                (roiB[i*imgB->widthStep + j] - roiA[i*imgA->widthStep + j]);
        }
    }
for( usePrevious = 0; usePrevious < 2; usePrevious++ )
{
/****************************************************************************************\
*                    Cases                                                               *
\****************************************************************************************/
    for ( k = 0; k < 4; k++ )
    {
        switch (k)
        {
        case 0:
            {
                criteria.type = CV_TERMCRIT_ITER;
                criteria.max_iter = 3;

                trsWrite( ATS_LST|ATS_CON,
                         "usePrevious = %d, criteria = ITER, max_iter = %d\n",
                         usePrevious, criteria.max_iter);

                break;
            }
        case 1:
            {
                criteria.type = CV_TERMCRIT_EPS;
                criteria.epsilon = 0.001f;
                trsWrite( ATS_LST|ATS_CON,
                         "usePrevious = %d, criteria = EPS, epsilon = %f\n",
                         usePrevious, criteria.epsilon);

                break;
            }
        case 2:
            {
                criteria.type = CV_TERMCRIT_EPS | CV_TERMCRIT_ITER;
                criteria.epsilon = 0.0001f;
                criteria.max_iter = 3;
                trsWrite( ATS_LST|ATS_CON,
                         "usePrevious = %d,"
                         "criteria = EPS|ITER,"
                         "epsilon = %f, max_iter = %d\n",
                         usePrevious, criteria.epsilon, criteria.max_iter);

                break;
            }
        case 3:
            {
                criteria.type = CV_TERMCRIT_EPS | CV_TERMCRIT_ITER;
                criteria.epsilon = 0.00001f;
                criteria.max_iter = 100;
                trsWrite( ATS_LST|ATS_CON,
                         "usePrevious = %d,"
                         "criteria = EPS|ITER,"
                         "epsilon = %f, max_iter = %d\n",
                         usePrevious, criteria.epsilon, criteria.max_iter);

                break;
            }
        }
        Stop = 0;
        
        /* Run CVL function */
        cvCalcOpticalFlowHS( imgA , imgB, usePrevious,
                             testVelocityX, testVelocityY,
                             lambda, criteria );

        /* Calc by other way */
        if (!usePrevious)
        {
            /* Filling initial velocity with zero */
            for (i = 0; i < lImageWidth * lImageHeight; i++ )
            {
                VelocityX[i] = 0 ;
                VelocityY[i] = 0 ;
            }
        }
        iteration = 0;
        while ( !Stop )
        {
            float* oldX;
            float* oldY;
            float* newX;
            float* newY;

            iteration++;

            if ( iteration & 1 )
            {
                oldX = VelocityX;
                oldY = VelocityY;
                newX = auxVelocityX;
                newY = auxVelocityY;
            }
            else
            {
                oldX = auxVelocityX;
                oldY = auxVelocityY;
                newX = VelocityX;
                newY = VelocityY;
            }

            for( i = 0; i < lImageHeight; i++)
            {
                for(j = 0; j< lImageWidth; j++)
                {
                    float aveX = 0;
                    float aveY = 0;
                    float dx,dy,dt;

                    aveX +=(j==0) ? oldX[ i*lImageWidth + j ] : oldX[ i*lImageWidth + j-1 ];
                    aveX +=(j==lImageWidth-1) ? oldX[ i*lImageWidth + j ] :
                                              oldX[ i*lImageWidth + j+1 ];
                    aveX +=(i==0) ? oldX[ i*lImageWidth + j ] : oldX[ (i-1)*lImageWidth + j ];
                    aveX +=(i==lImageHeight-1) ? oldX[ i*lImageWidth + j ] :
                                               oldX[ (i+1)*lImageWidth + j ];
                    aveX /=4;

                    aveY +=(j==0) ? oldY[ i*lImageWidth + j ] : oldY[ i*lImageWidth + j-1 ];
                    aveY +=(j==lImageWidth-1) ? oldY[ i*lImageWidth + j ] :
                                              oldY[ i*lImageWidth + j+1 ];
                    aveY +=(i==0) ? oldY[ i*lImageWidth + j ] : oldY[ (i-1)*lImageWidth + j ];
                    aveY +=(i==lImageHeight-1) ? oldY[ i*lImageWidth + j ] :
                                               oldY[ (i+1)*lImageWidth + j ];
                    aveY /=4;

                    dx = DerX[ i*lImageWidth + j ];
                    dy = DerY[ i*lImageWidth + j ];
                    dt = DerT[ i*lImageWidth + j ];

                    /* Horn & Schunck pure formulas */
                    newX[ i*lImageWidth + j ] = aveX - ( dx * aveX +
                                                       dy * aveY + dt ) * lambda * dx /
                                                       (1 + lambda * ( dx*dx + dy*dy ));

                    newY[ i*lImageWidth + j ] = aveY - ( dx * aveX +
                                                       dy * aveY + dt ) * lambda * dy /
                                                       (1 + lambda * ( dx*dx + dy*dy ));
                }
            }
            /* evaluate epsilon */
            epsilon = 0;
            for ( i = 0; i < lImageHeight; i++)
            {
                for ( j = 0; j < lImageWidth; j++)
                {
                    epsilon = MAX((float)fabs(newX[i*lImageWidth + j]
                                              - oldX[i*lImageWidth + j]), epsilon );
                    epsilon = MAX((float)fabs(newY[i*lImageWidth + j]
                                              - oldY[i*lImageWidth + j]), epsilon );
                }
            }

            switch (criteria.type)
            {
            case CV_TERMCRIT_ITER:
                Stop = (criteria.max_iter == iteration );break;
            case CV_TERMCRIT_EPS:
                Stop = (criteria.epsilon > epsilon );break;
            case CV_TERMCRIT_ITER|CV_TERMCRIT_EPS:
                Stop = ( ( criteria.epsilon > epsilon    ) ||
                         ( criteria.max_iter == iteration ));
                break;
            }
            if (Stop)
            {
                if ( (newX != VelocityX) && (newY != VelocityY) )
                {
                    memcpy( VelocityX, newX, lImageWidth * lImageHeight * sizeof(float) );
                    memcpy( VelocityY, newY, lImageWidth * lImageHeight * sizeof(float) );
                }
            }
        }
        trsWrite( ATS_LST|ATS_CON,
                         "%d iterations are made\n", iteration );

        for( i = 0; i < lImageHeight; i++)
        {
            for(j = 0; j< lImageWidth; j++)
            {
                float tvx = ((float*)(testVelocityX->imageData + i*testVelocityX->widthStep))[j];
                float tvy = ((float*)(testVelocityY->imageData + i*testVelocityY->widthStep))[j];

                if (( fabs( tvx - VelocityX[i*lImageWidth + j])>EPSILON )||
                    ( fabs( tvy - VelocityY[i*lImageWidth + j])>EPSILON ) )
                {
                    //trsWrite( ATS_LST | ATS_CON, " ValueX %f \n",
                    //          testVelocityX[i*lROIWidth + j] );
                    //trsWrite( ATS_LST | ATS_CON, " mustX  %f \n",
                    //          VelocityX[i*lROIWidth + j] );

                    //trsWrite( ATS_LST | ATS_CON, " ValueY %f \n",
                    //          testVelocityY[i*lROIWidth + j] );
                    //trsWrite( ATS_LST | ATS_CON, " mustY  %f \n",
                    //          VelocityY[i*lROIWidth + j] );

                    //trsWrite( ATS_LST | ATS_CON, " Coordinates %d %d\n", i, j );

                    lErrors++;
                }
            }
        }
    }/* for */
    /* Filling initial velocity with zero */
    cvZero( testVelocityX );
    cvZero( testVelocityY );
    for (i = 0; i < lImageWidth * lImageHeight; i++ )
    {
        VelocityX[i] = 0 ;
        VelocityY[i] = 0 ;
    }
}

    /* Free memory */
    cvFree( &VelocityX );
    cvFree( &VelocityY );
    cvFree( &auxVelocityX );
    cvFree( &auxVelocityY );


    cvFree( &DerX );
    cvFree( &DerY );
    cvFree( &DerT );

    cvReleaseImage( &imgA );
    cvReleaseImage( &imgB );
    cvReleaseImage( &testVelocityX );
    cvReleaseImage( &testVelocityY );


    if( lErrors == 0 ) return trsResult( TRS_OK, "No errors fixed for this text" );
    else return trsResult( TRS_FAIL, "Total fixed %d errors", lErrors );
} /*fmaCalcOpticalFlowHS*/

void InitACalcOpticalFlowHS( void )
{
    /* Registering test function */
    trsReg( FuncName, TestName, TestClass, fmaCalcOpticalFlowHS );
} /* InitACalcOpticalFlowHS */

#endif

/* End of file. */
