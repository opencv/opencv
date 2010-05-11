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
static char FuncName[] = "cvCalcOpticalFlowLK";
static char TestName[] = "Optical flow (Lucas & Kanade)";
static char TestClass[] = "Algorithm";

static long lImageWidth;
static long lImageHeight;
static long lWinWidth;
static long lWinHeight;

#define EPSILON 0.00001f

static int fmaCalcOpticalFlowLK( void )
{
    /* Some Variables */
    int* WH = NULL;
    int* WV = NULL;

    int W3[3] =  { 1,  2, 1 };
    int W5[5] =  { 1,  4,  6,  4, 1 };
    int W7[7] =  { 1,  6,  15, 20, 15, 6, 1 };
    int W9[9] =  { 1,  8,  28, 56, 70, 56, 28, 8, 1 };
    int W11[11] = {1, 10,  45, 120, 210, 252, 210, 120, 45, 10, 1 };

    int            i,j,m,k;
    uchar*         roiA;
    uchar*         roiB;

    float*         VelocityX;
    float*         VelocityY;

    float*         DerivativeX;
    float*         DerivativeY;
    float*         DerivativeT;

    long           lErrors = 0;
    CvSize        winSize;

    int HRad;
    int VRad;

    float A1, A2, B1, B2, C1, C2;

    static int  read_param = 0;

    /* Initialization global parameters */
    if( !read_param )
    {
        read_param = 1;
        /* Reading test-parameters */
        trslRead( &lImageHeight, "563", "Image height" );
        trslRead( &lImageWidth, "345", "Image width" );
        trslRead( &lWinHeight, "7", "win height 3/5/7/9/11 " );
        trslRead( &lWinWidth, "9", "win width 3/5/7/9/11 " );
    }

    /* Checking all sizes of source histogram in ranges */
    IplImage* imgA = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_8U, 1 );
    IplImage* imgB = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_8U, 1 );

    IplImage* testVelocityX = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_32F, 1 );
    IplImage* testVelocityY = cvCreateImage( cvSize(lImageWidth,lImageHeight), IPL_DEPTH_32F, 1 );

    VelocityX = (float*)cvAlloc(  lImageWidth * lImageHeight * sizeof(float) );
    VelocityY = (float*)cvAlloc(  lImageWidth * lImageHeight * sizeof(float) );

    DerivativeX = (float*)cvAlloc( lImageWidth * lImageHeight * sizeof(float) );
    DerivativeY = (float*)cvAlloc( lImageWidth * lImageHeight * sizeof(float) );
    DerivativeT = (float*)cvAlloc( lImageWidth * lImageHeight * sizeof(float) );

    winSize.height = lWinHeight;
    winSize.width =  lWinWidth;

    switch (lWinHeight)
    {
    case 3:
        WV = W3; break;
    case 5:
        WV = W5; break;
    case 7:
        WV = W7; break;
    case 9:
        WV = W9; break;
    case 11:
        WV = W11; break;
    }
    switch (lWinWidth)
    {
    case 3:
        WH = W3; break;
    case 5:
        WH = W5; break;
    case 7:
        WH = W7; break;
    case 9:
        WH = W9; break;
    case 11:
        WH = W11; break;

    }

    HRad = (winSize.width - 1)/2;
    VRad = (winSize.height - 1)/2;

    /* Filling images */
    ats1bInitRandom( 0, 255, (uchar*)imgA->imageData, lImageHeight * imgA->widthStep );
    ats1bInitRandom( 0, 255, (uchar*)imgB->imageData, imgA->widthStep * lImageHeight );

    /* Run CVL function */
    cvCalcOpticalFlowLK( imgA , imgB, winSize,
                         testVelocityX, testVelocityY );


    /* Calc by other way */
    roiA = (uchar*)imgA->imageData;
    roiB = (uchar*)imgB->imageData;  

    /* Calculate derivatives */
    for (i=0; i<imgA->height; i++)
    {
        for(j=0; j<imgA->width; j++)
        {
            int jr,jl,it,ib;

            if ( j==imgA->width-1 )
                jr = imgA->width-1;
            else jr = j + 1;

            if ( j==0 )
                jl = 0;
            else jl = j - 1;

            if ( i==(imgA->height - 1) )
                ib = imgA->height - 1;
            else ib = i + 1;

            if ( i==0 )
                it = 0;
            else it = i - 1;

            DerivativeX[ i*lImageWidth + j ] = (float)
                (roiA[ (it)*imgA->widthStep + jr ]
                - roiA[ (it)*imgA->widthStep + jl ]
                + 2*roiA[ (i)*imgA->widthStep + jr ]
                - 2*roiA[ (i)*imgA->widthStep + jl ]
                + roiA[ (ib)*imgA->widthStep + jr ]
                - roiA[ (ib)*imgA->widthStep + jl ]) ;

            DerivativeY[ i*lImageWidth + j ] = (float)
                ( roiA[ (ib)*imgA->widthStep + jl ]
                + 2*roiA[ (ib)*imgA->widthStep + j  ]
                + roiA[ (ib)*imgA->widthStep + jr ]
                - roiA[ (it)*imgA->widthStep + jl ]
                - 2*roiA[ (it)*imgA->widthStep + j  ]
                - roiA[ (it)*imgA->widthStep + jr ])  ;

            DerivativeT[ i*lImageWidth + j ] = (float)
                (roiB[i*imgB->widthStep + j] - roiA[i*imgA->widthStep + j])*8;
        }
    }

    for( i = 0; i < lImageHeight; i++)
    {
        for(j = 0; j< lImageWidth; j++)
        {
            A1 =0;
            A2 =0;
            B1 =0;
            B2 =0;
            C1= 0;
            C2= 0;

            for( k = -VRad ; k <= VRad ; k++ )
            {
                for( m = - HRad; m <= HRad ; m++ )
                {
                    int coord = (i+k)*lImageWidth + (j+m);
                    if ( (j+m<0)              || 
                         (j+m >lImageWidth-1) || 
                         ( (k+i)<0 )          || 
                         ( (k+i)>lImageHeight-1) )
                    {continue;}

                    A1+=WV[k+VRad]*WH[m+HRad]* DerivativeX[coord]*DerivativeY[coord];
                    A2+=WV[k+VRad]*WH[m+HRad]* DerivativeX[coord]*DerivativeX[coord];
                    B1+=WV[k+VRad]*WH[m+HRad]* DerivativeY[coord]*DerivativeY[coord];
                    B2+=WV[k+VRad]*WH[m+HRad]* DerivativeX[coord]*DerivativeY[coord];
                    C1+=WV[k+VRad]*WH[m+HRad]* DerivativeY[coord]*DerivativeT[coord];
                    C2+=WV[k+VRad]*WH[m+HRad]* DerivativeX[coord]*DerivativeT[coord];
                }
            }
            if (A1*B2 - A2*B1)
            {
                VelocityX[i*lImageWidth + j] = - (C1*B2 - C2*B1)/(A1*B2 - A2*B1);
                VelocityY[i*lImageWidth + j] = - (A1*C2 - A2*C1)/(A1*B2 - A2*B1);
            }
            else if ( (A1+A2)*(A1+A2) + (B1+B2)*(B1+B2) )
            {   /* Calculate Normal flow */
                VelocityX[i*lImageWidth + j] = -(A1+A2)*(C1+C2)/((A1+A2)*(A1+A2)+(B1+B2)*(B1+B2));
                VelocityY[i*lImageWidth + j] = -(B1+B2)*(C1+C2)/((A1+A2)*(A1+A2)+(B1+B2)*(B1+B2));
            }
            else
            {
                VelocityX[i*lImageWidth + j] = 0;
                VelocityY[i*lImageWidth + j] = 0;
            }
        }
    }

    for( i = 0; i < lImageHeight; i++)
    {
        for(j = 0; j< lImageWidth; j++)
        {
            float tvx = ((float*)(testVelocityX->imageData + i*testVelocityX->widthStep))[j];
            float tvy = ((float*)(testVelocityY->imageData + i*testVelocityY->widthStep))[j];



            if (( fabs(tvx - VelocityX[i*lImageWidth + j])>EPSILON )||
                ( fabs(tvy - VelocityY[i*lImageWidth + j])>EPSILON ) )
            {
                //trsWrite( ATS_LST | ATS_CON, " ValueX %f \n", tvx );
                //trsWrite( ATS_LST | ATS_CON, " mustX  %f \n", VelocityX[i*lImageWidth + j] );

                //trsWrite( ATS_LST | ATS_CON, " ValueY %f \n", tvy );
                //trsWrite( ATS_LST | ATS_CON, " mustY  %f \n", VelocityY[i*lImageWidth + j] );

                //trsWrite( ATS_LST | ATS_CON, " Coordinates %d %d\n", i, j );

                lErrors++;
            }
        }
    }
    cvFree( &VelocityX );
    cvFree( &VelocityY );
    
    cvFree( &DerivativeX );
    cvFree( &DerivativeY );
    cvFree( &DerivativeT );

    cvReleaseImage( &imgA );
    cvReleaseImage( &imgB );
    cvReleaseImage( &testVelocityX );
    cvReleaseImage( &testVelocityY );


    if( lErrors == 0 ) return trsResult( TRS_OK, "No errors fixed for this text" );
    else return trsResult( TRS_FAIL, "Total fixed %d errors", lErrors );
} /*fmaCalcOpticalFlowLK*/

void InitACalcOpticalFlowLK( void )
{
    /* Registering test function */
    trsReg( FuncName, TestName, TestClass, fmaCalcOpticalFlowLK );
} /* InitAACalcOpticalFlowLK */

#endif

/* End of file. */
