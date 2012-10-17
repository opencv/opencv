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

typedef struct
{
    float xx;
    float xy;
    float yy;
    float xt;
    float yt;
}
icvDerProduct;


#define CONV( A, B, C)  ((float)( A +  (B<<1)  + C ))
/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvCalcOpticalFlowLK_8u32fR ( Lucas & Kanade method )
//    Purpose: calculate Optical flow for 2 images using Lucas & Kanade algorithm
//    Context:
//    Parameters:
//            imgA,         // pointer to first frame ROI
//            imgB,         // pointer to second frame ROI
//            imgStep,      // width of single row of source images in bytes
//            imgSize,      // size of the source image ROI
//            winSize,      // size of the averaging window used for grouping
//            velocityX,    // pointer to horizontal and
//            velocityY,    // vertical components of optical flow ROI
//            velStep       // width of single row of velocity frames in bytes
//
//    Returns: CV_OK         - all ok
//             CV_OUTOFMEM_ERR  - insufficient memory for function work
//             CV_NULLPTR_ERR - if one of input pointers is NULL
//             CV_BADSIZE_ERR   - wrong input sizes interrelation
//
//    Notes:  1.Optical flow to be computed for every pixel in ROI
//            2.For calculating spatial derivatives we use 3x3 Sobel operator.
//            3.We use the following border mode.
//              The last row or column is replicated for the border
//              ( IPL_BORDER_REPLICATE in IPL ).
//
//
//F*/
static CvStatus CV_STDCALL
icvCalcOpticalFlowLK_8u32fR( uchar * imgA,
                             uchar * imgB,
                             int imgStep,
                             CvSize imgSize,
                             CvSize winSize,
                             float *velocityX,
                             float *velocityY, int velStep )
{
    /* Loops indexes */
    int i, j, k;

    /* Gaussian separable kernels */
    float GaussX[16];
    float GaussY[16];
    float *KerX;
    float *KerY;

    /* Buffers for Sobel calculations */
    float *MemX[2];
    float *MemY[2];

    float ConvX, ConvY;
    float GradX, GradY, GradT;

    int winWidth = winSize.width;
    int winHeight = winSize.height;

    int imageWidth = imgSize.width;
    int imageHeight = imgSize.height;

    int HorRadius = (winWidth - 1) >> 1;
    int VerRadius = (winHeight - 1) >> 1;

    int PixelLine;
    int ConvLine;

    int BufferAddress;

    int BufferHeight = 0;
    int BufferWidth;
    int BufferSize;

    /* buffers derivatives product */
    icvDerProduct *II;

    /* buffers for gaussian horisontal convolution */
    icvDerProduct *WII;

    /* variables for storing number of first pixel of image line */
    int Line1;
    int Line2;
    int Line3;

    /* we must have 2*2 linear system coeffs
       | A1B2  B1 |  {u}   {C1}   {0}
       |          |  { } + {  } = { }
       | A2  A1B2 |  {v}   {C2}   {0}
     */
    float A1B2, A2, B1, C1, C2;

    int pixNumber;

    /* auxiliary */
    int NoMem = 0;

    velStep /= sizeof(velocityX[0]);

    /* Checking bad arguments */
    if( imgA == NULL )
        return CV_NULLPTR_ERR;
    if( imgB == NULL )
        return CV_NULLPTR_ERR;

    if( imageHeight < winHeight )
        return CV_BADSIZE_ERR;
    if( imageWidth < winWidth )
        return CV_BADSIZE_ERR;

    if( winHeight >= 16 )
        return CV_BADSIZE_ERR;
    if( winWidth >= 16 )
        return CV_BADSIZE_ERR;

    if( !(winHeight & 1) )
        return CV_BADSIZE_ERR;
    if( !(winWidth & 1) )
        return CV_BADSIZE_ERR;

    BufferHeight = winHeight;
    BufferWidth = imageWidth;

    /****************************************************************************************/
    /* Computing Gaussian coeffs                                                            */
    /****************************************************************************************/
    GaussX[0] = 1;
    GaussY[0] = 1;
    for( i = 1; i < winWidth; i++ )
    {
        GaussX[i] = 1;
        for( j = i - 1; j > 0; j-- )
        {
            GaussX[j] += GaussX[j - 1];
        }
    }
    for( i = 1; i < winHeight; i++ )
    {
        GaussY[i] = 1;
        for( j = i - 1; j > 0; j-- )
        {
            GaussY[j] += GaussY[j - 1];
        }
    }
    KerX = &GaussX[HorRadius];
    KerY = &GaussY[VerRadius];

    /****************************************************************************************/
    /* Allocating memory for all buffers                                                    */
    /****************************************************************************************/
    for( k = 0; k < 2; k++ )
    {
        MemX[k] = (float *) cvAlloc( (imgSize.height) * sizeof( float ));

        if( MemX[k] == NULL )
            NoMem = 1;
        MemY[k] = (float *) cvAlloc( (imgSize.width) * sizeof( float ));

        if( MemY[k] == NULL )
            NoMem = 1;
    }

    BufferSize = BufferHeight * BufferWidth;

    II = (icvDerProduct *) cvAlloc( BufferSize * sizeof( icvDerProduct ));
    WII = (icvDerProduct *) cvAlloc( BufferSize * sizeof( icvDerProduct ));


    if( (II == NULL) || (WII == NULL) )
        NoMem = 1;

    if( NoMem )
    {
        for( k = 0; k < 2; k++ )
        {
            if( MemX[k] )
                cvFree( &MemX[k] );

            if( MemY[k] )
                cvFree( &MemY[k] );
        }
        if( II )
            cvFree( &II );
        if( WII )
            cvFree( &WII );

        return CV_OUTOFMEM_ERR;
    }

    /****************************************************************************************/
    /*        Calculate first line of memX and memY                                         */
    /****************************************************************************************/
    MemY[0][0] = MemY[1][0] = CONV( imgA[0], imgA[0], imgA[1] );
    MemX[0][0] = MemX[1][0] = CONV( imgA[0], imgA[0], imgA[imgStep] );

    for( j = 1; j < imageWidth - 1; j++ )
    {
        MemY[0][j] = MemY[1][j] = CONV( imgA[j - 1], imgA[j], imgA[j + 1] );
    }

    pixNumber = imgStep;
    for( i = 1; i < imageHeight - 1; i++ )
    {
        MemX[0][i] = MemX[1][i] = CONV( imgA[pixNumber - imgStep],
                                        imgA[pixNumber], imgA[pixNumber + imgStep] );
        pixNumber += imgStep;
    }

    MemY[0][imageWidth - 1] =
        MemY[1][imageWidth - 1] = CONV( imgA[imageWidth - 2],
                                        imgA[imageWidth - 1], imgA[imageWidth - 1] );

    MemX[0][imageHeight - 1] =
        MemX[1][imageHeight - 1] = CONV( imgA[pixNumber - imgStep],
                                         imgA[pixNumber], imgA[pixNumber] );


    /****************************************************************************************/
    /*    begin scan image, calc derivatives and solve system                               */
    /****************************************************************************************/

    PixelLine = -VerRadius;
    ConvLine = 0;
    BufferAddress = -BufferWidth;

    while( PixelLine < imageHeight )
    {
        if( ConvLine < imageHeight )
        {
            /*Here we calculate derivatives for line of image */
            int address;

            i = ConvLine;
            int L1 = i - 1;
            int L2 = i;
            int L3 = i + 1;

            int memYline = L3 & 1;

            if( L1 < 0 )
                L1 = 0;
            if( L3 >= imageHeight )
                L3 = imageHeight - 1;

            BufferAddress += BufferWidth;
            BufferAddress -= ((BufferAddress >= BufferSize) ? 0xffffffff : 0) & BufferSize;

            address = BufferAddress;

            Line1 = L1 * imgStep;
            Line2 = L2 * imgStep;
            Line3 = L3 * imgStep;

            /* Process first pixel */
            ConvX = CONV( imgA[Line1 + 1], imgA[Line2 + 1], imgA[Line3 + 1] );
            ConvY = CONV( imgA[Line3], imgA[Line3], imgA[Line3 + 1] );

            GradY = ConvY - MemY[memYline][0];
            GradX = ConvX - MemX[1][L2];

            MemY[memYline][0] = ConvY;
            MemX[1][L2] = ConvX;

            GradT = (float) (imgB[Line2] - imgA[Line2]);

            II[address].xx = GradX * GradX;
            II[address].xy = GradX * GradY;
            II[address].yy = GradY * GradY;
            II[address].xt = GradX * GradT;
            II[address].yt = GradY * GradT;
            address++;
            /* Process middle of line */
            for( j = 1; j < imageWidth - 1; j++ )
            {
                ConvX = CONV( imgA[Line1 + j + 1], imgA[Line2 + j + 1], imgA[Line3 + j + 1] );
                ConvY = CONV( imgA[Line3 + j - 1], imgA[Line3 + j], imgA[Line3 + j + 1] );

                GradY = ConvY - MemY[memYline][j];
                GradX = ConvX - MemX[(j - 1) & 1][L2];

                MemY[memYline][j] = ConvY;
                MemX[(j - 1) & 1][L2] = ConvX;

                GradT = (float) (imgB[Line2 + j] - imgA[Line2 + j]);

                II[address].xx = GradX * GradX;
                II[address].xy = GradX * GradY;
                II[address].yy = GradY * GradY;
                II[address].xt = GradX * GradT;
                II[address].yt = GradY * GradT;

                address++;
            }
            /* Process last pixel of line */
            ConvX = CONV( imgA[Line1 + imageWidth - 1], imgA[Line2 + imageWidth - 1],
                          imgA[Line3 + imageWidth - 1] );

            ConvY = CONV( imgA[Line3 + imageWidth - 2], imgA[Line3 + imageWidth - 1],
                          imgA[Line3 + imageWidth - 1] );


            GradY = ConvY - MemY[memYline][imageWidth - 1];
            GradX = ConvX - MemX[(imageWidth - 2) & 1][L2];

            MemY[memYline][imageWidth - 1] = ConvY;

            GradT = (float) (imgB[Line2 + imageWidth - 1] - imgA[Line2 + imageWidth - 1]);

            II[address].xx = GradX * GradX;
            II[address].xy = GradX * GradY;
            II[address].yy = GradY * GradY;
            II[address].xt = GradX * GradT;
            II[address].yt = GradY * GradT;
            address++;

            /* End of derivatives for line */

            /****************************************************************************************/
            /* ---------Calculating horizontal convolution of processed line----------------------- */
            /****************************************************************************************/
            address -= BufferWidth;
            /* process first HorRadius pixels */
            for( j = 0; j < HorRadius; j++ )
            {
                int jj;

                WII[address].xx = 0;
                WII[address].xy = 0;
                WII[address].yy = 0;
                WII[address].xt = 0;
                WII[address].yt = 0;

                for( jj = -j; jj <= HorRadius; jj++ )
                {
                    float Ker = KerX[jj];

                    WII[address].xx += II[address + jj].xx * Ker;
                    WII[address].xy += II[address + jj].xy * Ker;
                    WII[address].yy += II[address + jj].yy * Ker;
                    WII[address].xt += II[address + jj].xt * Ker;
                    WII[address].yt += II[address + jj].yt * Ker;
                }
                address++;
            }
            /* process inner part of line */
            for( j = HorRadius; j < imageWidth - HorRadius; j++ )
            {
                int jj;
                float Ker0 = KerX[0];

                WII[address].xx = 0;
                WII[address].xy = 0;
                WII[address].yy = 0;
                WII[address].xt = 0;
                WII[address].yt = 0;

                for( jj = 1; jj <= HorRadius; jj++ )
                {
                    float Ker = KerX[jj];

                    WII[address].xx += (II[address - jj].xx + II[address + jj].xx) * Ker;
                    WII[address].xy += (II[address - jj].xy + II[address + jj].xy) * Ker;
                    WII[address].yy += (II[address - jj].yy + II[address + jj].yy) * Ker;
                    WII[address].xt += (II[address - jj].xt + II[address + jj].xt) * Ker;
                    WII[address].yt += (II[address - jj].yt + II[address + jj].yt) * Ker;
                }
                WII[address].xx += II[address].xx * Ker0;
                WII[address].xy += II[address].xy * Ker0;
                WII[address].yy += II[address].yy * Ker0;
                WII[address].xt += II[address].xt * Ker0;
                WII[address].yt += II[address].yt * Ker0;

                address++;
            }
            /* process right side */
            for( j = imageWidth - HorRadius; j < imageWidth; j++ )
            {
                int jj;

                WII[address].xx = 0;
                WII[address].xy = 0;
                WII[address].yy = 0;
                WII[address].xt = 0;
                WII[address].yt = 0;

                for( jj = -HorRadius; jj < imageWidth - j; jj++ )
                {
                    float Ker = KerX[jj];

                    WII[address].xx += II[address + jj].xx * Ker;
                    WII[address].xy += II[address + jj].xy * Ker;
                    WII[address].yy += II[address + jj].yy * Ker;
                    WII[address].xt += II[address + jj].xt * Ker;
                    WII[address].yt += II[address + jj].yt * Ker;
                }
                address++;
            }
        }

        /****************************************************************************************/
        /*  Calculating velocity line                                                           */
        /****************************************************************************************/
        if( PixelLine >= 0 )
        {
            int USpace;
            int BSpace;
            int address;

            if( PixelLine < VerRadius )
                USpace = PixelLine;
            else
                USpace = VerRadius;

            if( PixelLine >= imageHeight - VerRadius )
                BSpace = imageHeight - PixelLine - 1;
            else
                BSpace = VerRadius;

            address = ((PixelLine - USpace) % BufferHeight) * BufferWidth;
            for( j = 0; j < imageWidth; j++ )
            {
                int addr = address;

                A1B2 = 0;
                A2 = 0;
                B1 = 0;
                C1 = 0;
                C2 = 0;

                for( i = -USpace; i <= BSpace; i++ )
                {
                    A2 += WII[addr + j].xx * KerY[i];
                    A1B2 += WII[addr + j].xy * KerY[i];
                    B1 += WII[addr + j].yy * KerY[i];
                    C2 += WII[addr + j].xt * KerY[i];
                    C1 += WII[addr + j].yt * KerY[i];

                    addr += BufferWidth;
                    addr -= ((addr >= BufferSize) ? 0xffffffff : 0) & BufferSize;
                }
                /****************************************************************************************\
                * Solve Linear System                                                                    *
                \****************************************************************************************/
                {
                    float delta = (A1B2 * A1B2 - A2 * B1);

                    if( delta )
                    {
                        /* system is not singular - solving by Kramer method */
                        float deltaX;
                        float deltaY;
                        float Idelta = 8 / delta;

                        deltaX = -(C1 * A1B2 - C2 * B1);
                        deltaY = -(A1B2 * C2 - A2 * C1);

                        velocityX[j] = deltaX * Idelta;
                        velocityY[j] = deltaY * Idelta;
                    }
                    else
                    {
                        /* singular system - find optical flow in gradient direction */
                        float Norm = (A1B2 + A2) * (A1B2 + A2) + (B1 + A1B2) * (B1 + A1B2);

                        if( Norm )
                        {
                            float IGradNorm = 8 / Norm;
                            float temp = -(C1 + C2) * IGradNorm;

                            velocityX[j] = (A1B2 + A2) * temp;
                            velocityY[j] = (B1 + A1B2) * temp;

                        }
                        else
                        {
                            velocityX[j] = 0;
                            velocityY[j] = 0;
                        }
                    }
                }
                /****************************************************************************************\
                * End of Solving Linear System                                                           *
                \****************************************************************************************/
            }                   /*for */
            velocityX += velStep;
            velocityY += velStep;
        }                       /*for */
        PixelLine++;
        ConvLine++;
    }

    /* Free memory */
    for( k = 0; k < 2; k++ )
    {
        cvFree( &MemX[k] );
        cvFree( &MemY[k] );
    }
    cvFree( &II );
    cvFree( &WII );

    return CV_OK;
} /*icvCalcOpticalFlowLK_8u32fR*/


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvCalcOpticalFlowLK
//    Purpose: Optical flow implementation
//    Context:
//    Parameters:
//             srcA, srcB - source image
//             velx, vely - destination image
//    Returns:
//
//    Notes:
//F*/
CV_IMPL void
cvCalcOpticalFlowLK( const void* srcarrA, const void* srcarrB, CvSize winSize,
                     void* velarrx, void* velarry )
{
    CvMat stubA, *srcA = cvGetMat( srcarrA, &stubA );
    CvMat stubB, *srcB = cvGetMat( srcarrB, &stubB );
    CvMat stubx, *velx = cvGetMat( velarrx, &stubx );
    CvMat stuby, *vely = cvGetMat( velarry, &stuby );

    if( !CV_ARE_TYPES_EQ( srcA, srcB ))
        CV_Error( CV_StsUnmatchedFormats, "Source images have different formats" );

    if( !CV_ARE_TYPES_EQ( velx, vely ))
        CV_Error( CV_StsUnmatchedFormats, "Destination images have different formats" );

    if( !CV_ARE_SIZES_EQ( srcA, srcB ) ||
        !CV_ARE_SIZES_EQ( velx, vely ) ||
        !CV_ARE_SIZES_EQ( srcA, velx ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( CV_MAT_TYPE( srcA->type ) != CV_8UC1 ||
        CV_MAT_TYPE( velx->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat, "Source images must have 8uC1 type and "
                                           "destination images must have 32fC1 type" );

    if( srcA->step != srcB->step || velx->step != vely->step )
        CV_Error( CV_BadStep, "source and destination images have different step" );

    IPPI_CALL( icvCalcOpticalFlowLK_8u32fR( (uchar*)srcA->data.ptr, (uchar*)srcB->data.ptr,
                                            srcA->step, cvGetMatSize( srcA ), winSize,
                                            velx->data.fl, vely->data.fl, velx->step ));
}

/* End of file. */
