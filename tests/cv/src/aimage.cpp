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

#define DEPTH_8U 0

/* Testing parameters */
static char test_desc[] = "Image Creation & access";
static char func_name[] = "cvCreateImage cvCreateImageHeader cvSetImageROI cvGetImageROI "
                          "cvSetImageCOI cvCreateImageData cvReleaseImageData "
                          "cvSetImageData cvCloneImage cvCopyImage cvInitImageHeader";

static int depths[] = { IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                        IPL_DEPTH_32S, IPL_DEPTH_32F, IPL_DEPTH_64F, 0};
static int channels[] = {1, 2, 3, 4, 0};

static char* imageData = (char*)cvAlloc(10000);

const int align = 4;

static int foaImage( void )
{
    CvSize size = cvSize(320, 200);
    int i, j;
    int Errors = 0;
    //Creating new image with different channels & depths
    for( i = 0; depths[i] != 0; i++ )  // cycle for depths
        for(j = 0; channels[j] != 0; j++)  // cycle for channels
        {
            if( depths[i] == IPL_DEPTH_1U && channels[j] != 1 ) // skip for IPL_DEPTH_1U
                continue;                                       // all non 1 channels
            IplImage* image = cvCreateImage( size, depths[i], channels[j] );
            if( image->width != size.width || image->height != size.height )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCreateImage: Size mismatch: act %d x %d      exp %d x %d\n",
                          image->width, image->height, size.width, size.height );
                Errors++;
            }
            if( size.width * (depths[i] & IPL_DEPTH_MASK) / 8 > image->widthStep ||
                (image->widthStep & 3) )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCreateImage: Wrong widthStep: act %d\n",
                          image->widthStep );
                Errors++;
            }

            cvReleaseImage( &image );
        }
    trsWrite( ATS_CON, "cvCreateImage: ... done\n" );

    //Creating new image header with different channels & depths
    for( i = 0; depths[i] != 0; i++ )  // cycle for depths
        for(j = 0; channels[j] != 0; j++)  // cycle for channels
        {
            if( depths[i] == IPL_DEPTH_1U && channels[j] != 1 ) // skip for IPL_DEPTH_1U
                continue;                                       // all non 1 channels
            if( depths[i] == (int)IPL_DEPTH_8S )
                continue;
            IplImage* image = cvCreateImageHeader( size, depths[i], channels[j] );
            if( image->width != size.width || image->height != size.height )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCreateImageHeader: Size mismatch: act %d x %d      exp %d x %d\n",
                          image->width, image->height, size.width, size.height );
                Errors++;
            }
            if( size.width * (depths[i] & IPL_DEPTH_MASK) / 8 > image->widthStep ||
                (image->widthStep & 3) )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCreateImageHeader: Wrong widthStep: act %d\n",
                          image->widthStep );
                Errors++;
            }
            if( image->imageData )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCreateImageHeader: imageData created :(\n" );
                Errors++;
            }

            cvSetImageROI( image, cvRect(1, 1, size.width - 1, size.height - 1) );
            if( image->roi->coi )
            {
                trsWrite( ATS_CON | ATS_LST, "cvSetImageROI: coi non zero\n" );
                Errors++;
            }

            CvRect rect = cvGetImageROI( image );
            if( rect.x != 1 || rect.y != 1 ||
                rect.width != size.width - 1 || rect.height != size.height - 1 )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvGetImageROI: wrong rect: act %d x %d x %d x %d       "
                          "exp %d x %d x %d x %d\n",
                          rect.x, rect.y, rect.width, rect.height,
                          1, 1, size.width - 1, size.height - 1 );
                Errors++;
            }

            cvSetImageCOI( image, 1 );
            if( image->roi->coi != 1 )
            {
                trsWrite( ATS_CON | ATS_LST, "cvSetImageCOI: soi non 1\n" );
                Errors++;
            }
            if( image->roi->xOffset != 1 || image->roi->yOffset != 1 ||
                image->roi->width != size.width  - 1 ||
                image->roi->height != size.height - 1)
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCreateImageHeader: Size mismatch: act %d x %d x %d x %d"
                          "exp %d x %d x %d x %d\n",
                          image->roi->xOffset, image->roi->yOffset,
                          image->roi->width, image->roi->height,
                          1, 1, size.width - 1, size.height - 1 );
                Errors++;
            }

            cvCreateImageData( image );
            if( !image->imageData )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCreateImageData: Wow :)... where is imageData ?....\n" );
                Errors++;
            }

            cvReleaseImageData( image );
            if( image->imageData )
            {
                trsWrite( ATS_CON | ATS_LST, "cvReleaseImageData: imageData non zero :(\n" );
                Errors++;
            }

            cvSetImageData( image, imageData, size.width * channels[j] * 8 ); // magic width step :)
            if( image->imageData != imageData )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvSetImageData: wrong pointer to imageData: act %x,  exp %x\n",
                          image->imageData, imageData );
                Errors++;
            }
            if( image->widthStep != size.width * channels[j] * 8 )
            {
                trsWrite( ATS_CON | ATS_LST, "cvSetImageData: wrong imageStep: act %d,   exp %d\n",
                          image->widthStep, size.width * channels[j] * 8 );
                Errors++;
            }

            cvReleaseImageHeader( &image );
        }
    trsWrite( ATS_CON, "cvCreateImageHeader: ... done\n" );
    trsWrite( ATS_CON, "cvSetImageROI: ... done\n" );
    trsWrite( ATS_CON, "cvGetImageROI: ... done\n" );
    trsWrite( ATS_CON, "cvSetImageCOI: ... done\n" );
    trsWrite( ATS_CON, "cvCreateImageData: ... done\n" );
    trsWrite( ATS_CON, "cvReleaseImageData: ... done\n" );
    trsWrite( ATS_CON, "cvSetImageData: ... done\n" );

    for( i = 0; depths[i] != 0; i++ )  // cycle for depths
        for(j = 0; channels[j] != 0; j++)  // cycle for channels
        {
            if( depths[i] == IPL_DEPTH_1U && channels[j] != 1 ) // skip for IPL_DEPTH_1U
                continue;                                       // all non 1 channels
            if( depths[i] == (int)IPL_DEPTH_8S )
                continue;
            IplImage* src = cvCreateImage( size, depths[i], channels[j] );
            //IplImage* dst = cvCreateImage( size, depths[i], channels[j] );
            IplImage* dst = 0;
            IplImage* clone = 0;

            cvSetImageROI( src, cvRect(1, 1, size.width - 1, size.height - 1) );

            for( int k = 0; k < src->widthStep * src->height; k++ )
                src->imageData[k] = (char)k;

            //cvCopy/*Image*/( src, dst );
            dst = cvCloneImage( src );
            clone = dst;

            if( clone->width != dst->width || clone->height != dst->height )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCopyImage/cvCloneImage: wrong destination size:"
                          "%d x %d  <>  %d %d\n",
                          clone->width, clone->height, dst->width, dst->height );
                Errors++;
            }
            if( clone->widthStep != src->widthStep )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCloneImage: wrong width step: act %d   exp %d\n",
                          clone->widthStep, src->widthStep );
                Errors++;
            }
            if( !clone->roi )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCloneImage: roi was lost\n" );
                Errors++;
            }
            else
            {
                if( clone->roi->xOffset != 1 || clone->roi->yOffset != 1 ||
                    clone->roi->width != size.width  - 1 ||
                    clone->roi->height != size.height - 1 )
                {
                    trsWrite( ATS_CON | ATS_LST,
                              "cvCloneImage: Size mismatch: act %d x %d x %d x %d"
                              "exp %d x %d x %d x %d\n",
                              clone->roi->xOffset, clone->roi->yOffset,
                              clone->roi->width, clone->roi->height,
                              1, 1, size.width - 1, size.height - 1 );
                    Errors++;
                }

            }
            if( depths[i] == IPL_DEPTH_32F )
            {
                src->depth = IPL_DEPTH_32S;
                dst->depth = IPL_DEPTH_32S;
            }
            else if( depths[i] == IPL_DEPTH_64F )
            {
                src->depth = IPL_DEPTH_32S;
                dst->depth = IPL_DEPTH_32S;

                src->width *= 2;
                dst->width *= 2;
            }

            if( cvNorm( src, dst, CV_L1 ) )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCopyImage: wrong destination image\n" );
                Errors++;
            }
            /*if( cvNorm( src, clone, CV_L1 ) )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCloneImage: wrong destination image\n" );
                Errors++;
            }*/

            cvReleaseImage( &src );
            cvReleaseImage( &clone );
        }
    trsWrite( ATS_CON, "cvCloneImage: ... done\n" );
    //trsWrite( ATS_CON, "cvCopyImage: ... done\n" );

    //Init new image header with different channels & depths
    for( i = 0; depths[i] != 0; i++ )  // cycle for depths
        for(j = 0; channels[j] != 0; j++)  // cycle for channels
        {
            if( depths[i] == IPL_DEPTH_1U && channels[j] != 1 ) // skip for IPL_DEPTH_1U
                continue;                                       // all non 1 channels
            IplImage image;
            cvInitImageHeader( &image, size, depths[i], channels[j], IPL_ORIGIN_TL, align );
            if( image.width != size.width || image.height != size.height )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvInitImageHeader: Size mismatch: act %d x %d      exp %d x %d\n",
                          image.width, image.height, size.width, size.height );
                Errors++;
            }
            if( ((size.width * channels[j] * (depths[i] & IPL_DEPTH_MASK) / 8 + align - 1) &
                ~(align - 1)) != image.widthStep )
            {
                trsWrite( ATS_CON | ATS_LST,
                          "cvCreateImageHeader: Wrong widthStep: act %d   exp: %d\n",
                          image.widthStep,
                          (size.width * (depths[i] & IPL_DEPTH_MASK) / 8 + align - 1) &
                          ~(align - 1) );
                Errors++;
            }
            if( image.imageData )
            {
                trsWrite( ATS_CON | ATS_LST, "cvCreateImageHeader: imageData created :(\n" );
                Errors++;
            }
        }
    trsWrite( ATS_CON, "cvInitImageHeader: ... done\n" );

    if( !Errors )
        return trsResult( TRS_OK, "Ok" );
    else
        return trsResult( TRS_FAIL, "%d errors" );
}



void InitAImage()
{
    /* Register test function */
    trsReg( func_name, test_desc, atsAlgoClass, foaImage );
} /* InitACanny */

#endif
