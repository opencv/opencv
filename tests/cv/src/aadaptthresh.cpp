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
 
//extern "C"{
//  #include "HighGUI.h"
//}


static char cTestName[] = "Image Adaptive Thresholding";
static char cTestClass[] = "Algorithm";
static char cFuncName[] = "cvAdaptThreshold";
static int  aAdaptThreshold()
{

    CvPoint *cp;
    int parameter1 = 3;
    double parameter2 = 10;
    int width = 128;
    int height = 128;
    int kp = 5;
    int nPoints2 = 20;

    int fi = 0;
    int a2 = 20;
    int b2 = 25,xc,yc;

    double pi = 3.1415926;

    double lower, upper;
    unsigned seed;
    char rand;
    AtsRandState state;

    long diff_binary, diff_binary_inv;
    
    int l,i,j;

    IplImage *imBinary, *imBinary_inv, *imTo_zero, *imTo_zero_inv, *imInput, *imOutput;
    CvSize size;

    int code = TRS_OK;

//  read tests params 
    if(!trsiRead( &width, "128", "image width" ))
        return TRS_UNDEF;
    if(!trsiRead( &height, "128", "image height" ))
        return TRS_UNDEF;

//  initialized image
    l = width*height*sizeof(uchar);

    cp = (CvPoint*) trsmAlloc(nPoints2*sizeof(CvPoint));

    xc = (int)( width/2.);
    yc = (int)( height/2.);

    kp = nPoints2;

    size.width = width;
    size.height = height;

    int xmin = width;
    int ymin = height;
    int xmax = 0;
    int ymax = 0;
    
    
    for(i=0;i<nPoints2;i++)
    {
        cp[i].x = (int)(a2*cos(2*pi*i/nPoints2)*cos(2*pi*fi/360.))-
        (int)(b2*sin(2*pi*i/nPoints2)*sin(2*pi*fi/360.))+xc;
        if(xmin> cp[i].x) xmin = cp[i].x;
        if(xmax< cp[i].x) xmax = cp[i].x;
        cp[i].y = (int)(a2*cos(2*pi*i/nPoints2)*sin(2*pi*fi/360.))+
                    (int)(b2*sin(2*pi*i/nPoints2)*cos(2*pi*fi/360.))+yc;
        if(ymin> cp[i].y) ymin = cp[i].y;
        if(ymax< cp[i].y) ymax = cp[i].y;
    }

    if(xmax>width||xmin<0||ymax>height||ymin<0) return TRS_FAIL;
    
//  IPL image moment calculation  
//  create image  
    imBinary = cvCreateImage( size, 8, 1 );
    imBinary_inv = cvCreateImage( size, 8, 1 );
    imTo_zero = cvCreateImage( size, 8, 1 );
    imTo_zero_inv = cvCreateImage( size, 8, 1 );
    imOutput = cvCreateImage( size, 8, 1 );
    imInput = cvCreateImage( size, 8, 1 );

    int bgrn = 50;
    int signal = 150;
    
    memset(imInput->imageData,bgrn,l);

    cvFillPoly(imInput, &cp, &kp, 1, cvScalarAll(signal));

//  do noise   
    upper = 22;
    lower = -upper;
    seed = 345753;
    atsRandInit( &state, lower, upper, seed );
    
    uchar *input = (uchar*)imInput->imageData;
    uchar *binary = (uchar*)imBinary->imageData;
    uchar *binary_inv = (uchar*)imBinary_inv->imageData;
    uchar *to_zero = (uchar*)imTo_zero->imageData;
    uchar *to_zero_inv = (uchar*)imTo_zero_inv->imageData;
    double *parameter = (double*)trsmAlloc(2*sizeof(double));

    int step = imInput->widthStep;

    for(i = 0; i<size.height; i++, input+=step, binary+=step, binary_inv+=step, to_zero+=step,to_zero_inv+=step)
    {
         for(j = 0; j<size.width; j++)
         {
                atsbRand8s( &state, &rand, 1);   
                if(input[j] == bgrn) 
                {
                    binary[j] = to_zero[j] = (uchar)0;
                    binary_inv[j] = (uchar)255;
                    to_zero_inv[j] = input [j] = (uchar)(bgrn + rand);
                }
                else 
                {
                    binary[j] = (uchar)255;
                    binary_inv[j] = to_zero_inv[j] = (uchar)0;
                    to_zero[j] = input[j] = (uchar)(signal + rand);
                }
        
         }
    }



    cvAdaptiveThreshold( imInput, imOutput, (double)255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, parameter1, parameter2 ); 
    diff_binary = atsCompare1Db( (uchar*)imOutput->imageData, (uchar*)imBinary->imageData, l, 5);

    cvAdaptiveThreshold( imInput, imOutput, (double)255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, parameter1, parameter2 ); 
    diff_binary_inv = atsCompare1Db( (uchar*)imOutput->imageData, (uchar*)imBinary_inv->imageData, l, 5);

    if( diff_binary > 5 || diff_binary_inv > 5 )
        code = TRS_FAIL;  
    
    cvReleaseImage(&imInput);
    cvReleaseImage(&imOutput);
    cvReleaseImage(&imBinary);
    cvReleaseImage(&imBinary_inv);
    cvReleaseImage(&imTo_zero);
    cvReleaseImage(&imTo_zero_inv);

    trsWrite( ATS_CON | ATS_LST | ATS_SUM, "diff_binary =%ld \n", diff_binary); 
    trsWrite( ATS_CON | ATS_LST | ATS_SUM, "diff_binary_inv =%ld \n", diff_binary_inv); 

    trsFree(parameter);
    trsFree(cp);
    return code;
}


void InitAAdaptThreshold( void )
{
/* Test Registartion */
    trsReg(cFuncName,cTestName,cTestClass,aAdaptThreshold); 
} /* InitAAdaptThreshold */

#endif

/* End of file. */
