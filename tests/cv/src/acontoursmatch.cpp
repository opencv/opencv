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
/*#include "conio.h"*/

static char cTestName[] = "Matching Contours";
static char cTestClass[] = "Algorithm";
static char cFuncName[] = "cvMatchContours";

static int aMatchContours(void)
{
    CvSeqBlock contour_blk1, contour_blk2;
    CvContour contour_h1, contour_h2;
    CvContoursMatchMethod method;
    int nPoints1 = 20, nPoints2 = 20; 
    int xc,yc,a1 = 10, b1 = 20, a2 = 15, b2 =30, fi = 0;
    int xmin,ymin,xmax,ymax; 
    int seq_type;
    double error_test,rezult, eps_rez = 0.1;
    double pi = 3.1415926;
    int i;
    int code = TRS_OK;

    int width=256,height=256;
    CvPoint *cp1,*cp2;

/* read tests params */

    if (!trsiRead(&nPoints1,"20","Number of points first contour"))
    return TRS_UNDEF;
    if (!trsiRead(&nPoints2,"20","Number of points second contour"))
    return TRS_UNDEF;

    if(nPoints1>0&&nPoints2>0)
    {
    if (!trsiRead(&a1,"10","first radius of the first elipse"))
    return TRS_UNDEF;
    if (!trsiRead(&b1,"20","second radius of the first elipse"))
    return TRS_UNDEF;
    if (!trsiRead(&a2,"15","first radius of the second elipse"))
    return TRS_UNDEF;
    if (!trsiRead(&b2,"30","second radius of the second elipse"))
    return TRS_UNDEF;
    if (!trsiRead(&fi,"0","second radius of the second elipse"))
    return TRS_UNDEF;

    xc = (int)(width/2.);
    yc = (int)(height/2.);
    xmin = width;
    ymin = height;
    xmax = 0;
    ymax = 0;

    cp1 = (CvPoint*) trsmAlloc(nPoints1*sizeof(CvPoint));
    cp2 = (CvPoint*) trsmAlloc(nPoints2*sizeof(CvPoint));

    for(i=0;i<nPoints1;i++)
    {
        cp1[i].x = (int)(a1*cos(2*pi*i/nPoints1))+xc; 
        cp1[i].y = (int)(b1*sin(2*pi*i/nPoints1))+yc;
        if(xmin> cp1[i].x) xmin = cp1[i].x;
        if(xmax< cp1[i].x) xmax = cp1[i].x;
        if(ymin> cp1[i].y) ymin = cp1[i].y;
        if(ymax< cp1[i].y) ymax = cp1[i].y;
    }

    if(xmax>width||xmin<0||ymax>height||ymin<0) return TRS_FAIL;

    for(i=0;i<nPoints2;i++)
    {
        cp2[i].x = (int)(a2*cos(2*pi*i/nPoints2)*cos(2*pi*fi/360.))-
                       (int)(b2*sin(2*pi*i/nPoints2)*sin(2*pi*fi/360.))+xc;

        cp2[i].y = (int)(a2*cos(2*pi*i/nPoints2)*sin(2*pi*fi/360.))+
                       (int)(b2*sin(2*pi*i/nPoints2)*cos(2*pi*fi/360.))+yc;

        if(xmin> cp2[i].x) xmin = cp2[i].x;
        if(xmax< cp2[i].x) xmax = cp2[i].x;
        if(ymin> cp2[i].y) ymin = cp2[i].y;
        if(ymax< cp2[i].y) ymax = cp2[i].y;
    } 
    if(xmax>width||xmin<0||ymax>height||ymin<0) return TRS_FAIL;

/*   contours initialazing */
    seq_type = CV_SEQ_POLYGON;
    cvMakeSeqHeaderForArray( seq_type, sizeof(CvContour), sizeof(CvPoint),
               (char*)cp1, nPoints1, (CvSeq*)&contour_h1, &contour_blk1);

    cvMakeSeqHeaderForArray( seq_type, sizeof(CvContour), sizeof(CvPoint),
              (char*)cp2, nPoints2, (CvSeq*)&contour_h2, &contour_blk2);

/*  countours matchig */
    error_test = 0.;

    for (i=1;i<=3;i++)
    {
        method = (CvContoursMatchMethod)i;
        rezult = cvMatchContours((CvSeq*)&contour_h1, (CvSeq*)&contour_h2, method);

        error_test+=rezult;
    }
    error_test = error_test/3.;

    if(error_test > eps_rez) code = TRS_FAIL;
    else code = TRS_OK;

    trsWrite( ATS_CON | ATS_LST | ATS_SUM, "contours matching error_test =%f \n",
               error_test);

    trsFree (cp2);
    trsFree (cp1);
    }


/*    _getch();     */
    return code;
}

void InitAMatchContours( void )
{
/* Test Registartion */
    trsReg(cFuncName,cTestName,cTestClass,aMatchContours); 
    
} /* InitAMatchContours */

/* End of file. */

#endif

