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

static char cTestName[] = "Binary tree create";
static char cTestClass[] = "Algorithm";
static char cFuncName[] = "cvCreateContourTree";

static int aCreateContourTree(void)
{
    CvSeqBlock contour_blk1;
    CvContour contour_h1;   /*  input contour */
    CvSeq *contour_h2;  /*  destination contour */
    CvContourTree *tree;     /*   created binary tree */
    CvMemStorage *storage;   /*   storage for contour and tree writing  */
    CvTermCriteria criteria; /*   criteria for the contour restoring */
/*    CvSeqReader reader;  //  points reader of contour */
/*  ippiTrianAttr vertex;*/

    int block_size = 10000;
    int nPoints1 = 20; 
    int xc,yc,a1 = 10, b1 = 20, fi = 0;
    int xmin,ymin,xmax,ymax; 
    double error_test;
    double pi = 3.1415926, eps_rez = 0.05;
    double threshold = 1.e-7;
    double rezult,error;
    int i, code = TRS_OK;
    int type_seq = 0; 
    int width=256,height=256;
    CvPoint *cp1;
    CvPoint *cp2;

/* read tests params */

    if (!trsiRead(&nPoints1,"20","Number of points first contour"))
    return TRS_UNDEF;

    if(nPoints1>0)
    {
    if (!trsiRead(&a1,"10","first radius of the first elipse"))
    return TRS_UNDEF;
    if (!trsiRead(&b1,"20","second radius of the first elipse"))
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
    cp2 = (CvPoint*) trsmAlloc(nPoints1*sizeof(CvPoint));

    for(i=0;i<nPoints1;i++)
     {
        cp1[i].x = (int)(a1*cos(2*pi*i/nPoints1))+xc; 
        cp1[i].y = (int)(b1*sin(2*pi*i/nPoints1))+yc;
        cp1[i].x = (int)(a1*cos(2*pi*i/nPoints1)*cos(2*pi*fi/360.))-
                              (int)(b1*sin(2*pi*i/nPoints1)*sin(2*pi*fi/360.))+xc;

        cp1[i].y = (int)(a1*cos(2*pi*i/nPoints1)*sin(2*pi*fi/360.))+
                                  (int)(b1*sin(2*pi*i/nPoints1)*cos(2*pi*fi/360.))+yc;

        if(xmin> cp1[i].x) xmin = cp1[i].x;
        if(xmax< cp1[i].x) xmax = cp1[i].x;
        if(ymin> cp1[i].y) ymin = cp1[i].y;
        if(ymax< cp1[i].y) ymax = cp1[i].y;
    }

    if(xmax>width||xmin<0||ymax>height||ymin<0)
        return TRS_FAIL;


    storage = cvCreateMemStorage( block_size );
/*   contours initialazing  */

    type_seq = CV_SEQ_POLYGON;
    cvMakeSeqHeaderForArray(type_seq, sizeof(CvContour), sizeof(CvPoint),
              (char*)cp1, nPoints1, (CvSeq*)&contour_h1, &contour_blk1);
    
/*  create countour's tree  */
    error_test = 0.;

    tree = cvCreateContourTree ((CvSeq*)&contour_h1, storage, threshold);

    trsWrite( ATS_CON | ATS_LST | ATS_SUM, "Contour's binary tree is created \n");

    error = 0;
    criteria.type = CV_TERMCRIT_ITER;
    criteria.max_iter = 100;
    contour_h2 = cvContourFromContourTree (tree, storage, criteria);
    rezult = cvMatchContours ((CvSeq*)&contour_h1, contour_h2,CV_CONTOURS_MATCH_I1);

    error+=rezult;

    criteria.type = CV_TERMCRIT_EPS;
    criteria.epsilon = (float)0.00001;
    contour_h2 = cvContourFromContourTree (tree, storage, criteria);
    rezult = cvMatchContours ((CvSeq*)&contour_h1, contour_h2, CV_CONTOURS_MATCH_I1);
    error+=rezult;

    criteria.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    criteria.epsilon = (float)0.00001;
    criteria.max_iter = 1;
    contour_h2 = cvContourFromContourTree (tree, storage, criteria);
    rezult = cvMatchContours ((CvSeq*)&contour_h1, contour_h2, CV_CONTOURS_MATCH_I1);
    error+=rezult;

    criteria.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    criteria.epsilon = 1000.;
    criteria.max_iter = 100;
    contour_h2 = cvContourFromContourTree (tree, storage, criteria);
    rezult = cvMatchContours ((CvSeq*)&contour_h1, contour_h2, CV_CONTOURS_MATCH_I1);
    error+=rezult;

    if(error > eps_rez )
        code = TRS_FAIL;
    else
        code = TRS_OK;

    trsWrite( ATS_CON | ATS_LST | ATS_SUM, "contour from contour tree is restored rezult= %f \n",rezult);

    cvCvtSeqToArray(contour_h2, (char*)cp2 );

    cvReleaseMemStorage ( &storage );

    trsFree (cp2);
    trsFree (cp1);
    
    }
    

/*    _getch();    */
    return code;
}

void InitACreateContourTree( void )
{
/* Test Registartion */
    trsReg(cFuncName,cTestName,cTestClass,aCreateContourTree); 
    
} /* InitACreateContourTree */

/* End of file. */

#endif
