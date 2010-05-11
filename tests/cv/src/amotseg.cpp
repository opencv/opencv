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

#define PUSHC(Y,X) { CurStack[CStIn].y=Y; \
                     CurStack[CStIn].x=X; \
                     CStIn++;}
#define PUSHN(Y,X) { NextStack[NStIn].y=Y; \
                     NextStack[NStIn].x=X; \
                     NStIn++;}

#define POP(Y,X)  {  CStIn--; \
    Y=CurStack[CStIn].y; \
    X=CurStack[CStIn].x;}

void testFill( float*    img,
               int       step,
               CvSize  imgSize,
               CvPoint seed_point,
               float     nv,
               float*    RP,
               int       itsstep,
               float     SegThresh,
               CvConnectedComp* region)
{
    CvPoint* CurStack = (CvPoint*)cvAlloc(imgSize.height*imgSize.width*sizeof(CvPoint));
    CvPoint* NextStack = (CvPoint*)cvAlloc(imgSize.height*imgSize.width*sizeof(CvPoint));
    CvPoint* Temp;
    int ownstep=step/4;
    int RPstep=itsstep/4;
    float thr = -SegThresh;
    float nthr = thr*2;
    int CStIn=0;
    int NStIn=0;
    int TempIn;
    int x,y;
    int XMax=0;
    int YMax=0;
    int XMin = imgSize.width;
    int YMin = imgSize.height;
    int Sum=0;
    
    PUSHC(seed_point.y,seed_point.x);
again:
    while(CStIn)
    {
        POP(y,x);
        XMax = MAX(XMax,x);
        YMax = MAX(YMax,y);
        XMin = MIN(XMin,x);
        YMin = MIN(YMin,y);
        if((y>0)&&(!RP[(y-1)*RPstep+x])&&
        (((img[(y-1)*ownstep+x]-img[y*ownstep+x])<0)&&
         ((img[(y-1)*ownstep+x]-img[y*ownstep+x])>=thr)))PUSHC(y-1,x);           
        if((y>0)&&(!RP[(y-1)*RPstep+x])&&
        (((img[(y-1)*ownstep+x]-img[y*ownstep+x])<=thr)&&
        ((img[(y-1)*ownstep+x]-img[y*ownstep+x])>=nthr)))PUSHN(y-1,x);
        if((y<imgSize.height-1)&&(!RP[(y+1)*RPstep+x])&&
        (((img[(y+1)*ownstep+x]-img[y*ownstep+x])<=0)&&
         ((img[(y+1)*ownstep+x]-img[y*ownstep+x])>=thr)))PUSHC(y+1,x);           
        if((y<imgSize.height-1)&&(!RP[(y+1)*RPstep+x])&&
        (((img[(y+1)*ownstep+x]-img[y*ownstep+x])<=thr)&&
        ((img[(y+1)*ownstep+x]-img[y*ownstep+x])>=nthr)))PUSHN(y+1,x);
        if((x>0)&&(!RP[y*RPstep+x-1])&&
        (((img[y*ownstep+x-1]-img[y*ownstep+x])<=0)&&
         ((img[y*ownstep+x-1]-img[y*ownstep+x])>=thr)))PUSHC(y,x-1);           
        if((x>0)&&(!RP[y*RPstep+x-1])&&
        (((img[y*ownstep+x-1]-img[y*ownstep+x])<=thr)&&
         ((img[y*ownstep+x-1]-img[y*ownstep+x])>=nthr)))PUSHN(y,x-1);
        if((x<imgSize.width-1)&&(!RP[y*RPstep+x+1])&&
        (((img[y*ownstep+x+1]-img[y*ownstep+x])<=0)&&
         ((img[y*ownstep+x+1]-img[y*ownstep+x])>=thr)))PUSHC(y,x+1);           
        if((x<imgSize.width-1)&&(!RP[y*RPstep+x+1])&&
        (((img[y*ownstep+x+1]-img[y*ownstep+x])<=thr)&&
         ((img[y*ownstep+x+1]-img[y*ownstep+x])>=nthr)))PUSHN(y,x+1);
        Sum++;
        RP[y*RPstep+x]=nv;
        img[y*ownstep+x] = -255;
    }
    if(NStIn)
    {
        Temp=CurStack;
        CurStack=NextStack;
        NextStack=Temp;
        TempIn=CStIn;
        CStIn=NStIn;
        NStIn=TempIn;
        goto again;
    }
    region->area = Sum;
    region->rect.x = XMin;
    region->rect.y = YMin;
    region->rect.width = XMax - XMin + 1;
    region->rect.height = YMax - YMin + 1;
    region->value = cvScalar(nv);
    cvFree(&CurStack);
    cvFree(&NextStack);
    return;
}

/* Testing parameters */
static char TestName[] = "Checking MotionSegmentation";
static char TestClass[] = "Algorithm";
static int lImageWidth;
static int lImageHeight;

static int  read_param = 0;
static int  data_types = 0;
static float thresh = 0;
static double EPSILON = 3;


static int fcaMotSeg( void )
{
    int step;
    float* src;
    AtsRandState state; 
    double Error = 0;
    int color = 1;
    CvSize roi;

    IplImage* mhi;
    IplImage* mask32f;
    IplImage* test32f;
    CvSeq* seq1 = NULL;
    CvSeq* seq2 = NULL;
    CvMemStorage* storage;
    CvSeqWriter writer;
    
    CvConnectedComp ConComp;
    
    storage = cvCreateMemStorage( 0 );
    cvClearMemStorage( storage );

    /* Initialization global parameters */
    if( !read_param )
    {
        read_param = 1;
        /* Determining which test are needed to run */
        trsCaseRead( &data_types,"/u/s/f/a", "a",
                    "u - unsigned char, s - signed char, f - float, a - all" );
        /* Reading test-parameters */
        trsiRead( &lImageHeight, "20", "Image height" );
        trsiRead( &lImageWidth, "20", "Image width" );
        trssRead( &thresh, "10", "Segmentation Threshold" );
    }
    if( data_types != 3 && data_types != 0 ) return TRS_UNDEF;
    
    /* Creating images for testing */
    mhi = cvCreateImage(cvSize(lImageWidth, lImageHeight), IPL_DEPTH_32F, 1);
    mask32f = cvCreateImage(cvSize(lImageWidth, lImageHeight), IPL_DEPTH_32F, 1);
    test32f = cvCreateImage(cvSize(lImageWidth, lImageHeight), IPL_DEPTH_32F, 1);

    atsRandInit(&state,40,100,60);
    atsFillRandomImageEx(mhi, &state );
    src = (float*)mhi->imageData;
    step = mhi->widthStep/4;
    int i;
    for(i=0; i<lImageHeight;i++)
    {
        for(int j=0; j<lImageWidth;j++)
        {
            if(src[i*step+j]>80)src[i*step+j]=80;
        }
    }
    cvZero(test32f);
    seq1 = cvSegmentMotion(mhi,mask32f,storage,80,thresh);
    cvStartWriteSeq( 0, sizeof( CvSeq ), sizeof( CvConnectedComp ), storage, &writer );
    roi.width = lImageWidth;
    roi.height = lImageHeight;
    for(i=1;i<lImageHeight-1;i++)
    {
        for(int j=1;j<lImageWidth-1;j++)
        {
            if(src[i*step+j]==80)
            {
                if((src[(i-1)*step+j]>=(80-thresh))&&(src[(i-1)*step+j]<80))
                {
                    CvPoint MinPoint;
                    MinPoint.x=j;
                    MinPoint.y=i-1;
                    testFill(src,
                             step*4,
                             roi,
                             MinPoint,
                             (float)color,
                             (float*)test32f->imageData,
                             test32f->widthStep,
                             thresh,
                             &ConComp);
                    ConComp.value = cvScalar(color);
                    CV_WRITE_SEQ_ELEM( ConComp, writer );
                    color+=1;
                }
                
                if((src[i*step+j-1]>=(80-thresh))&&(src[i*step+j-1]<80))
                {
                    CvPoint MinPoint;
                    MinPoint.x=j-1;
                    MinPoint.y=i;
                    testFill(src,
                             step*4,
                             roi,
                             MinPoint,
                             (float)color,
                             (float*)test32f->imageData,
                             test32f->widthStep,
                             thresh,
                             &ConComp);
                    ConComp.value = cvScalar(color);
                    CV_WRITE_SEQ_ELEM( ConComp, writer );
                    color+=1;
                }
                if((src[i*step+j+1]>=(80-thresh))&&(src[i*step+j+1]<80))
                {
                    CvPoint MinPoint;
                    MinPoint.x=j+1;
                    MinPoint.y=i;
                    testFill(src,
                             step*4,
                             roi,
                             MinPoint,
                             (float)color,
                             (float*)test32f->imageData,
                             test32f->widthStep,
                             thresh,
                             &ConComp);
                    ConComp.value = cvScalar(color);
                    CV_WRITE_SEQ_ELEM( ConComp, writer );
                    color+=1;
                }
                if((src[(i+1)*step+j]>=(80-thresh))&&(src[(i+1)*step+j]<80))
                {
                    CvPoint MinPoint;
                    MinPoint.x=j;
                    MinPoint.y=i+1;
                    testFill(src,
                             step*4,
                             roi,
                             MinPoint,
                             (float)color,
                             (float*)test32f->imageData,
                             test32f->widthStep,
                             thresh,
                             &ConComp);
                    ConComp.value = cvScalar(color);
                    CV_WRITE_SEQ_ELEM( ConComp, writer );
                    color+=1;
                }
                
            }   
         }
    }
    seq2 = cvEndWriteSeq( &writer );
    Error += cvNorm(test32f,mask32f,CV_C);
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&mhi);
    cvReleaseImage(&test32f);
    cvReleaseImage(&mask32f);
    /* Free Memory */
    
    if(Error>=EPSILON)return TRS_FAIL;
    return TRS_OK;
} /* fcaSobel8uC1R */

void InitAMotSeg(void)
{
    trsReg( "cvMotSeg", TestName, TestClass, fcaMotSeg);
} /* InitASobel */

#endif

/* End of file. */
