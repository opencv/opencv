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
///////////////////////////////////////////////
//// Created by Khudyakov V.A. bober@gorodok.net
//////////////////////////////////////////////
// FaceDetection.cpp: implementation of the FaceDetection class.
//
//////////////////////////////////////////////////////////////////////

#include "precomp.hpp"
#include "_facedetection.h"


int CV_CDECL CompareContourRect(const void* el1, const void* el2, void* userdata);

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

FaceDetection::FaceDetection()
{

    m_imgGray = NULL;
    m_imgThresh = NULL;
    m_mstgContours = NULL;
    memset(m_seqContours, 0, sizeof(CvSeq*) * MAX_LAYERS);
    m_mstgRects = NULL;
    m_seqRects = NULL;
    m_iNumLayers = 16;
    assert(m_iNumLayers <= MAX_LAYERS);
    m_pFaceList = new FaceDetectionList();
    


    m_bBoosting = false;

}// FaceDetection()

FaceDetection::~FaceDetection()
{
    if (m_imgGray)
        cvReleaseImage(&m_imgGray);

    if (m_imgThresh)
        cvReleaseImage(&m_imgThresh);

    if (m_mstgContours)
        cvReleaseMemStorage(&m_mstgContours);

    if (m_mstgRects)
        cvReleaseMemStorage(&m_mstgRects);
    

}// ~FaceDetection()

void FaceDetection::FindContours(IplImage* imgGray)
{
    ReallocImage(&m_imgThresh, cvGetSize(imgGray), 1);
    if (NULL == m_imgThresh)
        return;
    //
    int iNumLayers = m_iNumLayers;
    int iMinLevel = 0, iMaxLevel = 255, iStep = 255 / iNumLayers;
    ThresholdingParam(imgGray, iNumLayers, iMinLevel, iMaxLevel, iStep);
    // init
    cvReleaseMemStorage(&m_mstgContours);
    m_mstgContours = cvCreateMemStorage();
    if (NULL == m_mstgContours)
        return;
    memset(m_seqContours, 0, sizeof(CvSeq*) * MAX_LAYERS);

    cvReleaseMemStorage(&m_mstgRects);
    m_mstgRects = cvCreateMemStorage();
    if (NULL == m_mstgRects)
        return;
    m_seqRects = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvContourRect), m_mstgRects); 
    if (NULL == m_seqRects)
        return;
    // find contours
    for (int l = iMinLevel, i = 0; l < iMaxLevel; l += iStep, i++)
    {
        cvThreshold(imgGray, m_imgThresh, (double)l, (double)255, CV_THRESH_BINARY);
        if (cvFindContours(m_imgThresh, m_mstgContours, &m_seqContours[i], sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE))
            AddContours2Rect(m_seqContours[i], l, i);
    }
    // sort rects
    cvSeqSort(m_seqRects, CompareContourRect, NULL);
}// void FaceDetection::FindContours(IplImage* imgGray)

#define GIST_STEP   10
#define GIST_NUM    (256 / GIST_STEP)
#define GIST_MIN    32

void FaceDetection::ThresholdingParam(IplImage *imgGray, int iNumLayers, int &iMinLevel, int &iMaxLevel, int &iStep)
{
    assert(imgGray != NULL);
    assert(imgGray->nChannels == 1);
    int i, j;
    // create gistogramm
    uchar* buffImg = (uchar*)imgGray->imageData;
    int gistImg[GIST_NUM + 1] = {0};

    for (j = 0; j < imgGray->height; j ++)
    {
        for (i = 0; i < imgGray->width; i ++)
        {
            int ind = buffImg[i] / GIST_STEP;
            gistImg[ind] ++;
        }
        buffImg += imgGray->widthStep;
    }
    // params
    
    for (i = 0; i <= GIST_NUM; i ++)
    {
        if (gistImg[i] >= GIST_MIN)
            break;
    }
    
    iMinLevel = i * GIST_STEP;
    
    for (i = GIST_NUM; i >= 0; i --)
    {
        if (gistImg[i] >= GIST_MIN)
            break;
    }
    
    iMaxLevel = i * GIST_STEP;
    
    int dLevels = iMaxLevel - iMinLevel;
    if (dLevels <= 0)
    {
        iMinLevel = 0;
        iMaxLevel = 255;
    }
    else if (dLevels <= iNumLayers)
    {
        iMinLevel = iMaxLevel - iNumLayers;
        if (iMinLevel < 0)
        {
            iMinLevel = 0;
            iMaxLevel = iNumLayers;
        }
    }
    iStep = (iMaxLevel - iMinLevel) / iNumLayers;

}// void FaceDetection::ThresholdingParam(IplImage *imgGray, int iNumLayers, int &iMinLevel, int &iMaxLevel, int &iStep)

#ifndef MAX_ERROR
#define MAX_ERROR 0xFFFFFFFF
#endif //MAX_ERROR


void FaceDetection::CreateResults(CvSeq * lpSeq)
{
    
    Face * tmp;
    
    double Max  = 0;
    double CurStat = 0;
    
    FaceData tmpData;
    if (m_bBoosting)
    {
        tmp = m_pFaceList->GetData();
        tmp->CreateFace(&tmpData);

        CvFace tmpFace;
        tmpFace.MouthRect = tmpData.MouthRect;
        tmpFace.LeftEyeRect = tmpData.LeftEyeRect;
        tmpFace.RightEyeRect = tmpData.RightEyeRect;

        cvSeqPush(lpSeq,&tmpFace);

    }else
    {
        while ( (tmp = m_pFaceList->GetData()) != 0 )
        {
            CurStat = tmp->GetWeight();
            if (CurStat > Max)
                Max = CurStat;
        }
        
        while ( (tmp = m_pFaceList->GetData()) != 0 )
        {
            tmp->CreateFace(&tmpData);
            CurStat = tmp->GetWeight();
            
            if (CurStat == Max)
            {
                CvFace tmpFace;
                tmpFace.MouthRect = tmpData.MouthRect;
                tmpFace.LeftEyeRect = tmpData.LeftEyeRect;
                tmpFace.RightEyeRect = tmpData.RightEyeRect;
                cvSeqPush(lpSeq,&tmpFace);

                
            }
        }
    }
}// void FaceDetection::DrawResult(IplImage* img)

void FaceDetection::ResetImage()
{
        delete m_pFaceList;
        m_pFaceList = new FaceDetectionList();

}//FaceDetection::ResetImage

void FaceDetection::AddContours2Rect(CvSeq *seq, int color, int iLayer)
{
    assert(m_mstgRects != NULL);
    assert(m_seqRects != NULL);

    CvContourRect cr;
    for (CvSeq* external = seq; external; external = external->h_next)
    {
        cr.r = cvContourBoundingRect(external, 1 );
        cr.pCenter.x = cr.r.x + cr.r.width / 2;
        cr.pCenter.y = cr.r.y + cr.r.height / 2;
        cr.iNumber = iLayer;
        cr.iType = 6;
        cr.iFlags = 0;
        cr.seqContour = external;
        cr.iContourLength = external->total;
        cr.iColor = color;
        cvSeqPush(m_seqRects, &cr);
        for (CvSeq* internal = external->v_next; internal; internal = internal->h_next)
        {
            cr.r = cvContourBoundingRect(internal, 0);    
            cr.pCenter.x = cr.r.x + cr.r.width / 2;
            cr.pCenter.y = cr.r.y + cr.r.height / 2;
            cr.iNumber = iLayer;
            cr.iType = 12;
            cr.iFlags = 0;
            cr.seqContour = internal;
            cr.iContourLength = internal->total;
            cr.iColor = color;
            cvSeqPush(m_seqRects, &cr);
        }
    }
}// void FaceDetection::AddContours2Rect(CvSeq *seq, int color, int iLayer)

int CV_CDECL CompareContourRect(const void* el1, const void* el2, void* /*userdata*/)
{
    return (((CvContourRect*)el1)->pCenter.y - ((CvContourRect*)el2)->pCenter.y);
}// int CV_CDECL CompareContourRect(const void* el1, const void* el2, void* userdata)

void FaceDetection::FindFace(IplImage *img)
{
    // find all contours
    FindContours(img);
    //
    ResetImage();

    if (m_bBoosting)
        PostBoostingFindCandidats(img);
    else
        FindCandidats();    
    
}// void FaceDetection::FindFace(IplImage *img)


void FaceDetection::FindCandidats()
{
    bool bFound1 = false;
    MouthFaceTemplate * lpFaceTemplate1;
    RFace * lpFace1; 
    bool bInvalidRect1 = false;
    CvRect * lpRect1  = NULL;
    
    for (int i = 0; i < m_seqRects->total; i++)
    {
        CvContourRect* pRect = (CvContourRect*)cvGetSeqElem(m_seqRects, i);
        CvRect rect = pRect->r;
        if (rect.width >= 2*rect.height)
        {

            lpFaceTemplate1 = new MouthFaceTemplate(3,rect,3*(double)rect.width/(double)4,
                                                           3*(double)rect.width/(double)4,
                                                             (double)rect.width/(double)2,
                                                             (double)rect.width/(double)2);
    

            lpFace1 = new RFace(lpFaceTemplate1);
            
            for (int j = 0; j < m_seqRects->total; j++)
            {
                CvContourRect* pRect = (CvContourRect*)cvGetSeqElem(m_seqRects, j);
                
                if ( !bInvalidRect1 )
                {
                    lpRect1 = NULL;
                    lpRect1 = new CvRect();
                    *lpRect1 = pRect->r;
                }else
                {
                    delete lpRect1;
                    lpRect1 = new CvRect();
                    *lpRect1 = pRect->r;
                }
                
                
                if ( lpFace1->isFeature(lpRect1) )
                { 
                    bFound1 = true;
                    bInvalidRect1 = false;
                }else
                    bInvalidRect1 = true;
    

            }

            
            if (bFound1)
            {
                m_pFaceList->AddElem(lpFace1);
                bFound1 = false;
                lpFace1 = NULL;
            }else
            {
                delete lpFace1;
                lpFace1 = NULL;
            }

            
            delete lpFaceTemplate1;
        }
    
    }

}


void FaceDetection::PostBoostingFindCandidats(IplImage * FaceImage)
{
    BoostingFaceTemplate * lpFaceTemplate1;
    RFace * lpFace1; 
    bool bInvalidRect1 = false;
    CvRect * lpRect1  = NULL;
    
    if ( ( !FaceImage->roi ) )
        lpFaceTemplate1 = new BoostingFaceTemplate(3,cvRect(0,0,FaceImage->width,FaceImage->height));
    else
        lpFaceTemplate1 = new BoostingFaceTemplate(3,cvRect(FaceImage->roi->xOffset,FaceImage->roi->yOffset,
                                                            FaceImage->roi->width,FaceImage->roi->height));
    
    lpFace1 = new RFace(lpFaceTemplate1);

    for (int i = 0; i < m_seqRects->total; i++)
    {
        CvContourRect* pRect = (CvContourRect*)cvGetSeqElem(m_seqRects, i);
        
        if ( !bInvalidRect1 )
        {
            lpRect1 = NULL;
            lpRect1 = new CvRect();
            *lpRect1 = pRect->r;
        }else
        {
            delete lpRect1;
            lpRect1 = new CvRect();
            *lpRect1 = pRect->r;
        }
        
        
        if ( lpFace1->isFeature(lpRect1) )
        { 
            //bFound1 = true;
            bInvalidRect1 = false;
        }else
            bInvalidRect1 = true;

    
    }
    
    m_pFaceList->AddElem(lpFace1);
    
    delete lpFaceTemplate1;

}//void FaceDetection::PostBoostingFindCandidats(IplImage * FaceImage)

/////////////////////////
//class Face



//////
//FaceDetectionList Class
/////
FaceDetectionListElem::FaceDetectionListElem()
{
    m_pNext = this;
    m_pPrev = this;
    m_pFace = NULL;
}///FaceDetectionListElem::FaceDetectionListElem()

FaceDetectionListElem::FaceDetectionListElem(Face * pFace,FaceDetectionListElem * pHead)
{
    m_pNext = pHead;
    m_pPrev = pHead->m_pPrev;
    pHead->m_pPrev->m_pNext = this;
    pHead->m_pPrev = this;

    m_pFace = pFace;
}//FaceDetectionListElem::FaceDetectionListElem(Face * pFace)



FaceDetectionListElem::~FaceDetectionListElem()
{
    delete m_pFace;
    m_pNext->m_pPrev = m_pPrev;
    m_pPrev->m_pNext = m_pNext;

}//FaceDetectionListElem::~FaceDetectionListElem()

FaceDetectionList::FaceDetectionList()
{
    m_pHead = new FaceDetectionListElem();
    m_FacesCount = 0;
    m_pCurElem = m_pHead;
}//FaceDetectionList::FaceDetectionList()

FaceDetectionList::~FaceDetectionList()
{
    void * tmp;
    while((tmp = m_pHead->m_pNext->m_pFace) != 0)
        delete m_pHead->m_pNext;

    delete m_pHead;

}//FaceDetectionList::~FaceDetectionList()


int FaceDetectionList::AddElem(Face * pFace)
{
    new FaceDetectionListElem(pFace,m_pHead);
    return m_FacesCount++;
}//FaceDetectionList::AddElem(Face * pFace)

Face * FaceDetectionList::GetData()
{
    m_pCurElem = m_pCurElem->m_pNext;
    return m_pCurElem->m_pFace;
}//Face * FaceDetectionList::GetData()


