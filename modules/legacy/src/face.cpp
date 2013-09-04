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

#include "precomp.hpp"
#include "_facedetection.h"

Face::Face(FaceTemplate * lpFaceTemplate)
{
    //init number of face elements;
    m_lFaceFeaturesNumber = lpFaceTemplate->GetCount();

    //init array of numbers of foundet face elements of each type
    m_lplFaceFeaturesCount = new long[m_lFaceFeaturesNumber];
    memset(m_lplFaceFeaturesCount,0,m_lFaceFeaturesNumber*sizeof(long));

    //init array of ideal face features
    m_lpIdealFace = new FaceFeature[m_lFaceFeaturesNumber];

    //init array of founded features
    m_lppFoundedFaceFeatures = new FaceFeature*[m_lFaceFeaturesNumber];

    for (int i = 0;i < m_lFaceFeaturesNumber;i ++)
    {
        m_lppFoundedFaceFeatures[i] = (new FaceFeature[3*MAX_LAYERS]);
    }

    //set start weight 0
    m_dWeight = 0;

}//Face::Face(FaceTemplate * lpFaceTemplate)

Face::~Face()
{
    for (int i = 0;i < m_lFaceFeaturesNumber;i ++)
    {
        delete [] (m_lppFoundedFaceFeatures[i]);
    }
    delete [] m_lppFoundedFaceFeatures;


    delete [] m_lplFaceFeaturesCount;
    delete [] m_lpIdealFace;

}//Face::~Face()


#define UP_SCALE    1
#define DOWN_SCALE  2


////////////
//class RFace(rect based face)
////////////
RFace::RFace(FaceTemplate * lpFaceTemplate):Face(lpFaceTemplate)
{
    //init ideal face
    FaceFeature * lpTmp = lpFaceTemplate->GetFeatures();

    for (int j = 0;j < m_lFaceFeaturesNumber;j ++)
    {
        CvRect * lpTmpRect = NULL;
        lpTmpRect = new CvRect;
        *lpTmpRect = *(CvRect*)lpTmp[j].GetContour();

        m_lpIdealFace[j].SetContour( lpTmpRect );
        m_lpIdealFace[j].SetWeight( lpTmp[j].GetWeight() );
        m_lpIdealFace[j].SetFeature( lpTmp[j].isFaceFeature() );

    }

    m_bIsGenerated = false;
}//RFace::RFace(FaceTemplate * lpFaceTemplate)

RFace::~RFace()
{

}//RFace::~RFace()

inline bool RFace::isPointInRect(CvPoint p,CvRect rect)
{
    if ( (p.x >= rect.x) && (p.y >= rect.y) && (p.x <= rect.x + rect.width) && (p.y <= rect.y + rect.height) )
        return true;

    return false;
}//inline bool RFace::isPointInRect(CvPoint,CvRect rect)

double RFace::GetWeight()
{
    return m_dWeight;
}//double RFace::GetWeight()


bool RFace::CheckElem(void * lpCandidat,void * lpIdeal)
{

    CvRect IdealRect = *(CvRect*)lpIdeal;
    CvRect Rect = *(CvRect*)lpCandidat;

    if (Rect.height > Rect.width)
        return false;

    long SizeIdeal = IdealRect.width*IdealRect.height;
    long Size = Rect.width*Rect.height;

    if ( (Size > SizeIdeal) || ( Size < (SizeIdeal/5) ) )
        return false;

//  CvRect UpRect;
//  CvRect DownRect;
//  ResizeRect(IdealRect,&UpRect,UP_SCALE,7);
//  ResizeRect(IdealRect,&DownRect,DOWN_SCALE,7);

    long x = Rect.x + cvRound(Rect.width/2);
    long y = Rect.y + cvRound(Rect.height/2);

    if ( isPointInRect(cvPoint(x,y),IdealRect) )
        return true;

//  if ( isPointInRect(cvPoint(Rect.x,Rect.y),UpRect) &&
//       isPointInRect(cvPoint(Rect.x + Rect.width,Rect.y + Rect.height),UpRect ) &&
//       isPointInRect(cvPoint(DownRect.x,DownRect.y),Rect) &&
//       isPointInRect(cvPoint(DownRect.x + DownRect.width,DownRect.y + DownRect.height),Rect) )
//      return true;


//  if ( isPointInRect(cvPoint(Rect.x,Rect.y),IdealRect) &&
//       isPointInRect(cvPoint(Rect.x + Rect.width,Rect.y + Rect.height),IdealRect ) )
//      return true;

    return false;
}//inline bool RFace::CheckElem(CvRect rect)



void RFace::CalculateError(FaceData * lpFaceData)
{
    CvRect LeftEyeRect = lpFaceData->LeftEyeRect;
    CvRect RightEyeRect = lpFaceData->RightEyeRect;
    CvRect MouthRect = lpFaceData->MouthRect;

    long LeftSquare = LeftEyeRect.width*LeftEyeRect.height;
    long RightSquare = RightEyeRect.width*RightEyeRect.height;

    long dy = LeftEyeRect.y - RightEyeRect.y;

    long dx1 = LeftEyeRect.x + LeftEyeRect.width/2 - MouthRect.x;
    long dx2 = RightEyeRect.x + RightEyeRect.width/2 - MouthRect.x - MouthRect.width;


    lpFaceData->Error = (double)(LeftSquare - RightSquare)*(double)(LeftSquare - RightSquare)/((double)(LeftSquare + RightSquare)*(LeftSquare + RightSquare)) +
                        (double)(dy*dy)/((double)(LeftEyeRect.height + RightEyeRect.height)*(LeftEyeRect.height + RightEyeRect.height)) +
                        (double)(dx1*dx1)/((double)MouthRect.width*MouthRect.width) +
                        (double)(dx2*dx2)/((double)MouthRect.width*MouthRect.width);

}//void RFace::CalculateError(FaceData * lpFaceData)

#define MAX_ERROR 0xFFFFFFFF

void  RFace::CreateFace(void * lpData)
{
    FaceData Data;

    double Error = MAX_ERROR;
    double CurError = MAX_ERROR;

    FaceData * lpFaceData = (FaceData*)lpData;

    int im = 0;//mouth was find
    int jl = 0;//left eye was find
    int kr = 0;//right eye was find

    long MouthNumber = 0;
    long LeftEyeNumber = 0;
    long RightEyeNumber = 0;

    for (int i = 0;i < m_lplFaceFeaturesCount[0] + 1;i ++)
    {

        if ( !m_lplFaceFeaturesCount[0] )
            Data.MouthRect = *(CvRect*)m_lpIdealFace[0].GetContour();
        else
        {
            if ( i != m_lplFaceFeaturesCount[0] )
                Data.MouthRect = *(CvRect*)m_lppFoundedFaceFeatures[0][i].GetContour();
            im = 1;
        }


        for (int j = 0;j < m_lplFaceFeaturesCount[1] + 1;j ++)
        {

            if ( !m_lplFaceFeaturesCount[1] )
                Data.LeftEyeRect = *(CvRect*)m_lpIdealFace[1].GetContour();
            else
            {
                if (j != m_lplFaceFeaturesCount[1] )
                    Data.LeftEyeRect = *(CvRect*)m_lppFoundedFaceFeatures[1][j].GetContour();
                jl = 1;
            }


            for (int k = 0;k < m_lplFaceFeaturesCount[2] + 1;k ++)
            {

                if ( !m_lplFaceFeaturesCount[2] )
                    Data.RightEyeRect = *(CvRect*)m_lpIdealFace[2].GetContour();
                else
                {
                    if (k != m_lplFaceFeaturesCount[2] )
                        Data.RightEyeRect = *(CvRect*)m_lppFoundedFaceFeatures[2][k].GetContour();
                    kr = 1;
                }

                CalculateError(&Data);

                if ( (im + jl + kr) )
                {
                    Error = Data.Error/(im + jl + kr);
                }else
                    Error = MAX_ERROR;

                if (CurError > Error)
                {
                    CurError = Error;
                    MouthNumber = i;
                    LeftEyeNumber = j;
                    RightEyeNumber = k;
                }

            }


        }

    }

    if ( m_lplFaceFeaturesCount[0] )
        lpFaceData->MouthRect = *(CvRect*)m_lppFoundedFaceFeatures[0][MouthNumber].GetContour();
    else
        lpFaceData->MouthRect = *(CvRect*)m_lpIdealFace[0].GetContour();

    if ( m_lplFaceFeaturesCount[1] )
        lpFaceData->LeftEyeRect = *(CvRect*)m_lppFoundedFaceFeatures[1][LeftEyeNumber].GetContour();
    else
        lpFaceData->LeftEyeRect = *(CvRect*)m_lpIdealFace[1].GetContour();

    if ( m_lplFaceFeaturesCount[2] )
        lpFaceData->RightEyeRect = *(CvRect*)m_lppFoundedFaceFeatures[2][RightEyeNumber].GetContour();
    else
        lpFaceData->RightEyeRect = *(CvRect*)m_lpIdealFace[2].GetContour();

    lpFaceData->Error = CurError;

}//void * RFace::CreateFace()

void RFace::Show(IplImage * Image)
{
    for (int i = 0;i < m_lFaceFeaturesNumber;i ++)
    {
        if (m_lplFaceFeaturesCount[i])
        {
            for (int j = 0;j < m_lplFaceFeaturesCount[i];j ++)
            {
                CvRect rect = *(CvRect*)m_lppFoundedFaceFeatures[i][j].GetContour();
                CvPoint p1 = cvPoint(rect.x,rect.y);
                CvPoint p2 = cvPoint(rect.x + rect.width,rect.y + rect.height);
                cvRectangle(Image,p1,p2,CV_RGB(255,0,0),1);
            }
        }
    }

}//void RFace::Show(IplImage * Image)

void RFace::ShowIdeal(IplImage* Image)
{
    for (int i = 0;i < m_lFaceFeaturesNumber;i ++)
    {
        CvRect Rect = *(CvRect*)m_lpIdealFace[i].GetContour();
        CvPoint p1 = cvPoint(Rect.x,Rect.y);
        CvPoint p2 = cvPoint(Rect.x + Rect.width,Rect.y + Rect.height);
        cvRectangle(Image,p1,p2,CV_RGB(0,0,255),1);
    }
}//void RFace::ShowIdeal(IplImage* Image)


inline void RFace::ResizeRect(CvRect Rect,CvRect * lpRect,long lDir,long lD)
{
    if (lDir == UP_SCALE)
    {
        lpRect->x = Rect.x - lD;
        lpRect->y = Rect.y - lD;
        lpRect->width = Rect.width + 2*lD;
        lpRect->height = Rect.height + 2*lD;
    }
    if (lDir == DOWN_SCALE)
    {
        lpRect->x = Rect.x + lD;
        lpRect->y = Rect.y + lD;
        if (Rect.width - 2*lD >= 0)
        {
            lpRect->width = Rect.width - 2*lD;
        }else
            lpRect->width = 0;

        if (Rect.height - 2*lD >= 0)
        {
            lpRect->height = Rect.height - 2*lD;
        }else
            lpRect->height = 0;
    }

}// inline void RFace::ResizeRect(CvRect * lpRect,long lDir,long lD)
