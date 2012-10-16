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
// FaceDetection.h: interface for the FaceDetection class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _CVFACEDETECTION_H_
#define _CVFACEDETECTION_H_

#define MAX_LAYERS 64

class FaceFeature
{
public:
    FaceFeature(double dWeight,void * lpContour,bool bIsFeature);
    FaceFeature();
    virtual ~FaceFeature();
    inline bool isFaceFeature();
    inline void * GetContour();
    inline double GetWeight();
    inline void SetContour(void * lpContour);
    inline void SetWeight(double dWeight);
    inline void SetFeature(bool bIsFeature);
private:
    double m_dWeight;
    void * m_lpContour;
    bool m_bIsFaceFeature;
};//class FaceFeature

inline void FaceFeature::SetFeature(bool bIsFeature)
{
    m_bIsFaceFeature = bIsFeature;
}

inline bool FaceFeature::isFaceFeature()
{
    return m_bIsFaceFeature;
}//inline bool FaceFeature::isFaceFeature()

inline void * FaceFeature::GetContour()
{
    return m_lpContour;
}//inline void * FaceFeature::GetContour()

inline double FaceFeature::GetWeight()
{
    return m_dWeight;
}//inline long FaceFeature::GetWeight()

inline void FaceFeature::SetContour(void * lpContour)
{
    m_lpContour = lpContour;
}//inline void FaceFeature::SetContour(void * lpContour)

inline void FaceFeature::SetWeight(double  dWeight)
{
    m_dWeight = dWeight;
}//inline void FaceFeature::SetWeight(double * dWeight)



class FaceTemplate
{
public:
    FaceTemplate(long lFeatureCount) {m_lFeaturesCount = lFeatureCount;  m_lpFeaturesList = new FaceFeature[lFeatureCount];};
    virtual ~FaceTemplate();

    inline long GetCount();
    inline FaceFeature * GetFeatures();

protected:
    FaceFeature * m_lpFeaturesList;
private:
    long m_lFeaturesCount;
};//class FaceTemplate


inline long FaceTemplate::GetCount()
{
    return m_lFeaturesCount;
}//inline long FaceTemplate::GetCount()


inline FaceFeature * FaceTemplate::GetFeatures()
{
    return m_lpFeaturesList;
}//inline FaceFeature * FaceTemplate::GetFeatures()

////////////
//class RFaceTemplate
///////////

class MouthFaceTemplate:public FaceTemplate
{
public:
    inline MouthFaceTemplate(long lNumber,CvRect rect,double dEyeWidth,double dEyeHeight,double dDistanceBetweenEye,double dDistanceEyeAboveMouth);
    ~MouthFaceTemplate();
};//class MouthFaceTemplate:public FaceTemplate


inline MouthFaceTemplate::MouthFaceTemplate(long lNumber,CvRect rect,double dEyeWidth,double dEyeHeight,
                             double dDistanceBetweenEye,double dDistanceEyeAboveMouth):FaceTemplate(lNumber)
{

    CvRect MouthRect = rect;


    CvRect LeftEyeRect = cvRect(cvRound(rect.x - (dEyeWidth + dDistanceBetweenEye/(double)2 - (double)rect.width/(double)2)),
                                cvRound(rect.y - dDistanceEyeAboveMouth - dEyeHeight),
                                cvRound(dEyeWidth),
                                cvRound(dEyeHeight) );

    CvRect RightEyeRect = cvRect(cvRound(rect.x + (double)rect.width/(double)2 + dDistanceBetweenEye/(double)2),
                                 cvRound(rect.y - dDistanceEyeAboveMouth - dEyeHeight),
                                 cvRound(dEyeWidth),
                                 cvRound(dEyeHeight) );

//  CvRect NoseRect = cvRect(cvRound(rect.x + (double)rect.width/(double)4),
//                           cvRound(rect.y - (double)rect.width/(double)2 - (double)rect.height/(double)4),
//                           cvRound((double)rect.width/(double)2),
//                           cvRound((double)rect.width/(double)2) );
/*
    CvRect CheenRect = cvRect(rect.x,rect.y + 3*rect.height/2,rect.width,rect.height);

*/

    CvRect * lpMouthRect = new CvRect();
    *lpMouthRect = MouthRect;
    m_lpFeaturesList[0].SetContour(lpMouthRect);
    m_lpFeaturesList[0].SetWeight(1);
    m_lpFeaturesList[0].SetFeature(false);


    CvRect * lpLeftEyeRect = new CvRect();
    *lpLeftEyeRect = LeftEyeRect;
    m_lpFeaturesList[1].SetContour(lpLeftEyeRect);
    m_lpFeaturesList[1].SetWeight(1);
    m_lpFeaturesList[1].SetFeature(true);

    CvRect * lpRightEyeRect = new CvRect();
    *lpRightEyeRect = RightEyeRect;
    m_lpFeaturesList[2].SetContour(lpRightEyeRect);
    m_lpFeaturesList[2].SetWeight(1);
    m_lpFeaturesList[2].SetFeature(true);


//  CvRect * lpNoseRect = new CvRect();
//  *lpNoseRect = NoseRect;
//  m_lpFeaturesList[3].SetContour(lpNoseRect);
//  m_lpFeaturesList[3].SetWeight(0);
//  m_lpFeaturesList[3].SetFeature(true);

/*  CvRect * lpCheenRect = new CvRect();
    *lpCheenRect = CheenRect;
    m_lpFeaturesList[4].SetContour(lpCheenRect);
    m_lpFeaturesList[4].SetWeight(1);
    m_lpFeaturesList[4].SetFeature(false);

*/

}//constructor MouthFaceTemplate(long lNumFeatures,CvRect rect,double dEyeWidth,double dEyeHeight,double dDistanceBetweenEye,double dDistanceEyeAboveMouth);


typedef struct CvContourRect
{
    int     iNumber;
    int     iType;
    int     iFlags;
    CvSeq   *seqContour;
    int     iContourLength;
    CvRect  r;
    CvPoint pCenter;
    int     iColor;
} CvContourRect;

class Face
{
public:
    Face(FaceTemplate * lpFaceTemplate);
    virtual ~Face();

    inline bool isFeature(void * lpElem);

    virtual void Show(IplImage * /*Image*/){};
    virtual void ShowIdeal(IplImage* /*Image*/){};

    virtual void CreateFace(void * lpData) = 0;
    virtual bool CheckElem(void * lpCandidat,void * lpIdeal) = 0;
    virtual double GetWeight() = 0;
protected:
    FaceFeature * m_lpIdealFace;//ideal face definition
    long m_lFaceFeaturesNumber; //total number of diferent face features
    long * m_lplFaceFeaturesCount;//number of each features fouded for this face
    FaceFeature ** m_lppFoundedFaceFeatures;//founded features of curen face
    double m_dWeight;
};

inline bool Face::isFeature(void * lpElem)
{
    for (int i = 0;i < m_lFaceFeaturesNumber;i ++)
    {
        void * lpIdeal = m_lpIdealFace[i].GetContour();

        if ( CheckElem( lpElem,lpIdeal) )
        {
            if (m_lplFaceFeaturesCount[i] < 3*MAX_LAYERS)
            {
                double dWeight = m_lpIdealFace[i].GetWeight();
                bool bIsFeature = m_lpIdealFace[i].isFaceFeature();


                if (bIsFeature)
                {
                    m_lppFoundedFaceFeatures[i][m_lplFaceFeaturesCount[i]].SetWeight(dWeight);
                    m_lppFoundedFaceFeatures[i][m_lplFaceFeaturesCount[i]].SetContour(lpElem);
                    m_lppFoundedFaceFeatures[i][m_lplFaceFeaturesCount[i]].SetFeature(bIsFeature);
                    m_lplFaceFeaturesCount[i] ++;
                }

                m_dWeight += dWeight;

                if (bIsFeature)
                    return true;
            }
        }

    }

    return false;
}//inline bool RFace::isFeature(void * lpElem);


struct FaceData
{
    CvRect LeftEyeRect;
    CvRect RightEyeRect;
    CvRect MouthRect;
    double Error;
};//struct FaceData

class RFace:public Face
{
public:
    RFace(FaceTemplate * lpFaceTemplate);
    virtual ~RFace();
    virtual bool CheckElem(void * lpCandidat,void * lpIdeal);
    virtual void  CreateFace(void * lpData);
    virtual void Show(IplImage* Image);
    virtual void ShowIdeal(IplImage* Image);
    virtual double GetWeight();
private:
    bool isPointInRect(CvPoint p,CvRect rect);
    bool m_bIsGenerated;
    void ResizeRect(CvRect Rect,CvRect * lpRect,long lDir,long lD);
    void CalculateError(FaceData * lpFaceData);
};


class FaceDetectionListElem
{
public:
    FaceDetectionListElem();
    FaceDetectionListElem(Face * pFace,FaceDetectionListElem * pHead);
    virtual ~FaceDetectionListElem();
    FaceDetectionListElem * m_pNext;
    FaceDetectionListElem * m_pPrev;
    Face * m_pFace;
};//class FaceDetectionListElem

class FaceDetectionList
{
public:
    FaceDetectionList();
    int AddElem(Face * pFace);
    virtual ~FaceDetectionList();
    Face* GetData();
        long m_FacesCount;
private:
    FaceDetectionListElem * m_pHead;
    FaceDetectionListElem * m_pCurElem;
};//class FaceDetectionList


class FaceDetection
{
public:
    void FindFace(IplImage* img);
    void CreateResults(CvSeq * lpSeq);
    FaceDetection();
    virtual ~FaceDetection();
    void SetBoosting(bool bBoosting) {m_bBoosting = bBoosting;}
    bool isPostBoosting() {return m_bBoosting;}
protected:

    IplImage* m_imgGray;
    IplImage* m_imgThresh;
    int m_iNumLayers;
    CvMemStorage* m_mstgContours;
    CvSeq* m_seqContours[MAX_LAYERS];
    CvMemStorage* m_mstgRects;
    CvSeq* m_seqRects;

    bool m_bBoosting;
    FaceDetectionList * m_pFaceList;

protected:
    void ResetImage();
    void FindContours(IplImage* imgGray);
    void AddContours2Rect(CvSeq*  seq, int color, int iLayer);
    void ThresholdingParam(IplImage* imgGray, int iNumLayers, int& iMinLevel, int& iMaxLevel, int& iStep);
    void FindCandidats();
    void PostBoostingFindCandidats(IplImage * FaceImage);
};

inline void ReallocImage(IplImage** ppImage, CvSize sz, long lChNum)
{
    IplImage* pImage;
    if( ppImage == NULL )
        return;
    pImage = *ppImage;
    if( pImage != NULL )
    {
        if (pImage->width != sz.width || pImage->height != sz.height || pImage->nChannels != lChNum)
            cvReleaseImage( &pImage );
    }
    if( pImage == NULL )
        pImage = cvCreateImage( sz, IPL_DEPTH_8U, lChNum);
    *ppImage = pImage;
}

////////////
//class RFaceTemplate
///////////

class BoostingFaceTemplate:public FaceTemplate
{
public:
    inline BoostingFaceTemplate(long lNumber,CvRect rect);
    ~BoostingFaceTemplate() {};
};//class RFaceTemplate:public FaceTemplate


inline BoostingFaceTemplate::BoostingFaceTemplate(long lNumber,CvRect rect):FaceTemplate(lNumber)
{
    long EyeWidth = rect.width/5;
    long EyeHeight = EyeWidth;

    CvRect LeftEyeRect = cvRect(rect.x + EyeWidth,rect.y + rect.height/2 - EyeHeight,EyeWidth,EyeHeight);
    CvRect RightEyeRect = cvRect(rect.x + 3*EyeWidth,rect.y + rect.height/2 - EyeHeight,EyeWidth,EyeHeight);
    CvRect MouthRect = cvRect(rect.x + 3*EyeWidth/2,rect.y + 3*rect.height/4 - EyeHeight/2,2*EyeWidth,EyeHeight);

    CvRect * lpMouthRect = new CvRect();
    *lpMouthRect = MouthRect;
    m_lpFeaturesList[0].SetContour(lpMouthRect);
    m_lpFeaturesList[0].SetWeight(1);
    m_lpFeaturesList[0].SetFeature(true);

    CvRect * lpLeftEyeRect = new CvRect();
    *lpLeftEyeRect = LeftEyeRect;
    m_lpFeaturesList[1].SetContour(lpLeftEyeRect);
    m_lpFeaturesList[1].SetWeight(1);
    m_lpFeaturesList[1].SetFeature(true);

    CvRect * lpRightEyeRect = new CvRect();
    *lpRightEyeRect = RightEyeRect;
    m_lpFeaturesList[2].SetContour(lpRightEyeRect);
    m_lpFeaturesList[2].SetWeight(1);
    m_lpFeaturesList[2].SetFeature(true);

}//inline BoostingFaceTemplate::BoostingFaceTemplate(long lNumber,CvRect rect):FaceTemplate(lNumber)

#endif // !defined(AFX_FACEDETECTION_H__55865033_D8E5_4DD5_8925_34C2285BB1BE__INCLUDED_)
