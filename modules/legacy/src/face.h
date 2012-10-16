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

#ifndef __CVFACE_H__
#define __CVFACE_H__

#include "cvfacetemplate.h"

#define MAX_LAYERS 64

class Face
{
public:
    Face(FaceTemplate * lpFaceTemplate);
    virtual ~Face();

    inline bool isFeature(void * lpElem);

    virtual void Show(IplImage * /*Image*/){};
    virtual	void ShowIdeal(IplImage* /*Image*/){};

    virtual void CreateFace(void * lpData) = 0;
    virtual bool CheckElem(void * lpCandidat,void * lpIdeal) = 0;
    virtual double GetWeight() = 0;
protected:
    FaceFeature * m_lpIdealFace;             // Ideal face definition.
    long m_lFaceFeaturesNumber;              // Total number of different face features .
    long * m_lplFaceFeaturesCount;           // Count of each feature found on this face.
    FaceFeature ** m_lppFoundedFaceFeatures; // Features found on current face.
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


#endif //__FACE_H__

