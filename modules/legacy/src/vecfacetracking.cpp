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

/****************************************************************************************\
      Contour-based face feature tracking
      The code was created by Tatiana Cherepanova (tata@sl.iae.nsk.su)
\****************************************************************************************/

#include "precomp.hpp"
#include "_vectrack.h"

#define NUM_FACE_ELEMENTS   3
enum
{
    MOUTH = 0,
    LEYE = 1,
    REYE = 2,
};

#define MAX_LAYERS      64

const double pi = 3.1415926535;

struct CvFaceTracker;
struct CvTrackingRect;
class CvFaceElement;

void ThresholdingParam(IplImage *imgGray, int iNumLayers, int &iMinLevel, int &iMaxLevel, float &step, float& power, int iHistMin /*= HIST_MIN*/);
int ChoiceTrackingFace3(CvFaceTracker* pTF, const int nElements, const CvFaceElement* big_face, CvTrackingRect* face, int& new_energy);
int ChoiceTrackingFace2(CvFaceTracker* pTF, const int nElements, const CvFaceElement* big_face, CvTrackingRect* face, int& new_energy, int noel);
inline int GetEnergy(CvTrackingRect** ppNew, const CvTrackingRect* pPrev, CvPoint* ptTempl, CvRect* rTempl);
inline int GetEnergy2(CvTrackingRect** ppNew, const CvTrackingRect* pPrev, CvPoint* ptTempl, CvRect* rTempl, int* element);
inline double CalculateTransformationLMS3_0( CvPoint* pTemplPoints, CvPoint* pSrcPoints);
inline double CalculateTransformationLMS3( CvPoint* pTemplPoints,
                                   CvPoint* pSrcPoints,
                                   double*       pdbAverageScale,
                                   double*       pdbAverageRotate,
                                   double*       pdbAverageShiftX,
                                   double*       pdbAverageShiftY );

struct CvTrackingRect
{
    CvRect r;
    CvPoint ptCenter;
    int iColor;
    int iEnergy;
    int nRectsInThis;
    int nRectsOnLeft;
    int nRectsOnRight;
    int nRectsOnTop;
    int nRectsOnBottom;
    CvTrackingRect() { memset(this, 0, sizeof(CvTrackingRect)); };
    int Energy(const CvTrackingRect& prev)
    {
        int prev_color = 0 == prev.iColor ? iColor : prev.iColor;
        iEnergy =   1 * pow2(r.width - prev.r.width) +
            1 * pow2(r.height - prev.r.height) +
            1 * pow2(iColor - prev_color) / 4 +
            - 1 * nRectsInThis +
            - 0 * nRectsOnTop +
            + 0 * nRectsOnLeft +
            + 0 * nRectsOnRight +
            + 0 * nRectsOnBottom;
        return iEnergy;
    }
};

struct CvFaceTracker
{
    CvTrackingRect face[NUM_FACE_ELEMENTS];
    int iTrackingFaceType;
    double dbRotateDelta;
    double dbRotateAngle;
    CvPoint ptRotate;

    CvPoint ptTempl[NUM_FACE_ELEMENTS];
    CvRect rTempl[NUM_FACE_ELEMENTS];

    IplImage* imgGray;
    IplImage* imgThresh;
    CvMemStorage* mstgContours;
    CvFaceTracker()
    {
        ptRotate.x = 0;
        ptRotate.y = 0;
        dbRotateDelta = 0;
        dbRotateAngle = 0;
        iTrackingFaceType = -1;
        imgThresh = NULL;
        imgGray = NULL;
        mstgContours = NULL;
    };
    ~CvFaceTracker()
    {
        if (NULL != imgGray)
            delete imgGray;
        if (NULL != imgThresh)
            delete imgThresh;
        if (NULL != mstgContours)
            cvReleaseMemStorage(&mstgContours);
    };
    int Init(CvRect* pRects, IplImage* imgray)
    {
        for (int i = 0; i < NUM_FACE_ELEMENTS; i++)
        {
            face[i].r = pRects[i];
            face[i].ptCenter = Center(face[i].r);
            ptTempl[i] = face[i].ptCenter;
            rTempl[i] = face[i].r;
        }
        imgray = cvCreateImage(cvSize(imgray->width, imgray->height), 8, 1);
        imgThresh = cvCreateImage(cvSize(imgray->width, imgray->height), 8, 1);
        mstgContours = cvCreateMemStorage();
        if ((NULL == imgray) ||
            (NULL == imgThresh) ||
            (NULL == mstgContours))
            return 0;
        return 1;
    };
    int InitNextImage(IplImage* img)
    {
        CvSize sz(img->width, img->height);
        ReallocImage(&imgGray, sz, 1);
        ReallocImage(&imgThresh, sz, 1);
        ptRotate = face[MOUTH].ptCenter;
        float m[6];
        CvMat mat = cvMat( 2, 3, CV_32FC1, m );

        if (NULL == imgGray || NULL == imgThresh)
            return 0;

        /*m[0] = (float)cos(-dbRotateAngle*CV_PI/180.);
        m[1] = (float)sin(-dbRotateAngle*CV_PI/180.);
        m[2] = (float)ptRotate.x;
        m[3] = -m[1];
        m[4] = m[0];
        m[5] = (float)ptRotate.y;*/
        cv2DRotationMatrix( cvPointTo32f(ptRotate), -dbRotateAngle, 1., &mat );
        cvWarpAffine( img, imgGray, &mat );

        if (NULL == mstgContours)
            mstgContours = cvCreateMemStorage();
        else
            cvClearMemStorage(mstgContours);
        if (NULL == mstgContours)
            return 0;
        return 1;
    }
};

class CvFaceElement
{
public:
    CvSeq* m_seqRects;
    CvMemStorage* m_mstgRects;
    CvRect m_rROI;
    CvTrackingRect m_trPrev;
    inline CvFaceElement()
    {
        m_seqRects = NULL;
        m_mstgRects = NULL;
        m_rROI.x = 0;
        m_rROI.y = 0;
        m_rROI.width = 0;
        m_rROI.height = 0;
    };
    inline int Init(const CvRect& roi, const CvTrackingRect& prev, CvMemStorage* mstg = NULL)
    {
        m_rROI = roi;
        m_trPrev = prev;
        if (NULL != mstg)
            m_mstgRects = mstg;
        if (NULL == m_mstgRects)
            return 0;
        if (NULL == m_seqRects)
            m_seqRects = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvTrackingRect), m_mstgRects);
        else
            cvClearSeq(m_seqRects);
        if (NULL == m_seqRects)
            return 0;
        return 1;
    };
    void FindRects(IplImage* img, IplImage* thresh, int nLayers, int dMinSize);
protected:
    void FindContours(IplImage* img, IplImage* thresh, int nLayers, int dMinSize);
    void MergeRects(int d);
    void Energy();
}; //class CvFaceElement

inline int CV_CDECL CompareEnergy(const void* el1, const void* el2, void*)
{
    return ((CvTrackingRect*)el1)->iEnergy - ((CvTrackingRect*)el2)->iEnergy;
}// int CV_CDECL CompareEnergy(const void* el1, const void* el2, void*)

void CvFaceElement::FindRects(IplImage* img, IplImage* thresh, int nLayers, int dMinSize)
{
    FindContours(img, thresh, nLayers, dMinSize / 4);
    if (0 == m_seqRects->total)
        return;
    Energy();
    cvSeqSort(m_seqRects, CompareEnergy, NULL);
    CvTrackingRect* pR = (CvTrackingRect*)cvGetSeqElem(m_seqRects, 0);
    if (m_seqRects->total < 32)
    {
        MergeRects(dMinSize / 8);
        Energy();
        cvSeqSort(m_seqRects, CompareEnergy, NULL);
    }
    pR = (CvTrackingRect*)cvGetSeqElem(m_seqRects, 0);
    if ((pR->iEnergy > 100 && m_seqRects->total < 32) || (m_seqRects->total < 16))
    {
        MergeRects(dMinSize / 4);
        Energy();
        cvSeqSort(m_seqRects, CompareEnergy, NULL);
    }
    pR = (CvTrackingRect*)cvGetSeqElem(m_seqRects, 0);
    if ((pR->iEnergy > 100 && m_seqRects->total < 16) || (pR->iEnergy > 200 && m_seqRects->total < 32))
    {
        MergeRects(dMinSize / 2);
        Energy();
        cvSeqSort(m_seqRects, CompareEnergy, NULL);
    }

}// void CvFaceElement::FindRects(IplImage* img, IplImage* thresh, int nLayers, int dMinSize)

void CvFaceElement::FindContours(IplImage* img, IplImage* thresh, int nLayers, int dMinSize)
{
    CvSeq* seq;
    CvRect roi = m_rROI;
    Extend(roi, 1);
    cvSetImageROI(img, roi);
    cvSetImageROI(thresh, roi);
    // layers
    int colors[MAX_LAYERS] = {0};
    int iMinLevel = 0, iMaxLevel = 255;
    float step, power;
    ThresholdingParam(img, nLayers / 2, iMinLevel, iMaxLevel, step, power, 4);
    int iMinLevelPrev = iMinLevel;
    int iMaxLevelPrev = iMinLevel;
    if (m_trPrev.iColor != 0)
    {
        iMinLevelPrev = m_trPrev.iColor - nLayers / 2;
        iMaxLevelPrev = m_trPrev.iColor + nLayers / 2;
    }
    if (iMinLevelPrev < iMinLevel)
    {
        iMaxLevelPrev += iMinLevel - iMinLevelPrev;
        iMinLevelPrev = iMinLevel;
    }
    if (iMaxLevelPrev > iMaxLevel)
    {
        iMinLevelPrev -= iMaxLevelPrev - iMaxLevel;
        if (iMinLevelPrev < iMinLevel)
            iMinLevelPrev = iMinLevel;
        iMaxLevelPrev = iMaxLevel;
    }
    int n = nLayers;
    n -= (iMaxLevelPrev - iMinLevelPrev + 1) / 2;
    step = float(iMinLevelPrev - iMinLevel + iMaxLevel - iMaxLevelPrev) / float(n);
    int j = 0;
    float level;
    for (level = (float)iMinLevel; level < iMinLevelPrev && j < nLayers; level += step, j++)
        colors[j] = int(level + 0.5);
    for (level = (float)iMinLevelPrev; level < iMaxLevelPrev && j < nLayers; level += 2.0, j++)
        colors[j] = int(level + 0.5);
    for (level = (float)iMaxLevelPrev; level < iMaxLevel && j < nLayers; level += step, j++)
        colors[j] = int(level + 0.5);
    //
    for (int i = 0; i < nLayers; i++)
    {
        cvThreshold(img, thresh, colors[i], 255.0, CV_THRESH_BINARY);
        if (cvFindContours(thresh, m_mstgRects, &seq, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE))
        {
            CvTrackingRect cr;
            for (CvSeq* external = seq; external; external = external->h_next)
            {
                cr.r = cvContourBoundingRect(external);
                Move(cr.r, roi.x, roi.y);
                if (RectInRect(cr.r, m_rROI) && cr.r.width > dMinSize  && cr.r.height > dMinSize)
                {
                    cr.ptCenter = Center(cr.r);
                    cr.iColor = colors[i];
                    cvSeqPush(m_seqRects, &cr);
                }
                for (CvSeq* internal = external->v_next; internal; internal = internal->h_next)
                {
                    cr.r = cvContourBoundingRect(internal);
                    Move(cr.r, roi.x, roi.y);
                    if (RectInRect(cr.r, m_rROI) && cr.r.width > dMinSize  && cr.r.height > dMinSize)
                    {
                        cr.ptCenter = Center(cr.r);
                        cr.iColor = colors[i];
                        cvSeqPush(m_seqRects, &cr);
                    }
                }
            }
            cvClearSeq(seq);
        }
    }
    cvResetImageROI(img);
    cvResetImageROI(thresh);
}//void CvFaceElement::FindContours(IplImage* img, IplImage* thresh, int nLayers)

void CvFaceElement::MergeRects(int d)
{
    int nRects = m_seqRects->total;
    CvSeqReader reader, reader2;
    cvStartReadSeq( m_seqRects, &reader );
    int i, j;
    for (i = 0; i < nRects; i++)
    {
        CvTrackingRect* pRect1 = (CvTrackingRect*)(reader.ptr);
        cvStartReadSeq( m_seqRects, &reader2 );
        cvSetSeqReaderPos(&reader2, i + 1);
        for (j = i + 1; j < nRects; j++)
        {
            CvTrackingRect* pRect2 = (CvTrackingRect*)(reader2.ptr);
            if (abs(pRect1->ptCenter.y - pRect2->ptCenter.y) < d &&
                abs(pRect1->r.height - pRect2->r.height) < d)
            {
                CvTrackingRect rNew;
                rNew.iColor = (pRect1->iColor + pRect2->iColor + 1) / 2;
                rNew.r.x = min(pRect1->r.x, pRect2->r.x);
                rNew.r.y = min(pRect1->r.y, pRect2->r.y);
                rNew.r.width = max(pRect1->r.x + pRect1->r.width, pRect2->r.x + pRect2->r.width) - rNew.r.x;
                rNew.r.height = min(pRect1->r.y + pRect1->r.height, pRect2->r.y + pRect2->r.height) - rNew.r.y;
                if (rNew.r != pRect1->r && rNew.r != pRect2->r)
                {
                    rNew.ptCenter = Center(rNew.r);
                    cvSeqPush(m_seqRects, &rNew);
                }
            }
            CV_NEXT_SEQ_ELEM( sizeof(CvTrackingRect), reader2 );
        }
        CV_NEXT_SEQ_ELEM( sizeof(CvTrackingRect), reader );
    }
    // delete equal rects
    for (i = 0; i < m_seqRects->total; i++)
    {
        CvTrackingRect* pRect1 = (CvTrackingRect*)cvGetSeqElem(m_seqRects, i);
        int j_begin = i + 1;
        for (j = j_begin; j < m_seqRects->total;)
        {
            CvTrackingRect* pRect2 = (CvTrackingRect*)cvGetSeqElem(m_seqRects, j);
            if (pRect1->r == pRect2->r)
                cvSeqRemove(m_seqRects, j);
            else
                j++;
        }
    }

}//void CvFaceElement::MergeRects(int d)

void CvFaceElement::Energy()
{
    CvSeqReader reader, reader2;
    cvStartReadSeq( m_seqRects, &reader );
    for (int i = 0; i < m_seqRects->total; i++)
    {
        CvTrackingRect* pRect = (CvTrackingRect*)(reader.ptr);
        // outside and inside rects
        cvStartReadSeq( m_seqRects, &reader2 );
        for (int j = 0; j < m_seqRects->total; j++)
        {
            CvTrackingRect* pRect2 = (CvTrackingRect*)(reader2.ptr);
            if (i != j)
            {
                if (RectInRect(pRect2->r, pRect->r))
                    pRect->nRectsInThis ++;
                else if (pRect2->r.y + pRect2->r.height <= pRect->r.y)
                    pRect->nRectsOnTop ++;
                else if (pRect2->r.y >= pRect->r.y + pRect->r.height)
                    pRect->nRectsOnBottom ++;
                else if (pRect2->r.x + pRect2->r.width <= pRect->r.x)
                    pRect->nRectsOnLeft ++;
                else if (pRect2->r.x >= pRect->r.x + pRect->r.width)
                    pRect->nRectsOnRight ++;
            }
            CV_NEXT_SEQ_ELEM( sizeof(CvTrackingRect), reader2 );
        }
        // energy
        pRect->Energy(m_trPrev);
        CV_NEXT_SEQ_ELEM( sizeof(CvTrackingRect), reader );
    }
}//void CvFaceElement::Energy()

CV_IMPL CvFaceTracker*
cvInitFaceTracker(CvFaceTracker* pFaceTracker, const IplImage* imgGray, CvRect* pRects, int nRects)
{
    assert(NULL != imgGray);
    assert(NULL != pRects);
    assert(nRects >= NUM_FACE_ELEMENTS);
    if ((NULL == imgGray) ||
        (NULL == pRects) ||
        (nRects < NUM_FACE_ELEMENTS))
        return NULL;

    //int new_face = 0;
    CvFaceTracker* pFace = pFaceTracker;
    if (NULL == pFace)
    {
        pFace = new CvFaceTracker;
        if (NULL == pFace)
            return NULL;
        //new_face = 1;
    }
    pFace->Init(pRects, (IplImage*)imgGray);
    return pFace;
}//CvFaceTracker* InitFaceTracker(IplImage* imgGray, CvRect* pRects, int nRects)

CV_IMPL void
cvReleaseFaceTracker(CvFaceTracker** ppFaceTracker)
{
    if (NULL == *ppFaceTracker)
        return;
    delete *ppFaceTracker;
    *ppFaceTracker = NULL;
}//void ReleaseFaceTracker(CvFaceTracker** ppFaceTracker)


CV_IMPL int
cvTrackFace(CvFaceTracker* pFaceTracker, IplImage* imgGray, CvRect* pRects, int nRects, CvPoint* ptRotate, double* dbAngleRotate)
{
    assert(NULL != pFaceTracker);
    assert(NULL != imgGray);
    assert(NULL != pRects && nRects >= NUM_FACE_ELEMENTS);
    if ((NULL == pFaceTracker) ||
        (NULL == imgGray))
        return 0;
    pFaceTracker->InitNextImage(imgGray);
    *ptRotate = pFaceTracker->ptRotate;
    *dbAngleRotate = pFaceTracker->dbRotateAngle;

    int nElements = 16;
    double dx = pFaceTracker->face[LEYE].ptCenter.x - pFaceTracker->face[REYE].ptCenter.x;
    double dy = pFaceTracker->face[LEYE].ptCenter.y - pFaceTracker->face[REYE].ptCenter.y;
    double d_eyes = sqrt(dx*dx + dy*dy);
    int d = cvRound(0.25 * d_eyes);
    int dMinSize = d;
    int nRestarts = 0;

    int elem;

    CvFaceElement big_face[NUM_FACE_ELEMENTS];
START:
    // init
    for (elem = 0; elem < NUM_FACE_ELEMENTS; elem++)
    {
        CvRect r = pFaceTracker->face[elem].r;
        Extend(r, d);
        if (r.width < 4*d)
        {
            r.x -= (4*d - r.width) / 2;
            r.width += 4*d - r.width;
        }
        if (r.height < 3*d)
        {
            r.y -= (3*d - r.height) / 2;
            r.height += 3*d - r.height;
        }
        if (r.x < 1)
            r.x = 1;
        if (r.y < 1)
            r.y = 1;
        if (r.x + r.width > pFaceTracker->imgGray->width - 2)
            r.width = pFaceTracker->imgGray->width - 2 - r.x;
        if (r.y + r.height > pFaceTracker->imgGray->height - 2)
            r.height = pFaceTracker->imgGray->height - 2 - r.y;
        if (!big_face[elem].Init(r, pFaceTracker->face[elem], pFaceTracker->mstgContours))
            return 0;
    }
    // find contours
    for (elem = 0; elem < NUM_FACE_ELEMENTS; elem++)
        big_face[elem].FindRects(pFaceTracker->imgGray, pFaceTracker->imgThresh, 32, dMinSize);
    // candidats
    CvTrackingRect new_face[NUM_FACE_ELEMENTS];
    int new_energy = 0;
    int found = ChoiceTrackingFace3(pFaceTracker, nElements, big_face, new_face, new_energy);
    int restart = 0;
    int find2 = 0;
    int noel = -1;
    if (found)
    {
        if (new_energy > 100000 && -1 != pFaceTracker->iTrackingFaceType)
            find2 = 1;
        else if (new_energy > 150000)
        {
            int elements = 0;
            for (int el = 0; el < NUM_FACE_ELEMENTS; el++)
            {
                if (big_face[el].m_seqRects->total > 16 || (big_face[el].m_seqRects->total > 8 && new_face[el].iEnergy < 100))
                    elements++;
                else
                    noel = el;
            }
            if (2 == elements)
                find2 = 1;
            else
                restart = 1;
        }
    }
    else
    {
        if (-1 != pFaceTracker->iTrackingFaceType)
            find2 = 1;
        else
            restart = 1;
    }
RESTART:
    if (restart)
    {
        if (nRestarts++ < 2)
        {
            d = d + d/4;
            goto START;
        }
    }
    else if (find2)
    {
        if (-1 != pFaceTracker->iTrackingFaceType)
            noel = pFaceTracker->iTrackingFaceType;
        int found2 = ChoiceTrackingFace2(pFaceTracker, nElements, big_face, new_face, new_energy, noel);
        if (found2 && new_energy < 100000)
        {
            pFaceTracker->iTrackingFaceType = noel;
            found = 1;
        }
        else
        {
            restart = 1;
            goto RESTART;
        }
    }

    if (found)
    {
        // angle by mouth & eyes
        double vx_prev = double(pFaceTracker->face[LEYE].ptCenter.x + pFaceTracker->face[REYE].ptCenter.x) / 2.0 - pFaceTracker->face[MOUTH].ptCenter.x;
        double vy_prev = double(pFaceTracker->face[LEYE].ptCenter.y + pFaceTracker->face[REYE].ptCenter.y) / 2.0 - pFaceTracker->face[MOUTH].ptCenter.y;
        double vx_prev1 = vx_prev * cos(pFaceTracker->dbRotateDelta) - vy_prev * sin(pFaceTracker->dbRotateDelta);
        double vy_prev1 = vx_prev * sin(pFaceTracker->dbRotateDelta) + vy_prev * cos(pFaceTracker->dbRotateDelta);
        vx_prev = vx_prev1;
        vy_prev = vy_prev1;
        for (elem = 0; elem < NUM_FACE_ELEMENTS; elem++)
            pFaceTracker->face[elem] = new_face[elem];
        double vx = double(pFaceTracker->face[LEYE].ptCenter.x + pFaceTracker->face[REYE].ptCenter.x) / 2.0 - pFaceTracker->face[MOUTH].ptCenter.x;
        double vy = double(pFaceTracker->face[LEYE].ptCenter.y + pFaceTracker->face[REYE].ptCenter.y) / 2.0 - pFaceTracker->face[MOUTH].ptCenter.y;
        pFaceTracker->dbRotateDelta = 0;
        double n1_n2 = (vx * vx + vy * vy) * (vx_prev * vx_prev + vy_prev * vy_prev);
        if (n1_n2 != 0)
            pFaceTracker->dbRotateDelta = asin((vx * vy_prev - vx_prev * vy) / sqrt(n1_n2));
        pFaceTracker->dbRotateAngle -= pFaceTracker->dbRotateDelta;
    }
    else
    {
        pFaceTracker->dbRotateDelta = 0;
        pFaceTracker->dbRotateAngle = 0;
    }
    if ((pFaceTracker->dbRotateAngle >= pi/2 && pFaceTracker->dbRotateAngle > 0) ||
        (pFaceTracker->dbRotateAngle <= -pi/2 && pFaceTracker->dbRotateAngle < 0))
    {
        pFaceTracker->dbRotateDelta = 0;
        pFaceTracker->dbRotateAngle = 0;
        found = 0;
    }
    if (found)
    {
        for (int i = 0; i < NUM_FACE_ELEMENTS && i < nRects; i++)
            pRects[i] = pFaceTracker->face[i].r;
    }
    return found;
}//int FindFaceTracker(CvFaceTracker* pFaceTracker, IplImage* imgGray, CvRect* pRects, int nRects, CvPoint& ptRotate, double& dbAngleRotate)

void ThresholdingParam(IplImage *imgGray, int iNumLayers, int &iMinLevel, int &iMaxLevel, float &step, float& power, int iHistMin /*= HIST_MIN*/)
{
    assert(imgGray != NULL);
    assert(imgGray->nChannels == 1);
    int i, j;
    // create histogram
    int histImg[256] = {0};
    uchar* buffImg = (uchar*)imgGray->imageData;
    CvRect rROI = cvGetImageROI(imgGray);
    buffImg += rROI.y * imgGray->widthStep + rROI.x;
    for (j = 0; j < rROI.height; j++)
    {
        for (i = 0; i < rROI.width; i++)
            histImg[buffImg[i]] ++;
        buffImg += imgGray->widthStep;
    }
    // params
    for (i = 0; i < 256; i++)
    {
        if (histImg[i] > iHistMin)
            break;
    }
    iMinLevel = i;
    for (i = 255; i >= 0; i--)
    {
        if (histImg[i] > iHistMin)
            break;
    }
    iMaxLevel = i;
    if (iMaxLevel <= iMinLevel)
    {
        iMaxLevel = 255;
        iMinLevel = 0;
    }
    // power
    double black = 1;
    double white = 1;
    for (i = iMinLevel; i < (iMinLevel + iMaxLevel) / 2; i++)
        black += histImg[i];
    for (i = (iMinLevel + iMaxLevel) / 2; i < iMaxLevel; i++)
        white += histImg[i];
    power = float(black) / float(2 * white);
    //
    step = float(iMaxLevel - iMinLevel) / float(iNumLayers);
    if (step < 1.0)
        step = 1.0;
}// void ThresholdingParam(IplImage *imgGray, int iNumLayers, int &iMinLevel, int &iMaxLevel, int &iStep)

int ChoiceTrackingFace3(CvFaceTracker* pTF, const int nElements, const CvFaceElement* big_face, CvTrackingRect* face, int& new_energy)
{
    CvTrackingRect* curr_face[NUM_FACE_ELEMENTS] = {NULL};
    CvTrackingRect* new_face[NUM_FACE_ELEMENTS] = {NULL};
    new_energy = 0x7fffffff;
    int curr_energy = 0x7fffffff;
    int found = 0;
    int N = 0;
    CvSeqReader reader_m, reader_l, reader_r;
    cvStartReadSeq( big_face[MOUTH].m_seqRects, &reader_m );
    for (int i_mouth = 0; i_mouth < big_face[MOUTH].m_seqRects->total && i_mouth < nElements; i_mouth++)
    {
        curr_face[MOUTH] = (CvTrackingRect*)(reader_m.ptr);
        cvStartReadSeq( big_face[LEYE].m_seqRects, &reader_l );
        for (int i_left = 0; i_left < big_face[LEYE].m_seqRects->total && i_left < nElements; i_left++)
        {
            curr_face[LEYE] = (CvTrackingRect*)(reader_l.ptr);
            if (curr_face[LEYE]->r.y + curr_face[LEYE]->r.height < curr_face[MOUTH]->r.y)
            {
                cvStartReadSeq( big_face[REYE].m_seqRects, &reader_r );
                for (int i_right = 0; i_right < big_face[REYE].m_seqRects->total && i_right < nElements; i_right++)
                {
                    curr_face[REYE] = (CvTrackingRect*)(reader_r.ptr);
                    if (curr_face[REYE]->r.y + curr_face[REYE]->r.height < curr_face[MOUTH]->r.y &&
                        curr_face[REYE]->r.x > curr_face[LEYE]->r.x + curr_face[LEYE]->r.width)
                    {
                        curr_energy = GetEnergy(curr_face, pTF->face, pTF->ptTempl, pTF->rTempl);
                        if (curr_energy < new_energy)
                        {
                            for (int elem = 0; elem < NUM_FACE_ELEMENTS; elem++)
                                new_face[elem] = curr_face[elem];
                            new_energy = curr_energy;
                            found = 1;
                        }
                        N++;
                    }
                }
            }
        }
    }
    if (found)
    {
        for (int elem = 0; elem < NUM_FACE_ELEMENTS; elem++)
            face[elem] = *(new_face[elem]);
    }
    return found;
} // int ChoiceTrackingFace3(const CvTrackingRect* tr_face, CvTrackingRect* new_face, int& new_energy)

int ChoiceTrackingFace2(CvFaceTracker* pTF, const int nElements, const CvFaceElement* big_face, CvTrackingRect* face, int& new_energy, int noel)
{
    int element[NUM_FACE_ELEMENTS];
    for (int i = 0, elem = 0; i < NUM_FACE_ELEMENTS; i++)
    {
        if (i != noel)
        {
            element[elem] = i;
            elem ++;
        }
        else
            element[2] = i;
    }
    CvTrackingRect* curr_face[NUM_FACE_ELEMENTS] = {NULL};
    CvTrackingRect* new_face[NUM_FACE_ELEMENTS] = {NULL};
    new_energy = 0x7fffffff;
    int curr_energy = 0x7fffffff;
    int found = 0;
    int N = 0;
    CvSeqReader reader0, reader1;
    cvStartReadSeq( big_face[element[0]].m_seqRects, &reader0 );
    for (int i0 = 0; i0 < big_face[element[0]].m_seqRects->total && i0 < nElements; i0++)
    {
        curr_face[element[0]] = (CvTrackingRect*)(reader0.ptr);
        cvStartReadSeq( big_face[element[1]].m_seqRects, &reader1 );
        for (int i1 = 0; i1 < big_face[element[1]].m_seqRects->total && i1 < nElements; i1++)
        {
            curr_face[element[1]] = (CvTrackingRect*)(reader1.ptr);
            curr_energy = GetEnergy2(curr_face, pTF->face, pTF->ptTempl, pTF->rTempl, element);
            if (curr_energy < new_energy)
            {
                for (int elem = 0; elem < NUM_FACE_ELEMENTS; elem++)
                    new_face[elem] = curr_face[elem];
                new_energy = curr_energy;
                found = 1;
            }
            N++;
        }
    }
    if (found)
    {
        face[element[0]] = *(new_face[element[0]]);
        face[element[1]] = *(new_face[element[1]]);
        // 3 element find by template
        CvPoint templ_v01(pTF->ptTempl[element[1]].x - pTF->ptTempl[element[0]].x, pTF->ptTempl[element[1]].y - pTF->ptTempl[element[0]].y);
        CvPoint templ_v02(pTF->ptTempl[element[2]].x - pTF->ptTempl[element[0]].x, pTF->ptTempl[element[2]].y - pTF->ptTempl[element[0]].y);
        CvPoint prev_v01(pTF->face[element[1]].ptCenter.x - pTF->face[element[0]].ptCenter.x, pTF->face[element[1]].ptCenter.y - pTF->face[element[0]].ptCenter.y);
        CvPoint prev_v02(pTF->face[element[2]].ptCenter.x - pTF->face[element[0]].ptCenter.x, pTF->face[element[2]].ptCenter.y - pTF->face[element[0]].ptCenter.y);
        CvPoint new_v01(new_face[element[1]]->ptCenter.x - new_face[element[0]]->ptCenter.x, new_face[element[1]]->ptCenter.y - new_face[element[0]]->ptCenter.y);
        double templ_d01 = sqrt((double)templ_v01.x*templ_v01.x + templ_v01.y*templ_v01.y);
        double templ_d02 = sqrt((double)templ_v02.x*templ_v02.x + templ_v02.y*templ_v02.y);
        double prev_d01 = sqrt((double)prev_v01.x*prev_v01.x + prev_v01.y*prev_v01.y);
        double prev_d02 = sqrt((double)prev_v02.x*prev_v02.x + prev_v02.y*prev_v02.y);
        double new_d01 = sqrt((double)new_v01.x*new_v01.x + new_v01.y*new_v01.y);
        double scale = templ_d01 / new_d01;
        double new_d02 = templ_d02 / scale;
        double sin_a = double(prev_v01.x * prev_v02.y - prev_v01.y * prev_v02.x) / (prev_d01 * prev_d02);
        double cos_a = cos(asin(sin_a));
        double x = double(new_v01.x) * cos_a - double(new_v01.y) * sin_a;
        double y = double(new_v01.x) * sin_a + double(new_v01.y) * cos_a;
        x = x * new_d02 / new_d01;
        y = y * new_d02 / new_d01;
        CvPoint new_v02(int(x + 0.5), int(y + 0.5));
        face[element[2]].iColor = 0;
        face[element[2]].iEnergy = 0;
        face[element[2]].nRectsInThis = 0;
        face[element[2]].nRectsOnBottom = 0;
        face[element[2]].nRectsOnLeft = 0;
        face[element[2]].nRectsOnRight = 0;
        face[element[2]].nRectsOnTop = 0;
        face[element[2]].ptCenter.x = new_v02.x + new_face[element[0]]->ptCenter.x;
        face[element[2]].ptCenter.y = new_v02.y + new_face[element[0]]->ptCenter.y;
        face[element[2]].r.width = int(double(pTF->rTempl[element[2]].width) / (scale) + 0.5);
        face[element[2]].r.height = int(double(pTF->rTempl[element[2]].height) / (scale) + 0.5);
        face[element[2]].r.x = face[element[2]].ptCenter.x - (face[element[2]].r.width + 1) / 2;
        face[element[2]].r.y = face[element[2]].ptCenter.y - (face[element[2]].r.height + 1) / 2;
        assert(face[LEYE].r.x + face[LEYE].r.width <= face[REYE].r.x);
    }
    return found;
} // int ChoiceTrackingFace3(const CvTrackingRect* tr_face, CvTrackingRect* new_face, int& new_energy)

inline int GetEnergy(CvTrackingRect** ppNew, const CvTrackingRect* pPrev, CvPoint* ptTempl, CvRect* rTempl)
{
    int energy = 0;
    CvPoint ptNew[NUM_FACE_ELEMENTS];
    CvPoint ptPrev[NUM_FACE_ELEMENTS];
    for (int i = 0; i < NUM_FACE_ELEMENTS; i++)
    {
        ptNew[i] = ppNew[i]->ptCenter;
        ptPrev[i] = pPrev[i].ptCenter;
        energy += ppNew[i]->iEnergy - 2 * ppNew[i]->nRectsInThis;
    }
    double dx = 0, dy = 0, scale = 1, rotate = 0;
    double e_templ = CalculateTransformationLMS3(ptTempl, ptNew, &scale, &rotate, &dx, &dy);
    double e_prev = CalculateTransformationLMS3_0(ptPrev, ptNew);
    double w_eye = double(ppNew[LEYE]->r.width + ppNew[REYE]->r.width) * scale / 2.0;
    double h_eye = double(ppNew[LEYE]->r.height + ppNew[REYE]->r.height) * scale / 2.0;
    double w_mouth = double(ppNew[MOUTH]->r.width) * scale;
    double h_mouth = double(ppNew[MOUTH]->r.height) * scale;
    energy +=
        int(512.0 * (e_prev + 16.0 * e_templ)) +
        4 * pow2(ppNew[LEYE]->r.width - ppNew[REYE]->r.width) +
        4 * pow2(ppNew[LEYE]->r.height - ppNew[REYE]->r.height) +
        4 * (int)pow(w_eye - double(rTempl[LEYE].width + rTempl[REYE].width) / 2.0, 2) +
        2 * (int)pow(h_eye - double(rTempl[LEYE].height + rTempl[REYE].height) / 2.0, 2) +
        1 * (int)pow(w_mouth - double(rTempl[MOUTH].width), 2) +
        1 * (int)pow(h_mouth - double(rTempl[MOUTH].height), 2) +
        0;
    return energy;
}

inline int GetEnergy2(CvTrackingRect** ppNew, const CvTrackingRect* pPrev, CvPoint* ptTempl, CvRect* rTempl, int* element)
{
    CvPoint new_v(ppNew[element[0]]->ptCenter.x - ppNew[element[1]]->ptCenter.x,
        ppNew[element[0]]->ptCenter.y - ppNew[element[1]]->ptCenter.y);
    CvPoint prev_v(pPrev[element[0]].ptCenter.x - pPrev[element[1]].ptCenter.x,
        pPrev[element[0]].ptCenter.y - pPrev[element[1]].ptCenter.y);
    double new_d = sqrt((double)new_v.x*new_v.x + new_v.y*new_v.y);
    double prev_d = sqrt((double)prev_v.x*prev_v.x + prev_v.y*prev_v.y);
    double dx = ptTempl[element[0]].x - ptTempl[element[1]].x;
    double dy = ptTempl[element[0]].y - ptTempl[element[1]].y;
    double templ_d = sqrt(dx*dx + dy*dy);
    double scale_templ = new_d / templ_d;
    double w0 = (double)ppNew[element[0]]->r.width * scale_templ;
    double h0 = (double)ppNew[element[0]]->r.height * scale_templ;
    double w1 = (double)ppNew[element[1]]->r.width * scale_templ;
    double h1 = (double)ppNew[element[1]]->r.height * scale_templ;

    int energy = ppNew[element[0]]->iEnergy + ppNew[element[1]]->iEnergy +
        - 2 * (ppNew[element[0]]->nRectsInThis - ppNew[element[1]]->nRectsInThis) +
        (int)pow(w0 - (double)rTempl[element[0]].width, 2) +
        (int)pow(h0 - (double)rTempl[element[0]].height, 2) +
        (int)pow(w1 - (double)rTempl[element[1]].width, 2) +
        (int)pow(h1 - (double)rTempl[element[1]].height, 2) +
        (int)pow(new_d - prev_d, 2) +
        0;

    return energy;
}

inline double CalculateTransformationLMS3( CvPoint* pTemplPoints,
                                   CvPoint* pSrcPoints,
                                   double*       pdbAverageScale,
                                   double*       pdbAverageRotate,
                                   double*       pdbAverageShiftX,
                                   double*       pdbAverageShiftY )
{
//    double WS = 0;
    double dbAverageScale = 1;
    double dbAverageRotate = 0;
    double dbAverageShiftX = 0;
    double dbAverageShiftY = 0;
    double dbLMS = 0;

    assert( NULL != pTemplPoints);
    assert( NULL != pSrcPoints);

    double dbXt = double(pTemplPoints[0].x + pTemplPoints[1].x + pTemplPoints[2].x) / 3.0;
    double dbYt = double(pTemplPoints[0].y + pTemplPoints[1].y + pTemplPoints[2].y ) / 3.0;
    double dbXs = double(pSrcPoints[0].x + pSrcPoints[1].x + pSrcPoints[2].x) / 3.0;
    double dbYs = double(pSrcPoints[0].y + pSrcPoints[1].y + pSrcPoints[2].y) / 3.0;

    double dbXtXt = double(pow2(pTemplPoints[0].x) + pow2(pTemplPoints[1].x) + pow2(pTemplPoints[2].x)) / 3.0;
    double dbYtYt = double(pow2(pTemplPoints[0].y) + pow2(pTemplPoints[1].y) + pow2(pTemplPoints[2].y)) / 3.0;

    double dbXsXs = double(pow2(pSrcPoints[0].x) + pow2(pSrcPoints[1].x) + pow2(pSrcPoints[2].x)) / 3.0;
    double dbYsYs = double(pow2(pSrcPoints[0].y) + pow2(pSrcPoints[1].y) + pow2(pSrcPoints[2].y)) / 3.0;

    double dbXtXs = double(pTemplPoints[0].x * pSrcPoints[0].x +
        pTemplPoints[1].x * pSrcPoints[1].x +
        pTemplPoints[2].x * pSrcPoints[2].x) / 3.0;
    double dbYtYs = double(pTemplPoints[0].y * pSrcPoints[0].y +
        pTemplPoints[1].y * pSrcPoints[1].y +
        pTemplPoints[2].y * pSrcPoints[2].y) / 3.0;

    double dbXtYs = double(pTemplPoints[0].x * pSrcPoints[0].y +
        pTemplPoints[1].x * pSrcPoints[1].y +
        pTemplPoints[2].x * pSrcPoints[2].y) / 3.0;
    double dbYtXs = double(pTemplPoints[0].y * pSrcPoints[0].x +
        pTemplPoints[1].y * pSrcPoints[1].x +
        pTemplPoints[2].y * pSrcPoints[2].x ) / 3.0;

    dbXtXt -= dbXt * dbXt;
    dbYtYt -= dbYt * dbYt;

    dbXsXs -= dbXs * dbXs;
    dbYsYs -= dbYs * dbYs;

    dbXtXs -= dbXt * dbXs;
    dbYtYs -= dbYt * dbYs;

    dbXtYs -= dbXt * dbYs;
    dbYtXs -= dbYt * dbXs;

    dbAverageRotate = atan2( dbXtYs - dbYtXs, dbXtXs + dbYtYs );

    double cosR = cos(dbAverageRotate);
    double sinR = sin(dbAverageRotate);
    double del = dbXsXs + dbYsYs;
    if( del != 0 )
    {
        dbAverageScale = (double(dbXtXs + dbYtYs) * cosR + double(dbXtYs - dbYtXs) * sinR) / del;
        dbLMS = dbXtXt + dbYtYt - ((double)pow(dbXtXs + dbYtYs,2) + (double)pow(dbXtYs - dbYtXs,2)) / del;
    }

    dbAverageShiftX = double(dbXt) - dbAverageScale * (double(dbXs) * cosR + double(dbYs) * sinR);
    dbAverageShiftY = double(dbYt) - dbAverageScale * (double(dbYs) * cosR - double(dbXs) * sinR);

    if( pdbAverageScale != NULL ) *pdbAverageScale = dbAverageScale;
    if( pdbAverageRotate != NULL ) *pdbAverageRotate = dbAverageRotate;
    if( pdbAverageShiftX != NULL ) *pdbAverageShiftX = dbAverageShiftX;
    if( pdbAverageShiftY != NULL ) *pdbAverageShiftY = dbAverageShiftY;

    assert(dbLMS >= 0);
    return dbLMS;
}

inline double CalculateTransformationLMS3_0( CvPoint* pTemplPoints, CvPoint* pSrcPoints)
{
    double dbLMS = 0;

    assert( NULL != pTemplPoints);
    assert( NULL != pSrcPoints);

    double dbXt = double(pTemplPoints[0].x + pTemplPoints[1].x + pTemplPoints[2].x) / 3.0;
    double dbYt = double(pTemplPoints[0].y + pTemplPoints[1].y + pTemplPoints[2].y ) / 3.0;
    double dbXs = double(pSrcPoints[0].x + pSrcPoints[1].x + pSrcPoints[2].x) / 3.0;
    double dbYs = double(pSrcPoints[0].y + pSrcPoints[1].y + pSrcPoints[2].y) / 3.0;

    double dbXtXt = double(pow2(pTemplPoints[0].x) + pow2(pTemplPoints[1].x) + pow2(pTemplPoints[2].x)) / 3.0;
    double dbYtYt = double(pow2(pTemplPoints[0].y) + pow2(pTemplPoints[1].y) + pow2(pTemplPoints[2].y)) / 3.0;

    double dbXsXs = double(pow2(pSrcPoints[0].x) + pow2(pSrcPoints[1].x) + pow2(pSrcPoints[2].x)) / 3.0;
    double dbYsYs = double(pow2(pSrcPoints[0].y) + pow2(pSrcPoints[1].y) + pow2(pSrcPoints[2].y)) / 3.0;

    double dbXtXs = double(pTemplPoints[0].x * pSrcPoints[0].x +
        pTemplPoints[1].x * pSrcPoints[1].x +
        pTemplPoints[2].x * pSrcPoints[2].x) / 3.0;
    double dbYtYs = double(pTemplPoints[0].y * pSrcPoints[0].y +
        pTemplPoints[1].y * pSrcPoints[1].y +
        pTemplPoints[2].y * pSrcPoints[2].y) / 3.0;

    double dbXtYs = double(pTemplPoints[0].x * pSrcPoints[0].y +
        pTemplPoints[1].x * pSrcPoints[1].y +
        pTemplPoints[2].x * pSrcPoints[2].y) / 3.0;
    double dbYtXs = double(pTemplPoints[0].y * pSrcPoints[0].x +
        pTemplPoints[1].y * pSrcPoints[1].x +
        pTemplPoints[2].y * pSrcPoints[2].x ) / 3.0;

    dbXtXt -= dbXt * dbXt;
    dbYtYt -= dbYt * dbYt;

    dbXsXs -= dbXs * dbXs;
    dbYsYs -= dbYs * dbYs;

    dbXtXs -= dbXt * dbXs;
    dbYtYs -= dbYt * dbYs;

    dbXtYs -= dbXt * dbYs;
    dbYtXs -= dbYt * dbXs;

    double del = dbXsXs + dbYsYs;
    if( del != 0 )
        dbLMS = dbXtXt + dbYtYt - ((double)pow(dbXtXs + dbYtYs,2) + (double)pow(dbXtYs - dbYtXs,2)) / del;
    return dbLMS;
}
