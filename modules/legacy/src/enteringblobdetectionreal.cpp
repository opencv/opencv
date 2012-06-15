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

/*
This file contain implementation of virtual interface of CvBlobDetector
this implementation based on simple algorithm
new blob is detected when several successive frames contains connected componets
which have uniform motion with not high speed.
Also separation from border and already tracked blobs are considered.
*/

#include "precomp.hpp"

/* blob detector  based on real data (groundtruth data)*/
class CvBlobDetectorReal:public CvBlobDetector
{
protected:
    CvTestSeq*      m_pTestSeq;
    CvBlobSeq       m_DetectedBlobs;
    CvMemStorage*   m_pMem;

public:
    CvBlobDetectorReal(CvTestSeq* pTestSeq)
    {
        m_pTestSeq = pTestSeq;
        m_pMem = cvCreateMemStorage(0);
        SetModuleName("Real");
    }

    /* Destructor of BlobDetector: */
   ~CvBlobDetectorReal()
    {
        if(m_pMem) cvReleaseMemStorage(&m_pMem);
    }   /* cvReleaseBlobDetector */

    /* cvDetectNewBlobs:
     * Return 1 and fill blob pNewBlob with
     * blob parameters if new blob is detected:
     */
    int DetectNewBlob(IplImage* /*pImg*/, IplImage* /*pFGMask*/, CvBlobSeq* pNewBlobList, CvBlobSeq* /*pOldBlobList*/)
    {
        int         i;
        int         TestObjNum;
        IplImage*   pMask = NULL;
        IplImage*   pMaskCopy = NULL;
        CvSeq*      cnts = NULL;

        if(m_pTestSeq==NULL) return 0;
        TestObjNum = cvTestSeqGetObjectNum(m_pTestSeq);
        pMask = cvTestSeqGetFGMask(m_pTestSeq);
        if(pMask == NULL) return 0;
        pMaskCopy = cvCloneImage(pMask);
        assert(pMaskCopy);

        cvClearMemStorage(m_pMem);
        cvFindContours( pMaskCopy, m_pMem, &cnts, sizeof(CvContour), CV_RETR_EXTERNAL);
        cvReleaseImage(&pMaskCopy);

        for(i=0; i<TestObjNum; ++i)
        {   /* Check each object: */
            CvPoint2D32f    RealPos;
            CvPoint2D32f    RealSize;
            int             RealPosFlag = cvTestSeqGetObjectPos(m_pTestSeq,i,&RealPos);
            int             RealSizeFlag = cvTestSeqGetObjectSize(m_pTestSeq,i,&RealSize);

            if(!RealPosFlag) continue;
            if(m_DetectedBlobs.GetBlobByID(i)) continue;

            if(RealSizeFlag)
            {   /* Real size is known: */
                float W2 = RealSize.x * 0.5f;
                float H2 = RealSize.y * 0.5f;
                if( RealPos.x > W2 && RealPos.x < (pMask->width-W2) &&
                    RealPos.y > H2 && RealPos.y < (pMask->height-H2) )
                {   /* Yes!!  We found new blob, let's add it to list: */
                    CvBlob  NewBlob;
                    NewBlob.x = RealPos.x;
                    NewBlob.y = RealPos.y;
                    NewBlob.w = RealSize.x;
                    NewBlob.h = RealSize.y;
                    NewBlob.ID = i;
                    m_DetectedBlobs.AddBlob(&NewBlob);
                    pNewBlobList->AddBlob(&NewBlob);
                }
            }   /* Real size is known. */
            else
            {
                CvSeq*  cnt;
                if(m_DetectedBlobs.GetBlobByID(i)) continue;

                for(cnt=cnts; cnt; cnt=cnt->h_next)
                {
                    //CvBlob* pNewBlob = NULL;
                    CvBlob  NewBlob;
                    CvRect  r = cvBoundingRect( cnt );
                    float   x = RealPos.x - r.x;
                    float   y = RealPos.y - r.y;

                    if(x<0 || x > r.width || y < 0 || y > r.height ) continue;

                    if( r.x <= 1 ||
                        r.y <= 1 ||
                        r.x + r.width >= pMask->width - 2 ||
                        r.y + r.height >= pMask->height - 2 ) continue;

                    /* Yes!!  We found new blob, let's add it to list: */
                    NewBlob.x = RealPos.x;
                    NewBlob.y = RealPos.y;
                    NewBlob.w = (float)r.width;
                    NewBlob.h = (float)r.height;
                    NewBlob.ID = i;
                    m_DetectedBlobs.AddBlob(&NewBlob);
                    pNewBlobList->AddBlob(&NewBlob);
                }
            }   /* Check new blob entrance. */
        }   /* Check next object. */

        return pNewBlobList->GetBlobNum();

    }   /* cvDetectNewBlob */

    void Release(){delete this;};
};

/* Blob detector constructor: */
//CvBlobDetector* cvCreateBlobDetectorReal(CvTestSeq* pTestSeq){return new CvBlobDetectorReal(pTestSeq);}

