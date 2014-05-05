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
// Copyright (C) 2000, Intel Corporation, rights reserved.
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
This file contain simple implementation of BlobTrackerAuto virtual interface
This module just connected other low level 3 modules
(foreground estimator + BlobDetector + BlobTracker)
and some simple code to detect "lost tracking"
The track is lost when integral of foreground mask image by blob area has low value
*/
#include "precomp.hpp"
#include <time.h>

/* list of Blob Detection modules */
CvBlobDetector* cvCreateBlobDetectorSimple();

/* Get frequency for each module time working estimation: */
static double FREQ = 1000*cvGetTickFrequency();

#if 1
#define COUNTNUM 100
#define TIME_BEGIN() \
{\
    static double   _TimeSum = 0;\
    static int      _Count = 0;\
    static int      _CountBlob = 0;\
    int64           _TickCount = cvGetTickCount();\

#define TIME_END(_name_,_BlobNum_)    \
    _Count++;\
    _CountBlob+=_BlobNum_;\
    _TimeSum += (cvGetTickCount()-_TickCount)/FREQ;\
    if(m_TimesFile)if(_Count%COUNTNUM==0)\
    { \
        FILE* out = fopen(m_TimesFile,"at");\
        if(out)\
        {\
            fprintf(out,"ForFrame Frame: %d %s %f on %f blobs\n",_Count,_name_, _TimeSum/COUNTNUM,((float)_CountBlob)/COUNTNUM);\
            if(_CountBlob>0)fprintf(out,"ForBlob  Frame: %d %s - %f\n",_Count,_name_, _TimeSum/_CountBlob);\
            fclose(out);\
        }\
        _TimeSum = 0;\
        _CountBlob = 0;\
    }\
}
#else
#define TIME_BEGIN()
#define TIME_END(_name_)
#endif

/* Special extended blob structure for auto blob tracking: */
typedef struct CvBlobTrackAuto
{
    CvBlob  blob;
    int     BadFrames;
} CvBlobTrackAuto;

class CvBlobTrackerAuto1: public CvBlobTrackerAuto
{
public:
    CvBlobTrackerAuto1(CvBlobTrackerAutoParam1* param);
    ~CvBlobTrackerAuto1();
    CvBlob* GetBlob(int index){return m_BlobList.GetBlob(index);}
    CvBlob* GetBlobByID(int ID){return m_BlobList.GetBlobByID(ID);}
    int     GetBlobNum(){return m_BlobList.GetBlobNum();}
    virtual IplImage* GetFGMask(){return m_pFGMask;}
    float   GetState(int BlobID){return m_pBTA?m_pBTA->GetState(BlobID):0;}
    const char*   GetStateDesc(int BlobID){return m_pBTA?m_pBTA->GetStateDesc(BlobID):NULL;}
    /* Return 0 if trajectory is normal;
       return >0 if trajectory abnormal. */
    void Process(IplImage* pImg, IplImage* pMask = NULL);
    void Release(){delete this;}

private:
    IplImage*               m_pFGMask;
    int                     m_FGTrainFrames;
    CvFGDetector*           m_pFG; /* Pointer to foreground mask detector module. */
    CvBlobTracker*          m_pBT; /* Pointer to Blob tracker module. */
    int                     m_BTDel;
    int                     m_BTReal;
    CvBlobDetector*         m_pBD; /* Pointer to Blob detector module. */
    int                     m_BDDel;
    CvBlobTrackGen*         m_pBTGen;
    CvBlobTrackPostProc*    m_pBTPostProc;
    int                     m_UsePPData;
    CvBlobTrackAnalysis*    m_pBTA; /* Blob trajectory analyser. */
    CvBlobSeq               m_BlobList;
    int                     m_FrameCount;
    int                     m_NextBlobID;
    const char*                   m_TimesFile;

public:
    virtual void SaveState(CvFileStorage* fs)
    {
        cvWriteInt(fs,"FrameCount",m_FrameCount);
        cvWriteInt(fs,"NextBlobID",m_NextBlobID);
        m_BlobList.Write(fs,"BlobList");
    };

    virtual void LoadState(CvFileStorage* fs, CvFileNode* node)
    {
        CvFileNode* BlobListNode = cvGetFileNodeByName(fs,node,"BlobList");
        m_FrameCount = cvReadIntByName(fs,node, "FrameCount", m_FrameCount);
        m_NextBlobID = cvReadIntByName(fs,node, "NextBlobID", m_NextBlobID);
        if(BlobListNode)
        {
            m_BlobList.Load(fs,BlobListNode);
        }
    };
};

/* Auto Blob tracker creater (sole interface function for this file) */
CvBlobTrackerAuto* cvCreateBlobTrackerAuto1(CvBlobTrackerAutoParam1* param)
{
    return (CvBlobTrackerAuto*)new CvBlobTrackerAuto1(param);
}

/* Constructor of auto blob tracker: */
CvBlobTrackerAuto1::CvBlobTrackerAuto1(CvBlobTrackerAutoParam1* param):m_BlobList(sizeof(CvBlobTrackAuto))
{
    m_BlobList.AddFormat("i");
    m_TimesFile = NULL;
    AddParam("TimesFile",&m_TimesFile);

    m_NextBlobID = 0;
    m_pFGMask = NULL;
    m_FrameCount = 0;

    m_FGTrainFrames = param?param->FGTrainFrames:0;
    m_pFG = param?param->pFG:0;

    m_BDDel = 0;
    m_pBD = param?param->pBD:NULL;
    m_BTDel = 0;
    m_pBT = param?param->pBT:NULL;;
    m_BTReal = m_pBT?m_pBT->IsModuleName("BlobTrackerReal"):0;

    m_pBTGen = param?param->pBTGen:NULL;

    m_pBTA = param?param->pBTA:NULL;

    m_pBTPostProc = param?param->pBTPP:NULL;
    m_UsePPData = param?param->UsePPData:0;

    /* Create default submodules: */
    if(m_pBD==NULL)
    {
        m_pBD = cvCreateBlobDetectorSimple();
        m_BDDel = 1;
    }

    if(m_pBT==NULL)
    {
        m_pBT = cvCreateBlobTrackerMS();
        m_BTDel = 1;
    }

    SetModuleName("Auto1");

} /* CvBlobTrackerAuto1::CvBlobTrackerAuto1 */

/* Destructor for auto blob tracker: */
CvBlobTrackerAuto1::~CvBlobTrackerAuto1()
{
    if(m_BDDel)m_pBD->Release();
    if(m_BTDel)m_pBT->Release();
}

void CvBlobTrackerAuto1::Process(IplImage* pImg, IplImage* pMask)
{
    int         CurBlobNum = 0;
    IplImage*   pFG = pMask;

    /* Bump frame counter: */
    m_FrameCount++;

    if(m_TimesFile)
    {
        static int64  TickCount = cvGetTickCount();
        static double TimeSum = 0;
        static int Count = 0;
        Count++;

        if(Count%100==0)
        {
#ifndef WINCE
            time_t ltime;
            time( &ltime );
            char* stime = ctime( &ltime );
#else
            /* WINCE does not have above POSIX functions (time,ctime) */
            const char* stime = " wince ";
#endif
            FILE* out = fopen(m_TimesFile,"at");
            double Time;
            TickCount = cvGetTickCount()-TickCount;
            Time = TickCount/FREQ;
            TimeSum += Time;
            if(out){fprintf(out,"- %sFrame: %d ALL_TIME - %f\n",stime,Count,TimeSum/1000);fclose(out);}

            TimeSum = 0;
            TickCount = cvGetTickCount();
        }
    }

    /* Update BG model: */
    TIME_BEGIN()

    if(m_pFG)
    {   /* If FG detector is needed: */
        m_pFG->Process(pImg);
        pFG = m_pFG->GetMask();
    }   /* If FG detector is needed. */

    TIME_END("FGDetector",-1)

    m_pFGMask = pFG; /* For external use. */

    /*if(m_pFG && m_pFG->GetParam("DebugWnd") == 1)
    {// debug foreground result
        IplImage *pFG = m_pFG->GetMask();
        if(pFG)
        {
            cvNamedWindow("FG",0);
            cvShowImage("FG", pFG);
        }
    }*/

    /* Track blobs: */
    TIME_BEGIN()
    if(m_pBT)
    {
        m_pBT->Process(pImg, pFG);

        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Update data of tracked blob list: */
            CvBlob* pB = m_BlobList.GetBlob(i-1);
            int     BlobID = CV_BLOB_ID(pB);
            int     idx = m_pBT->GetBlobIndexByID(BlobID);
            m_pBT->ProcessBlob(idx, pB, pImg, pFG);
            pB->ID = BlobID;
        }
        CurBlobNum = m_pBT->GetBlobNum();
    }
    TIME_END("BlobTracker",CurBlobNum)

    /* This part should be removed: */
    if(m_BTReal && m_pBT)
    {   /* Update blob list (detect new blob for real blob tracker): */
        for(int i=m_pBT->GetBlobNum(); i>0; --i)
        {   /* Update data of tracked blob list: */
            CvBlob* pB = m_pBT->GetBlob(i-1);
            if(pB && m_BlobList.GetBlobByID(CV_BLOB_ID(pB)) == NULL )
            {
                CvBlobTrackAuto     NewB;
                NewB.blob = pB[0];
                NewB.BadFrames = 0;
                m_BlobList.AddBlob((CvBlob*)&NewB);
            }
        }   /* Next blob. */

        /* Delete blobs: */
        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Update tracked-blob list: */
            CvBlob* pB = m_BlobList.GetBlob(i-1);
            if(pB && m_pBT->GetBlobByID(CV_BLOB_ID(pB)) == NULL )
            {
                m_BlobList.DelBlob(i-1);
            }
        }   /* Next blob. */
    }   /* Update bloblist. */


    TIME_BEGIN()
    if(m_pBTPostProc)
    {   /* Post-processing module: */
        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Update tracked-blob list: */
            CvBlob* pB = m_BlobList.GetBlob(i-1);
            m_pBTPostProc->AddBlob(pB);
        }
        m_pBTPostProc->Process();

        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Update tracked-blob list: */
            CvBlob* pB = m_BlobList.GetBlob(i-1);
            int     BlobID = CV_BLOB_ID(pB);
            CvBlob* pBN = m_pBTPostProc->GetBlobByID(BlobID);

            if(pBN && m_UsePPData && pBN->w >= CV_BLOB_MINW && pBN->h >= CV_BLOB_MINH)
            {   /* Set new data for tracker: */
                m_pBT->SetBlobByID(BlobID, pBN );
            }

            if(pBN)
            {   /* Update blob list with results from postprocessing: */
                pB[0] = pBN[0];
            }
        }
    }   /* Post-processing module. */

    TIME_END("PostProcessing",CurBlobNum)

    /* Blob deleter (experimental and simple): */
    TIME_BEGIN()
    if(pFG)
    {   /* Blob deleter: */
        int i;
        if(!m_BTReal)for(i=m_BlobList.GetBlobNum();i>0;--i)
        {   /* Check all blobs on list: */
            CvBlobTrackAuto* pB = (CvBlobTrackAuto*)(m_BlobList.GetBlob(i-1));
            int     Good = 0;
            int     w=pFG->width;
            int     h=pFG->height;
            CvRect  r = CV_BLOB_RECT(pB);
            CvMat   mat;
            double  aver = 0;
            double  area = CV_BLOB_WX(pB)*CV_BLOB_WY(pB);
            if(r.x < 0){r.width += r.x;r.x = 0;}
            if(r.y < 0){r.height += r.y;r.y = 0;}
            if(r.x+r.width>=w){r.width = w-r.x-1;}
            if(r.y+r.height>=h){r.height = h-r.y-1;}

            if(r.width > 4 && r.height > 4 && r.x < w && r.y < h &&
                r.x >=0 && r.y >=0 &&
                r.x+r.width < w && r.y+r.height < h && area > 0)
            {
                aver = cvSum(cvGetSubRect(pFG,&mat,r)).val[0] / area;
                /* if mask in blob area exists then its blob OK*/
                if(aver > 0.1*255)Good = 1;
            }
            else
            {
                pB->BadFrames+=2;
            }

            if(Good)
            {
                pB->BadFrames = 0;
            }
            else
            {
                pB->BadFrames++;
            }
        }   /* Next blob: */

        /* Check error count: */
        for(i=0; i<m_BlobList.GetBlobNum(); ++i)
        {
            CvBlobTrackAuto* pB = (CvBlobTrackAuto*)m_BlobList.GetBlob(i);

            if(pB->BadFrames>3)
            {   /* Delete such objects */
                /* from tracker...     */
                m_pBT->DelBlobByID(CV_BLOB_ID(pB));

                /* ... and from local list: */
                m_BlobList.DelBlob(i);
                i--;
            }
        }   /* Check error count for next blob. */
    }   /*  Blob deleter. */

    TIME_END("BlobDeleter",m_BlobList.GetBlobNum())

    /* Update blobs: */
    TIME_BEGIN()
    if(m_pBT)
        m_pBT->Update(pImg, pFG);
    TIME_END("BlobTrackerUpdate",CurBlobNum)

    /* Detect new blob: */
    TIME_BEGIN()
    if(!m_BTReal && m_pBD && pFG && (m_FrameCount > m_FGTrainFrames) )
    {   /* Detect new blob: */
        static CvBlobSeq    NewBlobList;
        CvBlobTrackAuto     NewB;

        NewBlobList.Clear();

        if(m_pBD->DetectNewBlob(pImg, pFG, &NewBlobList, &m_BlobList))
        {   /* Add new blob to tracker and blob list: */
            int i;
            IplImage* pmask = pFG;

            /*if(0)if(NewBlobList.GetBlobNum()>0 && pFG )
            {// erode FG mask (only for FG_0 and MS1||MS2)
                pmask = cvCloneImage(pFG);
                cvErode(pFG,pmask,NULL,2);
            }*/

            for(i=0; i<NewBlobList.GetBlobNum(); ++i)
            {
                CvBlob* pBN = NewBlobList.GetBlob(i);

                if(pBN && pBN->w >= CV_BLOB_MINW && pBN->h >= CV_BLOB_MINH)
                {
                    pBN->ID = m_NextBlobID;

                    CvBlob* pB = m_pBT->AddBlob(pBN, pImg, pmask );
                    if(pB)
                    {
                        NewB.blob = pB[0];
                        NewB.BadFrames = 0;
                        m_BlobList.AddBlob((CvBlob*)&NewB);
                        m_NextBlobID++;
                    }
                }
            }   /* Add next blob from list of detected blob. */

            if(pmask != pFG) cvReleaseImage(&pmask);

        }   /* Create and add new blobs and trackers. */

    }   /*  Detect new blob. */

    TIME_END("BlobDetector",-1)

    TIME_BEGIN()
    if(m_pBTGen)
    {   /* Run track generator: */
        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Update data of tracked blob list: */
            CvBlob* pB = m_BlobList.GetBlob(i-1);
            m_pBTGen->AddBlob(pB);
        }
        m_pBTGen->Process(pImg, pFG);
    }   /* Run track generator: */
    TIME_END("TrajectoryGeneration",-1)

    TIME_BEGIN()
    if(m_pBTA)
    {   /* Trajectory analysis module: */
        int i;
        for(i=m_BlobList.GetBlobNum(); i>0; i--)
            m_pBTA->AddBlob(m_BlobList.GetBlob(i-1));

        m_pBTA->Process(pImg, pFG);

    }   /* Trajectory analysis module. */

    TIME_END("TrackAnalysis",m_BlobList.GetBlobNum())

} /* CvBlobTrackerAuto1::Process */
