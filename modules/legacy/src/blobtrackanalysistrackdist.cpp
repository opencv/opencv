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

#include "precomp.hpp"

typedef struct DefTrackPoint
{
    float x,y,r,vx,vy,v;
} DefTrackPoint;

class DefTrackRec
{
private:
    int     ID;
public:
    DefTrackRec(int id = 0,int BlobSize = sizeof(DefTrackPoint))
    {
        ID = id;
        m_pMem = cvCreateMemStorage();
        m_pSeq = cvCreateSeq(0,sizeof(CvSeq),BlobSize,m_pMem);
    }
    ~DefTrackRec()
    {
        cvReleaseMemStorage(&m_pMem);
    };
    inline DefTrackPoint* GetPoint(int PointIndex)
    {
        return (DefTrackPoint*)cvGetSeqElem(m_pSeq,PointIndex);
    };
    inline void DelPoint(int PointIndex)
    {
        cvSeqRemove(m_pSeq,PointIndex);
    };
    inline void Clear()
    {
        cvClearSeq(m_pSeq);
    };
    inline void AddPoint(float x, float y, float r)
    {
        DefTrackPoint   p = {x,y,r,0,0,0};
        int             Num = GetPointNum();

        if(Num > 0)
        {
            DefTrackPoint* pPrev = GetPoint(Num-1);
            float   Alpha = 0.8f;
            float  dx = x-pPrev->x;
            float  dy = y-pPrev->y;
            p.vx = Alpha*dx+(1-Alpha)*pPrev->vx;
            p.vy = Alpha*dy+(1-Alpha)*pPrev->vy;
            p.v  = Alpha*dx+(1-Alpha)*pPrev->v;
        }
        AddPoint(&p);
    }

    inline void AddPoint(DefTrackPoint* pB)
    {   /* Add point and recalculate last velocities: */
        int     wnd=3;
        int     Num;
        int     i;
        cvSeqPush(m_pSeq,pB);

        Num = GetPointNum();

        for(i=MAX(0,Num-wnd-1); i<Num; ++i)
        {   /* Next updating point: */
            DefTrackPoint*  p = GetPoint(i);
            int             j0 = i - wnd;
            int             j1 = i + wnd;

            if(j0<0) j0 = 0;
            if(j1>=Num)j1=Num-1;

            if(j1>j0)
            {
                float           dt = (float)(j1-j0);
                DefTrackPoint*  p0 = GetPoint(j0);
                DefTrackPoint*  p1 = GetPoint(j1);
                p->vx = (p1->x - p0->x) / dt;
                p->vy = (p1->y - p0->y) / dt;
                p->v = (float)sqrt(p->vx*p->vx+p->vy*p->vy);
            }
        } /* Next updating point. */

#if 0
        if(0)
        {   /* Debug: */
            int i;
            printf("Blob %d: ",ID);

            for(i=0; i<GetPointNum(); ++i)
            {
                DefTrackPoint*  p = GetPoint(i);
                printf(",(%.2f,%.2f,%f.2)",p->vx,p->vy,p->v);
            }
            printf("\n");
        }
#endif
    };
    inline int GetPointNum()
    {
        return m_pSeq->total;
    };
private:
    CvMemStorage*   m_pMem;
    CvSeq*          m_pSeq;
};

/* Fill array pIdxPairs by pair of index of correspondent blobs. */
/* Return number of pairs.                                       */
/* pIdxPairs must have size not less that 2*(pSeqNum+pSeqTNum)   */
/* pTmp is pointer to memory which size is pSeqNum*pSeqTNum*16   */
typedef struct DefMatch
{
    int     Idx;  /* Previous best blob index.          */
    int     IdxT; /* Previous best template blob index. */
    double  D;    /* Blob to blob distance sum.         */
} DefMatch;

static int cvTrackMatch(DefTrackRec* pSeq, int MaxLen, DefTrackRec* pSeqT, int* pIdxPairs, void* pTmp)
{
    int         NumPair = 0;
    DefMatch*   pMT = (DefMatch*)pTmp;
    int         Num = pSeq->GetPointNum();
    int         NumT = pSeqT->GetPointNum();
    int         i,it;
    int         i0=0; /* Last point in the track sequence. */

    if(MaxLen > 0 && Num > MaxLen)
    {   /* Set new point seq len and new last point in this seq: */
        Num = MaxLen;
        i0 = pSeq->GetPointNum() - Num;
    }

    for(i=0; i<Num; ++i)
    {   /* For each point row: */
        for(it=0; it<NumT; ++it)
        {   /* For each point template column: */
            DefTrackPoint*  pB = pSeq->GetPoint(i+i0);
            DefTrackPoint*  pBT = pSeqT->GetPoint(it);
            DefMatch*       pMT_cur = pMT + i*NumT + it;
            double          dx = pB->x-pBT->x;
            double          dy = pB->y-pBT->y;
            double          D = dx*dx+dy*dy;
            int             DI[3][2] = {{-1,-1},{-1,0},{0,-1}};
            int             iDI;

            pMT_cur->D = D;
            pMT_cur->Idx = -1;
            pMT_cur->IdxT = 0;

            if(i==0) continue;

            for(iDI=0; iDI<3; ++iDI)
            {
                int         i_prev = i+DI[iDI][0];
                int         it_prev = it+DI[iDI][1];

                if(i_prev >= 0 && it_prev>=0)
                {
                    double D_cur = D+pMT[NumT*i_prev+it_prev].D;

                    if(pMT_cur->D > D_cur || (pMT_cur->Idx<0) )
                    {   /* Set new best local way: */
                        pMT_cur->D = D_cur;
                        pMT_cur->Idx = i_prev;
                        pMT_cur->IdxT = it_prev;
                    }
                }
            } /* Check next direction. */
        } /* Fill next colum from table. */
    } /* Fill next row. */

    {   /* Back tracking. */
        /* Find best end in template: */
        int         it_best = 0;
        DefMatch*   pMT_best = pMT + (Num-1)*NumT;
        i = Num-1; /* set current i to last position */

        for(it=1; it<NumT; ++it)
        {
            DefMatch* pMT_new = pMT + it + i*NumT;

            if(pMT_best->D > pMT_new->D)
            {
                pMT_best->D = pMT_new->D;
                it_best = it;
            }
        } /* Find best end template point. */

        /* Back tracking whole sequence: */
        for(it = it_best;i>=0 && it>=0;)
        {
            DefMatch* pMT_new = pMT + it + i*NumT;
            pIdxPairs[2*NumPair] = i+i0;
            pIdxPairs[2*NumPair+1] = it;
            NumPair++;

            it = pMT_new->IdxT;
            i = pMT_new->Idx;
        }
    } /* End back tracing. */

    return NumPair;
} /* cvTrackMatch. */

typedef struct DefTrackForDist
{
    CvBlob                  blob;
    DefTrackRec*            pTrack;
    int                     LastFrame;
    float                   state;
    /* for debug */
    int                     close;
} DefTrackForDist;

class CvBlobTrackAnalysisTrackDist : public CvBlobTrackAnalysis
{
    /*---------------- Internal functions: --------------------*/
private:
    const char*               m_pDebugAVIName; /* For debugging. */
  //CvVideoWriter*      m_pDebugAVI;     /* For debugging. */
    IplImage*           m_pDebugImg;     /* For debugging. */

    char                m_DataFileName[1024];
    CvBlobSeq           m_Tracks;
    CvBlobSeq           m_TrackDataBase;
    int                 m_Frame;
    void*               m_pTempData;
    int                 m_TempDataSize;
    int                 m_TraceLen;
    float               m_AbnormalThreshold;
    float               m_PosThreshold;
    float               m_VelThreshold;
    inline void* ReallocTempData(int Size)
    {
        if(Size <= m_TempDataSize && m_pTempData) return m_pTempData;
        cvFree(&m_pTempData);
        m_TempDataSize = 0;
        m_pTempData = cvAlloc(Size);
        if(m_pTempData) m_TempDataSize = Size;
        return m_pTempData;
    } /* ReallocTempData. */

public:
    CvBlobTrackAnalysisTrackDist():m_Tracks(sizeof(DefTrackForDist)),m_TrackDataBase(sizeof(DefTrackForDist))
    {
        m_pDebugImg = 0;
        //m_pDebugAVI = 0;
        m_Frame = 0;
        m_pTempData = NULL;
        m_TempDataSize = 0;

        m_pDebugAVIName = NULL;
        AddParam("DebugAVI",&m_pDebugAVIName);
        CommentParam("DebugAVI","Name of AVI file to save images from debug window");

        m_TraceLen = 50;
        AddParam("TraceLen",&m_TraceLen);
        CommentParam("TraceLen","Length (in frames) of trajectory part that is used for comparison");

        m_AbnormalThreshold = 0.02f;
        AddParam("AbnormalThreshold",&m_AbnormalThreshold);
        CommentParam("AbnormalThreshold","If trajectory is equal with less then <AbnormalThreshold*DataBaseTrackNum> tracks then trajectory is abnormal");

        m_PosThreshold = 1.25;
        AddParam("PosThreshold",&m_PosThreshold);
        CommentParam("PosThreshold","Minimal allowd distance in blob width that is allowed");

        m_VelThreshold = 0.5;
        AddParam("VelThreshold",&m_VelThreshold);
        CommentParam("VelThreshold","Minimal allowed relative difference between blob speed");

        SetModuleName("TrackDist");

    } /* Constructor. */

    ~CvBlobTrackAnalysisTrackDist()
    {
        int i;
        for(i=m_Tracks.GetBlobNum(); i>0; --i)
        {
            DefTrackForDist* pF = (DefTrackForDist*)m_Tracks.GetBlob(i-1);
            delete pF->pTrack;
        }
        if(m_pDebugImg) cvReleaseImage(&m_pDebugImg);
        //if(m_pDebugAVI) cvReleaseVideoWriter(&m_pDebugAVI);
    } /* Destructor. */

    /*----------------- Interface: --------------------*/
    virtual void    AddBlob(CvBlob* pBlob)
    {
        DefTrackForDist* pF = (DefTrackForDist*)m_Tracks.GetBlobByID(CV_BLOB_ID(pBlob));

        if(pF == NULL)
        {   /* Create new TRack record: */
            DefTrackForDist F;
            F.state = 0;
            F.blob = pBlob[0];
            F.LastFrame = m_Frame;
            F.pTrack = new DefTrackRec(CV_BLOB_ID(pBlob));
            m_Tracks.AddBlob((CvBlob*)&F);
            pF = (DefTrackForDist*)m_Tracks.GetBlobByID(CV_BLOB_ID(pBlob));
        }

        assert(pF);
        assert(pF->pTrack);
        pF->pTrack->AddPoint(pBlob->x,pBlob->y,pBlob->w*0.5f);
        pF->blob = pBlob[0];
        pF->LastFrame = m_Frame;
    };

    virtual void Process(IplImage* pImg, IplImage* /*pFG*/)
    {
        double          MinTv = pImg->width/1440.0; /* minimal threshold for speed difference */
        double          MinTv2 = MinTv*MinTv;

        for(int i=m_Tracks.GetBlobNum(); i>0; --i)
        {
            DefTrackForDist* pF = (DefTrackForDist*)m_Tracks.GetBlob(i-1);
            pF->state = 0;

            if(pF->LastFrame == m_Frame || pF->LastFrame+1 == m_Frame)
            {   /* Process one blob trajectory: */
                int NumEq = 0;
                int it;

                for(it=m_TrackDataBase.GetBlobNum(); it>0; --it)
                {   /* Check template: */
                    DefTrackForDist*   pFT = (DefTrackForDist*)m_TrackDataBase.GetBlob(it-1);
                    int         Num = pF->pTrack->GetPointNum();
                    int         NumT = pFT->pTrack->GetPointNum();
                    int*        pPairIdx = (int*)ReallocTempData(sizeof(int)*2*(Num+NumT)+sizeof(DefMatch)*Num*NumT);
                    void*       pTmpData = pPairIdx+2*(Num+NumT);
                    int         PairNum = 0;
                    int         k;
                    int         Equal = 1;
                    int         UseVel = 0;
                    int         UsePos = 0;

                    if(i==it) continue;

                    /* Match track: */
                    PairNum = cvTrackMatch( pF->pTrack, m_TraceLen, pFT->pTrack, pPairIdx, pTmpData );
                    Equal = MAX(1,cvRound(PairNum*0.1));

                    UseVel = 3*pF->pTrack->GetPointNum() > m_TraceLen;
                    UsePos = 10*pF->pTrack->GetPointNum() > m_TraceLen;

                    {   /* Check continues: */
                        float   D;
                        int     DI = pPairIdx[0*2+0]-pPairIdx[(PairNum-1)*2+0];
                        int     DIt = pPairIdx[0*2+1]-pPairIdx[(PairNum-1)*2+1];
                        if(UseVel && DI != 0)
                        {
                            D = (float)(DI-DIt)/(float)DI;
                            if(fabs(D)>m_VelThreshold)Equal=0;
                            if(fabs(D)>m_VelThreshold*0.5)Equal/=2;
                        }
                    }   /* Check continues. */

                    for(k=0; Equal>0 && k<PairNum; ++k)
                    {   /* Compare with threshold: */
                        int             j = pPairIdx[k*2+0];
                        int             jt = pPairIdx[k*2+1];
                        DefTrackPoint*  pB = pF->pTrack->GetPoint(j);
                        DefTrackPoint*  pBT = pFT->pTrack->GetPoint(jt);
                        double          dx = pB->x-pBT->x;
                        double          dy = pB->y-pBT->y;
                        double          dvx = pB->vx - pBT->vx;
                        double          dvy = pB->vy - pBT->vy;
                      //double          dv = pB->v - pBT->v;
                        double          D = dx*dx+dy*dy;
                        double          Td = pBT->r*m_PosThreshold;
                        double          dv2 = dvx*dvx+dvy*dvy;
                        double          Tv2 = (pBT->vx*pBT->vx+pBT->vy*pBT->vy)*m_VelThreshold*m_VelThreshold;
                        double          Tvm = pBT->v*m_VelThreshold;


                        if(Tv2 < MinTv2) Tv2 = MinTv2;
                        if(Tvm < MinTv) Tvm = MinTv;

                        /* Check trajectory position: */
                        if(UsePos && D > Td*Td)
                        {
                            Equal--;
                        }
                        else
                        /* Check trajectory velocity. */
                        /* Don't consider trajectory tail because its unstable for velocity computation. */
                        if(UseVel && j>5 && jt>5 && dv2 > Tv2 )
                        {
                            Equal--;
                        }
                    } /* Compare with threshold. */

                    if(Equal>0)
                    {
                        NumEq++;
                        pFT->close++;
                    }
                } /* Next template. */

                {   /* Calculate state: */
                    float   T = m_TrackDataBase.GetBlobNum() * m_AbnormalThreshold; /* calc threshold */

                    if(T>0)
                    {
                        pF->state = (T - NumEq)/(T*0.2f) + 0.5f;
                    }
                    if(pF->state<0)pF->state=0;
                    if(pF->state>1)pF->state=1;

                    /*if(0)if(pF->state>0)
                    {// if abnormal blob
                        printf("Abnormal blob(%d) %d < %f, state=%f\n",CV_BLOB_ID(pF),NumEq,T, pF->state);
                    }*/
                }   /* Calculate state. */
            }   /*  Process one blob trajectory. */
            else
            {   /* Move track to tracks data base: */
                m_TrackDataBase.AddBlob((CvBlob*)pF);
                m_Tracks.DelBlob(i-1);
            }
        } /* Next blob. */


        if(m_Wnd)
        {   /* Debug output: */

            if(m_pDebugImg==NULL)
                m_pDebugImg = cvCloneImage(pImg);
            else
                cvCopy(pImg, m_pDebugImg);

            for(int i=m_TrackDataBase.GetBlobNum(); i>0; --i)
            {   /* Draw all elements in track data base:  */
                int         j;
                DefTrackForDist*   pF = (DefTrackForDist*)m_TrackDataBase.GetBlob(i-1);
                CvScalar    color = CV_RGB(0,0,0);
                if(!pF->close) continue;
                if(pF->close)
                {
                    color = CV_RGB(0,0,255);
                }
                else
                {
                    color = CV_RGB(0,0,128);
                }

                for(j=pF->pTrack->GetPointNum(); j>0; j--)
                {
                    DefTrackPoint* pB = pF->pTrack->GetPoint(j-1);
                    int r = 0;//MAX(cvRound(pB->r),1);
                    cvCircle(m_pDebugImg, cvPoint(cvRound(pB->x),cvRound(pB->y)), r, color);
                }
                pF->close = 0;
            }   /* Draw all elements in track data base. */

            for(int i=m_Tracks.GetBlobNum(); i>0; --i)
            {   /* Draw all elements for all trajectories: */
                DefTrackForDist*    pF = (DefTrackForDist*)m_Tracks.GetBlob(i-1);
                int                 j;
                int                 c = cvRound(pF->state*255);
                CvScalar            color = CV_RGB(c,255-c,0);
                CvPoint             p = cvPointFrom32f(CV_BLOB_CENTER(pF));
                int                 x = cvRound(CV_BLOB_RX(pF)), y = cvRound(CV_BLOB_RY(pF));
                CvSize              s = cvSize(MAX(1,x), MAX(1,y));

                cvEllipse( m_pDebugImg,
                    p,
                    s,
                    0, 0, 360,
                    CV_RGB(c,255-c,0), cvRound(1+(0*c)/255) );

                for(j=pF->pTrack->GetPointNum(); j>0; j--)
                {
                    DefTrackPoint* pB = pF->pTrack->GetPoint(j-1);
                    if(pF->pTrack->GetPointNum()-j > m_TraceLen) break;
                    cvCircle(m_pDebugImg, cvPoint(cvRound(pB->x),cvRound(pB->y)), 0, color);
                }
                pF->close = 0;

            }   /* Draw all elements for all trajectories. */

            //cvNamedWindow("Tracks",0);
            //cvShowImage("Tracks", m_pDebugImg);
        } /* Debug output. */

#if 0
        if(m_pDebugImg && m_pDebugAVIName)
        {
            if(m_pDebugAVI==NULL)
            {   /* Create avi file for writing: */
                m_pDebugAVI = cvCreateVideoWriter(
                    m_pDebugAVIName,
                    CV_FOURCC('x','v','i','d'),
                    25,
                    cvSize(m_pDebugImg->width,m_pDebugImg->height));

                if(m_pDebugAVI == NULL)
                {
                    printf("WARNING!!! Can not create AVI file %s for writing\n",m_pDebugAVIName);
                }
            }   /* Create avi file for writing. */

            if(m_pDebugAVI)cvWriteFrame( m_pDebugAVI, m_pDebugImg );
        }   /* Write debug window to AVI file. */
#endif
        m_Frame++;
    };
    float GetState(int BlobID)
    {
        DefTrackForDist* pF = (DefTrackForDist*)m_Tracks.GetBlobByID(BlobID);
        return pF?pF->state:0.0f;
    };

    /* Return 0 if trajectory is normal;
       return >0 if trajectory abnormal. */
    virtual const char*   GetStateDesc(int BlobID)
    {
        if(GetState(BlobID)>0.5) return "abnormal";
        return NULL;
    }

    virtual void    SetFileName(char* DataBaseName)
    {
        m_DataFileName[0] = m_DataFileName[1000] = 0;
        if(DataBaseName)
        {
            strncpy(m_DataFileName,DataBaseName,1000);
            strcat(m_DataFileName, ".yml");
        }
    };

    virtual void    Release(){ delete this; };
};



CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisTrackDist()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisTrackDist;}

