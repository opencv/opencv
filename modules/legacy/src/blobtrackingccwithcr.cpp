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

#include "precomp.hpp"

/* Blob (foreground-pixel connected-component) tracking with collision resolution.
 *
 * For entrypoints into the literature see:
 *
 *  A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking
 *  Arulampalam &t al, 2001, 15p
 *  http://www-clmc.usc.edu/publications/A/arulampalam-TSP2002.pdf
 *
 *  Particle Filters for Positioning, Navigation, and Tracking
 *  Gustafsson et al, 2002 12p
 *  http://www.control.isy.liu.se/~fredrik/reports/01SPpf4pos.pdf
 *
 *  Particle Filtering in High Clutter Environments
 *  Korhonen et al, 2005 4p
 *  http://www.cs.uku.fi/finsig05/papers/paper26_FINSIG05.pdf
 *
 *   Appearance Models for Occlusion Handling
 *   Andrew Senior &t al, 8p 2001
 *   http://www.research.ibm.com/peoplevision/PETS2001.pdf
 *
 */

/*============== BLOB TRACKERCC CLASS DECLARATION =============== */
typedef struct DefBlobTrackerCR
{
    CvBlob                      blob;
    CvBlobTrackPredictor*       pPredictor;
    CvBlob                      BlobPredict;
    CvBlob                      BlobPrev;
    int                         Collision;
    CvBlobSeq*                  pBlobHyp;
    CvBlobTrackerOne*           pResolver;
} DefBlobTrackerCR;

void cvFindBlobsByCCClasters(IplImage* pFG, CvBlobSeq* pBlobs, CvMemStorage* storage);

class CvBlobTrackerCCCR : public CvBlobTracker
{
private:
    float           m_AlphaSize;
    int             m_Collision;
    CvBlobSeq       m_BlobList;
    CvBlobSeq       m_BlobListNew;
    CvMemStorage*   m_pMem;
    CvBlobTrackerOne* (*m_CreateCR)();
    char            m_ModuleName[1024];


public:
    CvBlobTrackerCCCR(CvBlobTrackerOne* (*CreateCR)(), const char* CRName):m_BlobList(sizeof(DefBlobTrackerCR))
    {
        m_CreateCR = CreateCR;
        m_pMem = cvCreateMemStorage();

        m_Collision = 1; /* if 1 then collistion will be detected and processed */

        m_AlphaSize = 0.05f;
        AddParam("AlphaSize",&m_AlphaSize);
        CommentParam("AlphaSize", "Size update speed (0..1)");

        strcpy(m_ModuleName, "CCCR");
        if(CRName)strcat(m_ModuleName,CRName);
        SetModuleName(m_ModuleName);

        {
            CvBlobTrackerOne* pM = m_CreateCR();
            TransferParamsFromChild(pM,NULL);
            pM->Release();
        }
        SetParam("SizeVar",0);
    };

    ~CvBlobTrackerCCCR()
    {
        if(m_pMem)cvReleaseMemStorage(&m_pMem);
    };

    /* Blob functions: */
    virtual int     GetBlobNum() {return m_BlobList.GetBlobNum();};
    virtual CvBlob* GetBlob(int BlobIndex){return m_BlobList.GetBlob(BlobIndex);};
    virtual void    SetBlob(int BlobIndex, CvBlob* pBlob)
    {
        CvBlob* pB = m_BlobList.GetBlob(BlobIndex);
        if(pB) pB[0] = pBlob[0];
    };

    virtual CvBlob* GetBlobByID(int BlobID){return m_BlobList.GetBlobByID(BlobID);};
    virtual void    DelBlob(int BlobIndex)
    {
        DefBlobTrackerCR* pBT = (DefBlobTrackerCR*)m_BlobList.GetBlob(BlobIndex);
        if(pBT->pResolver)pBT->pResolver->Release();
        if(pBT->pPredictor)pBT->pPredictor->Release();
        delete pBT->pBlobHyp;
        m_BlobList.DelBlob(BlobIndex);
    };

    virtual void    DelBlobByID(int BlobID)
    {
        DefBlobTrackerCR* pBT = (DefBlobTrackerCR*)m_BlobList.GetBlobByID(BlobID);
        if(pBT->pResolver)pBT->pResolver->Release();
        if(pBT->pPredictor)pBT->pPredictor->Release();
        delete pBT->pBlobHyp;
        m_BlobList.DelBlobByID(BlobID);
    };

    virtual void    Release(){delete this;};

    /* Add new blob to track it and assign to this blob personal ID */
    /* pBlob - pinter to structure with blob parameters (ID is ignored)*/
    /* pImg - current image */
    /* pImgFG - current foreground mask */
    /* Return pointer to new added blob: */
    virtual CvBlob* AddBlob(CvBlob* pB, IplImage* pImg, IplImage* pImgFG = NULL )
    {
        DefBlobTrackerCR NewB;
        NewB.blob = pB[0];
        NewB.pBlobHyp = new CvBlobSeq;
        NewB.pPredictor = cvCreateModuleBlobTrackPredictKalman(); /* module for predict position */
        NewB.pPredictor->SetParam("DataNoisePos",0.001);
        NewB.pPredictor->ParamUpdate();
        NewB.pResolver = NULL;
        if(m_CreateCR)
        {
            NewB.pResolver = m_CreateCR();
            TransferParamsToChild(NewB.pResolver,NULL);
            NewB.pResolver->Init(pB, pImg, pImgFG);
        }
        m_BlobList.AddBlob((CvBlob*)&NewB);
        return m_BlobList.GetBlob(m_BlobList.GetBlobNum()-1);
    };

    virtual void    Process(IplImage* pImg, IplImage* pImgFG = NULL)
    {
        CvSeq*      cnts;
        CvSeq*      cnt;
        //CvMat*      pMC = NULL;

        if(m_BlobList.GetBlobNum() <= 0 ) return;

        /* Clear blob list for new blobs: */
        m_BlobListNew.Clear();

        assert(m_pMem);
        cvClearMemStorage(m_pMem);
        assert(pImgFG);

        {   /* One contour - one blob: */
            IplImage* pBin = cvCloneImage(pImgFG);
            assert(pBin);
            cvThreshold(pBin,pBin,128,255,CV_THRESH_BINARY);
            cvFindContours(pBin, m_pMem, &cnts, sizeof(CvContour), CV_RETR_EXTERNAL);

            /* Process each contour: */
            for(cnt = cnts; cnt; cnt=cnt->h_next)
            {
                CvBlob  NewBlob;

                /* Image moments: */
                double      M00,X,Y,XX,YY;
                CvMoments   m;
                CvRect      r = ((CvContour*)cnt)->rect;
                CvMat       mat;
                if(r.height < 3 || r.width < 3) continue;
                cvMoments( cvGetSubRect(pImgFG,&mat,r), &m, 0 );
                M00 = cvGetSpatialMoment( &m, 0, 0 );
                if(M00 <= 0 ) continue;
                X = cvGetSpatialMoment( &m, 1, 0 )/M00;
                Y = cvGetSpatialMoment( &m, 0, 1 )/M00;
                XX = (cvGetSpatialMoment( &m, 2, 0 )/M00) - X*X;
                YY = (cvGetSpatialMoment( &m, 0, 2 )/M00) - Y*Y;
                NewBlob = cvBlob(r.x+(float)X,r.y+(float)Y,(float)(4*sqrt(XX)),(float)(4*sqrt(YY)));
                m_BlobListNew.AddBlob(&NewBlob);

            }   /* Next contour. */

            cvReleaseImage(&pBin);
        }

        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Predict new blob position. */
            CvBlob*             pB = NULL;
            DefBlobTrackerCR*   pBT = (DefBlobTrackerCR*)m_BlobList.GetBlob(i-1);

            /* Update predictor. */
            pBT->pPredictor->Update(&(pBT->blob));
            pB = pBT->pPredictor->Predict();
            if(pB)
            {
                pBT->BlobPredict = pB[0];
            }
            pBT->BlobPrev = pBT->blob;
        }   /* Predict new blob position. */


        if(m_BlobList.GetBlobNum()>0 && m_BlobListNew.GetBlobNum()>0)
        {   /* Resolve new blob to old: */
            int NOld = m_BlobList.GetBlobNum();
            int NNew = m_BlobListNew.GetBlobNum();

            for(int i=0; i<NOld; i++)
            {   /* Set 0 collision and clear all hyp: */
                DefBlobTrackerCR* pF = (DefBlobTrackerCR*)m_BlobList.GetBlob(i);
                pF->Collision = 0;
                pF->pBlobHyp->Clear();
            }   /* Set 0 collision. */

            /* Create correspondence records: */
            for(int j=0; j<NNew; ++j)
            {
                CvBlob*             pB1 = m_BlobListNew.GetBlob(j);
                DefBlobTrackerCR*   pFLast = NULL;

                for(int i=0; i<NOld; i++)
                {   /* Check intersection: */
                    int Intersection = 0;
                    DefBlobTrackerCR* pF = (DefBlobTrackerCR*)m_BlobList.GetBlob(i);
                    CvBlob* pB2 = &(pF->BlobPredict);

                    if( fabs(pB1->x-pB2->x)<0.5*(pB1->w+pB2->w) &&
                        fabs(pB1->y-pB2->y)<0.5*(pB1->h+pB2->h) ) Intersection = 1;

                    if(Intersection)
                    {
                        if(pFLast)
                        {
                            pF->Collision = pFLast->Collision = 1;
                        }
                        pFLast = pF;
                        pF->pBlobHyp->AddBlob(pB1);
                    }
                }   /* Check intersection. */
            }   /*  Check next new blob. */
        }   /*  Resolve new blob to old. */

        for(int i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Track each blob. */
            CvBlob*             pB = m_BlobList.GetBlob(i-1);
            DefBlobTrackerCR*   pBT = (DefBlobTrackerCR*)pB;
            int                 BlobID = CV_BLOB_ID(pB);
          //CvBlob*             pBBest = NULL;
          //double              DistBest = -1;

            if(pBT->pResolver)
            {
                pBT->pResolver->SetCollision(pBT->Collision);
            }

            if(pBT->Collision)
            {   /* Tracking in collision: */
                if(pBT->pResolver)
                {
                    pB[0] = pBT->pResolver->Process(&(pBT->BlobPredict),pImg, pImgFG)[0];
                }
            }   /* Tracking in collision. */
            else
            {   /* Non-collision tracking: */
                CvBlob  NewCC = pBT->BlobPredict;
                if(pBT->pBlobHyp->GetBlobNum()==1)
                {   /* One blob to one CC: */
                    NewCC = pBT->pBlobHyp->GetBlob(0)[0];
                }
                else
                {   /* One blob several CC: */
                    CvBlob* pBBest = NULL;
                    double  DistBest = -1;
                    double  CMax = 0;
                    for(int j=pBT->pBlobHyp->GetBlobNum();j>0;--j)
                    {   /* Find best CC: */
                        CvBlob* pBNew = pBT->pBlobHyp->GetBlob(j-1);
                        if(pBT->pResolver)
                        {     /* Choose CC by confidence: */
//                            double  dx = fabs(CV_BLOB_X(pB)-CV_BLOB_X(pBNew));
//                            double  dy = fabs(CV_BLOB_Y(pB)-CV_BLOB_Y(pBNew));
                            double  C = pBT->pResolver->GetConfidence(pBNew,pImg, pImgFG);
                            if(C > CMax || pBBest == NULL)
                            {
                                CMax = C;
                                pBBest = pBNew;
                            }
                        }
                        else
                        {    /* Choose CC by distance: */
                            double  dx = fabs(CV_BLOB_X(pB)-CV_BLOB_X(pBNew));
                            double  dy = fabs(CV_BLOB_Y(pB)-CV_BLOB_Y(pBNew));
                            double  Dist = sqrt(dx*dx+dy*dy);
                            if(Dist < DistBest || pBBest == NULL)
                            {
                                DistBest = Dist;
                                pBBest = pBNew;
                            }
                        }
                    }   /* Find best CC. */
                    if(pBBest)
                        NewCC = pBBest[0];
                }   /* One blob several CC. */
                pB->x = NewCC.x;
                pB->y = NewCC.y;
                pB->w = (m_AlphaSize)*NewCC.w+(1-m_AlphaSize)*pB->w;
                pB->h = (m_AlphaSize)*NewCC.h+(1-m_AlphaSize)*pB->h;
                pBT->pResolver->SkipProcess(&(pBT->BlobPredict),pImg, pImgFG);
            }   /* Non-collision tracking. */

            pBT->pResolver->Update(pB, pImg, pImgFG);

            CV_BLOB_ID(pB)=BlobID;

        }   /* Track next blob. */

        if(m_Wnd)
        {
            IplImage* pI = cvCloneImage(pImg);
            for(int i=m_BlobListNew.GetBlobNum(); i>0; --i)
            {   /* Draw each new CC: */
                CvBlob* pB = m_BlobListNew.GetBlob(i-1);
                CvPoint p = cvPointFrom32f(CV_BLOB_CENTER(pB));
                int x = cvRound(CV_BLOB_RX(pB)), y = cvRound(CV_BLOB_RY(pB));
                CvSize  s = cvSize(MAX(1,x), MAX(1,y));
                //int c = 255;
                cvEllipse( pI,
                    p,
                    s,
                    0, 0, 360,
                    CV_RGB(255,255,0), 1 );
            }

            for(int i=m_BlobList.GetBlobNum(); i>0; --i)
            {   /* Draw each new CC: */
                DefBlobTrackerCR* pF = (DefBlobTrackerCR*)m_BlobList.GetBlob(i-1);
                CvBlob* pB = &(pF->BlobPredict);
                CvPoint p = cvPointFrom32f(CV_BLOB_CENTER(pB));
                int x = cvRound(CV_BLOB_RX(pB)), y = cvRound(CV_BLOB_RY(pB));
                CvSize  s = cvSize(MAX(1,x), MAX(1,y));
                cvEllipse( pI,
                    p,
                    s,
                    0, 0, 360,
                    CV_RGB(0,0,255), 1 );

                pB = &(pF->blob);
                p = cvPointFrom32f(CV_BLOB_CENTER(pB));
                x = cvRound(CV_BLOB_RX(pB)); y = cvRound(CV_BLOB_RY(pB));
                s = cvSize(MAX(1,x), MAX(1,y));
                cvEllipse( pI,
                    p,
                    s,
                    0, 0, 360,
                    CV_RGB(0,255,0), 1 );
            }

            //cvNamedWindow("CCwithCR",0);
            //cvShowImage("CCwithCR",pI);
            cvReleaseImage(&pI);
        }

    } /* Process. */

    virtual void SaveState(CvFileStorage* fs)
    {
        int     b,bN = m_BlobList.GetBlobNum();
        cvWriteInt(fs,"BlobNum",m_BlobList.GetBlobNum());
        cvStartWriteStruct(fs,"BlobList",CV_NODE_SEQ);

        for(b=0; b<bN; ++b)
        {
            DefBlobTrackerCR* pF = (DefBlobTrackerCR*)m_BlobList.GetBlob(b);
            cvStartWriteStruct(fs,NULL,CV_NODE_MAP);
            cvWriteInt(fs,"ID",CV_BLOB_ID(pF));
            cvStartWriteStruct(fs,"Blob",CV_NODE_SEQ|CV_NODE_FLOW);
            cvWriteRawData(fs,&(pF->blob),1,"ffffi");
            cvEndWriteStruct(fs);
            cvStartWriteStruct(fs,"BlobPredict",CV_NODE_SEQ|CV_NODE_FLOW);
            cvWriteRawData(fs,&(pF->BlobPredict),1,"ffffi");
            cvEndWriteStruct(fs);
            cvStartWriteStruct(fs,"BlobPrev",CV_NODE_SEQ|CV_NODE_FLOW);
            cvWriteRawData(fs,&(pF->BlobPrev),1,"ffffi");
            cvEndWriteStruct(fs);
            pF->pBlobHyp->Write(fs,"BlobHyp");
            cvWriteInt(fs,"Collision",pF->Collision);

            cvStartWriteStruct(fs,"Predictor",CV_NODE_MAP);
            pF->pPredictor->SaveState(fs);
            cvEndWriteStruct(fs);

            cvStartWriteStruct(fs,"Resolver",CV_NODE_MAP);
            pF->pResolver->SaveState(fs);
            cvEndWriteStruct(fs);
            cvEndWriteStruct(fs);
        }

        cvEndWriteStruct(fs);

    }   /* SaveState. */

    virtual void LoadState(CvFileStorage* fs, CvFileNode* node)
    {
        int         b,bN = cvReadIntByName(fs,node,"BlobNum",0);
        CvFileNode* pBlobListNode = cvGetFileNodeByName(fs,node,"BlobList");
        if(!CV_NODE_IS_SEQ(pBlobListNode->tag)) return;
        bN = pBlobListNode->data.seq->total;

        for(b=0; b<bN; ++b)
        {
            DefBlobTrackerCR*   pF = NULL;
            CvBlob              Blob;
            CvFileNode*         pSeqNode = NULL;
            CvFileNode*         pBlobNode = (CvFileNode*)cvGetSeqElem(pBlobListNode->data.seq,b);
            assert(pBlobNode);

            Blob.ID = cvReadIntByName(fs,pBlobNode,"ID",0);

            pSeqNode = cvGetFileNodeByName(fs, pBlobNode, "Blob");
            if(CV_NODE_IS_SEQ(pSeqNode->tag))
                cvReadRawData( fs, pSeqNode, &Blob, "ffffi" );

            AddBlob(&Blob,NULL,NULL);
            pF = (DefBlobTrackerCR*)m_BlobList.GetBlobByID(Blob.ID);

            pSeqNode = cvGetFileNodeByName(fs, pBlobNode, "BlobPredict");
            if(CV_NODE_IS_SEQ(pSeqNode->tag))
                cvReadRawData( fs, pSeqNode, &pF->BlobPredict, "ffffi" );

            pSeqNode = cvGetFileNodeByName(fs, pBlobNode, "BlobPrev");
            if(CV_NODE_IS_SEQ(pSeqNode->tag))
                cvReadRawData( fs, pSeqNode, &pF->BlobPrev, "ffffi" );

            pSeqNode = cvGetFileNodeByName(fs, pBlobNode, "BlobHyp");
            if(pSeqNode)
                pF->pBlobHyp->Load(fs,pSeqNode);
            pF->Collision = cvReadIntByName(fs, pBlobNode,"Collision",pF->Collision);

            pSeqNode = cvGetFileNodeByName(fs, pBlobNode, "Predictor");
            if(pSeqNode)
                pF->pPredictor->LoadState(fs,pSeqNode);

            pSeqNode = cvGetFileNodeByName(fs, pBlobNode, "Resolver");
            if(pSeqNode)
                pF->pResolver->LoadState(fs,pSeqNode);
        }   /* Read next blob. */
    }   /*  CCwithCR LoadState */

    //void SetCollision(int Collision){m_Collision = Collision;};
};

CvBlobTrackerOne* cvCreateBlobTrackerOneMSPF();
CvBlobTracker* cvCreateBlobTrackerCCMSPF()
{
    return (CvBlobTracker*) new CvBlobTrackerCCCR(cvCreateBlobTrackerOneMSPF,"MSPF");
}
/*============== BLOB TRACKERCC CLASS DECLARATION =============== */
