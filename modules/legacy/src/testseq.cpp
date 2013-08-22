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
This file contain implementation of virtual interface of CvTestSeq
*/

#include "precomp.hpp" /* virtual interface if CvTestSeq */


void cvAddNoise(IplImage* pImg, int noise_type, double Ampl, CvRandState* rnd_state = NULL);

#define FG_BG_THRESHOLD 3

#define SRC_TYPE_AVI 1
#define SRC_TYPE_IMAGE 0

/* Transformation structure: */
typedef struct CvTSTrans
{
    float           T[6]; /* geometry transformation */
    CvPoint2D32f    Shift;
    CvPoint2D32f    Scale;
    float           I;
    float           C;
    float           GN; /* standart deviation of added gaussian noise */
    float           NoiseAmp; /* amplifier of noise power */
    float           angle;
} CvTSTrans;

static void SET_TRANS_0(CvTSTrans *pT)
{
    memset(pT,0,sizeof(CvTSTrans));
    pT->C = 1;
    pT->Scale.x = 1;
    pT->Scale.y = 1;
    pT->T[4] = pT->T[0] = 1;
    pT->NoiseAmp = 1;
}

/* === Some definitions and functions for transformation update: ===*/
#define P_ANGLE 0
#define P_S     1
#define P_SX    2
#define P_SY    3
#define P_DX    4
#define P_DY    5
#define P_I     6
#define P_C     7
#define P_GN    8
#define P_NAmp  9
static const char*   param_name[] =    {"angle","s","sx","sy","dx","dy","I","C","GN","NoiseAmp", NULL};
static float   param_defval[] =  { 0,      1,  1,   1,   0,   0,   0,  1,  0,   1};
static void icvUpdateTrans(CvTSTrans* pTrans, int param, double val, float MaxX, float MaxY)
{
    assert(pTrans);
    if(param==P_ANGLE)
    {

        double  C = cos(3.1415926535897932384626433832795*val/180.0);
        double  S = sin(3.1415926535897932384626433832795*val/180.0);
        float*  T = pTrans->T;
        double  TR[6];
        int     i;
        pTrans->angle = (float)(pTrans->angle + val);
        TR[0] = C*T[0]-S*T[3];
        TR[1] = C*T[1]-S*T[4];
        TR[2] = C*T[2]-S*T[5];
        TR[3] = S*T[0]+C*T[3];
        TR[4] = S*T[1]+C*T[4];
        TR[5] = S*T[2]+C*T[5];
        for(i=0;i<6;++i)T[i]=(float)TR[i];
    }

    if(param==P_S)
    {
        int i;
        for(i=0;i<6;++i)pTrans->T[i] = (float)(pTrans->T[i]*val);
        pTrans->Scale.x = (float)(pTrans->Scale.x *val);
        pTrans->Scale.y = (float)(pTrans->Scale.y *val);
        pTrans->Shift.x = (float)(pTrans->Shift.x *val);
        pTrans->Shift.y = (float)(pTrans->Shift.y *val);
    }

    if(param==P_SX)
    {
        int i;
        for(i=0;i<3;++i)pTrans->T[i] = (float)(pTrans->T[i]*val);
        pTrans->Scale.x = (float)(pTrans->Scale.x*val);
        pTrans->Shift.x = (float)(pTrans->Shift.x*val);
    }

    if(param==P_SY)
    {
        int i;
        for(i=0;i<3;++i)pTrans->T[i+3] = (float)(pTrans->T[i+3]*val);
        pTrans->Scale.y = (float)(pTrans->Scale.y *val);
        pTrans->Shift.y = (float)(pTrans->Shift.y *val);
    }

    if(param==P_DX)
    {
        pTrans->Shift.x = (float)(pTrans->Shift.x +val);
        pTrans->T[2] = (float)(pTrans->T[2] +val*MaxX);
    }

    if(param==P_DY)
    {
        pTrans->Shift.y = (float)(pTrans->Shift.y +val);
        pTrans->T[5] = (float)(pTrans->T[5] +val*MaxY);
    }

    if(param==P_C)
    {
        pTrans->C = (float)(pTrans->C *val);
        pTrans->I = (float)(pTrans->I *val);
    }

    if(param==P_I) pTrans->I = (float)(pTrans->I +val);

    if(param==P_GN)
    {
        pTrans->GN = (float)sqrt(val*val+pTrans->GN*pTrans->GN);
    }

    if(param==P_NAmp) pTrans->NoiseAmp = (float)(pTrans->NoiseAmp *val);
}   /* icvUpdateTrans */

/* === END some defenitions and function for transformation update ===*/

typedef struct CvTestSeqElem
{
    const char*     pObjName;
    const char*     pFileName;
    int             type; /* video or image */
    CvPoint2D32f*   pPos; /* positions of object in sequence */
    int             PosNum;
    CvPoint2D32f*   pSize; /* sizes of object in sequence */
    int             SizeNum;
    CvTSTrans*      pTrans; /* transforation of image in sequence */
    int             TransNum;
    int             ShiftByPos;
    CvPoint2D32f    ShiftBegin;
    CvPoint2D32f    ShiftEnd;
    int             FrameBegin;
    int             FrameNum;
    IplImage*       pImg;
    IplImage*       pImgMask;
    void*           pAVI;
    //CvCapture*      pAVI;
    int             AVILen;
    int             BG; /* flag is it background (1) or foreground (0) */
    int             Mask; /* flag is it foreground mask (1) or usual video (0) */
    CvTestSeqElem   *next;
    int             noise_type;
    CvRandState     rnd_state;
    int             ObjID;
} CvTestSeqElem;

/* Test seq main structure: */
typedef struct CvTestSeq_
{
    int             ID;
    CvFileStorage*  pFileStorage;
    CvTestSeqElem*  pElemList;
    int             ListNum;
    IplImage*       pImg;
    IplImage*       pImgMask;
    int             CurFrame;
    int             FrameNum;
    int             noise_type;
    double          noise_ampl;
    float           IVar_DI;
    float           IVar_MinI;
    float           IVar_MaxI;
    float           IVar_CurDI;
    float           IVar_CurI;
    int             ObjNum;

} CvTestSeq_;


CvSize cvTestSeqGetImageSize(CvTestSeq* pTestSeq){return cvSize(((CvTestSeq_*)(pTestSeq))->pImg->width,((CvTestSeq_*)(pTestSeq))->pImg->height);}
int cvTestSeqFrameNum(CvTestSeq* pTestSeq){return ((CvTestSeq_*)(pTestSeq))->FrameNum;}

static void icvTestSeqCreateMask(IplImage* pImg,IplImage* pImgMask, int threshold)
{
    if(pImg->nChannels > 1)
    {
        cvCvtColor( pImg,pImgMask,CV_BGR2GRAY);
        cvThreshold(pImgMask,pImgMask,threshold,255,CV_THRESH_BINARY);
    }
    else
    {
        cvThreshold(pImg,pImgMask,threshold,255,CV_THRESH_BINARY);
    }
}   /* icvTestSeqCreateMask */


static void icvTestSeqQureyFrameElem(CvTestSeqElem* p, int /*frame*/)
{   /* Read next frame from avi for one record: */
    if(p->type == SRC_TYPE_AVI)
    {
        IplImage*   pI = NULL;
        //int         frameNum = p->AVILen;

        if(p->pAVI == NULL && p->pFileName)
        {   /* Open avi file if necessary: */
            p->pAVI = 0;//cvCaptureFromFile(p->pFileName);
            if(p->pAVI == NULL)
            {
                printf("WARNING!!! Can not open avi file %s\n",p->pFileName);
                return;
            }
        }   /* Open avi file if necessary. */

        assert(p->pAVI);
        //if(frame >= frameNum)
        {   /* Set new position: */
            //int N = frame%frameNum;

            /*if( N==0 ||
                N != (int)cvGetCaptureProperty(p->pAVI,CV_CAP_PROP_POS_FRAMES))
            {
                cvSetCaptureProperty(p->pAVI,CV_CAP_PROP_POS_FRAMES,N);
            }*/
        }   /* Set new position. */

        //pI = cvQueryFrame(p->pAVI);
        if(pI)
        {
            if(pI->origin != p->pImg->origin)
                cvFlip( pI, p->pImg, 0 );
            else
                cvCopy(pI, p->pImg);
        }

        if(p->pImg)
        {
            if(p->pImgMask==NULL)
            {
                p->pImgMask = cvCreateImage(
                    cvSize(p->pImg->width,p->pImg->height),
                    IPL_DEPTH_8U,1);
            }
            icvTestSeqCreateMask(p->pImg,p->pImgMask,p->Mask?128:FG_BG_THRESHOLD);
        }
    }

}   /* icvTestSeqQureyFrameElem */

/*------------- Recursive function to read all images, ------------------------*/
/*------------- videos and objects from config file.   ------------------------*/

static CvTestSeqElem* icvTestSeqReadElemAll(CvTestSeq_* pTS, CvFileStorage* fs, const char* name);

static void icvTestSeqAllocTrans(CvTestSeqElem* p)
{   /* Allocate transformation array if necessary */
    /* work with transformation */
    if(p->pTrans == NULL/* && p->FrameNum>0*/)
    {   /* Allocate transformation array: */
        int num = MAX(1,p->FrameNum);
        p->pTrans = (CvTSTrans*)cvAlloc(sizeof(CvTSTrans)*num);
        p->TransNum = num;
        while(num--)SET_TRANS_0(p->pTrans+num);
    }

    if(p->FrameNum > p->TransNum)
    {   /* Allocate new transformation array: */
        int         i;
        int         num = p->FrameNum;
        CvTSTrans*  pNewTrans = (CvTSTrans*)cvAlloc(sizeof(CvTSTrans)*num);

        for(i=0; i<num; ++i)
        {
            if(p->pTrans)
                pNewTrans[i] = p->pTrans[i%p->TransNum];
            else
                SET_TRANS_0(pNewTrans+i);
        }
        if(p->pTrans)cvFree(&p->pTrans);
        p->pTrans = pNewTrans;
        p->TransNum = num;
    }   /* Allocate new transformation array. */
}   /*  Allocate transformation array if necessary. */

static CvTestSeqElem* icvTestSeqReadElemOne(CvTestSeq_* pTS, CvFileStorage* fs, CvFileNode* node)
{
    int             noise_type = CV_NOISE_NONE;;
    CvTestSeqElem*  pElem = NULL;
    const char*     pVideoName = cvReadStringByName( fs, node,"Video", NULL);
    const char*     pVideoObjName = cvReadStringByName( fs, node,"VideoObj", NULL);

    if(pVideoName)
    {   /* Check to noise flag: */
        if( cv_stricmp(pVideoName,"noise_gaussian") == 0 ||
            cv_stricmp(pVideoName,"noise_normal") == 0) noise_type = CV_NOISE_GAUSSIAN;
        if( cv_stricmp(pVideoName,"noise_uniform") == 0) noise_type = CV_NOISE_UNIFORM;
        if( cv_stricmp(pVideoName,"noise_speckle") == 0) noise_type = CV_NOISE_SPECKLE;
        if( cv_stricmp(pVideoName,"noise_salt_and_pepper") == 0) noise_type = CV_NOISE_SALT_AND_PEPPER;
    }

    if((pVideoName || pVideoObjName ) && noise_type == CV_NOISE_NONE)
    {   /* Read other elements: */
        if(pVideoName) pElem = icvTestSeqReadElemAll(pTS, fs, pVideoName);
        if(pVideoObjName)
        {
            CvTestSeqElem* pE;
            pElem = icvTestSeqReadElemAll(pTS, fs, pVideoObjName);
            for(pE=pElem;pE;pE=pE->next)
            {
                pE->ObjID = pTS->ObjNum;
                pE->pObjName = pVideoObjName;
            }
            pTS->ObjNum++;
        }
    }   /* Read other elements. */
    else
    {   /* Create new element: */
        CvFileNode* pPosNode = cvGetFileNodeByName( fs, node,"Pos");
        CvFileNode* pSizeNode = cvGetFileNodeByName( fs, node,"Size");
        int AutoSize = (pSizeNode && CV_NODE_IS_STRING(pSizeNode->tag) && cv_stricmp("auto",cvReadString(pSizeNode,""))==0);
        int AutoPos = (pPosNode && CV_NODE_IS_STRING(pPosNode->tag) && cv_stricmp("auto",cvReadString(pPosNode,""))==0);
        const char* pFileName = cvReadStringByName( fs, node,"File", NULL);
        pElem = (CvTestSeqElem*)cvAlloc(sizeof(CvTestSeqElem));
        memset(pElem,0,sizeof(CvTestSeqElem));

        pElem->ObjID = -1;
        pElem->noise_type = noise_type;
        cvRandInit( &pElem->rnd_state, 1, 0, 0,CV_RAND_NORMAL);

        if(pFileName && pElem->noise_type == CV_NOISE_NONE)
        {   /* If AVI or BMP: */
            size_t  l = strlen(pFileName);
            pElem->pFileName = pFileName;

            pElem->type = SRC_TYPE_IMAGE;
            if(cv_stricmp(".avi",pFileName+l-4) == 0)pElem->type = SRC_TYPE_AVI;

            if(pElem->type == SRC_TYPE_IMAGE)
            {
                //pElem->pImg = cvLoadImage(pFileName);
                if(pElem->pImg)
                {
                    pElem->FrameNum = 1;
                    if(pElem->pImgMask)cvReleaseImage(&(pElem->pImgMask));

                    pElem->pImgMask = cvCreateImage(
                        cvSize(pElem->pImg->width,pElem->pImg->height),
                        IPL_DEPTH_8U,1);
                    icvTestSeqCreateMask(pElem->pImg,pElem->pImgMask,FG_BG_THRESHOLD);
                }
            }

            if(pElem->type == SRC_TYPE_AVI && pFileName)
            {
                //pElem->pAVI = cvCaptureFromFile(pFileName);

                if(pElem->pAVI)
                {
                    IplImage* pImg = 0;//cvQueryFrame(pElem->pAVI);
                    pElem->pImg = cvCloneImage(pImg);
                    pElem->pImg->origin = 0;
                    //cvSetCaptureProperty(pElem->pAVI,CV_CAP_PROP_POS_FRAMES,0);
                    pElem->FrameBegin = 0;
                    pElem->AVILen = pElem->FrameNum = 0;//(int)cvGetCaptureProperty(pElem->pAVI, CV_CAP_PROP_FRAME_COUNT);
                    //cvReleaseCapture(&pElem->pAVI);
                    pElem->pAVI = NULL;
                }
                else
                {
                    printf("WARNING!!! Cannot open avi file %s\n",pFileName);
                }
            }

        }   /* If AVI or BMP. */

        if(pPosNode)
        {   /* Read positions: */
            if(CV_NODE_IS_SEQ(pPosNode->tag))
            {
                int num = pPosNode->data.seq->total;
                pElem->pPos = (CvPoint2D32f*)cvAlloc(sizeof(float)*num);
                cvReadRawData( fs, pPosNode, pElem->pPos, "f" );
                pElem->PosNum = num/2;
                if(pElem->FrameNum == 0) pElem->FrameNum = pElem->PosNum;
            }
        }

        if(pSizeNode)
        {   /* Read sizes: */
            if(CV_NODE_IS_SEQ(pSizeNode->tag))
            {
                int num = pSizeNode->data.seq->total;
                pElem->pSize = (CvPoint2D32f*)cvAlloc(sizeof(float)*num);
                cvReadRawData( fs, pSizeNode, pElem->pSize, "f" );
                pElem->SizeNum = num/2;
            }
        }

        if(AutoPos || AutoSize)
        {   /* Auto size and pos: */
            int     i;
            int     num = (pElem->type == SRC_TYPE_AVI)?pElem->AVILen:1;
            if(AutoSize)
            {
                pElem->pSize = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f)*num);
                pElem->SizeNum = num;
            }
            if(AutoPos)
            {
                pElem->pPos = (CvPoint2D32f*)cvAlloc(sizeof(CvPoint2D32f)*num);
                pElem->PosNum = num;
            }

            for(i=0; i<num; ++i)
            {
                IplImage* pFG = NULL;
                CvPoint2D32f* pPos = AutoPos?(pElem->pPos + i):NULL;
                CvPoint2D32f* pSize = AutoSize?(pElem->pSize + i):NULL;

                icvTestSeqQureyFrameElem(pElem,i);
                pFG = pElem->pImgMask;

                if(pPos)
                {
                    pPos->x = 0.5f;
                    pPos->y = 0.5f;
                }
                if(pSize)
                {
                    pSize->x = 0;
                    pSize->y = 0;
                }

                if(pFG)
                {
                    double      M00;
                    CvMoments   m;
                    cvMoments( pElem->pImgMask, &m, 0 );
                    M00 = cvGetSpatialMoment( &m, 0, 0 );

                    if(M00 > 0 && pSize )
                    {
                        double X = cvGetSpatialMoment( &m, 1, 0 )/M00;
                        double Y = cvGetSpatialMoment( &m, 0, 1 )/M00;
                        double XX = (cvGetSpatialMoment( &m, 2, 0 )/M00) - X*X;
                        double YY = (cvGetSpatialMoment( &m, 0, 2 )/M00) - Y*Y;
                        pSize->x = (float)(4*sqrt(XX))/(pElem->pImgMask->width-1);
                        pSize->y = (float)(4*sqrt(YY))/(pElem->pImgMask->height-1);
                    }

                    if(M00 > 0 && pPos)
                    {
                        pPos->x = (float)(cvGetSpatialMoment( &m, 1, 0 )/(M00*(pElem->pImgMask->width-1)));
                        pPos->y = (float)(cvGetSpatialMoment( &m, 0, 1 )/(M00*(pElem->pImgMask->height-1)));
                    }

                    if(pPos)
                    {   /* Another way to calculate y pos
                         * using object median:
                         */
                        int y0=0, y1=pFG->height-1;
                        for(y0=0; y0<pFG->height; ++y0)
                        {
                            CvMat       tmp;
                            CvScalar    s = cvSum(cvGetRow(pFG, &tmp, y0));
                            if(s.val[0] > 255*7) break;
                        }

                        for(y1=pFG->height-1; y1>0; --y1)
                        {
                            CvMat tmp;
                            CvScalar s = cvSum(cvGetRow(pFG, &tmp, y1));
                            if(s.val[0] > 255*7) break;
                        }

                        pPos->y = (y0+y1)*0.5f/(pFG->height-1);
                    }
                }   /* pFG */
            }   /* Next frame. */

            //if(pElem->pAVI) cvReleaseCapture(&pElem->pAVI);

            pElem->pAVI = NULL;

        }   /* End auto position creation. */
    }   /*  Create new element. */

    if(pElem)
    {   /* Read transforms and: */
        int             FirstFrame, LastFrame;
        CvTestSeqElem*  p=pElem;
        CvFileNode*     pTransNode = NULL;
        CvFileNode*     pS = NULL;
        int             ShiftByPos = 0;
        int             KeyFrames[1024];
        CvSeq*          pTransSeq = NULL;
        int             KeyFrameNum = 0;

        pTransNode = cvGetFileNodeByName( fs, node,"Trans");

        while( pTransNode &&
               CV_NODE_IS_STRING(pTransNode->tag) &&
               cv_stricmp("auto",cvReadString(pTransNode,""))!=0)
        {   /* Trans is reference: */
            pTransNode = cvGetFileNodeByName( fs, NULL,cvReadString(pTransNode,""));
        }

        pS = cvGetFileNodeByName( fs, node,"Shift");
        ShiftByPos = 0;
        pTransSeq = pTransNode?(CV_NODE_IS_SEQ(pTransNode->tag)?pTransNode->data.seq:NULL):NULL;
        KeyFrameNum = pTransSeq?pTransSeq->total:1;

        if(   (pS && CV_NODE_IS_STRING(pS->tag) && cv_stricmp("auto",cvReadString(pS,""))==0)
            ||(pTransNode && CV_NODE_IS_STRING(pTransNode->tag) && cv_stricmp("auto",cvReadString(pTransNode,""))==0))
        {
            ShiftByPos = 1;
        }

        FirstFrame = pElem->FrameBegin;
        LastFrame = pElem->FrameBegin+pElem->FrameNum-1;

        /* Calculate length of video and reallocate
         * transformation array:
         */
        for(p=pElem; p; p=p->next)
        {
            int v;
            v = cvReadIntByName( fs, node, "BG", -1 );
            if(v!=-1)p->BG = v;
            v = cvReadIntByName( fs, node, "Mask", -1 );
            if(v!=-1)p->Mask = v;

            p->FrameBegin += cvReadIntByName( fs, node, "FrameBegin", 0 );
            p->FrameNum = cvReadIntByName( fs, node, "FrameNum", p->FrameNum );
            p->FrameNum = cvReadIntByName( fs, node, "Dur", p->FrameNum );
            {
                int lastFrame = cvReadIntByName( fs, node, "LastFrame", p->FrameBegin+p->FrameNum-1 );
                p->FrameNum = MIN(p->FrameNum,lastFrame - p->FrameBegin+1);
            }

            icvTestSeqAllocTrans(p);

            {   /* New range estimation: */
                int LF = p->FrameBegin+p->FrameNum-1;
                if(p==pElem || FirstFrame > p->FrameBegin)FirstFrame = p->FrameBegin;
                if(p==pElem || LastFrame < LF)LastFrame = LF;
            }   /* New range estimation. */
        }   /*  End allocate new transfrom array. */

        if(ShiftByPos)
        {
            for(p=pElem;p;p=p->next)
            {   /* Modify transformation to make autoshift: */
                int         i;
                int         num = p->FrameNum;
                assert(num <= p->TransNum);
                p->TransNum = MAX(1,num);

                for(i=0; i<num; ++i)
                {
                    CvTSTrans*  pT = p->pTrans+i;
                    //float   t = (num>1)?((float)i/(num-1)):0.0f;
                    float newx = p->pPos[i%p->PosNum].x;
                    float newy = p->pPos[i%p->PosNum].y;
                    pT->Shift.x = -newx*pT->Scale.x;
                    pT->Shift.y = -newy*pT->Scale.y;

                    if(p->pImg)
                    {
                        newx *= p->pImg->width-1;
                        newy *= p->pImg->height-1;
                    }

                    pT->T[2] = -(pT->T[0]*newx+pT->T[1]*newy);
                    pT->T[5] = -(pT->T[3]*newx+pT->T[4]*newy);
                }
            }   /* Modify transformation old. */
        }   /*  Next record. */

        /* Initialize frame number array: */
        KeyFrames[0] = FirstFrame;

        if(pTransSeq&&KeyFrameNum>1)
        {
            int i0,i1;
            for(int i=0; i<KeyFrameNum; ++i)
            {
                CvFileNode* pTN = (CvFileNode*)cvGetSeqElem(pTransSeq,i);
                KeyFrames[i] = cvReadIntByName(fs,pTN,"frame",-1);
            }

            if(KeyFrames[0]<0)KeyFrames[0]=FirstFrame;
            if(KeyFrames[KeyFrameNum-1]<0)KeyFrames[KeyFrameNum-1]=LastFrame;

            for(i0=0, i1=1; i1<KeyFrameNum;)
            {
                for(i1=i0+1; i1<KeyFrameNum && KeyFrames[i1]<0; i1++) {}

                assert(i1<KeyFrameNum);
                assert(i1>i0);

                for(int i=i0+1; i<i1; ++i)
                {
                    KeyFrames[i] = cvRound(KeyFrames[i0] + (float)(i-i0)*(float)(KeyFrames[i1] - KeyFrames[i0])/(float)(i1-i0));
                }
                i0 = i1;
                i1++;
            }   /* Next key run. */
        }   /*  Initialize frame number array. */

        if(pTransNode || pTransSeq)
        {   /* More complex transform. */
            int     param;
            CvFileNode* pTN = pTransSeq?(CvFileNode*)cvGetSeqElem(pTransSeq,0):pTransNode;

            for(p=pElem; p; p=p->next)
            {
                //int trans_num = p->TransNum;
                for(param=0; param_name[param]; ++param)
                {
                    const char*   name = param_name[param];
                    float   defv = param_defval[param];
                    if(KeyFrameNum==1)
                    {   /* Only one transform record: */
                        int     i;
                        double  val;
                        CvFileNode* fnode = cvGetFileNodeByName( fs, pTN,name);
                        if(fnode == NULL) continue;
                        val = cvReadReal(fnode,defv);

                        for(i=0; i<p->TransNum; ++i)
                        {
                            icvUpdateTrans(
                                p->pTrans+i, param, val,
                                p->pImg?(float)(p->pImg->width-1):1.0f,
                                p->pImg?(float)(p->pImg->height-1):1.0f);
                        }
                    }   /* Next record. */
                    else
                    {   /* Several transforms: */
                        int         i0,i1;
                        double      v0;
                        double      v1;

                        CvFileNode* pTN1 = (CvFileNode*)cvGetSeqElem(pTransSeq,0);
                        v0 = cvReadRealByName(fs, pTN1,name,defv);

                        for(i1=1,i0=0; i1<KeyFrameNum; ++i1)
                        {
                            int         f0,f1;
                            int         i;
                            CvFileNode* pTN2 = (CvFileNode*)cvGetSeqElem(pTransSeq,i1);
                            CvFileNode* pVN = cvGetFileNodeByName(fs,pTN2,name);

                            if(pVN)v1 = cvReadReal(pVN,defv);
                            else if(pVN == NULL && i1 == KeyFrameNum-1) v1 = defv;
                            else continue;

                            f0 = KeyFrames[i0];
                            f1 = KeyFrames[i1];

                            if(i1==(KeyFrameNum-1)) f1++;

                            for(i=f0; i<f1; ++i)
                            {
                                double   val;
                                double   t = (float)(i-f0);
                                int      li = i - p->FrameBegin;
                                if(li<0) continue;
                                if(li>= p->TransNum) break;
                                if(KeyFrames[i1]>KeyFrames[i0]) t /=(float)(KeyFrames[i1]-KeyFrames[i0]);
                                val = t*(v1-v0)+v0;

                                icvUpdateTrans(
                                    p->pTrans+li, param, val,
                                    p->pImg?(float)(p->pImg->width-1):1.0f,
                                    p->pImg?(float)(p->pImg->height-1):1.0f);

                            }   /* Next transform. */
                            i0 = i1;
                            v0 = v1;

                        }   /* Next value run. */
                    }   /*  Several transforms. */
                }   /*  Next parameter. */
            }   /*  Next record. */
        }   /*  More complex transform. */
    }   /*  Read transfroms. */

    return pElem;

}   /* icvTestSeqReadElemOne */

static CvTestSeqElem* icvTestSeqReadElemAll(CvTestSeq_* pTS, CvFileStorage* fs, const char* name)
{
    CvTestSeqElem*  pElem = NULL;
    CvFileNode*     node;

    if(name == NULL) return NULL;

    node = cvGetFileNodeByName( fs, NULL, name );

    if(node == NULL)
    {
        printf("WARNING!!! - Video %s does not exist!\n", name);
        return NULL;
    }

    printf("Read node %s\n",name);

    if(CV_NODE_IS_SEQ(node->tag))
    {   /* Read all element in sequence: */
        int             i;
        CvSeq*          seq = node->data.seq;
        CvTestSeqElem*  pElemLast = NULL;

        for(i=0; i<seq->total; ++i)
        {
            CvFileNode*     next_node = (CvFileNode*)cvGetSeqElem( seq, i );
            CvTestSeqElem*  pElemNew = icvTestSeqReadElemOne(pTS, fs, next_node );
            CvFileNode*     pDurNode = cvGetFileNodeByName( fs, next_node,"Dur");

            if(pElemNew == NULL )
            {
                printf("WARNING in parsing %s record!!! Cannot read array element\n", name);
                continue;
            }

            if(pElem && pElemLast)
            {
                pElemLast->next = pElemNew;
                if(pDurNode)
                {
                    pElemNew->FrameBegin = pElemLast->FrameBegin + pElemLast->FrameNum;
                }
            }
            else
            {
                pElem = pElemNew;
            }

            /* Find last element: */
            for(pElemLast=pElemNew;pElemLast && pElemLast->next;pElemLast= pElemLast->next) {}

        }   /* Next element. */
    }   /*  Read all element in sequence. */
    else
    {   /* Read one element: */
        pElem = icvTestSeqReadElemOne(pTS, fs, node );
    }

    return pElem;

}   /* icvTestSeqReadElemAll */

static void icvTestSeqReleaseAll(CvTestSeqElem** ppElemList)
{
    CvTestSeqElem* p = ppElemList[0];

    while(p)
    {
        CvTestSeqElem* pd = p;
        if(p->pAVI)
        {
            //cvReleaseCapture(&p->pAVI);
        }
        if(p->pImg)cvReleaseImage(&p->pImg);
        if(p->pImgMask)cvReleaseImage(&p->pImgMask);
        if(p->pPos)cvFree(&p->pPos);
        if(p->pTrans)cvFree(&p->pTrans);
        if(p->pSize)cvFree(&p->pSize);
        p=p->next;
        cvFree(&pd);

    }   /* Next element. */

    ppElemList[0] = NULL;

}   /* icvTestSeqReleaseAll */

CvTestSeq* cvCreateTestSeq(char* pConfigfile, char** videos, int numvideo, float Scale, int noise_type, double noise_ampl)
{
    int             size = sizeof(CvTestSeq_);
    CvTestSeq_*     pTS = (CvTestSeq_*)cvAlloc(size);
    CvFileStorage*  fs = cvOpenFileStorage( pConfigfile, NULL, CV_STORAGE_READ);
    int         i;

    if(pTS == NULL || fs == NULL) return NULL;
    memset(pTS,0,size);

    pTS->pFileStorage = fs;
    pTS->noise_ampl = noise_ampl;
    pTS->noise_type = noise_type;
    pTS->IVar_DI = 0;
    pTS->ObjNum = 0;

    /* Read all videos: */
    for (i=0; i<numvideo; ++i)
    {
        CvTestSeqElem*  pElemNew = icvTestSeqReadElemAll(pTS, fs, videos[i]);

        if(pTS->pElemList==NULL)pTS->pElemList = pElemNew;
        else
        {
            CvTestSeqElem* p = NULL;
            for(p=pTS->pElemList;p->next;p=p->next) {}
            p->next = pElemNew;
        }
    }   /* Read all videos. */

    {   /* Calculate elements and image size and video length: */
        CvTestSeqElem*  p = pTS->pElemList;
        int             num = 0;
        CvSize          MaxSize = {0,0};
        int             MaxFN = 0;

        for(p = pTS->pElemList; p; p=p->next, num++)
        {
            int     FN = p->FrameBegin+p->FrameNum;
            CvSize  S = {0,0};

            if(p->pImg && p->BG)
            {
                S.width = p->pImg->width;
                S.height = p->pImg->height;
            }

            if(MaxSize.width < S.width) MaxSize.width = S.width;
            if(MaxSize.height < S.height) MaxSize.height = S.height;
            if(MaxFN < FN)MaxFN = FN;
        }

        pTS->ListNum = num;

        if(MaxSize.width == 0)MaxSize.width = 320;
        if(MaxSize.height == 0)MaxSize.height = 240;

        MaxSize.width = cvRound(Scale*MaxSize.width);
        MaxSize.height = cvRound(Scale*MaxSize.height);

        pTS->pImg = cvCreateImage(MaxSize,IPL_DEPTH_8U,3);
        pTS->pImgMask = cvCreateImage(MaxSize,IPL_DEPTH_8U,1);
        pTS->FrameNum = MaxFN;

        for(p = pTS->pElemList; p; p=p->next)
        {
            if(p->FrameNum<=0)p->FrameNum=MaxFN;
        }
    }   /* Calculate elements and image size. */

    return (CvTestSeq*)pTS;

}   /* cvCreateTestSeq */

void cvReleaseTestSeq(CvTestSeq** ppTestSeq)
{
    CvTestSeq_* pTS = (CvTestSeq_*)ppTestSeq[0];

    icvTestSeqReleaseAll(&pTS->pElemList);
    if(pTS->pImg) cvReleaseImage(&pTS->pImg);
    if(pTS->pImgMask) cvReleaseImage(&pTS->pImgMask);
    if(pTS->pFileStorage)cvReleaseFileStorage(&pTS->pFileStorage);

    cvFree(ppTestSeq);

}   /* cvReleaseTestSeq */

void cvTestSeqSetFrame(CvTestSeq* pTestSeq, int n)
{
    CvTestSeq_*     pTS = (CvTestSeq_*)pTestSeq;
    pTS->CurFrame = n;
}

IplImage* cvTestSeqQueryFrame(CvTestSeq* pTestSeq)
{
    CvTestSeq_*     pTS = (CvTestSeq_*)pTestSeq;
    CvTestSeqElem*  p = pTS->pElemList;
    IplImage*       pImg = pTS->pImg;
    IplImage*       pImgAdd = cvCloneImage(pTS->pImg);
    IplImage*       pImgAddG = cvCreateImage(cvSize(pImgAdd->width,pImgAdd->height),IPL_DEPTH_8U,1);
    IplImage*       pImgMask = pTS->pImgMask;
    IplImage*       pImgMaskAdd = cvCloneImage(pTS->pImgMask);
    CvMat*          pT = cvCreateMat(2,3,CV_32F);

    if(pTS->CurFrame >= pTS->FrameNum) return NULL;
    cvZero(pImg);
    cvZero(pImgMask);

    for(p=pTS->pElemList; p; p=p->next)
    {
        int             DirectCopy = FALSE;
        int             frame = pTS->CurFrame - p->FrameBegin;
        //float           t = p->FrameNum>1?((float)frame/(p->FrameNum-1)):0;
        CvTSTrans*      pTrans = p->pTrans + frame%p->TransNum;

        assert(pTrans);

        if( p->FrameNum > 0 && (frame < 0 || frame >= p->FrameNum) )
        {   /* Current frame is out of range: */
            //if(p->pAVI)cvReleaseCapture(&p->pAVI);
            p->pAVI = NULL;
            continue;
        }

        cvZero(pImgAdd);
        cvZero(pImgAddG);
        cvZero(pImgMaskAdd);

        if(p->noise_type == CV_NOISE_NONE)
        {   /* For not noise:  */
            /* Get next frame: */
            icvTestSeqQureyFrameElem(p, frame);
            if(p->pImg == NULL) continue;

#if 1 /* transform using T filed in Trans */
            {   /* Calculate transform matrix: */
                float   W = (float)(pImgAdd->width-1);
                float   H = (float)(pImgAdd->height-1);
                float   W0 = (float)(p->pImg->width-1);
                float   H0 = (float)(p->pImg->height-1);
                cvZero(pT);
                {   /* Calcualte inverse matrix: */
                    CvMat   mat = cvMat(2,3,CV_32F, pTrans->T);
                    mat.width--;
                    pT->width--;
                    cvInvert(&mat, pT);
                    pT->width++;
                }

                CV_MAT_ELEM(pT[0], float, 0, 2) =
                    CV_MAT_ELEM(pT[0], float, 0, 0)*(W0/2-pTrans->T[2])+
                    CV_MAT_ELEM(pT[0], float, 0, 1)*(H0/2-pTrans->T[5]);

                CV_MAT_ELEM(pT[0], float, 1, 2) =
                    CV_MAT_ELEM(pT[0], float, 1, 0)*(W0/2-pTrans->T[2])+
                    CV_MAT_ELEM(pT[0], float, 1, 1)*(H0/2-pTrans->T[5]);

                CV_MAT_ELEM(pT[0], float, 0, 0) *= W0/W;
                CV_MAT_ELEM(pT[0], float, 0, 1) *= H0/H;
                CV_MAT_ELEM(pT[0], float, 1, 0) *= W0/W;
                CV_MAT_ELEM(pT[0], float, 1, 1) *= H0/H;

            }   /* Calculate transform matrix. */
#else
            {   /* Calculate transform matrix: */
                float   SX = (float)(p->pImg->width-1)/((pImgAdd->width-1)*pTrans->Scale.x);
                float   SY = (float)(p->pImg->height-1)/((pImgAdd->height-1)*pTrans->Scale.y);
                float   DX = pTrans->Shift.x;
                float   DY = pTrans->Shift.y;;
                cvZero(pT);
                ((float*)(pT->data.ptr+pT->step*0))[0]=SX;
                ((float*)(pT->data.ptr+pT->step*1))[1]=SY;
                ((float*)(pT->data.ptr+pT->step*0))[2]=SX*(pImgAdd->width-1)*(0.5f-DX);
                ((float*)(pT->data.ptr+pT->step*1))[2]=SY*(pImgAdd->height-1)*(0.5f-DY);
            }   /* Calculate transform matrix. */
#endif


            {   /* Check for direct copy: */
                DirectCopy = TRUE;
                if( fabs(CV_MAT_ELEM(pT[0],float,0,0)-1) > 0.00001) DirectCopy = FALSE;
                if( fabs(CV_MAT_ELEM(pT[0],float,1,0)) > 0.00001) DirectCopy = FALSE;
                if( fabs(CV_MAT_ELEM(pT[0],float,0,1)) > 0.00001) DirectCopy = FALSE;
                if( fabs(CV_MAT_ELEM(pT[0],float,0,1)) > 0.00001) DirectCopy = FALSE;
                if( fabs(CV_MAT_ELEM(pT[0],float,0,2)-(pImg->width-1)*0.5) > 0.5) DirectCopy = FALSE;
                if( fabs(CV_MAT_ELEM(pT[0],float,1,2)-(pImg->height-1)*0.5) > 0.5) DirectCopy = FALSE;
            }

            /* Extract image and mask: */
            if(p->pImg->nChannels == 1)
            {
                if(DirectCopy)
                {
                    cvCvtColor( p->pImg,pImgAdd,CV_GRAY2BGR);
                }
                else
                {
                    cvGetQuadrangleSubPix( p->pImg, pImgAddG, pT);
                    cvCvtColor( pImgAddG,pImgAdd,CV_GRAY2BGR);
                }
            }

            if(p->pImg->nChannels == 3)
            {
                if(DirectCopy)
                    cvCopy(p->pImg, pImgAdd);
                else
                    cvGetQuadrangleSubPix( p->pImg, pImgAdd, pT);
            }

            if(p->pImgMask)
            {
                if(DirectCopy)
                    cvCopy(p->pImgMask, pImgMaskAdd);
                else
                    cvGetQuadrangleSubPix( p->pImgMask, pImgMaskAdd, pT);

                cvThreshold(pImgMaskAdd,pImgMaskAdd,128,255,CV_THRESH_BINARY);
            }

            if(pTrans->C != 1 || pTrans->I != 0)
            {   /* Intensity transformation: */
                cvScale(pImgAdd, pImgAdd, pTrans->C,pTrans->I);
            }   /* Intensity transformation: */

            if(pTrans->GN > 0)
            {   /* Add noise: */
                IplImage* pImgN = cvCloneImage(pImgAdd);
                cvRandSetRange( &p->rnd_state, pTrans->GN, 0, -1 );
                cvRand(&p->rnd_state, pImgN);
                cvAdd(pImgN,pImgAdd,pImgAdd);
                cvReleaseImage(&pImgN);
            }   /* Add noise. */

            if(p->Mask)
            {   /* Update only mask: */
                cvOr(pImgMaskAdd, pImgMask, pImgMask);
            }
            else
            {   /* Add image and mask to exist main image and mask: */
                if(p->BG)
                {   /* If image is background: */
                    cvCopy( pImgAdd, pImg, NULL);
                }
                else
                {   /* If image is foreground: */
                    cvCopy( pImgAdd, pImg, pImgMaskAdd);
                    if(p->ObjID>=0)
                        cvOr(pImgMaskAdd, pImgMask, pImgMask);
                }
            }   /* Not mask. */
        }   /*  For not noise. */
        else
        {   /* Process noise video: */

            if( p->noise_type == CV_NOISE_GAUSSIAN ||
                p->noise_type == CV_NOISE_UNIFORM)

            {   /* Gaussan and uniform additive noise: */
                cvAddNoise(pImg,p->noise_type,pTrans->NoiseAmp * pTrans->C, &p->rnd_state);
            }   /* Gaussan and uniform additive noise. */

            if( p->noise_type == CV_NOISE_SPECKLE)
            {   /* Speckle -- multiplicative noise: */
                if(pTrans->I != 0)cvSubS(pImg,cvScalar(pTrans->I,pTrans->I,pTrans->I),pImg);
                cvAddNoise(pImg,p->noise_type,pTrans->NoiseAmp, &p->rnd_state);
                if(pTrans->I != 0)cvAddS(pImg,cvScalar(pTrans->I,pTrans->I,pTrans->I),pImg);
            }   /* Speckle -- multiplicative noise. */

            if( p->noise_type == CV_NOISE_SALT_AND_PEPPER)
            {   /* Salt and pepper: */
                cvAddNoise(pImg,p->noise_type,pTrans->NoiseAmp, &p->rnd_state);
            }   /* Salt and pepper. */
        }   /*  Process noise video.*/
    }   /*  Next item. */

    if(pImg)
    {
        if(pTS->noise_type != CV_NOISE_NONE)
        {   /* Add noise: */
            cvAddNoise(pImg,pTS->noise_type,pTS->noise_ampl);
        }

        if(pTS->IVar_DI != 0)
        {   /* Change intensity: */
            float   I = MIN(pTS->IVar_CurI,pTS->IVar_MaxI);
            I = MAX(I,pTS->IVar_MinI);
            cvScale(pImg,pImg,1,I);

            if(pTS->IVar_CurI >= pTS->IVar_MaxI)
                pTS->IVar_CurDI = (float)-fabs(pTS->IVar_DI);

            if(pTS->IVar_CurI <= pTS->IVar_MinI)
                pTS->IVar_CurDI = (float)+fabs(pTS->IVar_DI);

            pTS->IVar_CurI += pTS->IVar_CurDI;
        }
    }


    pTS->CurFrame++;
    cvReleaseImage(&pImgAdd);
    cvReleaseImage(&pImgAddG);
    cvReleaseImage(&pImgMaskAdd);
    cvReleaseMat(&pT);
    return pImg;

}   /*cvTestSeqQueryFrame*/

IplImage* cvTestSeqGetFGMask(CvTestSeq* pTestSeq)
{
    return ((CvTestSeq_*)pTestSeq)->pImgMask;
}

IplImage* cvTestSeqGetImage(CvTestSeq* pTestSeq)
{
    return ((CvTestSeq_*)pTestSeq)->pImg;
}

int cvTestSeqGetObjectNum(CvTestSeq* pTestSeq)
{
    //return ((CvTestSeq_*)pTestSeq)->ListNum;
    return ((CvTestSeq_*)pTestSeq)->ObjNum;
}

int cvTestSeqGetObjectPos(CvTestSeq* pTestSeq, int ObjIndex, CvPoint2D32f* pPos)
{
    CvTestSeq_*     pTS = (CvTestSeq_*)pTestSeq;
    CvTestSeqElem*  p = pTS->pElemList;
    if(pTS->CurFrame > pTS->FrameNum) return 0;

    for(p=pTS->pElemList; p; p=p->next)
    {
        int frame = pTS->CurFrame - p->FrameBegin - 1;
        if(ObjIndex==p->ObjID && frame >= 0 && frame < p->FrameNum) break;
    }

    if(p && p->pPos && p->PosNum>0)
    {
        CvTSTrans*  pTrans;
        int         frame = pTS->CurFrame - p->FrameBegin - 1;
        if(frame < 0 || frame >= p->FrameNum) return 0;
        //float t = (p->FrameNum>1)?((float)frame / (p->FrameNum-1)):0;
        pTrans = p->pTrans + frame%p->TransNum;
        pPos[0] = p->pPos[frame%p->PosNum];

#if 1   /* Transform using T filed in Trans: */
        {
            float x = pPos->x * (p->pImg?(p->pImg->width-1):1);
            float y = pPos->y * (p->pImg?(p->pImg->height-1):1);

            pPos->x = pTrans->T[0]*x+pTrans->T[1]*y+pTrans->T[2];
            pPos->y = pTrans->T[3]*x+pTrans->T[4]*y+pTrans->T[5];

            if(p->pImg)
            {
                pPos->x /= p->pImg->width-1;
                pPos->y /= p->pImg->height-1;
            }

        }


#else
        pPos->x = pPos->x * pTrans->Scale.x + pTrans->Shift.x;
        pPos->y = pPos->y * pTrans->Scale.y + pTrans->Shift.y;
#endif
        pPos->x *= pTS->pImg->width-1;
        pPos->y *= pTS->pImg->height-1;
        return 1;
    }
    return 0;

}   /* cvTestSeqGetObjectPos */

int cvTestSeqGetObjectSize(CvTestSeq* pTestSeq, int ObjIndex, CvPoint2D32f* pSize)
{
    CvTestSeq_*     pTS = (CvTestSeq_*)pTestSeq;
    CvTestSeqElem*  p = pTS->pElemList;
    if(pTS->CurFrame > pTS->FrameNum) return 0;

    for(p=pTS->pElemList; p; p=p->next)
    {
        int frame = pTS->CurFrame - p->FrameBegin - 1;
        if(ObjIndex==p->ObjID && frame >= 0 && frame < p->FrameNum) break;
    }

    if(p && p->pSize && p->SizeNum>0)
    {
        CvTSTrans*  pTrans;
        int         frame = pTS->CurFrame - p->FrameBegin - 1;

        if(frame < 0 || frame >= p->FrameNum) return 0;

        //float t = (p->FrameNum>1)?((float)frame / (p->FrameNum-1)):0;
        pTrans = p->pTrans + frame%p->TransNum;
        pSize[0] = p->pSize[frame%p->SizeNum];

#if 1   /* Transform using T filed in Trans: */
        {
            float x = pSize->x * (p->pImg?(p->pImg->width-1):1);
            float y = pSize->y * (p->pImg?(p->pImg->height-1):1);
            float   dx1, dx2;
            float   dy1, dy2;

            dx1 = (float)fabs(pTrans->T[0]*x+pTrans->T[1]*y);
            dy1 = (float)fabs(pTrans->T[3]*x+pTrans->T[4]*y);

            dx2 = (float)fabs(pTrans->T[0]*x - pTrans->T[1]*y);
            dy2 = (float)fabs(pTrans->T[3]*x - pTrans->T[4]*y);

            pSize->x = MAX(dx1,dx2);
            pSize->y = MAX(dy1,dy2);

            if(p->pImg)
            {
                pSize->x /= p->pImg->width-1;
                pSize->y /= p->pImg->height-1;
            }

        }


#else
        pSize->x = pSize->x * pTrans->Scale.x;
        pSize->y = pSize->y * pTrans->Scale.y;
#endif
        pSize->x *= pTS->pImg->width-1;
        pSize->y *= pTS->pImg->height-1;
        return 1;
    }

    return 0;

}   /* cvTestSeqGetObjectSize */

/* Add noise to finile image: */
void cvTestSeqAddNoise(CvTestSeq* pTestSeq, int noise_type, double noise_ampl)
{
    CvTestSeq_*     pTS = (CvTestSeq_*)pTestSeq;
    pTS->noise_type = noise_type;
    pTS->noise_ampl = noise_ampl;
}

/* Add Intensity variation: */
void cvTestSeqAddIntensityVariation(CvTestSeq* pTestSeq, float DI_per_frame, float MinI, float MaxI)
{
    CvTestSeq_* pTS = (CvTestSeq_*)pTestSeq;
    pTS->IVar_CurDI = pTS->IVar_DI = DI_per_frame;
    pTS->IVar_MaxI = MaxI;
    pTS->IVar_MinI = MinI;
}

void cvAddNoise(IplImage* pImg, int noise_type, double Ampl, CvRandState* rnd_state)
{   /* Add noise to image: */
    CvSize      S = cvSize(pImg->width,pImg->height);
    IplImage*   pImgAdd = cvCreateImage(S,pImg->depth,pImg->nChannels);
    static CvRandState local_rnd_state;
    static int  first = 1;

    if(first)
    {
        first = 0;
        cvRandInit( &local_rnd_state, 1, 0, 0,CV_RAND_NORMAL);
    }

    if(rnd_state == NULL)rnd_state = &local_rnd_state;

    if( noise_type == CV_NOISE_GAUSSIAN ||
        noise_type == CV_NOISE_UNIFORM)
    {   /* Gaussan and uniform additive noise: */
        int set_zero = 0;

        if( noise_type == CV_NOISE_GAUSSIAN)
        {
            rnd_state->disttype = CV_RAND_NORMAL;
            cvRandSetRange( rnd_state,  Ampl, 0, -1 );
            if(Ampl <= 0) set_zero = 1;
        }

        if( noise_type == CV_NOISE_UNIFORM)
        {
            double max_val =
                1.7320508075688772935274463415059 * Ampl;
            rnd_state->disttype = CV_RAND_UNI;
            cvRandSetRange( rnd_state, -max_val, max_val, -1 );
            if(max_val < 1) set_zero = 1;
        }

        if(!set_zero)
        {
            IplImage*   pImgNoise = cvCreateImage(S,IPL_DEPTH_32F,pImg->nChannels);
            IplImage*   pImgOrg = cvCreateImage(S,IPL_DEPTH_32F,pImg->nChannels);
            cvConvert(pImg, pImgOrg);
            cvRand(rnd_state, pImgNoise);
            cvAdd(pImgOrg,pImgNoise,pImgOrg);
            cvConvert(pImgOrg,pImg);
            cvReleaseImage(&pImgNoise);
            cvReleaseImage(&pImgOrg);
        }
    }   /* Gaussan and uniform additive noise. */

    if( noise_type == CV_NOISE_SPECKLE)
    {   /* Speckle -- multiplicative noise: */
        IplImage* pImgSP = cvCreateImage( S,IPL_DEPTH_32F, pImg->nChannels );
        IplImage* pImgTemp = cvCreateImage(S,IPL_DEPTH_32F, pImg->nChannels );
        rnd_state->disttype = CV_RAND_NORMAL;
        cvRandSetRange( rnd_state, Ampl, 0, -1 );
        cvRand(rnd_state, pImgSP);
        cvConvert(pImg,pImgTemp);
        cvMul(pImgSP,pImgTemp,pImgSP);
        cvAdd(pImgTemp,pImgSP,pImgTemp);
        cvConvert(pImgTemp,pImg);
        cvReleaseImage(&pImgSP);
        cvReleaseImage(&pImgTemp);
    }   /* Speckle -- multiplicative noise. */

    if( noise_type == CV_NOISE_SALT_AND_PEPPER && Ampl > 0)
    {   /* Salt and pepper: */
        IplImage* pImgMask = cvCreateImage( S,IPL_DEPTH_32F, 1 );
        IplImage* pImgMaskBin = cvCreateImage( S,IPL_DEPTH_8U, 1 );
        IplImage* pImgVal = cvCreateImage( S,IPL_DEPTH_8U, 1 );
        rnd_state->disttype = CV_RAND_UNI;

        /* Create mask: */
        cvRandSetRange( rnd_state, 0, 1, -1 );
        cvRand(rnd_state, pImgMask);
        cvThreshold(pImgMask,pImgMask, Ampl, 255, CV_THRESH_BINARY_INV );
        cvConvert(pImgMask,pImgMaskBin);

        /* Create vals: */
        cvRandSetRange( rnd_state, 0, 255, -1 );
        cvRand(rnd_state, pImgVal);
        cvThreshold(pImgVal,pImgVal,128, 255, CV_THRESH_BINARY );
        cvMerge(
            pImgAdd->nChannels>0?pImgVal:NULL,
            pImgAdd->nChannels>1?pImgVal:NULL,
            pImgAdd->nChannels>2?pImgVal:NULL,
            pImgAdd->nChannels>3?pImgVal:NULL,
            pImgAdd);
        cvCopy(pImgAdd, pImg, pImgMaskBin);
        cvReleaseImage(&pImgMask);
        cvReleaseImage(&pImgMaskBin);
        cvReleaseImage(&pImgVal);

    }   /* Salt and pepper. */

    cvReleaseImage(&pImgAdd);

}   /* cvAddNoise */
