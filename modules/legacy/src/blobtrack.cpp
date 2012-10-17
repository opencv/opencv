/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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
//   * The names of the copyright holders may not be used to endorse or promote products
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

//////////////////////////// CvVSModule /////////////////////////////

CvVSModule::CvVSModule()
{
    m_pNickName = NULL;
    m_pParamList = NULL;
    m_pModuleTypeName = NULL;
    m_pModuleName = NULL;
    m_Wnd = 0;
    AddParam("DebugWnd",&m_Wnd);
}

CvVSModule::~CvVSModule()
{
    CvDefParam* p = m_pParamList;
    for(;p;)
    {
        CvDefParam* pf = p;
        p=p->next;
        FreeParam(&pf);
    }
    m_pParamList=NULL;
    if(m_pModuleTypeName)free(m_pModuleTypeName);
    if(m_pModuleName)free(m_pModuleName);
}

void    CvVSModule::FreeParam(CvDefParam** pp)
{
    CvDefParam* p = pp[0];
    if(p->Str)free(p->Str);
    if(p->pName)free(p->pName);
    if(p->pComment)free(p->pComment);
    cvFree(pp);
}

CvDefParam* CvVSModule::NewParam(const char* name)
{
    CvDefParam* pNew = (CvDefParam*)cvAlloc(sizeof(CvDefParam));
    memset(pNew,0,sizeof(CvDefParam));
    pNew->pName = strdup(name);
    if(m_pParamList==NULL)
    {
        m_pParamList = pNew;
    }
    else
    {
        CvDefParam* p = m_pParamList;
        for(;p->next;p=p->next) ;
        p->next = pNew;
    }
    return pNew;
}

CvDefParam* CvVSModule::GetParamPtr(int index)
{
    CvDefParam* p = m_pParamList;
    for(;index>0 && p;index--,p=p->next) ;
    return p;
}

CvDefParam* CvVSModule::GetParamPtr(const char* name)
{
    CvDefParam* p = m_pParamList;
    for(;p;p=p->next)
    {
        if(cv_stricmp(p->pName,name)==0) break;
    }
    return p;
}

int  CvVSModule::IsParam(const char* name)
{
    return GetParamPtr(name)?1:0;
}

void CvVSModule::AddParam(const char* name, double* pAddr)
{
    NewParam(name)->pDouble = pAddr;
}

void CvVSModule::AddParam(const char* name, float* pAddr)
{
    NewParam(name)->pFloat=pAddr;
}

void CvVSModule::AddParam(const char* name, int* pAddr)
{
    NewParam(name)->pInt=pAddr;
}

void CvVSModule::AddParam(const char* name, const char** pAddr)
{
    CvDefParam* pP = NewParam(name);
    const char* p = pAddr?pAddr[0]:NULL;
    pP->pStr = pAddr?(char**)pAddr:&(pP->Str);
    if(p)
    {
        pP->Str = strdup(p);
        pP->pStr[0] = pP->Str;
    }
}

void CvVSModule::AddParam(const char* name)
{
    CvDefParam* p = NewParam(name);
    p->pDouble = &p->Double;
}

void CvVSModule::CommentParam(const char* name, const char* pComment)
{
    CvDefParam* p = GetParamPtr(name);
    if(p)p->pComment = pComment ? strdup(pComment) : 0;
}

void CvVSModule::SetTypeName(const char* name){m_pModuleTypeName = strdup(name);}

void CvVSModule::SetModuleName(const char* name){m_pModuleName = strdup(name);}

void CvVSModule::DelParam(const char* name)
{
    CvDefParam* p = m_pParamList;
    CvDefParam* pPrev = NULL;
    for(;p;p=p->next)
    {
        if(cv_stricmp(p->pName,name)==0) break;
        pPrev = p;
    }
    if(p)
    {
        if(pPrev)
        {
            pPrev->next = p->next;
        }
        else
        {
            m_pParamList = p->next;
        }
        FreeParam(&p);
    }
}/* DelParam */


const char* CvVSModule::GetParamName(int index)
{
    CvDefParam* p = GetParamPtr(index);
    return p?p->pName:NULL;
}
const char* CvVSModule::GetParamComment(const char* name)
{
    CvDefParam* p = GetParamPtr(name);
    if(p && p->pComment) return p->pComment;
    return NULL;
}
double CvVSModule::GetParam(const char* name)
{
    CvDefParam* p = GetParamPtr(name);
    if(p)
    {
        if(p->pDouble) return p->pDouble[0];
        if(p->pFloat) return p->pFloat[0];
        if(p->pInt) return p->pInt[0];
    }
    return 0;
};

const char* CvVSModule::GetParamStr(const char* name)
{
    CvDefParam* p = GetParamPtr(name);
    return p?p->Str:NULL;
}
void   CvVSModule::SetParam(const char* name, double val)
{
    CvDefParam* p = m_pParamList;
    for(;p;p=p->next)
    {
        if(cv_stricmp(p->pName,name) != 0) continue;
        if(p->pDouble)p->pDouble[0] = val;
        if(p->pFloat)p->pFloat[0] = (float)val;
        if(p->pInt)p->pInt[0] = cvRound(val);
    }
}
void   CvVSModule::SetParamStr(const char* name, const char* str)
{
    CvDefParam* p = m_pParamList;
    for(; p; p=p->next)
    {
        if(cv_stricmp(p->pName,name) != 0) continue;
        if(p->pStr)
        {
            if(p->Str)free(p->Str);
            p->Str = NULL;
            if(str)p->Str = strdup(str);
            p->pStr[0] = p->Str;
        }
    }
    /* Convert to double and set: */
    if(str) SetParam(name,atof(str));
}

void CvVSModule::TransferParamsFromChild(CvVSModule* pM, const char* prefix)
{
    char    tmp[1024];
    const char*   FN = NULL;
    int i;
    for(i=0;;++i)
    {
        const char* N = pM->GetParamName(i);
        if(N == NULL) break;
        FN = N;
        if(prefix)
        {
            strcpy(tmp,prefix);
            strcat(tmp,"_");
            FN = strcat(tmp,N);
        }

        if(!IsParam(FN))
        {
            if(pM->GetParamStr(N))
            {
                AddParam(FN,(const char**)NULL);
            }
            else
            {
                AddParam(FN);
            }
        }
        if(pM->GetParamStr(N))
        {
            const char* val = pM->GetParamStr(N);
            SetParamStr(FN,val);
        }
        else
        {
            double val = pM->GetParam(N);
            SetParam(FN,val);
        }
        CommentParam(FN, pM->GetParamComment(N));
    }/* transfer next param */
}/* Transfer params */

void CvVSModule::TransferParamsToChild(CvVSModule* pM, char* prefix)
{
    char    tmp[1024];
    int i;
    for(i=0;;++i)
    {
        const char* N = pM->GetParamName(i);
        if(N == NULL) break;
        if(prefix)
        {
            strcpy(tmp,prefix);
            strcat(tmp,"_");
            strcat(tmp,N);
        }
        else
        {
            strcpy(tmp,N);
        }

        if(IsParam(tmp))
        {
            if(GetParamStr(tmp))
                pM->SetParamStr(N,GetParamStr(tmp));
            else
                pM->SetParam(N,GetParam(tmp));
        }
    }/* Transfer next parameter */
    pM->ParamUpdate();
}/* Transfer params */

void CvVSModule::ParamUpdate(){}

const char*   CvVSModule::GetTypeName()
{
    return m_pModuleTypeName;
}

int     CvVSModule::IsModuleTypeName(const char* name)
{
    return m_pModuleTypeName?(cv_stricmp(m_pModuleTypeName,name)==0):0;
}

char*   CvVSModule::GetModuleName()
{
    return m_pModuleName;
}

int     CvVSModule::IsModuleName(const char* name)
{
    return m_pModuleName?(cv_stricmp(m_pModuleName,name)==0):0;
}

void CvVSModule::SetNickName(const char* pStr)
{
    if(m_pNickName)
        free(m_pNickName);

    m_pNickName = NULL;

    if(pStr)
        m_pNickName = strdup(pStr);
}

const char* CvVSModule::GetNickName()
{
    return m_pNickName ? m_pNickName : "unknown";
}

void CvVSModule::SaveState(CvFileStorage*)
{
}

void CvVSModule::LoadState(CvFileStorage*, CvFileNode*)
{
}

/////////////////////////////////////////////////////////////////////

void cvWriteStruct(CvFileStorage* fs, const char* name, void* addr, const char* desc, int num)
{
    cvStartWriteStruct(fs,name,CV_NODE_SEQ|CV_NODE_FLOW);
    cvWriteRawData(fs,addr,num,desc);
    cvEndWriteStruct(fs);
}

void cvReadStructByName(CvFileStorage* fs, CvFileNode* node, const char* name, void* addr, const char* desc)
{
    CvFileNode* pSeqNode = cvGetFileNodeByName(fs, node, name);
    if(pSeqNode==NULL)
    {
        printf("WARNING!!! Can't read structure %s\n",name);
    }
    else
    {
        if(CV_NODE_IS_SEQ(pSeqNode->tag))
        {
            cvReadRawData( fs, pSeqNode, addr, desc );
        }
        else
        {
            printf("WARNING!!! Structure %s is not sequence and can not be read\n",name);
        }
    }
}

////////////////////////////// CvFGDetector ///////////////////////////////////////////

CvFGDetector::CvFGDetector()
{
    SetTypeName("FGDetector");
}

void cvReleaseFGDetector(CvFGDetector** ppT )
{
    ppT[0]->Release();
    ppT[0] = 0;
}

///////////////////////////// CvBlobSeq ///////////////////////////////////////////////

CvBlobTrackSeq::CvBlobTrackSeq(int TrackSize)
{
    m_pMem = cvCreateMemStorage();
    m_pSeq = cvCreateSeq(0,sizeof(CvSeq),TrackSize,m_pMem);
}

CvBlobTrackSeq::~CvBlobTrackSeq()
{
    Clear();
    cvReleaseMemStorage(&m_pMem);
}

CvBlobTrack* CvBlobTrackSeq::GetBlobTrack(int TrackIndex)
{
    return (CvBlobTrack*)cvGetSeqElem(m_pSeq,TrackIndex);
}

CvBlobTrack* CvBlobTrackSeq::GetBlobTrackByID(int TrackID)
{
    int i;
    for(i=0; i<m_pSeq->total; ++i)
    {
        CvBlobTrack* pP = GetBlobTrack(i);
        if(pP && pP->TrackID == TrackID)
            return pP;
    }
    return NULL;
}

void CvBlobTrackSeq::DelBlobTrack(int TrackIndex)
{
    CvBlobTrack* pP = GetBlobTrack(TrackIndex);
    if(pP && pP->pBlobSeq) delete pP->pBlobSeq;
    cvSeqRemove(m_pSeq,TrackIndex);
}

void CvBlobTrackSeq::DelBlobTrackByID(int TrackID)
{
    int i;
    for(i=0; i<m_pSeq->total; ++i)
    {
        CvBlobTrack* pP = GetBlobTrack(i);
        if(TrackID == pP->TrackID)
        {
            DelBlobTrack(i);
            return;
        }
    }
}

void CvBlobTrackSeq::Clear()
{
    int i;
    for(i=GetBlobTrackNum();i>0;i--)
    {
        DelBlobTrack(i-1);
    }
    cvClearSeq(m_pSeq);
}

void CvBlobTrackSeq::AddBlobTrack(int TrackID, int StartFrame)
{
    CvBlobTrack N;
    N.TrackID = TrackID;
    N.StartFrame = StartFrame;
    N.pBlobSeq = new CvBlobSeq;
    cvSeqPush(m_pSeq,&N);
}

int CvBlobTrackSeq::GetBlobTrackNum()
{
    return m_pSeq->total;
}

void cvReleaseBlobDetector(CvBlobDetector** ppBD)
{
    ppBD[0]->Release();
    ppBD[0] = NULL;
}


///////////////////////////////////// CvObjectDetector /////////////////////////////////

CvObjectDetector::CvObjectDetector( const char* /*detector_file_name*/ )
{
}

CvObjectDetector::~CvObjectDetector()
{
}

/*
 * Release the current detector and load new detector from file
 * (if detector_file_name is not 0)
 * Return true on success:
 */
bool CvObjectDetector::Load( const char* /*detector_file_name*/ )
{
    return false;
}

/* Return min detector window size: */
CvSize CvObjectDetector::GetMinWindowSize() const
{
    return cvSize(0,0);
}

/* Return max border: */
int CvObjectDetector::GetMaxBorderSize() const
{
    return 0;
}

/*
 * Detect the object on the image and push the detected
 * blobs into <detected_blob_seq> which must be the sequence of <CvDetectedBlob>s
 */
void CvObjectDetector::Detect( const CvArr* /*img*/,
                               /* out */ CvBlobSeq* /*detected_blob_seq*/ )
{
}

//////////////////////////////// CvBlobTracker //////////////////////////////////////

CvBlobTracker::CvBlobTracker(){SetTypeName("BlobTracker");}

/* Process one blob (for multi hypothesis tracing): */
void CvBlobTracker::ProcessBlob(int BlobIndex, CvBlob* pBlob, IplImage* /*pImg*/, IplImage* /*pImgFG*/)
{
    CvBlob* pB;
    int ID = 0;
    assert(pBlob);
    //pBlob->ID;
    pB = GetBlob(BlobIndex);
    if(pB)
        pBlob[0] = pB[0];
    pBlob->ID = ID;
}

/* Get confidence/wieght/probability (0-1) for blob: */
double  CvBlobTracker::GetConfidence(int /*BlobIndex*/, CvBlob* /*pBlob*/, IplImage* /*pImg*/, IplImage* /*pImgFG*/)
{
    return 1;
}

double CvBlobTracker::GetConfidenceList(CvBlobSeq* pBlobList, IplImage* pImg, IplImage* pImgFG)
{
    int     b,bN = pBlobList->GetBlobNum();
    double  W = 1;
    for(b=0;b<bN;++b)
    {
        CvBlob* pB = pBlobList->GetBlob(b);
        int     BI = GetBlobIndexByID(pB->ID);
        W *= GetConfidence(BI,pB,pImg,pImgFG);
    }
    return W;
}

void CvBlobTracker::UpdateBlob(int /*BlobIndex*/, CvBlob* /*pBlob*/, IplImage* /*pImg*/, IplImage* /*pImgFG*/)
{
}

/* Update all blob models: */
void CvBlobTracker::Update(IplImage* pImg, IplImage* pImgFG)
{
    int i;
    for(i=GetBlobNum();i>0;i--)
    {
        CvBlob* pB=GetBlob(i-1);
        UpdateBlob(i-1, pB, pImg, pImgFG);
    }
}

/* Return pointer to blob by its unique ID: */
int     CvBlobTracker::GetBlobIndexByID(int BlobID)
{
    int i;
    for(i=GetBlobNum();i>0;i--)
    {
        CvBlob* pB=GetBlob(i-1);
        if(CV_BLOB_ID(pB) == BlobID) return i-1;
    }
    return -1;
}

/* Return pointer to blob by its unique ID: */
CvBlob* CvBlobTracker::GetBlobByID(int BlobID)
{
    return GetBlob(GetBlobIndexByID(BlobID));
}

/* Delete blob by its ID: */
void    CvBlobTracker::DelBlobByID(int BlobID)
{
    DelBlob(GetBlobIndexByID(BlobID));
}

/* Set new parameters for specified (by index) blob: */
void    CvBlobTracker::SetBlob(int /*BlobIndex*/, CvBlob* /*pBlob*/)
{
}

/* Set new parameters for specified (by ID) blob: */
void    CvBlobTracker::SetBlobByID(int BlobID, CvBlob* pBlob)
{
    SetBlob(GetBlobIndexByID(BlobID),pBlob);
}

/*  ===============  MULTI HYPOTHESIS INTERFACE ==================  */

/* Return number of position hyposetis of currently tracked blob: */
int     CvBlobTracker::GetBlobHypNum(int /*BlobIdx*/)
{
    return 1;
}

/* Return pointer to specified blob hypothesis by index blob: */
CvBlob* CvBlobTracker::GetBlobHyp(int BlobIndex, int /*hypothesis*/)
{
    return GetBlob(BlobIndex);
}

/* Set new parameters for specified (by index) blob hyp
 * (can be called several times for each hyp ):
 */
void    CvBlobTracker::SetBlobHyp(int /*BlobIndex*/, CvBlob* /*pBlob*/)
{
}

void cvReleaseBlobTracker(CvBlobTracker**ppT )
{
    ppT[0]->Release();
    ppT[0] = 0;
}

