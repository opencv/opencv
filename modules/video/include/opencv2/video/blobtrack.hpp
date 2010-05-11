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


#ifndef __OPENCV_VIDEOSURVEILLANCE_H__
#define __OPENCV_VIDEOSURVEILLANCE_H__

/* Turn off the functionality until cvaux/src/Makefile.am gets updated: */
//#if _MSC_VER >= 1200

#include <stdio.h>

#if _MSC_VER >= 1200 || defined __BORLANDC__
#define cv_stricmp stricmp
#define cv_strnicmp strnicmp
#if defined WINCE
#define strdup _strdup
#define stricmp _stricmp
#endif
#elif defined __GNUC__
#define cv_stricmp strcasecmp
#define cv_strnicmp strncasecmp
#else
#error Do not know how to make case-insensitive string comparison on this platform
#endif

//struct DefParam;
struct CvDefParam
{
    struct CvDefParam*    next;
    char*               pName;
    char*               pComment;
    double*             pDouble;
    double              Double;
    float*              pFloat;
    float               Float;
    int*                pInt;
    int                 Int;
    char**              pStr;
    char*               Str;
};

class CV_EXPORTS CvVSModule
{
private: /* Internal data: */
    CvDefParam*   m_pParamList;
    char*       m_pModuleTypeName;
    char*       m_pModuleName;
    char*       m_pNickName;
protected:
    int         m_Wnd;
public: /* Constructor and destructor: */
    CvVSModule()
    {
        m_pNickName = NULL;
        m_pParamList = NULL;
        m_pModuleTypeName = NULL;
        m_pModuleName = NULL;
        m_Wnd = 0;
        AddParam("DebugWnd",&m_Wnd);
    }
    virtual ~CvVSModule()
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
private: /* Internal functions: */
    void    FreeParam(CvDefParam** pp)
    {
        CvDefParam* p = pp[0];
        if(p->Str)free(p->Str);
        if(p->pName)free(p->pName);
        if(p->pComment)free(p->pComment);
        cvFree(pp);
    }
    CvDefParam* NewParam(const char* name)
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
    };

    CvDefParam* GetParamPtr(int index)
    {
        CvDefParam* p = m_pParamList;
        for(;index>0 && p;index--,p=p->next) ;
        return p;
    }
    CvDefParam* GetParamPtr(const char* name)
    {
        CvDefParam* p = m_pParamList;
        for(;p;p=p->next)
        {
            if(cv_stricmp(p->pName,name)==0) break;
        }
        return p;
    }
protected: /* INTERNAL INTERFACE */
    int  IsParam(const char* name)
    {
        return GetParamPtr(name)?1:0;
    };
    void AddParam(const char* name, double* pAddr)
    {
        NewParam(name)->pDouble = pAddr;
    };
    void AddParam(const char* name, float* pAddr)
    {
        NewParam(name)->pFloat=pAddr;
    };
    void AddParam(const char* name, int* pAddr)
    {
        NewParam(name)->pInt=pAddr;
    };
    void AddParam(const char* name, const char** pAddr)
    {
        CvDefParam* pP = NewParam(name);
        const char* p = pAddr?pAddr[0]:NULL;
        pP->pStr = pAddr?(char**)pAddr:&(pP->Str);
        if(p)
        {
            pP->Str = strdup(p);
            pP->pStr[0] = pP->Str;
        }
    };
    void AddParam(const char* name)
    {
        CvDefParam* p = NewParam(name);
        p->pDouble = &p->Double;
    };
    void CommentParam(const char* name, const char* pComment)
    {
        CvDefParam* p = GetParamPtr(name);
        if(p)p->pComment = pComment ? strdup(pComment) : 0;
    };
    void SetTypeName(const char* name){m_pModuleTypeName = strdup(name);}
    void SetModuleName(const char* name){m_pModuleName = strdup(name);}
    void DelParam(const char* name)
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

public: /* EXTERNAL INTERFACE */
    const char* GetParamName(int index)
    {
        CvDefParam* p = GetParamPtr(index);
        return p?p->pName:NULL;
    }
    const char* GetParamComment(const char* name)
    {
        CvDefParam* p = GetParamPtr(name);
        if(p && p->pComment) return p->pComment;
        return NULL;
    }
    double GetParam(const char* name)
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

    const char* GetParamStr(const char* name)
    {
        CvDefParam* p = GetParamPtr(name);
        return p?p->Str:NULL;
    }
    void   SetParam(const char* name, double val)
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
    void   SetParamStr(const char* name, const char* str)
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
    void TransferParamsFromChild(CvVSModule* pM, const char* prefix = NULL)
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

    void TransferParamsToChild(CvVSModule* pM, char* prefix = NULL)
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

    virtual void ParamUpdate(){};
    const char*   GetTypeName()
    {
        return m_pModuleTypeName;
    }
    int     IsModuleTypeName(const char* name)
    {
        return m_pModuleTypeName?(cv_stricmp(m_pModuleTypeName,name)==0):0;
    }
    char*   GetModuleName()
    {
        return m_pModuleName;
    }
    int     IsModuleName(const char* name)
    {
        return m_pModuleName?(cv_stricmp(m_pModuleName,name)==0):0;
    }
    void SetNickName(const char* pStr)
    {
        if(m_pNickName)
            free(m_pNickName);

        m_pNickName = NULL;

        if(pStr)
            m_pNickName = strdup(pStr);
    }
    const char* GetNickName()
    {
        return m_pNickName ? m_pNickName : "unknown";
    }
    virtual void SaveState(CvFileStorage*){};
    virtual void LoadState(CvFileStorage*, CvFileNode*){};

    virtual void Release() = 0;
};/* CvVMModule */
void inline cvWriteStruct(CvFileStorage* fs, const char* name, void* addr, const char* desc, int num=1)
{
    cvStartWriteStruct(fs,name,CV_NODE_SEQ|CV_NODE_FLOW);
    cvWriteRawData(fs,addr,num,desc);
    cvEndWriteStruct(fs);
}
void inline cvReadStructByName(CvFileStorage* fs, CvFileNode* node, const char* name, void* addr, const char* desc)
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


/* FOREGROUND DETECTOR INTERFACE */
class CV_EXPORTS CvFGDetector: public CvVSModule
{
public:
	CvFGDetector(){SetTypeName("FGDetector");};
    virtual IplImage* GetMask() = 0;
    /* Process current image: */
    virtual void    Process(IplImage* pImg) = 0;
    /* Release foreground detector: */
    virtual void    Release() = 0;
};
inline void cvReleaseFGDetector(CvFGDetector** ppT )
{
    ppT[0]->Release();
    ppT[0] = 0;
}
/* FOREGROUND DETECTOR INTERFACE */

CV_EXPORTS CvFGDetector* cvCreateFGDetectorBase(int type, void *param);


/* BLOB STRUCTURE*/
struct CvBlob
{
    float   x,y; /* blob position   */
    float   w,h; /* blob sizes      */
    int     ID;  /* blob ID         */
};

inline CvBlob cvBlob(float x,float y, float w, float h)
{
    CvBlob B = {x,y,w,h,0};
    return B;
}
#define CV_BLOB_MINW 5
#define CV_BLOB_MINH 5
#define CV_BLOB_ID(pB) (((CvBlob*)(pB))->ID)
#define CV_BLOB_CENTER(pB) cvPoint2D32f(((CvBlob*)(pB))->x,((CvBlob*)(pB))->y)
#define CV_BLOB_X(pB) (((CvBlob*)(pB))->x)
#define CV_BLOB_Y(pB) (((CvBlob*)(pB))->y)
#define CV_BLOB_WX(pB) (((CvBlob*)(pB))->w)
#define CV_BLOB_WY(pB) (((CvBlob*)(pB))->h)
#define CV_BLOB_RX(pB) (0.5f*CV_BLOB_WX(pB))
#define CV_BLOB_RY(pB) (0.5f*CV_BLOB_WY(pB))
#define CV_BLOB_RECT(pB) cvRect(cvRound(((CvBlob*)(pB))->x-CV_BLOB_RX(pB)),cvRound(((CvBlob*)(pB))->y-CV_BLOB_RY(pB)),cvRound(CV_BLOB_WX(pB)),cvRound(CV_BLOB_WY(pB)))
/* END BLOB STRUCTURE*/


/* simple BLOBLIST */
class CV_EXPORTS CvBlobSeq
{
public:
    CvBlobSeq(int BlobSize = sizeof(CvBlob))
    {
        m_pMem = cvCreateMemStorage();
        m_pSeq = cvCreateSeq(0,sizeof(CvSeq),BlobSize,m_pMem);
        strcpy(m_pElemFormat,"ffffi");
    }
    virtual ~CvBlobSeq()
    {
        cvReleaseMemStorage(&m_pMem);
    };
    virtual CvBlob* GetBlob(int BlobIndex)
    {
        return (CvBlob*)cvGetSeqElem(m_pSeq,BlobIndex);
    };
    virtual CvBlob* GetBlobByID(int BlobID)
    {
        int i;
        for(i=0; i<m_pSeq->total; ++i)
            if(BlobID == CV_BLOB_ID(GetBlob(i)))
                return GetBlob(i);
        return NULL;
    };
    virtual void DelBlob(int BlobIndex)
    {
        cvSeqRemove(m_pSeq,BlobIndex);
    };
    virtual void DelBlobByID(int BlobID)
    {
        int i;
        for(i=0; i<m_pSeq->total; ++i)
        {
            if(BlobID == CV_BLOB_ID(GetBlob(i)))
            {
                DelBlob(i);
                return;
            }
        }
    };
    virtual void Clear()
    {
        cvClearSeq(m_pSeq);
    };
    virtual void AddBlob(CvBlob* pB)
    {
        cvSeqPush(m_pSeq,pB);
    };
    virtual int GetBlobNum()
    {
        return m_pSeq->total;
    };
    virtual void Write(CvFileStorage* fs, const char* name)
    {
        const char*  attr[] = {"dt",m_pElemFormat,NULL};
        if(fs)
        {
            cvWrite(fs,name,m_pSeq,cvAttrList(attr,NULL));
        }
    }
    virtual void Load(CvFileStorage* fs, CvFileNode* node)
    {
        if(fs==NULL) return;
        CvSeq* pSeq = (CvSeq*)cvRead(fs, node);
        if(pSeq)
        {
            int i;
            cvClearSeq(m_pSeq);
            for(i=0;i<pSeq->total;++i)
            {
                void* pB = cvGetSeqElem( pSeq, i );
                cvSeqPush( m_pSeq, pB );
            }
        }
    }
    void AddFormat(const char* str){strcat(m_pElemFormat,str);}
protected:
    CvMemStorage*   m_pMem;
    CvSeq*          m_pSeq;
    char            m_pElemFormat[1024];
};
/* simple BLOBLIST */


/* simple TRACKLIST */
struct CvBlobTrack
{
    int         TrackID;
    int         StartFrame;
    CvBlobSeq*  pBlobSeq;
};

class CV_EXPORTS CvBlobTrackSeq
{
public:
    CvBlobTrackSeq(int TrackSize = sizeof(CvBlobTrack))
    {
        m_pMem = cvCreateMemStorage();
        m_pSeq = cvCreateSeq(0,sizeof(CvSeq),TrackSize,m_pMem);
    }
    virtual ~CvBlobTrackSeq()
    {
        Clear();
        cvReleaseMemStorage(&m_pMem);
    };
    virtual CvBlobTrack* GetBlobTrack(int TrackIndex)
    {
        return (CvBlobTrack*)cvGetSeqElem(m_pSeq,TrackIndex);
    };
    virtual CvBlobTrack* GetBlobTrackByID(int TrackID)
    {
        int i;
        for(i=0; i<m_pSeq->total; ++i)
        {
            CvBlobTrack* pP = GetBlobTrack(i);
            if(pP && pP->TrackID == TrackID)
                return pP;
        }
        return NULL;
    };
    virtual void DelBlobTrack(int TrackIndex)
    {
        CvBlobTrack* pP = GetBlobTrack(TrackIndex);
        if(pP && pP->pBlobSeq) delete pP->pBlobSeq;
        cvSeqRemove(m_pSeq,TrackIndex);
    };
    virtual void DelBlobTrackByID(int TrackID)
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
    };
    virtual void Clear()
    {
        int i;
        for(i=GetBlobTrackNum();i>0;i--)
        {
            DelBlobTrack(i-1);
        }
        cvClearSeq(m_pSeq);
    };
    virtual void AddBlobTrack(int TrackID, int StartFrame = 0)
    {
        CvBlobTrack N;
        N.TrackID = TrackID;
        N.StartFrame = StartFrame;
        N.pBlobSeq = new CvBlobSeq;
        cvSeqPush(m_pSeq,&N);
    };
    virtual int GetBlobTrackNum()
    {
        return m_pSeq->total;
    };
protected:
    CvMemStorage*   m_pMem;
    CvSeq*          m_pSeq;
};

/* simple TRACKLIST */


/* BLOB DETECTOR INTERFACE */
class CV_EXPORTS CvBlobDetector: public CvVSModule
{
public:
	CvBlobDetector(){SetTypeName("BlobDetector");};
    /* Try to detect new blob entrance based on foreground mask. */
    /* pFGMask - image of foreground mask */
    /* pNewBlob - pointer to CvBlob structure which will be filled if new blob entrance detected */
    /* pOldBlobList - pointer to blob list which already exist on image */
    virtual int DetectNewBlob(IplImage* pImg, IplImage* pImgFG, CvBlobSeq* pNewBlobList, CvBlobSeq* pOldBlobList) = 0;
    /* release blob detector */
    virtual void Release()=0;
};
/* Release any blob detector: */
inline void cvReleaseBlobDetector(CvBlobDetector** ppBD)
{
    ppBD[0]->Release();
    ppBD[0] = NULL;
}
/* END BLOB DETECTOR INTERFACE */

/* Declarations of constructors of implemented modules: */
CV_EXPORTS CvBlobDetector* cvCreateBlobDetectorSimple();
CV_EXPORTS CvBlobDetector* cvCreateBlobDetectorCC();


struct CV_EXPORTS CvDetectedBlob : public CvBlob
{
    float response;
};

CV_INLINE CvDetectedBlob cvDetectedBlob( float x, float y, float w, float h, int ID = 0, float response = 0.0F )
{
    CvDetectedBlob b;
    b.x = x; b.y = y; b.w = w; b.h = h; b.ID = ID; b.response = response;
    return b;
}


class CV_EXPORTS CvObjectDetector
{
public:
    CvObjectDetector( const char* /*detector_file_name*/ = 0 ) {};

    ~CvObjectDetector() {};

    /*
     * Release the current detector and load new detector from file
     * (if detector_file_name is not 0)
     * Return true on success:
     */
    bool Load( const char* /*detector_file_name*/ = 0 ) { return false; }

    /* Return min detector window size: */
    CvSize GetMinWindowSize() const { return cvSize(0,0); }

    /* Return max border: */
    int GetMaxBorderSize() const { return 0; }

    /*
     * Detect the object on the image and push the detected
     * blobs into <detected_blob_seq> which must be the sequence of <CvDetectedBlob>s
     */
    void Detect( const CvArr* /*img*/, /* out */ CvBlobSeq* /*detected_blob_seq*/ = 0 ) {};

protected:
    class CvObjectDetectorImpl* impl;
};


CV_INLINE CvRect cvRectIntersection( const CvRect r1, const CvRect r2 )
{
    CvRect r = cvRect( MAX(r1.x, r2.x), MAX(r1.y, r2.y), 0, 0 );

    r.width  = MIN(r1.x + r1.width, r2.x + r2.width) - r.x;
    r.height = MIN(r1.y + r1.height, r2.y + r2.height) - r.y;

    return r;
}


/*
 * CvImageDrawer
 *
 * Draw on an image the specified ROIs from the source image and
 * given blobs as ellipses or rectangles:
 */

struct CvDrawShape
{
    enum {RECT, ELLIPSE} shape;
    CvScalar color;
};

/*extern const CvDrawShape icv_shape[] =
{
    { CvDrawShape::ELLIPSE, CV_RGB(255,0,0) },
    { CvDrawShape::ELLIPSE, CV_RGB(0,255,0) },
    { CvDrawShape::ELLIPSE, CV_RGB(0,0,255) },
    { CvDrawShape::ELLIPSE, CV_RGB(255,255,0) },
    { CvDrawShape::ELLIPSE, CV_RGB(0,255,255) },
    { CvDrawShape::ELLIPSE, CV_RGB(255,0,255) }
};*/

class CV_EXPORTS CvImageDrawer
{
public:
    CvImageDrawer() : m_image(0) {}
    ~CvImageDrawer() { cvReleaseImage( &m_image ); }
    void SetShapes( const CvDrawShape* shapes, int num );
    /* <blob_seq> must be the sequence of <CvDetectedBlob>s */
    IplImage* Draw( const CvArr* src, CvBlobSeq* blob_seq = 0, const CvSeq* roi_seq = 0 );
    IplImage* GetImage() { return m_image; }
protected:
    //static const int MAX_SHAPES = sizeof(icv_shape) / sizeof(icv_shape[0]);;

    IplImage* m_image;
    CvDrawShape m_shape[16];
};



/* Trajectory generation module: */
class CV_EXPORTS CvBlobTrackGen: public CvVSModule
{
public:
	CvBlobTrackGen(){SetTypeName("BlobTrackGen");};
    virtual void    SetFileName(char* pFileName) = 0;
    virtual void    AddBlob(CvBlob* pBlob) = 0;
    virtual void    Process(IplImage* pImg = NULL, IplImage* pFG = NULL) = 0;
    virtual void    Release() = 0;
};

inline void cvReleaseBlobTrackGen(CvBlobTrackGen** pBTGen)
{
    if(*pBTGen)(*pBTGen)->Release();
    *pBTGen = 0;
}

/* Declarations of constructors of implemented modules: */
CV_EXPORTS CvBlobTrackGen* cvCreateModuleBlobTrackGen1();
CV_EXPORTS CvBlobTrackGen* cvCreateModuleBlobTrackGenYML();



/* BLOB TRACKER INTERFACE */
class CV_EXPORTS CvBlobTracker: public CvVSModule
{
public:
    CvBlobTracker(){SetTypeName("BlobTracker");};

    /* Add new blob to track it and assign to this blob personal ID */
    /* pBlob - pointer to structure with blob parameters (ID is ignored)*/
    /* pImg - current image */
    /* pImgFG - current foreground mask */
    /* Return pointer to new added blob: */
    virtual CvBlob* AddBlob(CvBlob* pBlob, IplImage* pImg, IplImage* pImgFG = NULL ) = 0;

    /* Return number of currently tracked blobs: */
    virtual int     GetBlobNum() = 0;

    /* Return pointer to specified by index blob: */
    virtual CvBlob* GetBlob(int BlobIndex) = 0;

    /* Delete blob by its index: */
    virtual void    DelBlob(int BlobIndex) = 0;

    /* Process current image and track all existed blobs: */
    virtual void    Process(IplImage* pImg, IplImage* pImgFG = NULL) = 0;

    /* Release blob tracker: */
    virtual void    Release() = 0;


    /* Process one blob (for multi hypothesis tracing): */
    virtual void ProcessBlob(int BlobIndex, CvBlob* pBlob, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL)
    {
        CvBlob* pB;
        int ID = 0;
        assert(pBlob);
        //pBlob->ID;
        pB = GetBlob(BlobIndex);
        if(pB)
            pBlob[0] = pB[0];
        pBlob->ID = ID;
    };

    /* Get confidence/wieght/probability (0-1) for blob: */
    virtual double  GetConfidence(int /*BlobIndex*/, CvBlob* /*pBlob*/, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL)
    {
        return 1;
    };

    virtual double GetConfidenceList(CvBlobSeq* pBlobList, IplImage* pImg, IplImage* pImgFG = NULL)
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
    };

    virtual void UpdateBlob(int /*BlobIndex*/, CvBlob* /*pBlob*/, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL){};

    /* Update all blob models: */
    virtual void Update(IplImage* pImg, IplImage* pImgFG = NULL)
    {
        int i;
        for(i=GetBlobNum();i>0;i--)
        {
            CvBlob* pB=GetBlob(i-1);
            UpdateBlob(i-1, pB, pImg, pImgFG);
        }

    };

    /* Return pointer to blob by its unique ID: */
    virtual int     GetBlobIndexByID(int BlobID)
    {
        int i;
        for(i=GetBlobNum();i>0;i--)
        {
            CvBlob* pB=GetBlob(i-1);
            if(CV_BLOB_ID(pB) == BlobID) return i-1;
        }
        return -1;
    };

    /* Return pointer to blob by its unique ID: */
    virtual CvBlob* GetBlobByID(int BlobID){return GetBlob(GetBlobIndexByID(BlobID));};

    /* Delete blob by its ID: */
    virtual void    DelBlobByID(int BlobID){DelBlob(GetBlobIndexByID(BlobID));};

    /* Set new parameters for specified (by index) blob: */
    virtual void    SetBlob(int /*BlobIndex*/, CvBlob* /*pBlob*/){};

    /* Set new parameters for specified (by ID) blob: */
    virtual void    SetBlobByID(int BlobID, CvBlob* pBlob)
    {
        SetBlob(GetBlobIndexByID(BlobID),pBlob);
    };

    /*  ===============  MULTI HYPOTHESIS INTERFACE ==================  */

    /* Return number of position hyposetis of currently tracked blob: */
    virtual int     GetBlobHypNum(int /*BlobIdx*/){return 1;};

    /* Return pointer to specified blob hypothesis by index blob: */
    virtual CvBlob* GetBlobHyp(int BlobIndex, int /*hypothesis*/){return GetBlob(BlobIndex);};

    /* Set new parameters for specified (by index) blob hyp
     * (can be called several times for each hyp ):
     */
    virtual void    SetBlobHyp(int /*BlobIndex*/, CvBlob* /*pBlob*/){};
};
inline void cvReleaseBlobTracker(CvBlobTracker**ppT )
{
    ppT[0]->Release();
    ppT[0] = 0;
}
/* BLOB TRACKER INTERFACE */

/*BLOB TRACKER ONE INTERFACE */
class CV_EXPORTS CvBlobTrackerOne:public CvVSModule
{
public:
    virtual void Init(CvBlob* pBlobInit, IplImage* pImg, IplImage* pImgFG = NULL) = 0;
    virtual CvBlob* Process(CvBlob* pBlobPrev, IplImage* pImg, IplImage* pImgFG = NULL) = 0;
    virtual void Release() =  0;

    /* Non-required methods: */
    virtual void SkipProcess(CvBlob* /*pBlobPrev*/, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL){};
    virtual void Update(CvBlob* /*pBlob*/, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL){};
    virtual void SetCollision(int /*CollisionFlag*/){}; /* call in case of blob collision situation*/
    virtual double GetConfidence(CvBlob* /*pBlob*/, IplImage* /*pImg*/,
                                 IplImage* /*pImgFG*/ = NULL, IplImage* /*pImgUnusedReg*/ = NULL)
    {
        return 1;
    };
};
inline void cvReleaseBlobTrackerOne(CvBlobTrackerOne **ppT )
{
    ppT[0]->Release();
    ppT[0] = 0;
}
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerList(CvBlobTrackerOne* (*create)());
/*BLOB TRACKER ONE INTERFACE */

/* Declarations of constructors of implemented modules: */

/* Some declarations for specific MeanShift tracker: */
#define PROFILE_EPANECHNIKOV    0
#define PROFILE_DOG             1
struct CvBlobTrackerParamMS
{
    int     noOfSigBits;
    int     appearance_profile;
    int     meanshift_profile;
    float   sigma;
};

CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMS1(CvBlobTrackerParamMS* param);
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMS2(CvBlobTrackerParamMS* param);
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMS1ByList();

/* Some declarations for specific Likelihood tracker: */
struct CvBlobTrackerParamLH
{
    int     HistType; /* see Prob.h */
    int     ScaleAfter;
};

/* Without scale optimization: */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerLHR(CvBlobTrackerParamLH* /*param*/ = NULL);

/* With scale optimization: */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerLHRS(CvBlobTrackerParamLH* /*param*/ = NULL);

/* Simple blob tracker based on connected component tracking: */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerCC();

/* Connected component tracking and mean-shift particle filter collion-resolver: */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerCCMSPF();

/* Blob tracker that integrates meanshift and connected components: */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMSFG();
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMSFGS();

/* Meanshift without connected-components */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMS();

/* Particle filtering via Bhattacharya coefficient, which        */
/* is roughly the dot-product of two probability densities.      */
/* See: Real-Time Tracking of Non-Rigid Objects using Mean Shift */
/*      Comanicius, Ramesh, Meer, 2000, 8p                       */
/*      http://citeseer.ist.psu.edu/321441.html                  */
CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerMSPF();

/* =========== tracker integrators trackers =============*/

/* Integrator based on Particle Filtering method: */
//CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerIPF();

/* Rule based integrator: */
//CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerIRB();

/* Integrator based on data fusion using particle filtering: */
//CV_EXPORTS CvBlobTracker* cvCreateBlobTrackerIPFDF();




/* Trajectory postprocessing module: */
class CV_EXPORTS CvBlobTrackPostProc: public CvVSModule
{
public:
	CvBlobTrackPostProc(){SetTypeName("BlobTrackPostProc");};
    virtual void    AddBlob(CvBlob* pBlob) = 0;
    virtual void    Process() = 0;
    virtual int     GetBlobNum() = 0;
    virtual CvBlob* GetBlob(int index) = 0;
    virtual void    Release() = 0;

    /* Additional functionality: */
    virtual CvBlob* GetBlobByID(int BlobID)
    {
        int i;
        for(i=GetBlobNum();i>0;i--)
        {
            CvBlob* pB=GetBlob(i-1);
            if(pB->ID==BlobID) return pB;
        }
        return NULL;
    };
};

inline void cvReleaseBlobTrackPostProc(CvBlobTrackPostProc** pBTPP)
{
    if(pBTPP == NULL) return;
    if(*pBTPP)(*pBTPP)->Release();
    *pBTPP = 0;
}

/* Trajectory generation module: */
class CV_EXPORTS CvBlobTrackPostProcOne: public CvVSModule
{
public:
	CvBlobTrackPostProcOne(){SetTypeName("BlobTrackPostOne");};
    virtual CvBlob* Process(CvBlob* pBlob) = 0;
    virtual void    Release() = 0;
};

/* Create blob tracking post processing module based on simle module: */
CV_EXPORTS CvBlobTrackPostProc* cvCreateBlobTrackPostProcList(CvBlobTrackPostProcOne* (*create)());


/* Declarations of constructors of implemented modules: */
CV_EXPORTS CvBlobTrackPostProc* cvCreateModuleBlobTrackPostProcKalman();
CV_EXPORTS CvBlobTrackPostProc* cvCreateModuleBlobTrackPostProcTimeAverRect();
CV_EXPORTS CvBlobTrackPostProc* cvCreateModuleBlobTrackPostProcTimeAverExp();


/* PREDICTORS */
/* blob PREDICTOR */
class CvBlobTrackPredictor: public CvVSModule
{
public:
	CvBlobTrackPredictor(){SetTypeName("BlobTrackPredictor");};
    virtual CvBlob* Predict() = 0;
    virtual void    Update(CvBlob* pBlob) = 0;
    virtual void    Release() = 0;
};
CV_EXPORTS CvBlobTrackPredictor* cvCreateModuleBlobTrackPredictKalman();



/* Trajectory analyser module: */
class CV_EXPORTS CvBlobTrackAnalysis: public CvVSModule
{
public:
	CvBlobTrackAnalysis(){SetTypeName("BlobTrackAnalysis");};
    virtual void    AddBlob(CvBlob* pBlob) = 0;
    virtual void    Process(IplImage* pImg, IplImage* pFG) = 0;
    virtual float   GetState(int BlobID) = 0;
    /* return 0 if trajectory is normal
       return >0 if trajectory abnormal */
    virtual const char*   GetStateDesc(int /*BlobID*/){return NULL;};
    virtual void    SetFileName(char* /*DataBaseName*/){};
    virtual void    Release() = 0;
};


inline void cvReleaseBlobTrackAnalysis(CvBlobTrackAnalysis** pBTPP)
{
    if(pBTPP == NULL) return;
    if(*pBTPP)(*pBTPP)->Release();
    *pBTPP = 0;
}

/* Feature-vector generation module: */
class CV_EXPORTS CvBlobTrackFVGen : public CvVSModule
{
public:
	CvBlobTrackFVGen(){SetTypeName("BlobTrackFVGen");};
    virtual void    AddBlob(CvBlob* pBlob) = 0;
    virtual void    Process(IplImage* pImg, IplImage* pFG) = 0;
    virtual void    Release() = 0;
    virtual int     GetFVSize() = 0;
    virtual int     GetFVNum() = 0;
    virtual float*  GetFV(int index, int* pFVID) = 0; /* Returns pointer to FV, if return 0 then FV not created */
    virtual float*  GetFVVar(){return NULL;}; /* Returns pointer to array of variation of values of FV, if returns 0 then FVVar does not exist. */
    virtual float*  GetFVMin() = 0; /* Returns pointer to array of minimal values of FV, if returns 0 then FVrange does not exist */
    virtual float*  GetFVMax() = 0; /* Returns pointer to array of maximal values of FV, if returns 0 then FVrange does not exist */
};


/* Trajectory Analyser module: */
class CV_EXPORTS CvBlobTrackAnalysisOne
{
public:
    virtual ~CvBlobTrackAnalysisOne() {};
    virtual int     Process(CvBlob* pBlob, IplImage* pImg, IplImage* pFG) = 0;
    /* return 0 if trajectory is normal
       return >0 if trajectory abnormal */
    virtual void    Release() = 0;
};

/* Create blob tracking post processing module based on simle module: */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateBlobTrackAnalysisList(CvBlobTrackAnalysisOne* (*create)());

/* Declarations of constructors of implemented modules: */

/* Based on histogram analysis of 2D FV (x,y): */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistP();

/* Based on histogram analysis of 4D FV (x,y,vx,vy): */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistPV();

/* Based on histogram analysis of 5D FV (x,y,vx,vy,state): */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistPVS();

/* Based on histogram analysis of 4D FV (startpos,stoppos): */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistSS();



/* Based on SVM classifier analysis of 2D FV (x,y): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMP();

/* Based on SVM classifier analysis of 4D FV (x,y,vx,vy): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMPV();

/* Based on SVM classifier analysis of 5D FV (x,y,vx,vy,state): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMPVS();

/* Based on SVM classifier analysis of 4D FV (startpos,stoppos): */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMSS();

/* Track analysis based on distance between tracks: */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisTrackDist();

/* Analyzer based on reation Road and height map: */
//CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysis3DRoadMap();

/* Analyzer that makes OR decision using set of analyzers: */
CV_EXPORTS CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisIOR();

/* Estimator of human height: */
class CV_EXPORTS CvBlobTrackAnalysisHeight: public CvBlobTrackAnalysis
{
public:
    virtual double  GetHeight(CvBlob* pB) = 0;
};
//CV_EXPORTS CvBlobTrackAnalysisHeight* cvCreateModuleBlobTrackAnalysisHeightScale();



/* AUTO BLOB TRACKER INTERFACE -- pipeline of 3 modules: */
class CV_EXPORTS CvBlobTrackerAuto: public CvVSModule
{
public:
	CvBlobTrackerAuto(){SetTypeName("BlobTrackerAuto");};
    virtual void        Process(IplImage* pImg, IplImage* pMask = NULL) = 0;
    virtual CvBlob*     GetBlob(int index) = 0;
    virtual CvBlob*     GetBlobByID(int ID) = 0;
    virtual int         GetBlobNum() = 0;
    virtual IplImage*   GetFGMask(){return NULL;};
    virtual float       GetState(int BlobID) = 0;
    virtual const char*       GetStateDesc(int BlobID) = 0;
    /* return 0 if trajectory is normal;
     * return >0 if trajectory abnormal. */
    virtual void    Release() = 0;
};
inline void cvReleaseBlobTrackerAuto(CvBlobTrackerAuto** ppT)
{
    ppT[0]->Release();
    ppT[0] = 0;
}
/* END AUTO BLOB TRACKER INTERFACE */


/* Constructor functions and data for specific BlobTRackerAuto modules: */

/* Parameters of blobtracker auto ver1: */
struct CvBlobTrackerAutoParam1
{
    int                     FGTrainFrames; /* Number of frames needed for FG (foreground) detector to train.        */

    CvFGDetector*           pFG;           /* FGDetector module. If this field is NULL the Process FG mask is used. */

    CvBlobDetector*         pBD;           /* Selected blob detector module. 					    */
                                           /* If this field is NULL default blobdetector module will be created.    */

    CvBlobTracker*          pBT;           /* Selected blob tracking module.					    */
                                           /* If this field is NULL default blobtracker module will be created.     */

    CvBlobTrackGen*         pBTGen;        /* Selected blob trajectory generator.				    */
                                           /* If this field is NULL no generator is used.                           */

    CvBlobTrackPostProc*    pBTPP;         /* Selected blob trajectory postprocessing module.			    */
                                           /* If this field is NULL no postprocessing is done.                      */

    int                     UsePPData;

    CvBlobTrackAnalysis*    pBTA;          /* Selected blob trajectory analysis module.                             */
                                           /* If this field is NULL no track analysis is done.                      */
};

/* Create blob tracker auto ver1: */
CV_EXPORTS CvBlobTrackerAuto* cvCreateBlobTrackerAuto1(CvBlobTrackerAutoParam1* param = NULL);

/* Simple loader for many auto trackers by its type : */
inline CvBlobTrackerAuto* cvCreateBlobTrackerAuto(int type, void* param)
{
    if(type == 0) return cvCreateBlobTrackerAuto1((CvBlobTrackerAutoParam1*)param);
    return 0;
}



struct CvTracksTimePos
{
    int len1,len2;
    int beg1,beg2;
    int end1,end2;
    int comLen; //common length for two tracks
    int shift1,shift2;
};

/*CV_EXPORTS int cvCompareTracks( CvBlobTrackSeq *groundTruth,
                   CvBlobTrackSeq *result,
                   FILE *file);*/


/* Constructor functions:  */

CV_EXPORTS void cvCreateTracks_One(CvBlobTrackSeq *TS);
CV_EXPORTS void cvCreateTracks_Same(CvBlobTrackSeq *TS1, CvBlobTrackSeq *TS2);
CV_EXPORTS void cvCreateTracks_AreaErr(CvBlobTrackSeq *TS1, CvBlobTrackSeq *TS2, int addW, int addH);


/* HIST API */
class CV_EXPORTS CvProb
{
public:
    virtual ~CvProb() {};

    /* Calculate probability value: */
    virtual double Value(int* /*comp*/, int /*x*/ = 0, int /*y*/ = 0){return -1;};

    /* Update histograpp Pnew = (1-W)*Pold + W*Padd*/
    /* W weight of new added prob */
    /* comps - matrix of new fetature vectors used to update prob */
    virtual void AddFeature(float W, int* comps, int x =0, int y = 0) = 0;
    virtual void Scale(float factor = 0, int x = -1, int y = -1) = 0;
    virtual void Release() = 0;
};
inline void cvReleaseProb(CvProb** ppProb){ppProb[0]->Release();ppProb[0]=NULL;}
/* HIST API */

/* Some Prob: */
CV_EXPORTS CvProb* cvCreateProbS(int dim, CvSize size, int sample_num);
CV_EXPORTS CvProb* cvCreateProbMG(int dim, CvSize size, int sample_num);
CV_EXPORTS CvProb* cvCreateProbMG2(int dim, CvSize size, int sample_num);
CV_EXPORTS CvProb* cvCreateProbHist(int dim, CvSize size);

#define CV_BT_HIST_TYPE_S     0
#define CV_BT_HIST_TYPE_MG    1
#define CV_BT_HIST_TYPE_MG2   2
#define CV_BT_HIST_TYPE_H     3
inline CvProb* cvCreateProb(int type, int dim, CvSize size = cvSize(1,1), void* /*param*/ = NULL)
{
    if(type == CV_BT_HIST_TYPE_S) return cvCreateProbS(dim, size, -1);
    if(type == CV_BT_HIST_TYPE_MG) return cvCreateProbMG(dim, size, -1);
    if(type == CV_BT_HIST_TYPE_MG2) return cvCreateProbMG2(dim, size, -1);
    if(type == CV_BT_HIST_TYPE_H) return cvCreateProbHist(dim, size);
    return NULL;
}



/* Noise type definitions: */
#define CV_NOISE_NONE               0
#define CV_NOISE_GAUSSIAN           1
#define CV_NOISE_UNIFORM            2
#define CV_NOISE_SPECKLE            3
#define CV_NOISE_SALT_AND_PEPPER    4

/* Add some noise to image: */
/* pImg - (input) image without noise */
/* pImg - (output) image with noise */
/* noise_type - type of added noise */
/*  CV_NOISE_GAUSSIAN - pImg += n , n - is gaussian noise with Ampl standart deviation */
/*  CV_NOISE_UNIFORM - pImg += n , n - is uniform noise with Ampl standart deviation */
/*  CV_NOISE_SPECKLE - pImg += n*pImg , n - is gaussian noise with Ampl standart deviation */
/*  CV_NOISE_SALT_AND_PAPPER - pImg = pImg with blacked and whited pixels,
            Ampl is density of brocken pixels (0-there are not broken pixels, 1 - all pixels are broken)*/
/* Ampl - "amplitude" of noise */
//CV_EXPORTS void cvAddNoise(IplImage* pImg, int noise_type, double Ampl, CvRNG* rnd_state = NULL);

/*================== GENERATOR OF TEST VIDEO SEQUENCE ===================== */
typedef void CvTestSeq;

/* pConfigfile - Name of file (yml or xml) with description of test sequence */
/* videos - array of names of test videos described in "pConfigfile" file */
/* numvideos - size of "videos" array */
CV_EXPORTS CvTestSeq* cvCreateTestSeq(char* pConfigfile, char** videos, int numvideo, float Scale = 1, int noise_type = CV_NOISE_NONE, double noise_ampl = 0);
CV_EXPORTS void cvReleaseTestSeq(CvTestSeq** ppTestSeq);

/* Generate next frame from test video seq and return pointer to it: */
CV_EXPORTS IplImage* cvTestSeqQueryFrame(CvTestSeq* pTestSeq);

/* Return pointer to current foreground mask: */
CV_EXPORTS IplImage* cvTestSeqGetFGMask(CvTestSeq* pTestSeq);

/* Return pointer to current image: */
CV_EXPORTS IplImage* cvTestSeqGetImage(CvTestSeq* pTestSeq);

/* Return frame size of result test video: */
CV_EXPORTS CvSize cvTestSeqGetImageSize(CvTestSeq* pTestSeq);

/* Return number of frames result test video: */
CV_EXPORTS int cvTestSeqFrameNum(CvTestSeq* pTestSeq);

/* Return number of existing objects.
 * This is general number of any objects.
 * For example number of trajectories may be equal or less than returned value:
 */
CV_EXPORTS int cvTestSeqGetObjectNum(CvTestSeq* pTestSeq);

/* Return 0 if there is not position for current defined on current frame */
/* Return 1 if there is object position and pPos was filled */
CV_EXPORTS int cvTestSeqGetObjectPos(CvTestSeq* pTestSeq, int ObjIndex, CvPoint2D32f* pPos);
CV_EXPORTS int cvTestSeqGetObjectSize(CvTestSeq* pTestSeq, int ObjIndex, CvPoint2D32f* pSize);

/* Add noise to final image: */
CV_EXPORTS void cvTestSeqAddNoise(CvTestSeq* pTestSeq, int noise_type = CV_NOISE_NONE, double noise_ampl = 0);

/* Add Intensity variation: */
CV_EXPORTS void cvTestSeqAddIntensityVariation(CvTestSeq* pTestSeq, float DI_per_frame, float MinI, float MaxI);
CV_EXPORTS void cvTestSeqSetFrame(CvTestSeq* pTestSeq, int n);

#endif

/* End of file. */
