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

#define MAX_FV_SIZE 5
#define BLOB_NUM    5

typedef struct DefBlobFVN
{
    CvBlob  blob;
    CvBlob  BlobSeq[BLOB_NUM];
    int     state;
    int     LastFrame;
    int     FrameNum;
} DefBlobFVN;

class CvBlobTrackFVGenN: public CvBlobTrackFVGen
{
private:
    CvBlobSeq       m_BlobList;
    CvMemStorage*   m_pMem;
    CvSeq*          m_pFVSeq;
    float           m_FVMax[MAX_FV_SIZE];
    float           m_FVMin[MAX_FV_SIZE];
    float           m_FVVar[MAX_FV_SIZE];
    int             m_Dim;
    int             m_Frame;
    int             m_State;
    int             m_ClearFlag;
    void Clear()
    {
        if(m_pMem)
        {
            cvClearMemStorage(m_pMem);
            m_pFVSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(float)*(m_Dim+1), m_pMem);
            m_ClearFlag = 1;
        }
    }
public:
    CvBlobTrackFVGenN(int dim = 2 ):m_BlobList(sizeof(DefBlobFVN))
    {
        int i;
        assert(dim <= MAX_FV_SIZE);
        m_Dim = dim;
        for(i=0; i<m_Dim; ++i)
        {
            m_FVVar[i] = 0.01f;
            m_FVMax[i] = 1;
            m_FVMin[i] = 0;
        }
        m_Frame = 0;
        m_State = 0;
        m_pMem = cvCreateMemStorage();
        m_pFVSeq = NULL;
        Clear();

        switch(dim) {
        case 2: SetModuleName("P"); break;
        case 4: SetModuleName("PV"); break;
        case 5: SetModuleName("PVS"); break;
        }
    };

    ~CvBlobTrackFVGenN()
    {
        if(m_pMem)cvReleaseMemStorage(&m_pMem);
    };

    void AddBlob(CvBlob* pBlob)
    {
        float       FV[MAX_FV_SIZE+1];
        DefBlobFVN* pFVBlob = (DefBlobFVN*)m_BlobList.GetBlobByID(CV_BLOB_ID(pBlob));

        if(!m_ClearFlag) Clear();

        if(pFVBlob==NULL)
        {
            DefBlobFVN BlobNew;
            BlobNew.blob = pBlob[0];
            BlobNew.LastFrame = m_Frame;
            BlobNew.state = 0;;
            BlobNew.FrameNum = 0;
            m_BlobList.AddBlob((CvBlob*)&BlobNew);
            pFVBlob = (DefBlobFVN*)m_BlobList.GetBlobByID(CV_BLOB_ID(pBlob));
        } /* Add new record if necessary. */

        pFVBlob->blob = pBlob[0];

        /* Shift: */
        for(int i=(BLOB_NUM-1); i>0; --i)
        {
            pFVBlob->BlobSeq[i] = pFVBlob->BlobSeq[i-1];
        }

        pFVBlob->BlobSeq[0] = pBlob[0];

        if(m_Dim>0)
        {   /* Calculate FV position: */
            FV[0] = CV_BLOB_X(pBlob);
            FV[1] = CV_BLOB_Y(pBlob);
        }

        if(m_Dim<=2)
        {   /* Add new FV if position is enough: */
            *(int*)(FV+m_Dim) = CV_BLOB_ID(pBlob);
            cvSeqPush( m_pFVSeq, FV );
        }
        else if(pFVBlob->FrameNum > BLOB_NUM)
        {   /* Calculate velocity for more complex FV: */
            float       AverVx = 0;
            float       AverVy = 0;
            {   /* Average velocity: */
                CvBlob* pBlobSeq = pFVBlob->BlobSeq;
                for(int i=1;i<BLOB_NUM;++i)
                {
                    AverVx += CV_BLOB_X(pBlobSeq+i-1)-CV_BLOB_X(pBlobSeq+i);
                    AverVy += CV_BLOB_Y(pBlobSeq+i-1)-CV_BLOB_Y(pBlobSeq+i);
                }
                AverVx /= BLOB_NUM-1;
                AverVy /= BLOB_NUM-1;

                FV[2] = AverVx;
                FV[3] = AverVy;
            }

            if(m_Dim>4)
            {   /* State duration: */
                float T = (CV_BLOB_WX(pBlob)+CV_BLOB_WY(pBlob))*0.01f;

                if( fabs(AverVx) < T && fabs(AverVy) < T)
                    pFVBlob->state++;
                else
                    pFVBlob->state=0;
                FV[4] = (float)pFVBlob->state;
            } /* State duration. */

            /* Add new FV: */
            *(int*)(FV+m_Dim) = CV_BLOB_ID(pBlob);
            cvSeqPush( m_pFVSeq, FV );

        } /* If velocity is calculated. */

        pFVBlob->FrameNum++;
        pFVBlob->LastFrame = m_Frame;
    };  /* AddBlob */

    void Process(IplImage* pImg, IplImage* /*pFG*/)
    {
        int i;
        if(!m_ClearFlag) Clear();
        for(i=m_BlobList.GetBlobNum(); i>0; --i)
        {   /* Delete unused blob: */
            DefBlobFVN* pFVBlob = (DefBlobFVN*)m_BlobList.GetBlob(i-1);
            if(pFVBlob->LastFrame < m_Frame)
            {
                m_BlobList.DelBlob(i-1);
            }
        } /* Check next blob in list. */

        m_FVMin[0] = 0;
        m_FVMin[1] = 0;
        m_FVMax[0] = (float)(pImg->width-1);
        m_FVMax[1] = (float)(pImg->height-1);
        m_FVVar[0] = m_FVMax[0]*0.01f;
        m_FVVar[1] = m_FVMax[1]*0.01f;
        m_FVVar[2] = (float)(pImg->width-1)/1440.0f;
        m_FVMax[2] = (float)(pImg->width-1)*0.02f;
        m_FVMin[2] = -m_FVMax[2];
        m_FVVar[3] = (float)(pImg->width-1)/1440.0f;
        m_FVMax[3] = (float)(pImg->height-1)*0.02f;
        m_FVMin[3] = -m_FVMax[3];
        m_FVMax[4] = 25*32.0f; /* max state is 32 sec */
        m_FVMin[4] = 0;
        m_FVVar[4] = 10;

        m_Frame++;
        m_ClearFlag = 0;
    };
    virtual void    Release(){delete this;};
    virtual int     GetFVSize(){return m_Dim;};
    virtual int     GetFVNum()
    {
        return m_pFVSeq->total;
    };

    virtual float*  GetFV(int index, int* pFVID)
    {
        float* pFV = (float*)cvGetSeqElem( m_pFVSeq, index );
        if(pFVID)pFVID[0] = *(int*)(pFV+m_Dim);
        return pFV;
    };
    virtual float*  GetFVMin(){return m_FVMin;}; /* returned pointer to array of minimal values of FV, if return 0 then FVrange is not exist */
    virtual float*  GetFVMax(){return m_FVMax;}; /* returned pointer to array of maximal values of FV, if return 0 then FVrange is not exist */
    virtual float*  GetFVVar(){return m_FVVar;}; /* returned pointer to array of maximal values of FV, if return 0 then FVrange is not exist */
};/* CvBlobTrackFVGenN */

inline CvBlobTrackFVGen* cvCreateFVGenP(){return (CvBlobTrackFVGen*)new CvBlobTrackFVGenN(2);}
inline CvBlobTrackFVGen* cvCreateFVGenPV(){return (CvBlobTrackFVGen*)new CvBlobTrackFVGenN(4);}
inline CvBlobTrackFVGen* cvCreateFVGenPVS(){return (CvBlobTrackFVGen*)new CvBlobTrackFVGenN(5);}
#undef MAX_FV_SIZE

#define MAX_FV_SIZE 4
class CvBlobTrackFVGenSS: public CvBlobTrackFVGen
{
private:
    CvBlobSeq       m_BlobList;
    CvMemStorage*   m_pMem;
    CvSeq*          m_pFVSeq;
    float           m_FVMax[MAX_FV_SIZE];
    float           m_FVMin[MAX_FV_SIZE];
    float           m_FVVar[MAX_FV_SIZE];
    int             m_Dim;
    int             m_Frame;
    int             m_State;
    int             m_ClearFlag;
    void Clear()
    {
        cvClearMemStorage(m_pMem);
        m_pFVSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(float)*(m_Dim+1), m_pMem);
        m_ClearFlag = 1;
    }
public:
    CvBlobTrackFVGenSS(int dim = 2 ):m_BlobList(sizeof(DefBlobFVN))
    {
        int i;
        assert(dim <= MAX_FV_SIZE);
        m_Dim = dim;
        for(i=0;i<m_Dim;++i)
        {
            m_FVVar[i] = 0.01f;
            m_FVMax[i] = 1;
            m_FVMin[i] = 0;
        }
        m_Frame = 0;
        m_State = 0;
        m_pMem = cvCreateMemStorage();
        m_pFVSeq = NULL;

        SetModuleName("SS");
    };
    ~CvBlobTrackFVGenSS()
    {
        if(m_pMem)cvReleaseMemStorage(&m_pMem);
    };

    void AddBlob(CvBlob* pBlob)
    {
        //float       FV[MAX_FV_SIZE+1];
        DefBlobFVN* pFVBlob = (DefBlobFVN*)m_BlobList.GetBlobByID(CV_BLOB_ID(pBlob));

        if(!m_ClearFlag) Clear();

        if(pFVBlob==NULL)
        {
            DefBlobFVN BlobNew;
            BlobNew.blob = pBlob[0];
            BlobNew.LastFrame = m_Frame;
            BlobNew.state = 0;;
            BlobNew.FrameNum = 0;
            m_BlobList.AddBlob((CvBlob*)&BlobNew);
            pFVBlob = (DefBlobFVN*)m_BlobList.GetBlobByID(CV_BLOB_ID(pBlob));
        } /* Add new record if necessary. */

        /* Shift: */
        for(int i=(BLOB_NUM-1); i>0; --i)
        {
            pFVBlob->BlobSeq[i] = pFVBlob->BlobSeq[i-1];
        }

        pFVBlob->BlobSeq[0] = pBlob[0];

        if(pFVBlob->FrameNum > BLOB_NUM)
        {   /* Average velocity: */
            CvBlob* pBlobSeq = pFVBlob->BlobSeq;
            float   T = (CV_BLOB_WX(pBlob)+CV_BLOB_WY(pBlob))*0.01f;
            float   AverVx = 0;
            float   AverVy = 0;
            for(int i=1; i<BLOB_NUM; ++i)
            {
                AverVx += CV_BLOB_X(pBlobSeq+i-1)-CV_BLOB_X(pBlobSeq+i);
                AverVy += CV_BLOB_Y(pBlobSeq+i-1)-CV_BLOB_Y(pBlobSeq+i);
            }
            AverVx /= BLOB_NUM-1;
            AverVy /= BLOB_NUM-1;

            if( fabs(AverVx) < T && fabs(AverVy) < T)
                pFVBlob->state++;
            else
                pFVBlob->state=0;
        }

        if(pFVBlob->state == 5)
        {   /* Object is stopped:  */
            float   FV[MAX_FV_SIZE];
            FV[0] = pFVBlob->blob.x;
            FV[1] = pFVBlob->blob.y;
            FV[2] = pFVBlob->BlobSeq[0].x;
            FV[3] = pFVBlob->BlobSeq[0].y;
            *(int*)(FV+m_Dim) = CV_BLOB_ID(pBlob);
            cvSeqPush( m_pFVSeq, FV );
        } /* Object is stopped. */

        pFVBlob->FrameNum++;
        pFVBlob->LastFrame = m_Frame;
    };  /* AddBlob */
    void Process(IplImage* pImg, IplImage* /*pFG*/)
    {
        int i;

        if(!m_ClearFlag) Clear();

        for(i=m_BlobList.GetBlobNum();i>0;--i)
        {   /* Delete unused blob: */
            DefBlobFVN* pFVBlob = (DefBlobFVN*)m_BlobList.GetBlob(i-1);
            if(pFVBlob->LastFrame < m_Frame)
            {
                float   FV[MAX_FV_SIZE+1];
                FV[0] = pFVBlob->blob.x;
                FV[1] = pFVBlob->blob.y;
                FV[2] = pFVBlob->BlobSeq[0].x;
                FV[3] = pFVBlob->BlobSeq[0].y;
                *(int*)(FV+m_Dim) = CV_BLOB_ID(pFVBlob);
                cvSeqPush( m_pFVSeq, FV );
                m_BlobList.DelBlob(i-1);
            }
        } /* Check next blob in list. */

        /* Set max min range: */
        m_FVMin[0] = 0;
        m_FVMin[1] = 0;
        m_FVMin[2] = 0;
        m_FVMin[3] = 0;
        m_FVMax[0] = (float)(pImg->width-1);
        m_FVMax[1] = (float)(pImg->height-1);
        m_FVMax[2] = (float)(pImg->width-1);
        m_FVMax[3] = (float)(pImg->height-1);
        m_FVVar[0] = m_FVMax[0]*0.01f;
        m_FVVar[1] = m_FVMax[1]*0.01f;
        m_FVVar[2] = m_FVMax[2]*0.01f;
        m_FVVar[3] = m_FVMax[3]*0.01f;

        m_Frame++;
        m_ClearFlag = 0;
    };
    virtual void    Release(){delete this;};
    virtual int     GetFVSize(){return m_Dim;};
    virtual int     GetFVNum()
    {
        return m_pFVSeq->total;
    };

    virtual float*  GetFV(int index, int* pFVID)
    {
        float* pFV = (float*)cvGetSeqElem( m_pFVSeq, index );
        if(pFVID)pFVID[0] = *(int*)(pFV+m_Dim);
        return pFV;
    };

    virtual float*  GetFVMin(){return m_FVMin;}; /* returned pointer to array of minimal values of FV, if return 0 then FVrange is not exist */
    virtual float*  GetFVMax(){return m_FVMax;}; /* returned pointer to array of maximal values of FV, if return 0 then FVrange is not exist */
    virtual float*  GetFVVar(){return m_FVVar;}; /* returned pointer to array of maximal values of FV, if return 0 then FVrange is not exist */
};/* CvBlobTrackFVGenSS */

inline CvBlobTrackFVGen* cvCreateFVGenSS(){return (CvBlobTrackFVGen*)new CvBlobTrackFVGenSS;}

/*======================= TRAJECTORY ANALYZER MODULES =====================*/
/* Trajectory Analyser module */
#define SPARSE  0
#define ND      1
#define BYSIZE  -1
class DefMat
{
private:
    CvSparseMatIterator m_SparseIterator;
    CvSparseNode*       m_pSparseNode;
    int*                m_IDXs;
    int                 m_Dim;

public:
    CvSparseMat*        m_pSparse;
    CvMatND*            m_pND;
    int                 m_Volume;
    int                 m_Max;
    DefMat(int dim = 0, int* sizes = NULL, int type = SPARSE)
    {
        /* Create sparse or ND matrix but not both: */
        m_pSparseNode = NULL;
        m_pSparse = NULL;
        m_pND = NULL;
        m_Volume = 0;
        m_Max = 0;
        m_IDXs = NULL;
        m_Dim = 0;
        if(dim>0 && sizes != 0)
            Realloc(dim, sizes, type);
    }
    ~DefMat()
    {
        if(m_pSparse)cvReleaseSparseMat(&m_pSparse);
        if(m_pND)cvReleaseMatND(&m_pND);
        if(m_IDXs) cvFree(&m_IDXs);
    }

    void Realloc(int dim, int* sizes, int type = SPARSE)
    {
        if(m_pSparse)cvReleaseSparseMat(&m_pSparse);
        if(m_pND)cvReleaseMatND(&m_pND);

        if(type == BYSIZE )
        {
            int size = 0;
            int i;
            for(size=1,i=0;i<dim;++i)
            {
                size *= sizes[i];
            }
            size *= sizeof(int);
            if(size > (2<<20))
            { /* if size > 1M */
                type = SPARSE;
            }
            else
            {
                type = ND;
            }
        } /* Define matrix type. */

        if(type == SPARSE)
        {
            m_pSparse = cvCreateSparseMat( dim, sizes, CV_32SC1 );
            m_Dim = dim;
        }
        if(type == ND )
        {
            m_pND = cvCreateMatND( dim, sizes, CV_32SC1 );
            cvZero(m_pND);
            m_IDXs = (int*)cvAlloc(sizeof(int)*dim);
            m_Dim = dim;
        }
        m_Volume = 0;
        m_Max = 0;
    }
    void Save(const char* File)
    {
        if(m_pSparse)cvSave(File, m_pSparse );
        if(m_pND)cvSave(File, m_pND );
    }
    void Save(CvFileStorage* fs, const char* name)
    {
        if(m_pSparse)
        {
            cvWrite(fs, name, m_pSparse );
        }
        else if(m_pND)
        {
            cvWrite(fs, name, m_pND );
        }
    }
    void Load(const char* File)
    {
        CvFileStorage* fs = cvOpenFileStorage( File, NULL, CV_STORAGE_READ );
        if(fs)
        {
            void* ptr;
            if(m_pSparse) cvReleaseSparseMat(&m_pSparse);
            if(m_pND) cvReleaseMatND(&m_pND);
            m_Volume = 0;
            m_Max = 0;
            ptr = cvLoad(File);
            if(ptr && CV_IS_MATND_HDR(ptr)) m_pND = (CvMatND*)ptr;
            if(ptr && CV_IS_SPARSE_MAT_HDR(ptr)) m_pSparse = (CvSparseMat*)ptr;
            cvReleaseFileStorage(&fs);
        }
        AfterLoad();
    } /* Load. */

    void Load(CvFileStorage* fs, CvFileNode* node, const char* name)
    {
        CvFileNode* n = cvGetFileNodeByName(fs,node,name);
        void* ptr = n?cvRead(fs,n):NULL;
        if(ptr)
        {
            if(m_pSparse) cvReleaseSparseMat(&m_pSparse);
            if(m_pND) cvReleaseMatND(&m_pND);
            m_Volume = 0;
            m_Max = 0;
            if(CV_IS_MATND_HDR(ptr)) m_pND = (CvMatND*)ptr;
            if(CV_IS_SPARSE_MAT_HDR(ptr)) m_pSparse = (CvSparseMat*)ptr;
        }
        else
        {
            printf("WARNING!!! Can't load %s matrix\n",name);
        }
        AfterLoad();
    } /* Load. */

    void AfterLoad()
    {
        m_Volume = 0;
        m_Max = 0;
        if(m_pSparse)
        {   /* Calculate Volume of loaded hist: */
            CvSparseMatIterator mat_iterator;
            CvSparseNode* node = cvInitSparseMatIterator( m_pSparse, &mat_iterator );

            for( ; node != 0; node = cvGetNextSparseNode( &mat_iterator ))
            {
                int val = *(int*)CV_NODE_VAL( m_pSparse, node ); /* get value of the element
                                                                (assume that the type is CV_32SC1) */
                m_Volume += val;
                if(m_Max < val)m_Max = val;
            }
        } /* Calculate Volume of loaded hist. */

        if(m_pND)
        {   /* Calculate Volume of loaded hist: */
            CvMat   mat;
            double  max_val;
            double  vol;
            cvGetMat( m_pND, &mat, NULL, 1 );

            vol = cvSum(&mat).val[0];
            m_Volume = cvRound(vol);
            cvMinMaxLoc( &mat, NULL, &max_val);
            m_Max = cvRound(max_val);
            /* MUST BE WRITTEN LATER */
        } /* Calculate Volume of loaded hist. */
    } /* AfterLoad. */

    int* GetPtr(int* indx)
    {
        if(m_pSparse) return (int*)cvPtrND( m_pSparse, indx, NULL, 1, NULL);
        if(m_pND) return  (int*)cvPtrND( m_pND, indx, NULL, 1, NULL);
        return NULL;
    } /* GetPtr. */

    int GetVal(int* indx)
    {
        int* p = GetPtr(indx);
        if(p)return p[0];
        return -1;
    } /* GetVal. */

    int Add(int* indx, int val)
    {
        int  NewVal;
        int* pVal = GetPtr(indx);
        if(pVal == NULL) return -1;
        pVal[0] += val;
        NewVal = pVal[0];
        m_Volume += val;
        if(m_Max < NewVal)m_Max = NewVal;
        return NewVal;
    } /* Add. */

    void Add(DefMat* pMatAdd)
    {
        int*    pIDXS = NULL;
        int     Val = 0;
        for(Val = pMatAdd->GetNext(&pIDXS, 1 );pIDXS;Val=pMatAdd->GetNext(&pIDXS, 0 ))
        {
            Add(pIDXS,Val);
        }
    } /* Add. */

    int SetMax(int* indx, int val)
    {
        int  NewVal;
        int* pVal = GetPtr(indx);
        if(pVal == NULL) return -1;
        if(val > pVal[0])
        {
            m_Volume += val-pVal[0];
            pVal[0] = val;
        }
        NewVal = pVal[0];
        if(m_Max < NewVal)m_Max = NewVal;
        return NewVal;
    } /* Add. */

    int GetNext(int** pIDXS, int init = 0)
    {
        int     Val = 0;
        pIDXS[0] = NULL;
        if(m_pSparse)
        {
            m_pSparseNode = (init || m_pSparseNode==NULL)?
                    cvInitSparseMatIterator( m_pSparse, &m_SparseIterator ):
                        cvGetNextSparseNode( &m_SparseIterator );

                    if(m_pSparseNode)
                    {
                        int* pVal = (int*)CV_NODE_VAL( m_pSparse, m_pSparseNode );
                        if(pVal)Val = pVal[0];
                        pIDXS[0] = CV_NODE_IDX( m_pSparse, m_pSparseNode );
                    }
        }/* Sparse matrix. */

        if(m_pND)
        {
            int i;
            if(init)
            {
                for(i=0;i<m_Dim;++i)
                {
                    m_IDXs[i] = cvGetDimSize( m_pND, i )-1;
                }
                pIDXS[0] = m_IDXs;
                Val = GetVal(m_IDXs);
            }
            else
            {
                for(i=0;i<m_Dim;++i)
                {
                    if((m_IDXs[i]--)>0)
                        break;
                    m_IDXs[i] = cvGetDimSize( m_pND, i )-1;
                }
                if(i==m_Dim)
                {
                    pIDXS[0] = NULL;
                }
                else
                {
                    pIDXS[0] = m_IDXs;
                    Val = GetVal(m_IDXs);
                }

            } /* Get next ND. */

        } /* Sparse matrix. */

        return Val;

    }; /* GetNext. */
};

#define FV_NUM 10
#define FV_SIZE 10
typedef struct DefTrackFG
{
    CvBlob                  blob;
    //    CvBlobTrackFVGen*       pFVGen;
    int                     LastFrame;
    float                   state;
    DefMat*                 pHist;
} DefTrackFG;
class CvBlobTrackAnalysisHist : public CvBlobTrackAnalysis
{
    /*---------------- Internal functions: --------------------*/
private:
    int                 m_BinNumParam;
    int                 m_SmoothRadius;
    const char*         m_SmoothKernel;
    float               m_AbnormalThreshold;
    int                 m_TrackNum;
    int                 m_Frame;
    int                 m_BinNum;
    char                m_DataFileName[1024];
    int                 m_Dim;
    int*                m_Sizes;
    DefMat              m_HistMat;
    int                 m_HistVolumeSaved;
    int*                m_pFVi;
    int*                m_pFViVar;
    int*                m_pFViVarRes;
    CvBlobSeq           m_TrackFGList;
    //CvBlobTrackFVGen*   (*m_CreateFVGen)();
    CvBlobTrackFVGen*   m_pFVGen;
    void SaveHist()
    {
        if(m_DataFileName[0])
        {
            m_HistMat.Save(m_DataFileName);
            m_HistVolumeSaved = m_HistMat.m_Volume;
        }
    };
    void LoadHist()
    {
        if(m_DataFileName[0])m_HistMat.Load(m_DataFileName);
        m_HistVolumeSaved = m_HistMat.m_Volume;
    }
    void AllocData()
    {   /* AllocData: */
        m_pFVi = (int*)cvAlloc(sizeof(int)*m_Dim);
        m_pFViVar = (int*)cvAlloc(sizeof(int)*m_Dim);
        m_pFViVarRes = (int*)cvAlloc(sizeof(int)*m_Dim);
        m_Sizes = (int*)cvAlloc(sizeof(int)*m_Dim);

        {   /* Create init sparse matrix: */
            int     i;
            for(i=0;i<m_Dim;++i)m_Sizes[i] = m_BinNum;
            m_HistMat.Realloc(m_Dim,m_Sizes,SPARSE);
            m_HistVolumeSaved = 0;
        } /* Create init sparse matrix. */
    } /* AllocData. */

    void FreeData()
    {   /* FreeData. */
        int i;
        for(i=m_TrackFGList.GetBlobNum();i>0;--i)
        {
            //DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlob(i-1);
            //            pF->pFVGen->Release();
            m_TrackFGList.DelBlob(i-1);
        }
        cvFree(&m_pFVi);
        cvFree(&m_pFViVar);
        cvFree(&m_pFViVarRes);
        cvFree(&m_Sizes);
    } /* FreeData. */

    virtual void ParamUpdate()
    {
        if(m_BinNum != m_BinNumParam)
        {
            FreeData();
            m_BinNum = m_BinNumParam;
            AllocData();
        }
    }
public:
    CvBlobTrackAnalysisHist(CvBlobTrackFVGen*   (*createFVGen)()):m_TrackFGList(sizeof(DefTrackFG))
    {
        m_pFVGen = createFVGen();
        m_Dim = m_pFVGen->GetFVSize();
        m_Frame = 0;
        m_pFVi = 0;
        m_TrackNum = 0;
        m_BinNum = 32;
        m_DataFileName[0] = 0;

        m_AbnormalThreshold = 0.02f;
        AddParam("AbnormalThreshold",&m_AbnormalThreshold);
        CommentParam("AbnormalThreshold","If trajectory histogram value is lesst then <AbnormalThreshold*DataBaseTrackNum> then trajectory is abnormal");

        m_SmoothRadius = 1;
        AddParam("SmoothRadius",&m_SmoothRadius);
        CommentParam("AbnormalThreshold","Radius (in bins) for histogram smoothing");

        m_SmoothKernel = "L";
        AddParam("SmoothKernel",&m_SmoothKernel);
        CommentParam("SmoothKernel","L - Linear, G - Gaussian");


        m_BinNumParam = m_BinNum;
        AddParam("BinNum",&m_BinNumParam);
        CommentParam("BinNum","Number of bin for each dimention of feature vector");

        AllocData();
        SetModuleName("Hist");

    } /* Constructor. */

    ~CvBlobTrackAnalysisHist()
    {
        SaveHist();
        FreeData();
        m_pFVGen->Release();
    } /* Destructor. */

    /*----------------- Interface: --------------------*/
    virtual void    AddBlob(CvBlob* pBlob)
    {
        DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlobByID(CV_BLOB_ID(pBlob));
        if(pF == NULL)
        { /* create new filter */
            DefTrackFG F;
            F.state = 0;
            F.blob = pBlob[0];
            F.LastFrame = m_Frame;
            //            F.pFVGen = m_CreateFVGen();
            F.pHist = new DefMat(m_Dim,m_Sizes,SPARSE);
            m_TrackFGList.AddBlob((CvBlob*)&F);
            pF = (DefTrackFG*)m_TrackFGList.GetBlobByID(CV_BLOB_ID(pBlob));
        }

        assert(pF);
        pF->blob = pBlob[0];
        pF->LastFrame = m_Frame;
        m_pFVGen->AddBlob(pBlob);
    };
    virtual void Process(IplImage* pImg, IplImage* pFG)
    {
        m_pFVGen->Process(pImg, pFG);
        int SK = m_SmoothKernel[0];

        for(int i=0; i<m_pFVGen->GetFVNum(); ++i)
        {
            int         BlobID = 0;
            float*      pFV = m_pFVGen->GetFV(i,&BlobID);
            float*      pFVMax = m_pFVGen->GetFVMax();
            float*      pFVMin = m_pFVGen->GetFVMin();
            DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlobByID(BlobID);
            int         HistVal = 1;

            if(pFV==NULL) break;

            pF->LastFrame = m_Frame;

            {   /* Binarize FV: */
                int         j;
            for(j=0; j<m_Dim; ++j)
            {
                int     index;
                float   f0 = pFVMin?pFVMin[j]:0;
                float   f1 = pFVMax?pFVMax[j]:1;
                assert(f1>f0);
                index = cvRound((m_BinNum-1)*(pFV[j]-f0)/(f1-f0));
                if(index<0)index=0;
                if(index>=m_BinNum)index=m_BinNum-1;
                m_pFVi[j] = index;
            }
            }

            HistVal = m_HistMat.GetVal(m_pFVi);/* get bin value*/
            pF->state = 0;
            {   /* Calculate state: */
                float   T = m_HistMat.m_Max*m_AbnormalThreshold; /* calc threshold */

                if(m_TrackNum>0) T = 256.0f * m_TrackNum*m_AbnormalThreshold;
                if(T>0)
                {
                    pF->state = (T - HistVal)/(T*0.2f) + 0.5f;
                }
                if(pF->state<0)pF->state=0;
                if(pF->state>1)pF->state=1;
            }

            {   /* If it is a new FV then add it to trajectory histogram: */
                int flag = 1;
                int r = m_SmoothRadius;

                //                    printf("BLob %3d NEW FV [", CV_BLOB_ID(pF));
                //                    for(i=0;i<m_Dim;++i) printf("%d,", m_pFVi[i]);
                //                    printf("]");

                for(int k=0; k<m_Dim; ++k)
                {
                    m_pFViVar[k]=-r;
                }

                while(flag)
                {
                    float   dist = 0;
                    int     HistAdd = 0;
                    int     good = 1;
                    for(int k=0; k<m_Dim; ++k)
                    {
                        m_pFViVarRes[k] = m_pFVi[k]+m_pFViVar[k];
                        if(m_pFViVarRes[k]<0) good= 0;
                        if(m_pFViVarRes[k]>=m_BinNum) good= 0;
                        dist += m_pFViVar[k]*m_pFViVar[k];
                    }/* Calculate next dimension. */

                    if(SK=='G' || SK=='g')
                    {
                        double dist2 = dist/(r*r);
                        HistAdd = cvRound(256*exp(-dist2)); /* Hist Add for (dist=1) = 25.6*/
                    }
                    else if(SK=='L' || SK=='l')
                    {
                        dist = (float)(sqrt(dist)/(r+1));
                        HistAdd = cvRound(256*(1-dist));
                    }
                    else
                    {
                        HistAdd = 255; /* Flat smoothing. */
                    }

                    if(good && HistAdd>0)
                    {   /* Update histogram: */
                        assert(pF->pHist);
                        pF->pHist->SetMax(m_pFViVarRes, HistAdd);
                    }   /* Update histogram. */

                    int idx = 0;
                    for( ; idx<m_Dim; ++idx)
                    {   /* Next config: */
                        if((m_pFViVar[idx]++) < r)
                            break;
                        m_pFViVar[idx] = -r;
                    }   /* Increase next dimension variable. */
                    if(idx==m_Dim)break;
                }   /* Next variation. */
            } /* If new FV. */
        } /* Next FV. */

        {   /* Check all blobs on list: */
            int i;
            for(i=m_TrackFGList.GetBlobNum(); i>0; --i)
            {   /* Add histogram and delete blob from list: */
                DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlob(i-1);
                if(pF->LastFrame+3 < m_Frame && pF->pHist)
                {
                    m_HistMat.Add(pF->pHist);
                    delete pF->pHist;
                    m_TrackNum++;
                    m_TrackFGList.DelBlob(i-1);
                }
            }/* next blob */
        }

        m_Frame++;

        if(m_Wnd)
        {   /* Debug output: */
            int*        idxs = NULL;
            int         Val = 0;
            IplImage*   pI = cvCloneImage(pImg);

            cvZero(pI);

            for(Val = m_HistMat.GetNext(&idxs,1); idxs; Val=m_HistMat.GetNext(&idxs,0))
            {   /* Draw all elements: */
                if(!idxs) break;
                if(Val == 0) continue;

                float vf = (float)Val/(m_HistMat.m_Max?m_HistMat.m_Max:1);
                int x = cvRound((float)(pI->width-1)*(float)idxs[0] / (float)m_BinNum);
                int y = cvRound((float)(pI->height-1)*(float)idxs[1] / (float)m_BinNum);

                cvCircle(pI, cvPoint(x,y), cvRound(vf*pI->height/(m_BinNum*2)),CV_RGB(255,0,0),CV_FILLED);
                if(m_Dim > 3)
                {
                    int dx = -2*(idxs[2]-m_BinNum/2);
                    int dy = -2*(idxs[3]-m_BinNum/2);
                    cvLine(pI,cvPoint(x,y),cvPoint(x+dx,y+dy),CV_RGB(0,cvRound(vf*255),1));
                }
                if( m_Dim==4 &&
                        m_pFVGen->GetFVMax()[0]==m_pFVGen->GetFVMax()[2] &&
                        m_pFVGen->GetFVMax()[1]==m_pFVGen->GetFVMax()[3])
                {
                    int x1 = cvRound((float)(pI->width-1)*(float)idxs[2] / (float)m_BinNum);
                    int y1 = cvRound((float)(pI->height-1)*(float)idxs[3] / (float)m_BinNum);
                    cvCircle(pI, cvPoint(x1,y1), cvRound(vf*pI->height/(m_BinNum*2)),CV_RGB(0,0,255),CV_FILLED);
                }
            } /* Draw all elements. */

            for(int i=m_TrackFGList.GetBlobNum();i>0;--i)
            {
                DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlob(i-1);
                DefMat* pHist = pF?pF->pHist:NULL;

                if(pHist==NULL) continue;

                for(Val = pHist->GetNext(&idxs,1);idxs;Val=pHist->GetNext(&idxs,0))
                {   /* Draw all elements: */
                    float   vf;
                int     x,y;

                if(!idxs) break;
                if(Val == 0) continue;

                vf = (float)Val/(pHist->m_Max?pHist->m_Max:1);
                x = cvRound((float)(pI->width-1)*(float)idxs[0] / (float)m_BinNum);
                y = cvRound((float)(pI->height-1)*(float)idxs[1] / (float)m_BinNum);

                cvCircle(pI, cvPoint(x,y), cvRound(2*vf),CV_RGB(0,0,cvRound(255*vf)),CV_FILLED);
                if(m_Dim > 3)
                {
                    int dx = -2*(idxs[2]-m_BinNum/2);
                    int dy = -2*(idxs[3]-m_BinNum/2);
                    cvLine(pI,cvPoint(x,y),cvPoint(x+dx,y+dy),CV_RGB(0,0,255));
                }
                if( m_Dim==4 &&
                        m_pFVGen->GetFVMax()[0]==m_pFVGen->GetFVMax()[2] &&
                        m_pFVGen->GetFVMax()[1]==m_pFVGen->GetFVMax()[3])
                { /* if SS feature vector */
                    int x1 = cvRound((float)(pI->width-1)*(float)idxs[2] / (float)m_BinNum);
                    int y1 = cvRound((float)(pI->height-1)*(float)idxs[3] / (float)m_BinNum);
                    cvCircle(pI, cvPoint(x1,y1), cvRound(vf*pI->height/(m_BinNum*2)),CV_RGB(0,0,255),CV_FILLED);
                }
                } /* Draw all elements. */
            } /* Next track. */

            //cvNamedWindow("Hist",0);
            //cvShowImage("Hist", pI);
            cvReleaseImage(&pI);
        }
    };

    float GetState(int BlobID)
    {
        DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlobByID(BlobID);
        return pF?pF->state:0.0f;
    };

    /* Return 0 if trajectory is normal;
       rreturn >0 if trajectory abnormal. */
    virtual const char*   GetStateDesc(int BlobID)
    {
        if(GetState(BlobID)>0.5) return "abnormal";
        return NULL;
    }

    virtual void    SetFileName(char* DataBaseName)
    {
        if(m_HistMat.m_Volume!=m_HistVolumeSaved)SaveHist();
        m_DataFileName[0] = m_DataFileName[1000] = 0;

        if(DataBaseName)
        {
            strncpy(m_DataFileName,DataBaseName,1000);
            strcat(m_DataFileName, ".yml");
        }
        LoadHist();
    };

    virtual void SaveState(CvFileStorage* fs)
    {
        int b, bN = m_TrackFGList.GetBlobNum();
        cvWriteInt(fs,"BlobNum",bN);
        cvStartWriteStruct(fs,"BlobList",CV_NODE_SEQ);

        for(b=0; b<bN; ++b)
        {
            DefTrackFG* pF = (DefTrackFG*)m_TrackFGList.GetBlob(b);
            cvStartWriteStruct(fs,NULL,CV_NODE_MAP);
            cvWriteStruct(fs,"Blob", &(pF->blob), "ffffi");
            cvWriteInt(fs,"LastFrame",pF->LastFrame);
            cvWriteReal(fs,"State",pF->state);
            pF->pHist->Save(fs, "Hist");
            cvEndWriteStruct(fs);
        }
        cvEndWriteStruct(fs);
        m_HistMat.Save(fs, "Hist");
    };

    virtual void LoadState(CvFileStorage* fs, CvFileNode* node)
    {
        CvFileNode* pBLN = cvGetFileNodeByName(fs,node,"BlobList");

        if(pBLN && CV_NODE_IS_SEQ(pBLN->tag))
        {
            int b, bN = pBLN->data.seq->total;
            for(b=0; b<bN; ++b)
            {
                DefTrackFG* pF = NULL;
                CvBlob      Blob;
                CvFileNode* pBN = (CvFileNode*)cvGetSeqElem(pBLN->data.seq,b);

                assert(pBN);
                cvReadStructByName(fs, pBN, "Blob", &Blob, "ffffi");
                AddBlob(&Blob);
                pF = (DefTrackFG*)m_TrackFGList.GetBlobByID(Blob.ID);
                if(pF==NULL) continue;
                assert(pF);
                pF->state = (float)cvReadIntByName(fs,pBN,"State",cvRound(pF->state));
                assert(pF->pHist);
                pF->pHist->Load(fs,pBN,"Hist");
            }
        }

        m_HistMat.Load(fs, node, "Hist");
    }; /* LoadState */


    virtual void    Release(){ delete this; };

};



CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistP()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisHist(cvCreateFVGenP);}

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistPV()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisHist(cvCreateFVGenPV);}

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistPVS()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisHist(cvCreateFVGenPVS);}

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisHistSS()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisHist(cvCreateFVGenSS);}

typedef struct DefTrackSVM
{
    CvBlob                  blob;
    //    CvBlobTrackFVGen*       pFVGen;
    int                     LastFrame;
    float                   state;
    CvBlob                  BlobLast;
    CvSeq*                  pFVSeq;
    CvMemStorage*           pMem;
} DefTrackSVM;

class CvBlobTrackAnalysisSVM : public CvBlobTrackAnalysis
{
    /*---------------- Internal functions: --------------------*/
private:
    int                 m_TrackNum;
    int                 m_Frame;
    char                m_DataFileName[1024];
    int                 m_Dim;
    float*              m_pFV;
    //CvStatModel*        m_pStatModel;
    void*               m_pStatModel;
    CvBlobSeq           m_Tracks;
    CvMat*              m_pTrainData;
    int                 m_LastTrainDataSize;
    //    CvBlobTrackFVGen*   (*m_CreateFVGen)();
    CvBlobTrackFVGen*   m_pFVGen;
    float               m_NU;
    float               m_RBFWidth;
    IplImage*           m_pStatImg; /* for debug purpose */
    CvSize              m_ImgSize;
    void RetrainStatModel()
    {
        ///////// !!!!! TODO !!!!! Repair /////////////
#if 0
        float               nu = 0;
        CvSVMModelParams    SVMParams = {0};
        CvStatModel*        pM = NULL;


        memset(&SVMParams,0,sizeof(SVMParams));
        SVMParams.svm_type = CV_SVM_ONE_CLASS;
        SVMParams.kernel_type = CV_SVM_RBF;
        SVMParams.gamma = 2.0/(m_RBFWidth*m_RBFWidth);
        SVMParams.nu = m_NU;
        SVMParams.degree = 3;
        SVMParams.criteria = cvTermCriteria(CV_TERMCRIT_EPS, 100, 1e-3 );
        SVMParams.C = 1;
        SVMParams.p = 0.1;


        if(m_pTrainData == NULL) return;
        {
            int64       TickCount = cvGetTickCount();
            printf("Frame: %d\n           Retrain SVM\nData Size = %d\n",m_Frame, m_pTrainData->rows);
            pM = cvTrainSVM( m_pTrainData,CV_ROW_SAMPLE, NULL, (CvStatModelParams*)&SVMParams, NULL, NULL);
            TickCount = cvGetTickCount() - TickCount ;
            printf("SV Count = %d\n",((CvSVMModel*)pM)->sv_total);
            printf("Processing Time = %.1f(ms)\n",TickCount/(1000*cvGetTickFrequency()));

        }
        if(pM==NULL) return;
        if(m_pStatModel) cvReleaseStatModel(&m_pStatModel);
        m_pStatModel = pM;

        if(m_pTrainData && m_Wnd)
        {
            float       MaxVal = 0;
            IplImage*   pW = cvCreateImage(m_ImgSize,IPL_DEPTH_32F,1);
            IplImage*   pI = cvCreateImage(m_ImgSize,IPL_DEPTH_8U,1);
            float*      pFVVar = m_pFVGen->GetFVVar();
            int     i;
            cvZero(pW);

            for(i=0; i<m_pTrainData->rows; ++i)
            {   /* Draw all elements: */
                float*          pFV = (float*)(m_pTrainData->data.ptr + m_pTrainData->step*i);
                int             x = cvRound(pFV[0]*pFVVar[0]);
                int             y = cvRound(pFV[1]*pFVVar[1]);
                float           r;

                if(x<0)x=0;
                if(x>=pW->width)x=pW->width-1;
                if(y<0)y=0;
                if(y>=pW->height)y=pW->height-1;

                r = ((float*)(pW->imageData + y*pW->widthStep))[x];
                r++;
                ((float*)(pW->imageData + y*pW->widthStep))[x] = r;

                if(r>MaxVal)MaxVal=r;
            } /* Next point. */

            if(MaxVal>0)cvConvertScale(pW,pI,255/MaxVal,0);
            cvNamedWindow("SVMData",0);
            cvShowImage("SVMData",pI);
            cvSaveImage("SVMData.bmp",pI);
            cvReleaseImage(&pW);
            cvReleaseImage(&pI);
        } /* Prepare for debug. */

        if(m_pStatModel && m_Wnd && m_Dim == 2)
        {
            float*      pFVVar = m_pFVGen->GetFVVar();
            int x,y;
            if(m_pStatImg==NULL)
            {
                m_pStatImg = cvCreateImage(m_ImgSize,IPL_DEPTH_8U,1);
            }
            cvZero(m_pStatImg);

            for(y=0; y<m_pStatImg->height; y+=1) for(x=0; x<m_pStatImg->width; x+=1)
            {   /* Draw all elements: */
                float           res;
            uchar*  pData = (uchar*)m_pStatImg->imageData + x + y*m_pStatImg->widthStep;
            CvMat           FVmat;
            float           xy[2] = {x/pFVVar[0],y/pFVVar[1]};
            cvInitMatHeader( &FVmat, 1, 2, CV_32F, xy );
            res = cvStatModelPredict( m_pStatModel, &FVmat, NULL );
            pData[0]=((res>0.5)?255:0);
            } /* Next point. */

            cvNamedWindow("SVMMask",0);
            cvShowImage("SVMMask",m_pStatImg);
            cvSaveImage("SVMMask.bmp",m_pStatImg);
        } /* Prepare for debug. */
#endif
    };
    void SaveStatModel()
    {
        if(m_DataFileName[0])
        {
            if(m_pTrainData)cvSave(m_DataFileName, m_pTrainData);
        }
    };
    void LoadStatModel()
    {
        if(m_DataFileName[0])
        {
            CvMat* pTrainData = (CvMat*)cvLoad(m_DataFileName);
            if(CV_IS_MAT(pTrainData) && pTrainData->width == m_Dim)
            {
                if(m_pTrainData) cvReleaseMat(&m_pTrainData);
                m_pTrainData = pTrainData;
                RetrainStatModel();
            }
        }
    }
public:
    CvBlobTrackAnalysisSVM(CvBlobTrackFVGen*   (*createFVGen)()):m_Tracks(sizeof(DefTrackSVM))
    {
        m_pFVGen = createFVGen();
        m_Dim = m_pFVGen->GetFVSize();
        m_pFV = (float*)cvAlloc(sizeof(float)*m_Dim);
        m_Frame = 0;
        m_TrackNum = 0;
        m_pTrainData = NULL;
        m_pStatModel = NULL;
        m_DataFileName[0] = 0;
        m_pStatImg = NULL;
        m_LastTrainDataSize = 0;

        m_NU = 0.2f;
        AddParam("Nu",&m_NU);
        CommentParam("Nu","Parameters that tunes SVM border elastic");

        m_RBFWidth = 1;
        AddParam("RBFWidth",&m_RBFWidth);
        CommentParam("RBFWidth","Parameters that tunes RBF kernel function width.");

        SetModuleName("SVM");

    } /* Constructor. */

    ~CvBlobTrackAnalysisSVM()
    {
        int i;
        SaveStatModel();
        for(i=m_Tracks.GetBlobNum();i>0;--i)
        {
            DefTrackSVM* pF = (DefTrackSVM*)m_Tracks.GetBlob(i-1);
            if(pF->pMem) cvReleaseMemStorage(&pF->pMem);
            //pF->pFVGen->Release();
        }
        if(m_pStatImg)cvReleaseImage(&m_pStatImg);
        cvFree(&m_pFV);
    } /* Destructor. */

    /*----------------- Interface: --------------------*/
    virtual void    AddBlob(CvBlob* pBlob)
    {
        DefTrackSVM* pF = (DefTrackSVM*)m_Tracks.GetBlobByID(CV_BLOB_ID(pBlob));

        m_pFVGen->AddBlob(pBlob);

        if(pF == NULL)
        {   /* Create new record: */
            DefTrackSVM F;
            F.state = 0;
            F.blob = pBlob[0];
            F.LastFrame = m_Frame;
            //F.pFVGen = m_CreateFVGen();
            F.pMem = cvCreateMemStorage();
            F.pFVSeq = cvCreateSeq(0,sizeof(CvSeq),sizeof(float)*m_Dim,F.pMem);

            F.BlobLast.x = -1;
            F.BlobLast.y = -1;
            F.BlobLast.w = -1;
            F.BlobLast.h = -1;
            m_Tracks.AddBlob((CvBlob*)&F);
            pF = (DefTrackSVM*)m_Tracks.GetBlobByID(CV_BLOB_ID(pBlob));
        }

        assert(pF);
        pF->blob = pBlob[0];
        pF->LastFrame = m_Frame;
    };

    virtual void Process(IplImage* pImg, IplImage* pFG)
    {
        int     i;
        float*  pFVVar = m_pFVGen->GetFVVar();

        m_pFVGen->Process(pImg, pFG);
        m_ImgSize = cvSize(pImg->width,pImg->height);

        for(i=m_pFVGen->GetFVNum(); i>0; --i)
        {
            int             BlobID = 0;
            float*          pFV = m_pFVGen->GetFV(i,&BlobID);
            DefTrackSVM*    pF = (DefTrackSVM*)m_Tracks.GetBlobByID(BlobID);

            if(pF && pFV)
            {   /* Process: */
                float   dx,dy;
            CvMat   FVmat;

            pF->state = 0;

            if(m_pStatModel)
            {
                int j;
                for(j=0; j<m_Dim; ++j)
                {
                    m_pFV[j] = pFV[j]/pFVVar[j];
                }

                cvInitMatHeader( &FVmat, 1, m_Dim, CV_32F, m_pFV );
                //pF->state = cvStatModelPredict( m_pStatModel, &FVmat, NULL )<0.5;
                pF->state = 1.f;
            }

            dx = (pF->blob.x - pF->BlobLast.x);
            dy = (pF->blob.y - pF->BlobLast.y);

            if(pF->BlobLast.x<0 || (dx*dx+dy*dy) >= 2*2)
            {   /* Add feature vector to train data base: */
                pF->BlobLast = pF->blob;
                cvSeqPush(pF->pFVSeq,pFV);
            }
            } /* Process one blob. */
        } /* Next FV. */

        for(i=m_Tracks.GetBlobNum(); i>0; --i)
        {   /* Check each blob record: */
            DefTrackSVM* pF = (DefTrackSVM*)m_Tracks.GetBlob(i-1);

            if(pF->LastFrame+3 < m_Frame )
            {   /* Retrain stat model and delete blob filter: */
                int                 mult = 1+m_Dim;
                int                 old_height = m_pTrainData?m_pTrainData->height:0;
                int                 height = old_height + pF->pFVSeq->total*mult;
                CvMat*              pTrainData = cvCreateMat(height, m_Dim, CV_32F);
                int                 j;
                if(m_pTrainData && pTrainData)
                {   /* Create new train data matrix: */
                    int h = pTrainData->height;
                    pTrainData->height = MIN(pTrainData->height, m_pTrainData->height);
                    cvCopy(m_pTrainData,pTrainData);
                    pTrainData->height = h;
                }

                for(j=0; j<pF->pFVSeq->total; ++j)
                {   /* Copy new data to train data: */
                    float*  pFVvar = m_pFVGen->GetFVVar();
                    float*  pFV = (float*)cvGetSeqElem(pF->pFVSeq,j);
                    int     k;

                    for(k=0; k<mult; ++k)
                    {
                        int t;
                        float*  pTD = (float*)CV_MAT_ELEM_PTR( pTrainData[0], old_height+j*mult+k, 0);
                        memcpy(pTD,pFV,sizeof(float)*m_Dim);

                        if(pFVvar)for(t=0;t<m_Dim;++t)
                        {   /* Scale FV: */
                            pTD[t] /= pFVvar[t];
                        }

                        if(k>0)
                        {   /* Variate: */
                            for(t=0; t<m_Dim; ++t)
                            {
                                pTD[t] += m_RBFWidth*0.5f*(1-2.0f*rand()/(float)RAND_MAX);
                            }
                        }
                    }
                } /* Next new datum. */

                if(m_pTrainData) cvReleaseMat(&m_pTrainData);
                m_pTrainData = pTrainData;

                /* delete track record */
                cvReleaseMemStorage(&pF->pMem);
                m_TrackNum++;
                m_Tracks.DelBlob(i-1);

            } /* End delete. */
        } /* Next track. */

        /* Retrain data each 1 minute if new data exist: */
        if(m_Frame%(25*60) == 0 && m_pTrainData && m_pTrainData->rows > m_LastTrainDataSize)
        {
            RetrainStatModel();
        }

        m_Frame++;

        if(m_Wnd && m_Dim==2)
        {   /* Debug output: */
            int         x,y;
            IplImage*   pI = cvCloneImage(pImg);

            if(m_pStatModel && m_pStatImg)

                for(y=0; y<pI->height; y+=2)
                {
                    uchar*  pStatData = (uchar*)m_pStatImg->imageData + y*m_pStatImg->widthStep;
                    uchar*  pData = (uchar*)pI->imageData + y*pI->widthStep;

                    for(x=0;x<pI->width;x+=2)
                    {   /* Draw all elements: */
                        int d = pStatData[x];
                        d = (d<<8) | (d^0xff);
                        *(ushort*)(pData + x*3) = (ushort)d;
                    }
                } /* Next line. */

            //cvNamedWindow("SVMMap",0);
            //cvShowImage("SVMMap", pI);
            cvReleaseImage(&pI);
        } /* Debug output. */
    };
    float GetState(int BlobID)
    {
        DefTrackSVM* pF = (DefTrackSVM*)m_Tracks.GetBlobByID(BlobID);
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
        if(m_pTrainData)SaveStatModel();
        m_DataFileName[0] = m_DataFileName[1000] = 0;
        if(DataBaseName)
        {
            strncpy(m_DataFileName,DataBaseName,1000);
            strcat(m_DataFileName, ".yml");
        }
        LoadStatModel();
    };


    virtual void    Release(){ delete this; };

}; /* CvBlobTrackAnalysisSVM. */

#if 0
CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMP()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisSVM(cvCreateFVGenP);}

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMPV()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisSVM(cvCreateFVGenPV);}

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMPVS()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisSVM(cvCreateFVGenPVS);}

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisSVMSS()
{return (CvBlobTrackAnalysis*) new CvBlobTrackAnalysisSVM(cvCreateFVGenSS);}
#endif
