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

/*======================= FILTER LIST SHELL =====================*/
#define MAX_ANS 16
#define MAX_DESC 1024
class CvBlobTrackAnalysisIOR : public CvBlobTrackAnalysis
{
protected:
    struct  DefAn
    {
        const char*                   pName;
        CvBlobTrackAnalysis*    pAn;
    } m_Ans[MAX_ANS];
    int m_AnNum;
    char m_Desc[MAX_DESC];

public:
    CvBlobTrackAnalysisIOR()
    {
        m_AnNum = 0;
        SetModuleName("IOR");
    }

    ~CvBlobTrackAnalysisIOR()
    {
    };

    virtual void    AddBlob(CvBlob* pBlob)
    {
        int i;
        for(i=0; i<m_AnNum; ++i)
        {
            m_Ans[i].pAn->AddBlob(pBlob);
        } /* Next analyzer. */
    };

    virtual void    Process(IplImage* pImg, IplImage* pFG)
    {
        int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(i=0; i<m_AnNum; ++i)
        {
            m_Ans[i].pAn->Process(pImg, pFG);
        } /* Next analyzer. */
    };

    float GetState(int BlobID)
    {
        int state = 0;
        int i;
        for(i=0; i<m_AnNum; ++i)
        {
            state |= (m_Ans[i].pAn->GetState(BlobID) > 0.5);
        } /* Next analyzer. */

        return (float)state;
    };

    virtual const char*   GetStateDesc(int BlobID)
    {
        int     rest = MAX_DESC-1;
        int     i;
        m_Desc[0] = 0;

        for(i=0; i<m_AnNum; ++i)
        {
            const char* str = m_Ans[i].pAn->GetStateDesc(BlobID);

            if(str && strlen(m_Ans[i].pName) + strlen(str)+4 < (size_t)rest)
            {
                strcat(m_Desc,m_Ans[i].pName);
                strcat(m_Desc,": ");
                strcat(m_Desc,str);
                strcat(m_Desc,"\n");
                rest = MAX_DESC - (int)strlen(m_Desc) - 1;
            }
        } /* Next analyzer. */

        if(m_Desc[0]!=0)return m_Desc;

        return NULL;
    };

    virtual void SetFileName(char* /*DataBaseName*/)
    {
    };

    int AddAnalyzer(CvBlobTrackAnalysis* pA, const char* pName)
    {
        if(m_AnNum<MAX_ANS)
        {
            //int i;
            m_Ans[m_AnNum].pName = pName;
            m_Ans[m_AnNum].pAn = pA;
            TransferParamsFromChild(m_Ans[m_AnNum].pAn, pName);
            m_AnNum++;
            return 1;
        }
        else
        {
            printf("Can not add track analyzer %s! (not more that %d analyzers)\n",pName,MAX_ANS);
            return 0;
        }
    }
    void    Release()
    {
        int i;
        for(i=0; i<m_AnNum; ++i)
        {
            m_Ans[i].pAn->Release();
        } /* Next analyzer. */

        delete this;
    };
}; /* CvBlobTrackAnalysisIOR. */

CvBlobTrackAnalysis* cvCreateModuleBlobTrackAnalysisIOR()
{
    CvBlobTrackAnalysisIOR* pIOR = new CvBlobTrackAnalysisIOR();
    CvBlobTrackAnalysis* pA = NULL;

    pA = cvCreateModuleBlobTrackAnalysisHistPVS();
    pIOR->AddAnalyzer(pA, "HIST");

    //pA = (CvBlobTrackAnalysis*)cvCreateModuleBlobTrackAnalysisHeightScale();
    //pIOR->AddAnalyzer(pA, "SCALE");

    return (CvBlobTrackAnalysis*)pIOR;
}/* cvCreateCvBlobTrackAnalysisIOR */
/* ======================== Analyser modules ============================= */
