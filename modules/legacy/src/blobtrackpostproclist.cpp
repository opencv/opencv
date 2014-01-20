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

/*======================= FILTER LIST SHELL =====================*/
typedef struct DefBlobFilter
{
    CvBlob                  blob;
    CvBlobTrackPostProcOne* pFilter;
    int                     m_LastFrame;
} DefBlobFilter;

class CvBlobTrackPostProcList : public CvBlobTrackPostProc
{
protected:
    CvBlobTrackPostProcOne* (*m_CreatePostProc)();
    CvBlobSeq               m_BlobFilterList;
    int                     m_Frame;

public:
    CvBlobTrackPostProcList(CvBlobTrackPostProcOne* (*create)()):m_BlobFilterList(sizeof(DefBlobFilter))
    {
        m_Frame = 0;
        m_CreatePostProc = create;
        CvBlobTrackPostProcOne* pM = create();
        TransferParamsFromChild(pM,NULL);
        pM->Release();
        SetModuleName("List");
    }

   ~CvBlobTrackPostProcList()
    {
        int i;
        for(i=m_BlobFilterList.GetBlobNum();i>0;--i)
        {
            DefBlobFilter* pF = (DefBlobFilter*)m_BlobFilterList.GetBlob(i-1);
            pF->pFilter->Release();
        }
    }

    virtual void    AddBlob(CvBlob* pBlob)
    {
        DefBlobFilter* pF = (DefBlobFilter*)m_BlobFilterList.GetBlobByID(CV_BLOB_ID(pBlob));
        if(pF == NULL)
        {   /* Create new filter: */
            DefBlobFilter F;
            F.blob = pBlob[0];
            F.m_LastFrame = m_Frame;
            F.pFilter = m_CreatePostProc();
            TransferParamsToChild(F.pFilter,NULL);
            m_BlobFilterList.AddBlob((CvBlob*)&F);
            pF = (DefBlobFilter*)m_BlobFilterList.GetBlobByID(CV_BLOB_ID(pBlob));
        }

        assert(pF);
        pF->blob = pBlob[0];
        pF->m_LastFrame = m_Frame;
    }

    virtual void    Process()
    {
        int i;
        for(i=m_BlobFilterList.GetBlobNum(); i>0; --i)
        {
            DefBlobFilter* pF = (DefBlobFilter*)m_BlobFilterList.GetBlob(i-1);

            if(pF->m_LastFrame == m_Frame)
            {   /* Process: */
                int ID = CV_BLOB_ID(pF);
                pF->blob = *(pF->pFilter->Process(&(pF->blob)));
                CV_BLOB_ID(pF) = ID;
            }
            else
            {   /* Delete blob filter: */
                pF->pFilter->Release();
                m_BlobFilterList.DelBlob(i-1);
            }
        }   /* Next blob. */
        m_Frame++;
    }

    int     GetBlobNum(){return m_BlobFilterList.GetBlobNum();}
    CvBlob* GetBlob(int index){return m_BlobFilterList.GetBlob(index);}
    void    Release(){delete this;}

    /* Additional functionality: */
    CvBlob* GetBlobByID(int BlobID){return m_BlobFilterList.GetBlobByID(BlobID);}

};  /* CvBlobTrackPostProcList */

CvBlobTrackPostProc* cvCreateBlobTrackPostProcList(CvBlobTrackPostProcOne* (*create)())
{
    return (CvBlobTrackPostProc*) new CvBlobTrackPostProcList(create);
}
/*======================= FILTER LIST SHELL =====================*/
