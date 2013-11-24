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

typedef struct DefBlobTrack
{
    CvBlob      blob;
    CvBlobSeq*  pSeq;
    int         FrameBegin;
    int         FrameLast;
    int         Saved; /* flag */
} DefBlobTrack;


static void SaveTrack(DefBlobTrack* pTrack, char* pFileName, int norm = 0)
{   /* Save blob track: */
    int         j;
    FILE*       out = NULL;
    CvBlobSeq*  pS = pTrack->pSeq;
    CvBlob*     pB0 = pS?pS->GetBlob(0):NULL;

    if(pFileName == NULL) return;
    if(pTrack == NULL) return;

    out = fopen(pFileName,"at");
    if(out == NULL)
    {
        printf("Warning! Cannot open %s file for track output\n", pFileName);
        return;
    }

    fprintf(out,"%d",pTrack->FrameBegin);

    if(pS) for(j=0; j<pS->GetBlobNum(); ++j)
    {
        CvBlob* pB = pS->GetBlob(j);

        fprintf(out,", %.1f, %.1f", CV_BLOB_X(pB),CV_BLOB_Y(pB));

        if(CV_BLOB_WX(pB0)>0)
            fprintf(out,", %.2f",CV_BLOB_WX(pB)/(norm?CV_BLOB_WX(pB0):1));

        if(CV_BLOB_WY(pB0)>0)
            fprintf(out,", %.2f",CV_BLOB_WY(pB)/(norm?CV_BLOB_WY(pB0):1));
    }
    fprintf(out,"\n");
    fclose(out);
    pTrack->Saved = 1;
}   /* Save blob track. */

class CvBlobTrackGen1:public CvBlobTrackGen
{

public:
    CvBlobTrackGen1(int BlobSizeNorm = 0):m_TrackList(sizeof(DefBlobTrack))
    {
        m_BlobSizeNorm = BlobSizeNorm;
        m_Frame = 0;
        m_pFileName = NULL;

        SetModuleName("Gen1");
    };

    ~CvBlobTrackGen1()
    {
        int i;
        for(i=m_TrackList.GetBlobNum();i>0;--i)
        {
            DefBlobTrack* pTrack = (DefBlobTrack*)m_TrackList.GetBlob(i-1);
            if(!pTrack->Saved)
            {   /* Save track: */
                SaveTrack(pTrack, m_pFileName, m_BlobSizeNorm);
            }   /* Save track. */

            /* Delete sequence: */
            delete pTrack->pSeq;

            pTrack->pSeq = NULL;

        }   /* Check next track. */
    }   /*  Destructor. */

    void    SetFileName(char* pFileName){m_pFileName = pFileName;};

    void    AddBlob(CvBlob* pBlob)
    {
        DefBlobTrack* pTrack = (DefBlobTrack*)m_TrackList.GetBlobByID(CV_BLOB_ID(pBlob));

        if(pTrack==NULL)
        {   /* Add new track: */
            DefBlobTrack    Track;
            Track.blob = pBlob[0];
            Track.FrameBegin = m_Frame;
            Track.pSeq = new CvBlobSeq;
            Track.Saved = 0;
            m_TrackList.AddBlob((CvBlob*)&Track);
            pTrack = (DefBlobTrack*)m_TrackList.GetBlobByID(CV_BLOB_ID(pBlob));
        }   /* Add new track. */

        assert(pTrack);
        pTrack->FrameLast = m_Frame;
        assert(pTrack->pSeq);
        pTrack->pSeq->AddBlob(pBlob);
    };

    void    Process(IplImage* /*pImg*/ = NULL, IplImage* /*pFG*/ = NULL)
    {
        int i;

        for(i=m_TrackList.GetBlobNum(); i>0; --i)
        {
            DefBlobTrack* pTrack = (DefBlobTrack*)m_TrackList.GetBlob(i-1);

            if(pTrack->FrameLast < m_Frame && !pTrack->Saved)
            {   /* Save track: */
                SaveTrack(pTrack, m_pFileName, m_BlobSizeNorm);
                if(pTrack->Saved)
                {   /* delete sequence */
                    delete pTrack->pSeq;
                    pTrack->pSeq = NULL;
                    m_TrackList.DelBlob(i-1);
                }
            }   /* Save track. */
        }   /*  Check next track. */
        m_Frame++;
    }

    void Release()
    {
        delete this;
    }

protected:
    int         m_Frame;
    char*       m_pFileName;
    CvBlobSeq   m_TrackList;
    int         m_BlobSizeNorm;
};  /* class CvBlobTrackGen1 */


CvBlobTrackGen* cvCreateModuleBlobTrackGen1()
{
    return (CvBlobTrackGen*) new CvBlobTrackGen1(0);
}
