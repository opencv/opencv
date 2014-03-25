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

typedef struct DefBlobTrack
{
    CvBlob      blob;
    CvBlobSeq*  pSeq;
    int         FrameBegin;
    int         FrameLast;
    int         Saved; /* flag */
} DefBlobTrack;

class CvBlobTrackGenYML:public CvBlobTrackGen
{
protected:
    int         m_Frame;
    char*       m_pFileName;
    CvBlobSeq   m_TrackList;
    CvSize      m_Size;

    void SaveAll()
    {
        int     ObjNum = m_TrackList.GetBlobNum();
        int     i;
        char    video_name[1024+1];
        char*   struct_name = NULL;
        CvFileStorage* storage = cvOpenFileStorage(m_pFileName,NULL,CV_STORAGE_WRITE_TEXT);

        if(storage==NULL)
        {
            printf("WARNING!!! Cannot open %s file for trajectory output.", m_pFileName);
        }

        for(i=0; i<1024 && m_pFileName[i]!='.' && m_pFileName[i]!=0; ++i) video_name[i]=m_pFileName[i];
        video_name[i] = 0;

        for(;i>0; i--)
        {
            if(video_name[i-1] == '\\') break;
            if(video_name[i-1] == '/') break;
            if(video_name[i-1] == ':') break;
        }
        struct_name = video_name + i;

        cvStartWriteStruct(storage, struct_name, CV_NODE_SEQ);

        for(i=0; i<ObjNum; ++i)
        {
            char            obj_name[1024];
            DefBlobTrack*   pTrack = (DefBlobTrack*)m_TrackList.GetBlob(i);
            if(pTrack==NULL) continue;
            sprintf(obj_name,"%s_obj%d",struct_name,i);
            cvStartWriteStruct(storage, NULL, CV_NODE_MAP);
            cvWriteInt(storage, "FrameBegin", pTrack->FrameBegin);
            cvWriteString(storage, "VideoObj", obj_name);
            cvEndWriteStruct( storage );
            pTrack->Saved = 1;
        }
        cvEndWriteStruct( storage );

        for(i=0;i<ObjNum;++i)
        {
            char            obj_name[1024];
            DefBlobTrack*   pTrack = (DefBlobTrack*)m_TrackList.GetBlob(i);
            CvBlobSeq*      pSeq = pTrack->pSeq;
            sprintf(obj_name,"%s_obj%d",struct_name,i);
            cvStartWriteStruct(storage, obj_name, CV_NODE_MAP);

            {   /* Write position: */
                int             j;
                CvPoint2D32f    p;
                cvStartWriteStruct(storage, "Pos", CV_NODE_SEQ|CV_NODE_FLOW);
                for(j=0;j<pSeq->GetBlobNum();++j)
                {
                    CvBlob* pB = pSeq->GetBlob(j);
                    p.x = pB->x/(m_Size.width-1);
                    p.y = pB->y/(m_Size.height-1);
                    cvWriteRawData(storage, &p, 1 ,"ff");
                }
                cvEndWriteStruct( storage );
            }

            {   /* Write size: */
                int             j;
                CvPoint2D32f    p;
                cvStartWriteStruct(storage, "Size", CV_NODE_SEQ|CV_NODE_FLOW);

                for(j=0; j<pSeq->GetBlobNum(); ++j)
                {
                    CvBlob* pB = pSeq->GetBlob(j);
                    p.x = pB->w/(m_Size.width-1);
                    p.y = pB->h/(m_Size.height-1);
                    cvWriteRawData(storage, &p, 1 ,"ff");
                }
                cvEndWriteStruct( storage );
            }
            cvEndWriteStruct( storage );
        }
        cvReleaseFileStorage(&storage);

    }   /* Save All */

public:
    CvBlobTrackGenYML():m_TrackList(sizeof(DefBlobTrack))
    {
        m_Frame = 0;
        m_pFileName = NULL;
        m_Size = cvSize(2,2);

        SetModuleName("YML");
    }

    ~CvBlobTrackGenYML()
    {
        int i;
        SaveAll();

        for(i=m_TrackList.GetBlobNum(); i>0; --i)
        {
            DefBlobTrack* pTrack = (DefBlobTrack*)m_TrackList.GetBlob(i-1);
            /* Delete sequence: */
            delete pTrack->pSeq;
            pTrack->pSeq = NULL;
        }   /* Check next track. */

    }   /* Destructor. */

    void    SetFileName(char* pFileName){m_pFileName = pFileName;}
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
    }
    void    Process(IplImage* pImg = NULL, IplImage* /*pFG*/ = NULL)
    {
        int i;
        m_Size = cvSize(pImg->width,pImg->height);

        for(i=m_TrackList.GetBlobNum(); i>0; --i)
        {
            DefBlobTrack* pTrack = (DefBlobTrack*)m_TrackList.GetBlob(i-1);

            if(pTrack->FrameLast < m_Frame && !pTrack->Saved)
            {   /* Save track: */
                SaveAll();
            }   /* Save track. */

        }   /* Check next track. */

        m_Frame++;
    }

    void Release()
    {
        delete this;
    }
};  /* class CvBlobTrackGenYML */


CvBlobTrackGen* cvCreateModuleBlobTrackGenYML()
{
    return (CvBlobTrackGen*) new CvBlobTrackGenYML;
}
