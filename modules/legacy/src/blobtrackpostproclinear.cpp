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

/*======================= TIME AVERAGING FILTER =========================*/
#define TIME_WND 5
class CvBlobTrackPostProcTimeAver:public CvBlobTrackPostProcOne
{

protected:
    CvBlob      m_Blob;
    CvBlob      m_pBlobs[TIME_WND];
    float       m_Weights[TIME_WND];
    int         m_Frame;

public:
    CvBlobTrackPostProcTimeAver( int KernelType = 0)
    {
        int i;
        m_Frame = 0;
        for(i=0;i<TIME_WND;++i)
        {
            m_Weights[i] = 1;
            if(KernelType == 1)
            {
                m_Weights[i] = (float)exp((-2.3*i)/(TIME_WND-1)); /* last weight is 0.1 of first weight */
            }
        }

        SetModuleName("TimeAver");
    };

   ~CvBlobTrackPostProcTimeAver(){};

    CvBlob* Process(CvBlob* pBlob)
    {
        float   WSum = 0;
        int     i;
        int idx = m_Frame % TIME_WND;
        int size = MIN((m_Frame+1), TIME_WND);
        m_pBlobs[idx] = pBlob[0];
        m_Blob.x = m_Blob.y = m_Blob.w = m_Blob.h = 0;

        for(i=0; i<size; ++i)
        {
            float   W = m_Weights[i];
            int     index  = (m_Frame - i + TIME_WND) % TIME_WND;
            m_Blob.x += W*m_pBlobs[index].x;
            m_Blob.y += W*m_pBlobs[index].y;
            m_Blob.w += W*m_pBlobs[index].w;
            m_Blob.h += W*m_pBlobs[index].h;
            WSum += W;
        }
        assert(WSum>0);

        m_Blob.x /= WSum;
        m_Blob.y /= WSum;
        m_Blob.w /= WSum;
        m_Blob.h /= WSum;

        m_Frame++;
        return &m_Blob;
    };

    void Release()
    {
        delete this;
    }
};  /* class CvBlobTrackPostProcTimeAver */

static CvBlobTrackPostProcOne* cvCreateModuleBlobTrackPostProcTimeAverRectOne()
{
    return (CvBlobTrackPostProcOne*) new CvBlobTrackPostProcTimeAver(0);
}

static CvBlobTrackPostProcOne* cvCreateModuleBlobTrackPostProcTimeAverExpOne()
{
    return (CvBlobTrackPostProcOne*) new CvBlobTrackPostProcTimeAver(1);
}

CvBlobTrackPostProc* cvCreateModuleBlobTrackPostProcTimeAverRect()
{
    return cvCreateBlobTrackPostProcList(cvCreateModuleBlobTrackPostProcTimeAverRectOne);
}

CvBlobTrackPostProc* cvCreateModuleBlobTrackPostProcTimeAverExp()
{
    return cvCreateBlobTrackPostProcList(cvCreateModuleBlobTrackPostProcTimeAverExpOne);
}
/*======================= KALMAN FILTER =========================*/
