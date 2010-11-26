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

/*======================= KALMAN FILTER AS TRACKER =========================*/
/* State vector is (x,y,w,h,dx,dy,dw,dh). */
/* Measurement is (x,y,w,h) */

/* Dynamic matrix A: */
const float A8[] = { 1, 0, 0, 0, 1, 0, 0, 0,
                     0, 1, 0, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 0, 0, 1, 0,
                     0, 0, 0, 1, 0, 0, 0, 1,
                     0, 0, 0, 0, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 1};

/* Measurement matrix H: */
const float H8[] = { 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 0, 0, 0};

/* Matices for zero size velocity: */
/* Dynamic matrix A: */
const float A6[] = { 1, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 1,
                     0, 0, 1, 0, 0, 0,
                     0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 1};

/* Measurement matrix H: */
const float H6[] = { 1, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0,
                     0, 0, 0, 1, 0, 0};

#define STATE_NUM 6
#define A A6
#define H H6
class CvBlobTrackerOneKalman:public CvBlobTrackerOne
{
private:
    CvBlob      m_Blob;
    CvKalman*   m_pKalman;
    int         m_Frame;

public:
    CvBlobTrackerOneKalman()
    {
        m_Frame = 0;
        m_pKalman = cvCreateKalman(STATE_NUM,4);
        memcpy( m_pKalman->transition_matrix->data.fl, A, sizeof(A));
        memcpy( m_pKalman->measurement_matrix->data.fl, H, sizeof(H));
        cvSetIdentity( m_pKalman->process_noise_cov, cvRealScalar(1e-5) );
        cvSetIdentity( m_pKalman->measurement_noise_cov, cvRealScalar(1e-1) );
    //    CV_MAT_ELEM(*m_pKalman->measurement_noise_cov, float, 2,2) *= (float)pow(20,2);
    //    CV_MAT_ELEM(*m_pKalman->measurement_noise_cov, float, 3,3) *= (float)pow(20,2);
        cvSetIdentity( m_pKalman->error_cov_post, cvRealScalar(1));
        cvZero(m_pKalman->state_post);
        cvZero(m_pKalman->state_pre);

        SetModuleName("Kalman");
    }

    ~CvBlobTrackerOneKalman()
    {
        cvReleaseKalman(&m_pKalman);
    }

    virtual void Init(CvBlob* pBlob, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL)
    {
        m_Blob = pBlob[0];
        m_pKalman->state_post->data.fl[0] = CV_BLOB_X(pBlob);
        m_pKalman->state_post->data.fl[1] = CV_BLOB_Y(pBlob);
        m_pKalman->state_post->data.fl[2] = CV_BLOB_WX(pBlob);
        m_pKalman->state_post->data.fl[3] = CV_BLOB_WY(pBlob);
    }

    virtual CvBlob* Process(CvBlob* pBlob, IplImage* /*pImg*/, IplImage* /*pImgFG*/ = NULL)
    {
        CvBlob* pBlobRes = &m_Blob;
        float   Z[4];
        CvMat   Zmat = cvMat(4,1,CV_32F,Z);
        m_Blob = pBlob[0];

        if(m_Frame < 2)
        {   /* First call: */
            m_pKalman->state_post->data.fl[0+4] = CV_BLOB_X(pBlob)-m_pKalman->state_post->data.fl[0];
            m_pKalman->state_post->data.fl[1+4] = CV_BLOB_Y(pBlob)-m_pKalman->state_post->data.fl[1];
            if(m_pKalman->DP>6)
            {
                m_pKalman->state_post->data.fl[2+4] = CV_BLOB_WX(pBlob)-m_pKalman->state_post->data.fl[2];
                m_pKalman->state_post->data.fl[3+4] = CV_BLOB_WY(pBlob)-m_pKalman->state_post->data.fl[3];
            }
            m_pKalman->state_post->data.fl[0] = CV_BLOB_X(pBlob);
            m_pKalman->state_post->data.fl[1] = CV_BLOB_Y(pBlob);
            m_pKalman->state_post->data.fl[2] = CV_BLOB_WX(pBlob);
            m_pKalman->state_post->data.fl[3] = CV_BLOB_WY(pBlob);
            memcpy(m_pKalman->state_pre->data.fl,m_pKalman->state_post->data.fl,sizeof(float)*STATE_NUM);
        }
        else
        {   /* Another call: */
            Z[0] = CV_BLOB_X(pBlob);
            Z[1] = CV_BLOB_Y(pBlob);
            Z[2] = CV_BLOB_WX(pBlob);
            Z[3] = CV_BLOB_WY(pBlob);
            cvKalmanCorrect(m_pKalman,&Zmat);
            cvKalmanPredict(m_pKalman,0);
            cvMatMulAdd(m_pKalman->measurement_matrix, m_pKalman->state_pre, NULL, &Zmat);
            CV_BLOB_X(pBlobRes) = Z[0];
            CV_BLOB_Y(pBlobRes) = Z[1];
            CV_BLOB_WX(pBlobRes) = Z[2];
            CV_BLOB_WY(pBlobRes) = Z[3];
        }
        m_Frame++;
        return pBlobRes;
    }
    virtual void Release()
    {
        delete this;
    }
};  /* class CvBlobTrackerOneKalman */

static CvBlobTrackerOne* cvCreateModuleBlobTrackerOneKalman()
{
    return (CvBlobTrackerOne*) new CvBlobTrackerOneKalman;
}

CvBlobTracker* cvCreateBlobTrackerKalman()
{
    return cvCreateBlobTrackerList(cvCreateModuleBlobTrackerOneKalman);
}
