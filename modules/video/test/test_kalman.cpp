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

#include "test_precomp.hpp"

using namespace cv;

class CV_KalmanTest : public cvtest::BaseTest
{
public:
    CV_KalmanTest();
protected:
    void run(int);
};


CV_KalmanTest::CV_KalmanTest()
{
}

void CV_KalmanTest::run( int )
{
    int code = cvtest::TS::OK;
    const int Dim = 7;
    const int Steps = 100;
    const double max_init = 1;
    const double max_noise = 0.1;
    
    const double EPSILON = 1.000;
    RNG& rng = ts->get_rng();
    CvKalman* Kalm;
    int i, j;
    
    CvMat* Sample = cvCreateMat(Dim,1,CV_32F);
    CvMat* Temp = cvCreateMat(Dim,1,CV_32F);
    
    Kalm = cvCreateKalman(Dim, Dim);
    CvMat Dyn = cvMat(Dim,Dim,CV_32F,Kalm->DynamMatr);
    CvMat Mes = cvMat(Dim,Dim,CV_32F,Kalm->MeasurementMatr);
    CvMat PNC = cvMat(Dim,Dim,CV_32F,Kalm->PNCovariance);
    CvMat MNC = cvMat(Dim,Dim,CV_32F,Kalm->MNCovariance);
    CvMat PriErr = cvMat(Dim,Dim,CV_32F,Kalm->PriorErrorCovariance);
    CvMat PostErr = cvMat(Dim,Dim,CV_32F,Kalm->PosterErrorCovariance);
    CvMat PriState = cvMat(Dim,1,CV_32F,Kalm->PriorState);
    CvMat PostState = cvMat(Dim,1,CV_32F,Kalm->PosterState);
    cvSetIdentity(&PNC);
    cvSetIdentity(&PriErr);
    cvSetIdentity(&PostErr);
    cvSetZero(&MNC);
    cvSetZero(&PriState);
    cvSetZero(&PostState);
    cvSetIdentity(&Mes);
    cvSetIdentity(&Dyn);
    Mat _Sample = cvarrToMat(Sample);
    cvtest::randUni(rng, _Sample, cvScalarAll(-max_init), cvScalarAll(max_init));
    cvKalmanCorrect(Kalm, Sample);
    for(i = 0; i<Steps; i++)
    {
        cvKalmanPredict(Kalm);
        for(j = 0; j<Dim; j++)
        {
            float t = 0;
            for(int k=0; k<Dim; k++)
            {
                t += Dyn.data.fl[j*Dim+k]*Sample->data.fl[k];
            }
            Temp->data.fl[j]= (float)(t+(cvtest::randReal(rng)*2-1)*max_noise);
        }
        cvCopy( Temp, Sample );
        cvKalmanCorrect(Kalm,Temp);
    }

    Mat _state_post = cvarrToMat(Kalm->state_post);
    code = cvtest::cmpEps2( ts, _Sample, _state_post, EPSILON, false, "The final estimated state" );

    cvReleaseMat(&Sample);
    cvReleaseMat(&Temp);
    cvReleaseKalman(&Kalm);
    
    if( code < 0 )
        ts->set_failed_test_info( code );
}

TEST(Video_Kalman, accuracy) { CV_KalmanTest test; test.safe_run(); }

/* End of file. */
