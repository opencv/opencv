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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include "opencv2/video/tracking_c.h"


/////////////////////////// Meanshift & CAMShift ///////////////////////////

CV_IMPL int
cvMeanShift( const void* imgProb, CvRect windowIn,
             CvTermCriteria criteria, CvConnectedComp* comp )
{
    cv::Mat img = cv::cvarrToMat(imgProb);
    cv::Rect window = windowIn;
    int iters = cv::meanShift(img, window, criteria);

    if( comp )
    {
        comp->rect = window;
        comp->area = cvRound(cv::sum(img(window))[0]);
    }

    return iters;
}


CV_IMPL int
cvCamShift( const void* imgProb, CvRect windowIn,
            CvTermCriteria criteria,
            CvConnectedComp* comp,
            CvBox2D* box )
{
    cv::Mat img = cv::cvarrToMat(imgProb);
    cv::Rect window = windowIn;
    cv::RotatedRect rr = cv::CamShift(img, window, criteria);

    if( comp )
    {
        comp->rect = window;
        cv::Rect roi = rr.boundingRect() & cv::Rect(0, 0, img.cols, img.rows);
        comp->area = cvRound(cv::sum(img(roi))[0]);
    }

    if( box )
        *box = rr;

    return rr.size.width*rr.size.height > 0.f ? 1 : -1;
}


///////////////////////// Motion Templates ////////////////////////////

CV_IMPL void
cvUpdateMotionHistory( const void* silhouette, void* mhimg,
                       double timestamp, double mhi_duration )
{
    cv::Mat silh = cv::cvarrToMat(silhouette), mhi = cv::cvarrToMat(mhimg);
    cv::updateMotionHistory(silh, mhi, timestamp, mhi_duration);
}


CV_IMPL void
cvCalcMotionGradient( const CvArr* mhimg, CvArr* maskimg,
                      CvArr* orientation,
                      double delta1, double delta2,
                      int aperture_size )
{
    cv::Mat mhi = cv::cvarrToMat(mhimg);
    const cv::Mat mask = cv::cvarrToMat(maskimg), orient = cv::cvarrToMat(orientation);
    cv::calcMotionGradient(mhi, mask, orient, delta1, delta2, aperture_size);
}


CV_IMPL double
cvCalcGlobalOrientation( const void* orientation, const void* maskimg, const void* mhimg,
                         double curr_mhi_timestamp, double mhi_duration )
{
    cv::Mat mhi = cv::cvarrToMat(mhimg);
    cv::Mat mask = cv::cvarrToMat(maskimg), orient = cv::cvarrToMat(orientation);
    return cv::calcGlobalOrientation(orient, mask, mhi, curr_mhi_timestamp, mhi_duration);
}


CV_IMPL CvSeq*
cvSegmentMotion( const CvArr* mhimg, CvArr* segmaskimg, CvMemStorage* storage,
                 double timestamp, double segThresh )
{
    cv::Mat mhi = cv::cvarrToMat(mhimg);
    const cv::Mat segmask = cv::cvarrToMat(segmaskimg);
    std::vector<cv::Rect> brs;
    cv::segmentMotion(mhi, segmask, brs, timestamp, segThresh);
    CvSeq* seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvConnectedComp), storage);

    CvConnectedComp comp;
    memset(&comp, 0, sizeof(comp));
    for( size_t i = 0; i < brs.size(); i++ )
    {
        cv::Rect roi = brs[i];
        float compLabel = (float)(i+1);
        int x, y, area = 0;

        cv::Mat part = segmask(roi);
        for( y = 0; y < roi.height; y++ )
        {
            const float* partptr = part.ptr<float>(y);
            for( x = 0; x < roi.width; x++ )
                area += partptr[x] == compLabel;
        }

        comp.value = cv::Scalar(compLabel);
        comp.rect = roi;
        comp.area = area;
        cvSeqPush(seq, &comp);
    }

    return seq;
}


///////////////////////////////// Kalman ///////////////////////////////

CV_IMPL CvKalman*
cvCreateKalman( int DP, int MP, int CP )
{
    CvKalman *kalman = 0;

    if( DP <= 0 || MP <= 0 )
        CV_Error( CV_StsOutOfRange,
        "state and measurement vectors must have positive number of dimensions" );

    if( CP < 0 )
        CP = DP;

    /* allocating memory for the structure */
    kalman = (CvKalman *)cvAlloc( sizeof( CvKalman ));
    memset( kalman, 0, sizeof(*kalman));

    kalman->DP = DP;
    kalman->MP = MP;
    kalman->CP = CP;

    kalman->state_pre = cvCreateMat( DP, 1, CV_32FC1 );
    cvZero( kalman->state_pre );

    kalman->state_post = cvCreateMat( DP, 1, CV_32FC1 );
    cvZero( kalman->state_post );

    kalman->transition_matrix = cvCreateMat( DP, DP, CV_32FC1 );
    cvSetIdentity( kalman->transition_matrix );

    kalman->process_noise_cov = cvCreateMat( DP, DP, CV_32FC1 );
    cvSetIdentity( kalman->process_noise_cov );

    kalman->measurement_matrix = cvCreateMat( MP, DP, CV_32FC1 );
    cvZero( kalman->measurement_matrix );

    kalman->measurement_noise_cov = cvCreateMat( MP, MP, CV_32FC1 );
    cvSetIdentity( kalman->measurement_noise_cov );

    kalman->error_cov_pre = cvCreateMat( DP, DP, CV_32FC1 );

    kalman->error_cov_post = cvCreateMat( DP, DP, CV_32FC1 );
    cvZero( kalman->error_cov_post );

    kalman->gain = cvCreateMat( DP, MP, CV_32FC1 );

    if( CP > 0 )
    {
        kalman->control_matrix = cvCreateMat( DP, CP, CV_32FC1 );
        cvZero( kalman->control_matrix );
    }

    kalman->temp1 = cvCreateMat( DP, DP, CV_32FC1 );
    kalman->temp2 = cvCreateMat( MP, DP, CV_32FC1 );
    kalman->temp3 = cvCreateMat( MP, MP, CV_32FC1 );
    kalman->temp4 = cvCreateMat( MP, DP, CV_32FC1 );
    kalman->temp5 = cvCreateMat( MP, 1, CV_32FC1 );

#if 1
    kalman->PosterState = kalman->state_pre->data.fl;
    kalman->PriorState = kalman->state_post->data.fl;
    kalman->DynamMatr = kalman->transition_matrix->data.fl;
    kalman->MeasurementMatr = kalman->measurement_matrix->data.fl;
    kalman->MNCovariance = kalman->measurement_noise_cov->data.fl;
    kalman->PNCovariance = kalman->process_noise_cov->data.fl;
    kalman->KalmGainMatr = kalman->gain->data.fl;
    kalman->PriorErrorCovariance = kalman->error_cov_pre->data.fl;
    kalman->PosterErrorCovariance = kalman->error_cov_post->data.fl;
#endif

    return kalman;
}


CV_IMPL void
cvReleaseKalman( CvKalman** _kalman )
{
    CvKalman *kalman;

    if( !_kalman )
        CV_Error( CV_StsNullPtr, "" );

    kalman = *_kalman;
    if( !kalman )
        return;

    /* freeing the memory */
    cvReleaseMat( &kalman->state_pre );
    cvReleaseMat( &kalman->state_post );
    cvReleaseMat( &kalman->transition_matrix );
    cvReleaseMat( &kalman->control_matrix );
    cvReleaseMat( &kalman->measurement_matrix );
    cvReleaseMat( &kalman->process_noise_cov );
    cvReleaseMat( &kalman->measurement_noise_cov );
    cvReleaseMat( &kalman->error_cov_pre );
    cvReleaseMat( &kalman->gain );
    cvReleaseMat( &kalman->error_cov_post );
    cvReleaseMat( &kalman->temp1 );
    cvReleaseMat( &kalman->temp2 );
    cvReleaseMat( &kalman->temp3 );
    cvReleaseMat( &kalman->temp4 );
    cvReleaseMat( &kalman->temp5 );

    memset( kalman, 0, sizeof(*kalman));

    /* deallocating the structure */
    cvFree( _kalman );
}


CV_IMPL const CvMat*
cvKalmanPredict( CvKalman* kalman, const CvMat* control )
{
    if( !kalman )
        CV_Error( CV_StsNullPtr, "" );

    /* update the state */
    /* x'(k) = A*x(k) */
    cvMatMulAdd( kalman->transition_matrix, kalman->state_post, 0, kalman->state_pre );

    if( control && kalman->CP > 0 )
        /* x'(k) = x'(k) + B*u(k) */
        cvMatMulAdd( kalman->control_matrix, control, kalman->state_pre, kalman->state_pre );

    /* update error covariance matrices */
    /* temp1 = A*P(k) */
    cvMatMulAdd( kalman->transition_matrix, kalman->error_cov_post, 0, kalman->temp1 );

    /* P'(k) = temp1*At + Q */
    cvGEMM( kalman->temp1, kalman->transition_matrix, 1, kalman->process_noise_cov, 1,
                     kalman->error_cov_pre, CV_GEMM_B_T );

    /* handle the case when there will be measurement before the next predict */
    cvCopy(kalman->state_pre, kalman->state_post);

    return kalman->state_pre;
}


CV_IMPL const CvMat*
cvKalmanCorrect( CvKalman* kalman, const CvMat* measurement )
{
    if( !kalman || !measurement )
        CV_Error( CV_StsNullPtr, "" );

    /* temp2 = H*P'(k) */
    cvMatMulAdd( kalman->measurement_matrix, kalman->error_cov_pre, 0, kalman->temp2 );
    /* temp3 = temp2*Ht + R */
    cvGEMM( kalman->temp2, kalman->measurement_matrix, 1,
            kalman->measurement_noise_cov, 1, kalman->temp3, CV_GEMM_B_T );

    /* temp4 = inv(temp3)*temp2 = Kt(k) */
    cvSolve( kalman->temp3, kalman->temp2, kalman->temp4, CV_SVD );

    /* K(k) */
    cvTranspose( kalman->temp4, kalman->gain );

    /* temp5 = z(k) - H*x'(k) */
    cvGEMM( kalman->measurement_matrix, kalman->state_pre, -1, measurement, 1, kalman->temp5 );

    /* x(k) = x'(k) + K(k)*temp5 */
    cvMatMulAdd( kalman->gain, kalman->temp5, kalman->state_pre, kalman->state_post );

    /* P(k) = P'(k) - K(k)*temp2 */
    cvGEMM( kalman->gain, kalman->temp2, -1, kalman->error_cov_pre, 1,
                     kalman->error_cov_post, 0 );

    return kalman->state_post;
}

///////////////////////////////////// Optical Flow ////////////////////////////////

CV_IMPL void
cvCalcOpticalFlowPyrLK( const void* arrA, const void* arrB,
                        void* /*pyrarrA*/, void* /*pyrarrB*/,
                        const CvPoint2D32f * featuresA,
                        CvPoint2D32f * featuresB,
                        int count, CvSize winSize, int level,
                        char *status, float *error,
                        CvTermCriteria criteria, int flags )
{
    if( count <= 0 )
        return;
    CV_Assert( featuresA && featuresB );
    cv::Mat A = cv::cvarrToMat(arrA), B = cv::cvarrToMat(arrB);
    cv::Mat ptA(count, 1, CV_32FC2, (void*)featuresA);
    cv::Mat ptB(count, 1, CV_32FC2, (void*)featuresB);
    cv::Mat st, err;

    if( status )
        st = cv::Mat(count, 1, CV_8U, (void*)status);
    if( error )
        err = cv::Mat(count, 1, CV_32F, (void*)error);
    cv::calcOpticalFlowPyrLK( A, B, ptA, ptB, st,
                              error ? cv::_OutputArray(err) : (cv::_OutputArray)cv::noArray(),
                              winSize, level, criteria, flags);
}


CV_IMPL void cvCalcOpticalFlowFarneback(
            const CvArr* _prev, const CvArr* _next,
            CvArr* _flow, double pyr_scale, int levels,
            int winsize, int iterations, int poly_n,
            double poly_sigma, int flags )
{
    cv::Mat prev = cv::cvarrToMat(_prev), next = cv::cvarrToMat(_next);
    cv::Mat flow = cv::cvarrToMat(_flow);
    CV_Assert( flow.size() == prev.size() && flow.type() == CV_32FC2 );
    cv::calcOpticalFlowFarneback( prev, next, flow, pyr_scale, levels,
        winsize, iterations, poly_n, poly_sigma, flags );
}


CV_IMPL int
cvEstimateRigidTransform( const CvArr* arrA, const CvArr* arrB, CvMat* arrM, int full_affine )
{
    cv::Mat matA = cv::cvarrToMat(arrA), matB = cv::cvarrToMat(arrB);
    const cv::Mat matM0 = cv::cvarrToMat(arrM);

    cv::Mat matM = cv::estimateRigidTransform(matA, matB, full_affine != 0);
    if( matM.empty() )
    {
        matM = cv::cvarrToMat(arrM);
        matM.setTo(cv::Scalar::all(0));
        return 0;
    }
    matM.convertTo(matM0, matM0.type());
    return 1;
}
