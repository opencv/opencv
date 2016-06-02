/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_TRACKING_C_H__
#define __OPENCV_TRACKING_C_H__

#include "opencv2/imgproc/types_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup video_c
  @{
*/

/****************************************************************************************\
*                                  Motion Analysis                                       *
\****************************************************************************************/

/************************************ optical flow ***************************************/

#define CV_LKFLOW_PYR_A_READY       1
#define CV_LKFLOW_PYR_B_READY       2
#define CV_LKFLOW_INITIAL_GUESSES   4
#define CV_LKFLOW_GET_MIN_EIGENVALS 8

/* It is Lucas & Kanade method, modified to use pyramids.
   Also it does several iterations to get optical flow for
   every point at every pyramid level.
   Calculates optical flow between two images for certain set of points (i.e.
   it is a "sparse" optical flow, which is opposite to the previous 3 methods) */
CVAPI(void)  cvCalcOpticalFlowPyrLK( const CvArr*  prev, const CvArr*  curr,
                                     CvArr*  prev_pyr, CvArr*  curr_pyr,
                                     const CvPoint2D32f* prev_features,
                                     CvPoint2D32f* curr_features,
                                     int       count,
                                     CvSize    win_size,
                                     int       level,
                                     char*     status,
                                     float*    track_error,
                                     CvTermCriteria criteria,
                                     int       flags );


/* Modification of a previous sparse optical flow algorithm to calculate
   affine flow */
CVAPI(void)  cvCalcAffineFlowPyrLK( const CvArr*  prev, const CvArr*  curr,
                                    CvArr*  prev_pyr, CvArr*  curr_pyr,
                                    const CvPoint2D32f* prev_features,
                                    CvPoint2D32f* curr_features,
                                    float* matrices, int  count,
                                    CvSize win_size, int  level,
                                    char* status, float* track_error,
                                    CvTermCriteria criteria, int flags );

/* Estimate rigid transformation between 2 images or 2 point sets */
CVAPI(int)  cvEstimateRigidTransform( const CvArr* A, const CvArr* B,
                                      CvMat* M, int full_affine );

/* Estimate optical flow for each pixel using the two-frame G. Farneback algorithm */
CVAPI(void) cvCalcOpticalFlowFarneback( const CvArr* prev, const CvArr* next,
                                        CvArr* flow, double pyr_scale, int levels,
                                        int winsize, int iterations, int poly_n,
                                        double poly_sigma, int flags );

/********************************* motion templates *************************************/

/****************************************************************************************\
*        All the motion template functions work only with single channel images.         *
*        Silhouette image must have depth IPL_DEPTH_8U or IPL_DEPTH_8S                   *
*        Motion history image must have depth IPL_DEPTH_32F,                             *
*        Gradient mask - IPL_DEPTH_8U or IPL_DEPTH_8S,                                   *
*        Motion orientation image - IPL_DEPTH_32F                                        *
*        Segmentation mask - IPL_DEPTH_32F                                               *
*        All the angles are in degrees, all the times are in milliseconds                *
\****************************************************************************************/

/* Updates motion history image given motion silhouette */
CVAPI(void)    cvUpdateMotionHistory( const CvArr* silhouette, CvArr* mhi,
                                      double timestamp, double duration );

/* Calculates gradient of the motion history image and fills
   a mask indicating where the gradient is valid */
CVAPI(void)    cvCalcMotionGradient( const CvArr* mhi, CvArr* mask, CvArr* orientation,
                                     double delta1, double delta2,
                                     int aperture_size CV_DEFAULT(3));

/* Calculates average motion direction within a selected motion region
   (region can be selected by setting ROIs and/or by composing a valid gradient mask
   with the region mask) */
CVAPI(double)  cvCalcGlobalOrientation( const CvArr* orientation, const CvArr* mask,
                                        const CvArr* mhi, double timestamp,
                                        double duration );

/* Splits a motion history image into a few parts corresponding to separate independent motions
   (e.g. left hand, right hand) */
CVAPI(CvSeq*)  cvSegmentMotion( const CvArr* mhi, CvArr* seg_mask,
                                CvMemStorage* storage,
                                double timestamp, double seg_thresh );

/****************************************************************************************\
*                                       Tracking                                         *
\****************************************************************************************/

/* Implements CAMSHIFT algorithm - determines object position, size and orientation
   from the object histogram back project (extension of meanshift) */
CVAPI(int)  cvCamShift( const CvArr* prob_image, CvRect  window,
                        CvTermCriteria criteria, CvConnectedComp* comp,
                        CvBox2D* box CV_DEFAULT(NULL) );

/* Implements MeanShift algorithm - determines object position
   from the object histogram back project */
CVAPI(int)  cvMeanShift( const CvArr* prob_image, CvRect  window,
                         CvTermCriteria criteria, CvConnectedComp* comp );

/*
standard Kalman filter (in G. Welch' and G. Bishop's notation):

  x(k)=A*x(k-1)+B*u(k)+w(k)  p(w)~N(0,Q)
  z(k)=H*x(k)+v(k),   p(v)~N(0,R)
*/
typedef struct CvKalman
{
    int MP;                     /* number of measurement vector dimensions */
    int DP;                     /* number of state vector dimensions */
    int CP;                     /* number of control vector dimensions */

    /* backward compatibility fields */
#if 1
    float* PosterState;         /* =state_pre->data.fl */
    float* PriorState;          /* =state_post->data.fl */
    float* DynamMatr;           /* =transition_matrix->data.fl */
    float* MeasurementMatr;     /* =measurement_matrix->data.fl */
    float* MNCovariance;        /* =measurement_noise_cov->data.fl */
    float* PNCovariance;        /* =process_noise_cov->data.fl */
    float* KalmGainMatr;        /* =gain->data.fl */
    float* PriorErrorCovariance;/* =error_cov_pre->data.fl */
    float* PosterErrorCovariance;/* =error_cov_post->data.fl */
    float* Temp1;               /* temp1->data.fl */
    float* Temp2;               /* temp2->data.fl */
#endif

    CvMat* state_pre;           /* predicted state (x'(k)):
                                    x(k)=A*x(k-1)+B*u(k) */
    CvMat* state_post;          /* corrected state (x(k)):
                                    x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) */
    CvMat* transition_matrix;   /* state transition matrix (A) */
    CvMat* control_matrix;      /* control matrix (B)
                                   (it is not used if there is no control)*/
    CvMat* measurement_matrix;  /* measurement matrix (H) */
    CvMat* process_noise_cov;   /* process noise covariance matrix (Q) */
    CvMat* measurement_noise_cov; /* measurement noise covariance matrix (R) */
    CvMat* error_cov_pre;       /* priori error estimate covariance matrix (P'(k)):
                                    P'(k)=A*P(k-1)*At + Q)*/
    CvMat* gain;                /* Kalman gain matrix (K(k)):
                                    K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)*/
    CvMat* error_cov_post;      /* posteriori error estimate covariance matrix (P(k)):
                                    P(k)=(I-K(k)*H)*P'(k) */
    CvMat* temp1;               /* temporary matrices */
    CvMat* temp2;
    CvMat* temp3;
    CvMat* temp4;
    CvMat* temp5;
} CvKalman;

/* Creates Kalman filter and sets A, B, Q, R and state to some initial values */
CVAPI(CvKalman*) cvCreateKalman( int dynam_params, int measure_params,
                                 int control_params CV_DEFAULT(0));

/* Releases Kalman filter state */
CVAPI(void)  cvReleaseKalman( CvKalman** kalman);

/* Updates Kalman filter by time (predicts future state of the system) */
CVAPI(const CvMat*)  cvKalmanPredict( CvKalman* kalman,
                                      const CvMat* control CV_DEFAULT(NULL));

/* Updates Kalman filter by measurement
   (corrects state of the system and internal matrices) */
CVAPI(const CvMat*)  cvKalmanCorrect( CvKalman* kalman, const CvMat* measurement );

#define cvKalmanUpdateByTime  cvKalmanPredict
#define cvKalmanUpdateByMeasurement cvKalmanCorrect

/** @} video_c */

#ifdef __cplusplus
} // extern "C"
#endif


#endif // __OPENCV_TRACKING_C_H__
