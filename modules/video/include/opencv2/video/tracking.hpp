/*! \file tracking.hpp
 \brief The Object and Feature Tracking
 */

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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_TRACKING_HPP__
#define __OPENCV_TRACKING_HPP__

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************************\
*                                  Motion Analysis                                       *
\****************************************************************************************/

/************************************ optical flow ***************************************/

/* Calculates optical flow for 2 images using classical Lucas & Kanade algorithm */
CVAPI(void)  cvCalcOpticalFlowLK( const CvArr* prev, const CvArr* curr,
                                  CvSize win_size, CvArr* velx, CvArr* vely );

/* Calculates optical flow for 2 images using block matching algorithm */
CVAPI(void)  cvCalcOpticalFlowBM( const CvArr* prev, const CvArr* curr,
                                  CvSize block_size, CvSize shift_size,
                                  CvSize max_range, int use_previous,
                                  CvArr* velx, CvArr* vely );

/* Calculates Optical flow for 2 images using Horn & Schunck algorithm */
CVAPI(void)  cvCalcOpticalFlowHS( const CvArr* prev, const CvArr* curr,
                                  int use_previous, CvArr* velx, CvArr* vely,
                                  double lambda, CvTermCriteria criteria );

#define  CV_LKFLOW_PYR_A_READY       1
#define  CV_LKFLOW_PYR_B_READY       2
#define  CV_LKFLOW_INITIAL_GUESSES   4
#define  CV_LKFLOW_GET_MIN_EIGENVALS 8

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
    
#ifdef __cplusplus
}

namespace cv
{
    
//! updates motion history image using the current silhouette
CV_EXPORTS_W void updateMotionHistory( InputArray silhouette, InputOutputArray mhi,
                                       double timestamp, double duration );

//! computes the motion gradient orientation image from the motion history image
CV_EXPORTS_W void calcMotionGradient( InputArray mhi, OutputArray mask,
                                      OutputArray orientation,
                                      double delta1, double delta2,
                                      int apertureSize=3 );

//! computes the global orientation of the selected motion history image part
CV_EXPORTS_W double calcGlobalOrientation( InputArray orientation, InputArray mask,
                                           InputArray mhi, double timestamp,
                                           double duration );

CV_EXPORTS_W void segmentMotion(InputArray mhi, OutputArray segmask,
                                vector<Rect>& boundingRects,
                                double timestamp, double segThresh);

//! updates the object tracking window using CAMSHIFT algorithm
CV_EXPORTS_W RotatedRect CamShift( InputArray probImage, CV_IN_OUT Rect& window,
                                   TermCriteria criteria );

//! updates the object tracking window using meanshift algorithm
CV_EXPORTS_W int meanShift( InputArray probImage, CV_IN_OUT Rect& window,
                            TermCriteria criteria );

/*!
 Kalman filter.
 
 The class implements standard Kalman filter \url{http://en.wikipedia.org/wiki/Kalman_filter}.
 However, you can modify KalmanFilter::transitionMatrix, KalmanFilter::controlMatrix and
 KalmanFilter::measurementMatrix to get the extended Kalman filter functionality.
*/
class CV_EXPORTS_W KalmanFilter
{
public:
    //! the default constructor
    CV_WRAP KalmanFilter();
    //! the full constructor taking the dimensionality of the state, of the measurement and of the control vector
    CV_WRAP KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);
    //! re-initializes Kalman filter. The previous content is destroyed.
    void init(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);

    //! computes predicted state
    CV_WRAP const Mat& predict(const Mat& control=Mat());
    //! updates the predicted state from the measurement
    CV_WRAP const Mat& correct(const Mat& measurement);

    Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    Mat transitionMatrix;   //!< state transition matrix (A)
    Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
    Mat measurementMatrix;  //!< measurement matrix (H)
    Mat processNoiseCov;    //!< process noise covariance matrix (Q)
    Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
    Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
    Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
    Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
    
    // temporary matrices
    Mat temp1;
    Mat temp2;
    Mat temp3;
    Mat temp4;
    Mat temp5;
};


enum { OPTFLOW_USE_INITIAL_FLOW=4, OPTFLOW_FARNEBACK_GAUSSIAN=256 };

//! computes sparse optical flow using multi-scale Lucas-Kanade algorithm
CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                           InputArray prevPts, CV_OUT InputOutputArray nextPts,
                           OutputArray status, OutputArray err,
                           Size winSize=Size(15,15), int maxLevel=3,
                           TermCriteria criteria=TermCriteria(
                            TermCriteria::COUNT+TermCriteria::EPS,
                            30, 0.01),
                           double derivLambda=0.5,
                           int flags=0 );

//! computes dense optical flow using Farneback algorithm
CV_EXPORTS_W void calcOpticalFlowFarneback( InputArray prev, InputArray next,
                           CV_OUT InputOutputArray flow, double pyr_scale, int levels, int winsize,
                           int iterations, int poly_n, double poly_sigma, int flags );

//! estimates the best-fit Euqcidean, similarity, affine or perspective transformation
// that maps one 2D point set to another or one image to another.
CV_EXPORTS_W Mat estimateRigidTransform( InputArray src, InputArray dst,
                                         bool fullAffine);
    
}

#endif

#endif
