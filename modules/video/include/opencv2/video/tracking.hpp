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

#ifndef __OPENCV_TRACKING_HPP__
#define __OPENCV_TRACKING_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

enum { OPTFLOW_USE_INITIAL_FLOW     = 4,
       OPTFLOW_LK_GET_MIN_EIGENVALS = 8,
       OPTFLOW_FARNEBACK_GAUSSIAN   = 256
     };

//! updates the object tracking window using CAMSHIFT algorithm
CV_EXPORTS_W RotatedRect CamShift( InputArray probImage, CV_IN_OUT Rect& window,
                                   TermCriteria criteria );

//! updates the object tracking window using meanshift algorithm
CV_EXPORTS_W int meanShift( InputArray probImage, CV_IN_OUT Rect& window, TermCriteria criteria );

//! constructs a pyramid which can be used as input for calcOpticalFlowPyrLK
CV_EXPORTS_W int buildOpticalFlowPyramid( InputArray img, OutputArrayOfArrays pyramid,
                                          Size winSize, int maxLevel, bool withDerivatives = true,
                                          int pyrBorder = BORDER_REFLECT_101,
                                          int derivBorder = BORDER_CONSTANT,
                                          bool tryReuseInputImage = true );

//! computes sparse optical flow using multi-scale Lucas-Kanade algorithm
CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                                        InputArray prevPts, InputOutputArray nextPts,
                                        OutputArray status, OutputArray err,
                                        Size winSize = Size(21,21), int maxLevel = 3,
                                        TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                        int flags = 0, double minEigThreshold = 1e-4 );

//! computes dense optical flow using Farneback algorithm
CV_EXPORTS_W void calcOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
                                            double pyr_scale, int levels, int winsize,
                                            int iterations, int poly_n, double poly_sigma,
                                            int flags );

//! estimates the best-fit Euqcidean, similarity, affine or perspective transformation
// that maps one 2D point set to another or one image to another.
CV_EXPORTS_W Mat estimateRigidTransform( InputArray src, InputArray dst, bool fullAffine );


enum
{
    MOTION_TRANSLATION = 0,
    MOTION_EUCLIDEAN   = 1,
    MOTION_AFFINE      = 2,
    MOTION_HOMOGRAPHY  = 3
};

//! estimates the best-fit Translation, Euclidean, Affine or Perspective Transformation
// with respect to Enhanced Correlation Coefficient criterion that maps one image to
// another (area-based alignment)
//
// see reference:
// Evangelidis, G. E., Psarakis, E.Z., Parametric Image Alignment using
// Enhanced Correlation Coefficient Maximization, PAMI, 30(8), 2008
CV_EXPORTS_W double findTransformECC( InputArray templateImage, InputArray inputImage,
                                      InputOutputArray warpMatrix, int motionType = MOTION_AFFINE,
                                      TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001));

/*!
 Kalman filter.

 The class implements standard Kalman filter http://en.wikipedia.org/wiki/Kalman_filter.
 However, you can modify KalmanFilter::transitionMatrix, KalmanFilter::controlMatrix and
 KalmanFilter::measurementMatrix to get the extended Kalman filter functionality.
*/
class CV_EXPORTS_W KalmanFilter
{
public:
    //! the default constructor
    CV_WRAP KalmanFilter();
    //! the full constructor taking the dimensionality of the state, of the measurement and of the control vector
    CV_WRAP KalmanFilter( int dynamParams, int measureParams, int controlParams = 0, int type = CV_32F );
    //! re-initializes Kalman filter. The previous content is destroyed.
    void init( int dynamParams, int measureParams, int controlParams = 0, int type = CV_32F );

    //! computes predicted state
    CV_WRAP const Mat& predict( const Mat& control = Mat() );
    //! updates the predicted state from the measurement
    CV_WRAP const Mat& correct( const Mat& measurement );

    CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
    CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
    CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
    CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
    CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
    CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/
    CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
    CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)

    // temporary matrices
    Mat temp1;
    Mat temp2;
    Mat temp3;
    Mat temp4;
    Mat temp5;
};



class CV_EXPORTS_W DenseOpticalFlow : public Algorithm
{
public:
    CV_WRAP virtual void calc( InputArray I0, InputArray I1, InputOutputArray flow ) = 0;
    CV_WRAP virtual void collectGarbage() = 0;
};

// Implementation of the Zach, Pock and Bischof Dual TV-L1 Optical Flow method
//
// see reference:
//   [1] C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
//   [2] Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_DualTVL1();

} // cv

#endif
