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

#ifndef OPENCV_CALIB3D_C_API_H
#define OPENCV_CALIB3D_C_API_H

#include "opencv2/core/core_c.h"
#include "opencv2/calib3d/calib3d_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************************\
*                      Camera Calibration, Pose Estimation and Stereo                    *
\****************************************************************************************/

void cvConvertPointsHomogeneous( const CvMat* src, CvMat* dst );

/* For each input point on one of images
   computes parameters of the corresponding
   epipolar line on the other image */
void cvComputeCorrespondEpilines( const CvMat* points,
                                  int which_image,
                                  const CvMat* fundamental_matrix,
                                  CvMat* correspondent_lines );

/* Finds perspective transformation between the object plane and image (view) plane */
int cvFindHomography( const CvMat* src_points,
                      const CvMat* dst_points,
                      CvMat* homography,
                      int method CV_DEFAULT(0),
                      double ransacReprojThreshold CV_DEFAULT(3),
                      CvMat* mask CV_DEFAULT(0),
                      int maxIters CV_DEFAULT(2000),
                      double confidence CV_DEFAULT(0.995));

/* Computes initial estimate of the intrinsic camera parameters
   in case of planar calibration target (e.g. chessboard) */
void cvInitIntrinsicParams2D( const CvMat* object_points,
                              const CvMat* image_points,
                              const CvMat* npoints, CvSize image_size,
                              CvMat* camera_matrix,
                              double aspect_ratio CV_DEFAULT(1.) );

/* Finds intrinsic and extrinsic camera parameters
   from a few views of known calibration pattern */
double cvCalibrateCamera2( const CvMat* object_points,
                                const CvMat* image_points,
                                const CvMat* point_counts,
                                CvSize image_size,
                                CvMat* camera_matrix,
                                CvMat* distortion_coeffs,
                                CvMat* rotation_vectors CV_DEFAULT(NULL),
                                CvMat* translation_vectors CV_DEFAULT(NULL),
                                int flags CV_DEFAULT(0),
                                CvTermCriteria term_crit CV_DEFAULT(cvTermCriteria(
                                    CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,DBL_EPSILON)) );

/* Computes the transformation from one camera coordinate system to another one
   from a few correspondent views of the same calibration target. Optionally, calibrates
   both cameras */
double cvStereoCalibrate( const CvMat* object_points, const CvMat* image_points1,
                          const CvMat* image_points2, const CvMat* npoints,
                          CvMat* camera_matrix1, CvMat* dist_coeffs1,
                          CvMat* camera_matrix2, CvMat* dist_coeffs2,
                          CvSize image_size, CvMat* R, CvMat* T,
                          CvMat* E CV_DEFAULT(0), CvMat* F CV_DEFAULT(0),
                          int flags CV_DEFAULT(CV_CALIB_FIX_INTRINSIC),
                          CvTermCriteria term_crit CV_DEFAULT(cvTermCriteria(
                              CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,1e-6)) );

#define CV_CALIB_ZERO_DISPARITY 1024

/* Computes 3D rotations (+ optional shift) for each camera coordinate system to make both
   views parallel (=> to make all the epipolar lines horizontal or vertical) */
void cvStereoRectify( const CvMat* camera_matrix1, const CvMat* camera_matrix2,
                      const CvMat* dist_coeffs1, const CvMat* dist_coeffs2,
                      CvSize image_size, const CvMat* R, const CvMat* T,
                      CvMat* R1, CvMat* R2, CvMat* P1, CvMat* P2,
                      CvMat* Q CV_DEFAULT(0),
                      int flags CV_DEFAULT(CV_CALIB_ZERO_DISPARITY),
                      double alpha CV_DEFAULT(-1),
                      CvSize new_image_size CV_DEFAULT(cvSize(0,0)),
                      CvRect* valid_pix_ROI1 CV_DEFAULT(0),
                      CvRect* valid_pix_ROI2 CV_DEFAULT(0));

/* Computes rectification transformations for uncalibrated pair of images using a set
   of point correspondences */
int cvStereoRectifyUncalibrated( const CvMat* points1, const CvMat* points2,
                                 const CvMat* F, CvSize img_size,
                                 CvMat* H1, CvMat* H2,
                                 double threshold CV_DEFAULT(5));

/** @brief Computes the original (undistorted) feature coordinates
   from the observed (distorted) coordinates
@see cv::undistortPoints
*/
void cvUndistortPoints( const CvMat* src, CvMat* dst,
                        const CvMat* camera_matrix,
                        const CvMat* dist_coeffs,
                        const CvMat* R CV_DEFAULT(0),
                        const CvMat* P CV_DEFAULT(0));

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* OPENCV_CALIB3D_C_API_H */
