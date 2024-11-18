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

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* OPENCV_CALIB3D_C_API_H */
