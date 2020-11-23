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
//#include "opencv2/calib3d/calib3d_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup calib3d_c
  @{
  */

/****************************************************************************************\
*                      Camera Calibration, Pose Estimation and Stereo                    *
\****************************************************************************************/

typedef struct CvPOSITObject CvPOSITObject;

/* Allocates and initializes CvPOSITObject structure before doing cvPOSIT */
CvPOSITObject*  cvCreatePOSITObject( CvPoint3D32f* points, int point_count );


/* Runs POSIT (POSe from ITeration) algorithm for determining 3d position of
   an object given its model and projection in a weak-perspective case */
void  cvPOSIT(  CvPOSITObject* posit_object, CvPoint2D32f* image_points,
                       double focal_length, CvTermCriteria criteria,
                       float* rotation_matrix, float* translation_vector);

/* Releases CvPOSITObject structure */
void  cvReleasePOSITObject( CvPOSITObject**  posit_object );

/* updates the number of RANSAC iterations */
int cvRANSACUpdateNumIters( double p, double err_prob,
                                   int model_points, int max_iters );

void cvConvertPointsHomogeneous( const CvMat* src, CvMat* dst );

/* Calculates fundamental matrix given a set of corresponding points */
/*#define CV_FM_7POINT 1
#define CV_FM_8POINT 2

#define CV_LMEDS 4
#define CV_RANSAC 8

#define CV_FM_LMEDS_ONLY  CV_LMEDS
#define CV_FM_RANSAC_ONLY CV_RANSAC
#define CV_FM_LMEDS CV_LMEDS
#define CV_FM_RANSAC CV_RANSAC*/

int cvFindFundamentalMat( const CvMat* points1, const CvMat* points2,
                          CvMat* fundamental_matrix,
                          int method CV_DEFAULT(CV_FM_RANSAC),
                          double param1 CV_DEFAULT(3.), double param2 CV_DEFAULT(0.99),
                          CvMat* status CV_DEFAULT(NULL) );

/* For each input point on one of images
   computes parameters of the corresponding
   epipolar line on the other image */
void cvComputeCorrespondEpilines( const CvMat* points,
                                  int which_image,
                                  const CvMat* fundamental_matrix,
                                  CvMat* correspondent_lines );

/* Triangulation functions */

void cvTriangulatePoints(CvMat* projMatr1, CvMat* projMatr2,
                         CvMat* projPoints1, CvMat* projPoints2,
                         CvMat* points4D);

void cvCorrectMatches(CvMat* F, CvMat* points1, CvMat* points2,
                      CvMat* new_points1, CvMat* new_points2);


/* Computes the optimal new camera matrix according to the free scaling parameter alpha:
   alpha=0 - only valid pixels will be retained in the undistorted image
   alpha=1 - all the source image pixels will be retained in the undistorted image
*/
void cvGetOptimalNewCameraMatrix( const CvMat* camera_matrix,
                                  const CvMat* dist_coeffs,
                                  CvSize image_size, double alpha,
                                  CvMat* new_camera_matrix,
                                  CvSize new_imag_size CV_DEFAULT(cvSize(0,0)),
                                  CvRect* valid_pixel_ROI CV_DEFAULT(0),
                                  int center_principal_point CV_DEFAULT(0));

/* Converts rotation vector to rotation matrix or vice versa */
int cvRodrigues2( const CvMat* src, CvMat* dst,
                  CvMat* jacobian CV_DEFAULT(0) );

/* Finds perspective transformation between the object plane and image (view) plane */
int cvFindHomography( const CvMat* src_points,
                      const CvMat* dst_points,
                      CvMat* homography,
                      int method CV_DEFAULT(0),
                      double ransacReprojThreshold CV_DEFAULT(3),
                      CvMat* mask CV_DEFAULT(0),
                      int maxIters CV_DEFAULT(2000),
                      double confidence CV_DEFAULT(0.995));

/* Computes RQ decomposition for 3x3 matrices */
void cvRQDecomp3x3( const CvMat *matrixM, CvMat *matrixR, CvMat *matrixQ,
                    CvMat *matrixQx CV_DEFAULT(NULL),
                    CvMat *matrixQy CV_DEFAULT(NULL),
                    CvMat *matrixQz CV_DEFAULT(NULL),
                    CvPoint3D64f *eulerAngles CV_DEFAULT(NULL));

/* Computes projection matrix decomposition */
void cvDecomposeProjectionMatrix( const CvMat *projMatr, CvMat *calibMatr,
                                  CvMat *rotMatr, CvMat *posVect,
                                  CvMat *rotMatrX CV_DEFAULT(NULL),
                                  CvMat *rotMatrY CV_DEFAULT(NULL),
                                  CvMat *rotMatrZ CV_DEFAULT(NULL),
                                  CvPoint3D64f *eulerAngles CV_DEFAULT(NULL));

/* Computes d(AB)/dA and d(AB)/dB */
void cvCalcMatMulDeriv( const CvMat* A, const CvMat* B, CvMat* dABdA, CvMat* dABdB );

/* Computes r3 = rodrigues(rodrigues(r2)*rodrigues(r1)),
   t3 = rodrigues(r2)*t1 + t2 and the respective derivatives */
void cvComposeRT( const CvMat* _rvec1, const CvMat* _tvec1,
                  const CvMat* _rvec2, const CvMat* _tvec2,
                  CvMat* _rvec3, CvMat* _tvec3,
                  CvMat* dr3dr1 CV_DEFAULT(0), CvMat* dr3dt1 CV_DEFAULT(0),
                  CvMat* dr3dr2 CV_DEFAULT(0), CvMat* dr3dt2 CV_DEFAULT(0),
                  CvMat* dt3dr1 CV_DEFAULT(0), CvMat* dt3dt1 CV_DEFAULT(0),
                  CvMat* dt3dr2 CV_DEFAULT(0), CvMat* dt3dt2 CV_DEFAULT(0) );

/* Projects object points to the view plane using
   the specified extrinsic and intrinsic camera parameters */
void cvProjectPoints2( const CvMat* object_points, const CvMat* rotation_vector,
                       const CvMat* translation_vector, const CvMat* camera_matrix,
                       const CvMat* distortion_coeffs, CvMat* image_points,
                       CvMat* dpdrot CV_DEFAULT(NULL), CvMat* dpdt CV_DEFAULT(NULL),
                       CvMat* dpdf CV_DEFAULT(NULL), CvMat* dpdc CV_DEFAULT(NULL),
                       CvMat* dpddist CV_DEFAULT(NULL),
                       double aspect_ratio CV_DEFAULT(0));

/* Finds extrinsic camera parameters from
   a few known corresponding point pairs and intrinsic parameters */
void cvFindExtrinsicCameraParams2( const CvMat* object_points,
                                const CvMat* image_points,
                                const CvMat* camera_matrix,
                                const CvMat* distortion_coeffs,
                                CvMat* rotation_vector,
                                CvMat* translation_vector,
                                int use_extrinsic_guess CV_DEFAULT(0) );

/* Computes various useful characteristics of the camera from the data computed by
   cvCalibrateCamera2 */
void cvCalibrationMatrixValues( const CvMat *camera_matrix,
                                CvSize image_size,
                                double aperture_width CV_DEFAULT(0),
                                double aperture_height CV_DEFAULT(0),
                                double *fovx CV_DEFAULT(NULL),
                                double *fovy CV_DEFAULT(NULL),
                                double *focal_length CV_DEFAULT(NULL),
                                CvPoint2D64f *principal_point CV_DEFAULT(NULL),
                                double *pixel_aspect_ratio CV_DEFAULT(NULL));

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

CvRect cvGetValidDisparityROI( CvRect roi1, CvRect roi2, int minDisparity,
                               int numberOfDisparities, int SADWindowSize );

void cvValidateDisparity( CvArr* disparity, const CvArr* cost,
                          int minDisparity, int numberOfDisparities,
                          int disp12MaxDiff CV_DEFAULT(1) );

/** @brief Transforms the input image to compensate lens distortion
@see cv::undistort
*/
void cvUndistort2( const CvArr* src, CvArr* dst,
                   const CvMat* camera_matrix,
                   const CvMat* distortion_coeffs,
                   const CvMat* new_camera_matrix CV_DEFAULT(0) );

/** @brief Computes transformation map from intrinsic camera parameters
   that can used by cvRemap
*/
void cvInitUndistortMap( const CvMat* camera_matrix,
                         const CvMat* distortion_coeffs,
                         CvArr* mapx, CvArr* mapy );

/** @brief Computes undistortion+rectification map for a head of stereo camera
@see cv::initUndistortRectifyMap
*/
void cvInitUndistortRectifyMap( const CvMat* camera_matrix,
                                const CvMat* dist_coeffs,
                                const CvMat *R, const CvMat* new_camera_matrix,
                                CvArr* mapx, CvArr* mapy );

/** @brief Computes the original (undistorted) feature coordinates
   from the observed (distorted) coordinates
@see cv::undistortPoints
*/
void cvUndistortPoints( const CvMat* src, CvMat* dst,
                        const CvMat* camera_matrix,
                        const CvMat* dist_coeffs,
                        const CvMat* R CV_DEFAULT(0),
                        const CvMat* P CV_DEFAULT(0));

/** @} calib3d_c */

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* OPENCV_CALIB3D_C_API_H */
