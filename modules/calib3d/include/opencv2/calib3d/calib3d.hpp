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

#ifndef __OPENCV_CALIB3D_HPP__
#define __OPENCV_CALIB3D_HPP__

#include "opencv2/core/core.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************************\
*                      Camera Calibration, Pose Estimation and Stereo                    *
\****************************************************************************************/

typedef struct CvPOSITObject CvPOSITObject;

/* Allocates and initializes CvPOSITObject structure before doing cvPOSIT */
CVAPI(CvPOSITObject*)  cvCreatePOSITObject( CvPoint3D32f* points, int point_count );


/* Runs POSIT (POSe from ITeration) algorithm for determining 3d position of
   an object given its model and projection in a weak-perspective case */
CVAPI(void)  cvPOSIT(  CvPOSITObject* posit_object, CvPoint2D32f* image_points,
                       double focal_length, CvTermCriteria criteria,
                       float* rotation_matrix, float* translation_vector);

/* Releases CvPOSITObject structure */
CVAPI(void)  cvReleasePOSITObject( CvPOSITObject**  posit_object );

/* updates the number of RANSAC iterations */
CVAPI(int) cvRANSACUpdateNumIters( double p, double err_prob,
                                   int model_points, int max_iters );

CVAPI(void) cvConvertPointsHomogeneous( const CvMat* src, CvMat* dst );

/* Calculates fundamental matrix given a set of corresponding points */
#define CV_FM_7POINT 1
#define CV_FM_8POINT 2
#define CV_FM_LMEDS_ONLY  8
#define CV_FM_RANSAC_ONLY 4
#define CV_FM_LMEDS 8
#define CV_FM_RANSAC 4
CVAPI(int) cvFindFundamentalMat( const CvMat* points1, const CvMat* points2,
                                 CvMat* fundamental_matrix,
                                 int method CV_DEFAULT(CV_FM_RANSAC),
                                 double param1 CV_DEFAULT(3.), double param2 CV_DEFAULT(0.99),
                                 CvMat* status CV_DEFAULT(NULL) );

/* For each input point on one of images
   computes parameters of the corresponding
   epipolar line on the other image */
CVAPI(void) cvComputeCorrespondEpilines( const CvMat* points,
                                         int which_image,
                                         const CvMat* fundamental_matrix,
                                         CvMat* correspondent_lines );

/* Triangulation functions */

CVAPI(void) cvTriangulatePoints(CvMat* projMatr1, CvMat* projMatr2,
                                CvMat* projPoints1, CvMat* projPoints2,
                                CvMat* points4D);

CVAPI(void) cvCorrectMatches(CvMat* F, CvMat* points1, CvMat* points2,
                             CvMat* new_points1, CvMat* new_points2);

    
/* Computes the optimal new camera matrix according to the free scaling parameter alpha:
   alpha=0 - only valid pixels will be retained in the undistorted image
   alpha=1 - all the source image pixels will be retained in the undistorted image
*/
CVAPI(void) cvGetOptimalNewCameraMatrix( const CvMat* camera_matrix,
                                         const CvMat* dist_coeffs,
                                         CvSize image_size, double alpha,
                                         CvMat* new_camera_matrix,
                                         CvSize new_imag_size CV_DEFAULT(cvSize(0,0)),
                                         CvRect* valid_pixel_ROI CV_DEFAULT(0) );

/* Converts rotation vector to rotation matrix or vice versa */
CVAPI(int) cvRodrigues2( const CvMat* src, CvMat* dst,
                         CvMat* jacobian CV_DEFAULT(0) );

#define CV_LMEDS 4
#define CV_RANSAC 8

/* Finds perspective transformation between the object plane and image (view) plane */
CVAPI(int) cvFindHomography( const CvMat* src_points,
                             const CvMat* dst_points,
                             CvMat* homography,
                             int method CV_DEFAULT(0),
                             double ransacReprojThreshold CV_DEFAULT(3),
                             CvMat* mask CV_DEFAULT(0));

/* Computes RQ decomposition for 3x3 matrices */
CVAPI(void) cvRQDecomp3x3( const CvMat *matrixM, CvMat *matrixR, CvMat *matrixQ,
                           CvMat *matrixQx CV_DEFAULT(NULL),
                           CvMat *matrixQy CV_DEFAULT(NULL),
                           CvMat *matrixQz CV_DEFAULT(NULL),
                           CvPoint3D64f *eulerAngles CV_DEFAULT(NULL));

/* Computes projection matrix decomposition */
CVAPI(void) cvDecomposeProjectionMatrix( const CvMat *projMatr, CvMat *calibMatr,
                                         CvMat *rotMatr, CvMat *posVect,
                                         CvMat *rotMatrX CV_DEFAULT(NULL),
                                         CvMat *rotMatrY CV_DEFAULT(NULL),
                                         CvMat *rotMatrZ CV_DEFAULT(NULL),
                                         CvPoint3D64f *eulerAngles CV_DEFAULT(NULL));

/* Computes d(AB)/dA and d(AB)/dB */
CVAPI(void) cvCalcMatMulDeriv( const CvMat* A, const CvMat* B, CvMat* dABdA, CvMat* dABdB );

/* Computes r3 = rodrigues(rodrigues(r2)*rodrigues(r1)),
   t3 = rodrigues(r2)*t1 + t2 and the respective derivatives */
CVAPI(void) cvComposeRT( const CvMat* _rvec1, const CvMat* _tvec1,
                         const CvMat* _rvec2, const CvMat* _tvec2,
                         CvMat* _rvec3, CvMat* _tvec3,
                         CvMat* dr3dr1 CV_DEFAULT(0), CvMat* dr3dt1 CV_DEFAULT(0),
                         CvMat* dr3dr2 CV_DEFAULT(0), CvMat* dr3dt2 CV_DEFAULT(0),
                         CvMat* dt3dr1 CV_DEFAULT(0), CvMat* dt3dt1 CV_DEFAULT(0),
                         CvMat* dt3dr2 CV_DEFAULT(0), CvMat* dt3dt2 CV_DEFAULT(0) );

/* Projects object points to the view plane using
   the specified extrinsic and intrinsic camera parameters */
CVAPI(void) cvProjectPoints2( const CvMat* object_points, const CvMat* rotation_vector,
                              const CvMat* translation_vector, const CvMat* camera_matrix,
                              const CvMat* distortion_coeffs, CvMat* image_points,
                              CvMat* dpdrot CV_DEFAULT(NULL), CvMat* dpdt CV_DEFAULT(NULL),
                              CvMat* dpdf CV_DEFAULT(NULL), CvMat* dpdc CV_DEFAULT(NULL),
                              CvMat* dpddist CV_DEFAULT(NULL),
                              double aspect_ratio CV_DEFAULT(0));

/* Finds extrinsic camera parameters from
   a few known corresponding point pairs and intrinsic parameters */
CVAPI(void) cvFindExtrinsicCameraParams2( const CvMat* object_points,
                                          const CvMat* image_points,
                                          const CvMat* camera_matrix,
                                          const CvMat* distortion_coeffs,
                                          CvMat* rotation_vector,
                                          CvMat* translation_vector,
                                          int use_extrinsic_guess CV_DEFAULT(0) );

/* Computes initial estimate of the intrinsic camera parameters
   in case of planar calibration target (e.g. chessboard) */
CVAPI(void) cvInitIntrinsicParams2D( const CvMat* object_points,
                                     const CvMat* image_points,
                                     const CvMat* npoints, CvSize image_size,
                                     CvMat* camera_matrix,
                                     double aspect_ratio CV_DEFAULT(1.) );

#define CV_CALIB_CB_ADAPTIVE_THRESH  1
#define CV_CALIB_CB_NORMALIZE_IMAGE  2
#define CV_CALIB_CB_FILTER_QUADS     4
#define CV_CALIB_CB_FAST_CHECK       8

// Performs a fast check if a chessboard is in the input image. This is a workaround to 
// a problem of cvFindChessboardCorners being slow on images with no chessboard
// - src: input image
// - size: chessboard size
// Returns 1 if a chessboard can be in this image and findChessboardCorners should be called, 
// 0 if there is no chessboard, -1 in case of error
CVAPI(int) cvCheckChessboard(IplImage* src, CvSize size);
    
    /* Detects corners on a chessboard calibration pattern */
CVAPI(int) cvFindChessboardCorners( const void* image, CvSize pattern_size,
                                    CvPoint2D32f* corners,
                                    int* corner_count CV_DEFAULT(NULL),
                                    int flags CV_DEFAULT(CV_CALIB_CB_ADAPTIVE_THRESH+
                                        CV_CALIB_CB_NORMALIZE_IMAGE) );

/* Draws individual chessboard corners or the whole chessboard detected */
CVAPI(void) cvDrawChessboardCorners( CvArr* image, CvSize pattern_size,
                                     CvPoint2D32f* corners,
                                     int count, int pattern_was_found );

#define CV_CALIB_USE_INTRINSIC_GUESS  1
#define CV_CALIB_FIX_ASPECT_RATIO     2
#define CV_CALIB_FIX_PRINCIPAL_POINT  4
#define CV_CALIB_ZERO_TANGENT_DIST    8
#define CV_CALIB_FIX_FOCAL_LENGTH 16
#define CV_CALIB_FIX_K1  32
#define CV_CALIB_FIX_K2  64
#define CV_CALIB_FIX_K3  128
#define CV_CALIB_FIX_K4  2048
#define CV_CALIB_FIX_K5  4096
#define CV_CALIB_FIX_K6  8192
#define CV_CALIB_RATIONAL_MODEL 16384

/* Finds intrinsic and extrinsic camera parameters
   from a few views of known calibration pattern */
CVAPI(double) cvCalibrateCamera2( const CvMat* object_points,
                                const CvMat* image_points,
                                const CvMat* point_counts,
                                CvSize image_size,
                                CvMat* camera_matrix,
                                CvMat* distortion_coeffs,
                                CvMat* rotation_vectors CV_DEFAULT(NULL),
                                CvMat* translation_vectors CV_DEFAULT(NULL),
                                int flags CV_DEFAULT(0) );

/* Computes various useful characteristics of the camera from the data computed by
   cvCalibrateCamera2 */
CVAPI(void) cvCalibrationMatrixValues( const CvMat *camera_matrix,
                                CvSize image_size,
                                double aperture_width CV_DEFAULT(0),
                                double aperture_height CV_DEFAULT(0),
                                double *fovx CV_DEFAULT(NULL),
                                double *fovy CV_DEFAULT(NULL),
                                double *focal_length CV_DEFAULT(NULL),
                                CvPoint2D64f *principal_point CV_DEFAULT(NULL),
                                double *pixel_aspect_ratio CV_DEFAULT(NULL));

#define CV_CALIB_FIX_INTRINSIC  256
#define CV_CALIB_SAME_FOCAL_LENGTH 512

/* Computes the transformation from one camera coordinate system to another one
   from a few correspondent views of the same calibration target. Optionally, calibrates
   both cameras */
CVAPI(double) cvStereoCalibrate( const CvMat* object_points, const CvMat* image_points1,
                               const CvMat* image_points2, const CvMat* npoints,
                               CvMat* camera_matrix1, CvMat* dist_coeffs1,
                               CvMat* camera_matrix2, CvMat* dist_coeffs2,
                               CvSize image_size, CvMat* R, CvMat* T,
                               CvMat* E CV_DEFAULT(0), CvMat* F CV_DEFAULT(0),
                               CvTermCriteria term_crit CV_DEFAULT(cvTermCriteria(
                                   CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,1e-6)),
                               int flags CV_DEFAULT(CV_CALIB_FIX_INTRINSIC));

#define CV_CALIB_ZERO_DISPARITY 1024

/* Computes 3D rotations (+ optional shift) for each camera coordinate system to make both
   views parallel (=> to make all the epipolar lines horizontal or vertical) */
CVAPI(void) cvStereoRectify( const CvMat* camera_matrix1, const CvMat* camera_matrix2,
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
CVAPI(int) cvStereoRectifyUncalibrated( const CvMat* points1, const CvMat* points2,
                                        const CvMat* F, CvSize img_size,
                                        CvMat* H1, CvMat* H2,
                                        double threshold CV_DEFAULT(5));



/* stereo correspondence parameters and functions */

#define CV_STEREO_BM_NORMALIZED_RESPONSE  0
#define CV_STEREO_BM_XSOBEL               1

/* Block matching algorithm structure */
typedef struct CvStereoBMState
{
    // pre-filtering (normalization of input images)
    int preFilterType; // =CV_STEREO_BM_NORMALIZED_RESPONSE now
    int preFilterSize; // averaging window size: ~5x5..21x21
    int preFilterCap; // the output of pre-filtering is clipped by [-preFilterCap,preFilterCap]

    // correspondence using Sum of Absolute Difference (SAD)
    int SADWindowSize; // ~5x5..21x21
    int minDisparity;  // minimum disparity (can be negative)
    int numberOfDisparities; // maximum disparity - minimum disparity (> 0)

    // post-filtering
    int textureThreshold;  // the disparity is only computed for pixels
                           // with textured enough neighborhood
    int uniquenessRatio;   // accept the computed disparity d* only if
                           // SAD(d) >= SAD(d*)*(1 + uniquenessRatio/100.)
                           // for any d != d*+/-1 within the search range.
    int speckleWindowSize; // disparity variation window
    int speckleRange; // acceptable range of variation in window

    int trySmallerWindows; // if 1, the results may be more accurate,
                           // at the expense of slower processing 
    CvRect roi1, roi2;
    int disp12MaxDiff;

    // temporary buffers
    CvMat* preFilteredImg0;
    CvMat* preFilteredImg1;
    CvMat* slidingSumBuf;
    CvMat* cost;
    CvMat* disp;
} CvStereoBMState;

#define CV_STEREO_BM_BASIC 0
#define CV_STEREO_BM_FISH_EYE 1
#define CV_STEREO_BM_NARROW 2

CVAPI(CvStereoBMState*) cvCreateStereoBMState(int preset CV_DEFAULT(CV_STEREO_BM_BASIC),
                                              int numberOfDisparities CV_DEFAULT(0));

CVAPI(void) cvReleaseStereoBMState( CvStereoBMState** state );

CVAPI(void) cvFindStereoCorrespondenceBM( const CvArr* left, const CvArr* right,
                                          CvArr* disparity, CvStereoBMState* state );
    
CVAPI(CvRect) cvGetValidDisparityROI( CvRect roi1, CvRect roi2, int minDisparity,
                                      int numberOfDisparities, int SADWindowSize );
    
CVAPI(void) cvValidateDisparity( CvArr* disparity, const CvArr* cost,
                                 int minDisparity, int numberOfDisparities,
                                 int disp12MaxDiff CV_DEFAULT(1) );  

/* Kolmogorov-Zabin stereo-correspondence algorithm (a.k.a. KZ1) */
#define CV_STEREO_GC_OCCLUDED  SHRT_MAX

typedef struct CvStereoGCState
{
    int Ithreshold;
    int interactionRadius;
    float K, lambda, lambda1, lambda2;
    int occlusionCost;
    int minDisparity;
    int numberOfDisparities;
    int maxIters;

    CvMat* left;
    CvMat* right;
    CvMat* dispLeft;
    CvMat* dispRight;
    CvMat* ptrLeft;
    CvMat* ptrRight;
    CvMat* vtxBuf;
    CvMat* edgeBuf;
} CvStereoGCState;

CVAPI(CvStereoGCState*) cvCreateStereoGCState( int numberOfDisparities, int maxIters );
CVAPI(void) cvReleaseStereoGCState( CvStereoGCState** state );

CVAPI(void) cvFindStereoCorrespondenceGC( const CvArr* left, const CvArr* right,
                                          CvArr* disparityLeft, CvArr* disparityRight,
                                          CvStereoGCState* state,
                                          int useDisparityGuess CV_DEFAULT(0) );

/* Reprojects the computed disparity image to the 3D space using the specified 4x4 matrix */
CVAPI(void)  cvReprojectImageTo3D( const CvArr* disparityImage,
                                   CvArr* _3dImage, const CvMat* Q,
                                   int handleMissingValues CV_DEFAULT(0) );


#ifdef __cplusplus
}

//////////////////////////////////////////////////////////////////////////////////////////

class CV_EXPORTS CvLevMarq
{
public:
    CvLevMarq();
    CvLevMarq( int nparams, int nerrs, CvTermCriteria criteria=
              cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    ~CvLevMarq();
    void init( int nparams, int nerrs, CvTermCriteria criteria=
              cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,DBL_EPSILON),
              bool completeSymmFlag=false );
    bool update( const CvMat*& param, CvMat*& J, CvMat*& err );
    bool updateAlt( const CvMat*& param, CvMat*& JtJ, CvMat*& JtErr, double*& errNorm );
    
    void clear();
    void step();
    enum { DONE=0, STARTED=1, CALC_J=2, CHECK_ERR=3 };
    
    cv::Ptr<CvMat> mask;
    cv::Ptr<CvMat> prevParam;
    cv::Ptr<CvMat> param;
    cv::Ptr<CvMat> J;
    cv::Ptr<CvMat> err;
    cv::Ptr<CvMat> JtJ;
    cv::Ptr<CvMat> JtJN;
    cv::Ptr<CvMat> JtErr;
    cv::Ptr<CvMat> JtJV;
    cv::Ptr<CvMat> JtJW;
    double prevErrNorm, errNorm;
    int lambdaLg10;
    CvTermCriteria criteria;
    int state;
    int iters;
    bool completeSymmFlag;
};

namespace cv
{

//! converts rotation vector to rotation matrix or vice versa using Rodrigues transformation
CV_EXPORTS_W void Rodrigues(const Mat& src, CV_OUT Mat& dst);
    
//! converts rotation vector to rotation matrix or vice versa using Rodrigues transformation. Also computes the Jacobian matrix
CV_EXPORTS_AS(RodriguesJ) void Rodrigues(const Mat& src, CV_OUT Mat& dst, CV_OUT Mat& jacobian);

//! type of the robust estimation algorithm
enum
{
    LMEDS=4, //!< least-median algorithm
    RANSAC=8 //!< RANSAC algorithm
};

//! computes the best-fit perspective transformation mapping srcPoints to dstPoints.
CV_EXPORTS_AS(findHomographyAndOutliers) Mat findHomography( const Mat& srcPoints,
                               const Mat& dstPoints,
                               vector<uchar>& mask, int method=0,
                               double ransacReprojThreshold=3 );

//! computes the best-fit perspective transformation mapping srcPoints to dstPoints.
CV_EXPORTS_W Mat findHomography( const Mat& srcPoints,
                               const Mat& dstPoints,
                               int method=0, double ransacReprojThreshold=3 );

//! Computes RQ decomposition of 3x3 matrix
CV_EXPORTS void RQDecomp3x3( const Mat& M, Mat& R, Mat& Q );
    
//! Computes RQ decomposition of 3x3 matrix. Also, decomposes the output orthogonal matrix into the 3 primitive rotation matrices
CV_EXPORTS_W Vec3d RQDecomp3x3( const Mat& M, Mat& R, Mat& Q,
                              CV_OUT Mat& Qx, CV_OUT Mat& Qy, CV_OUT Mat& Qz );

//! Decomposes the projection matrix into camera matrix and the rotation martix and the translation vector
CV_EXPORTS void decomposeProjectionMatrix( const Mat& projMatrix, Mat& cameraMatrix,
                                           Mat& rotMatrix, Mat& transVect );
    
//! Decomposes the projection matrix into camera matrix and the rotation martix and the translation vector. The rotation matrix is further decomposed
CV_EXPORTS_W void decomposeProjectionMatrix( const Mat& projMatrix, CV_OUT Mat& cameraMatrix,
                                           CV_OUT Mat& rotMatrix, CV_OUT Mat& transVect,
                                           CV_OUT Mat& rotMatrixX, CV_OUT Mat& rotMatrixY,
                                           CV_OUT Mat& rotMatrixZ, CV_OUT Vec3d& eulerAngles );

//! computes derivatives of the matrix product w.r.t each of the multiplied matrix coefficients
CV_EXPORTS_W void matMulDeriv( const Mat& A, const Mat& B, CV_OUT Mat& dABdA, CV_OUT Mat& dABdB );

//! composes 2 [R|t] transformations together
CV_EXPORTS_W void composeRT( const Mat& rvec1, const Mat& tvec1,
                           const Mat& rvec2, const Mat& tvec2,
                           CV_OUT Mat& rvec3, CV_OUT Mat& tvec3 );

//! composes 2 [R|t] transformations together. Also computes the derivatives of the result w.r.t the arguments
CV_EXPORTS_AS(composeRT_J) void composeRT( const Mat& rvec1, const Mat& tvec1,
                           const Mat& rvec2, const Mat& tvec2,
                           CV_OUT Mat& rvec3, CV_OUT Mat& tvec3,
                           CV_OUT Mat& dr3dr1, CV_OUT Mat& dr3dt1,
                           CV_OUT Mat& dr3dr2, CV_OUT Mat& dr3dt2,
                           CV_OUT Mat& dt3dr1, CV_OUT Mat& dt3dt1,
                           CV_OUT Mat& dt3dr2, CV_OUT Mat& dt3dt2 );

//! projects points from the model coordinate space to the image coordinates. Takes the intrinsic and extrinsic camera parameters into account
CV_EXPORTS_W void projectPoints( const Mat& objectPoints,
                               const Mat& rvec, const Mat& tvec,
                               const Mat& cameraMatrix,
                               const Mat& distCoeffs,
                               CV_OUT vector<Point2f>& imagePoints );

//! projects points from the model coordinate space to the image coordinates. Also computes derivatives of the image coordinates w.r.t the intrinsic and extrinsic camera parameters
CV_EXPORTS_AS(projectPointsJ) void projectPoints( const Mat& objectPoints,
                               const Mat& rvec, const Mat& tvec,
                               const Mat& cameraMatrix,
                               const Mat& distCoeffs,
                               CV_OUT vector<Point2f>& imagePoints,
                               CV_OUT Mat& dpdrot, CV_OUT Mat& dpdt, CV_OUT Mat& dpdf,
                               CV_OUT Mat& dpdc, CV_OUT Mat& dpddist,
                               double aspectRatio=0 );

//! computes the camera pose from a few 3D points and the corresponding projections. The outliers are not handled.
CV_EXPORTS_W void solvePnP( const Mat& objectPoints,
                            const Mat& imagePoints,
                            const Mat& cameraMatrix,
                            const Mat& distCoeffs,
                            CV_OUT Mat& rvec, CV_OUT Mat& tvec,
                            bool useExtrinsicGuess=false );

//! initializes camera matrix from a few 3D points and the corresponding projections.
CV_EXPORTS_W Mat initCameraMatrix2D( const vector<vector<Point3f> >& objectPoints,
                                   const vector<vector<Point2f> >& imagePoints,
                                   Size imageSize, double aspectRatio=1. );


enum { CALIB_CB_ADAPTIVE_THRESH = 1, CALIB_CB_NORMALIZE_IMAGE = 2,
       CALIB_CB_FILTER_QUADS = 4, CALIB_CB_FAST_CHECK = 8 };

//! finds checkerboard pattern of the specified size in the image
CV_EXPORTS_W bool findChessboardCorners( const Mat& image, Size patternSize,
                                         CV_OUT vector<Point2f>& corners,
                                         int flags=CALIB_CB_ADAPTIVE_THRESH+
                                              CALIB_CB_NORMALIZE_IMAGE );

//! draws the checkerboard pattern (found or partly found) in the image
CV_EXPORTS_W void drawChessboardCorners( Mat& image, Size patternSize,
                                         const Mat& corners,
                                         bool patternWasFound );

CV_EXPORTS void drawChessboardCorners( Mat& image, Size patternSize,
                                       const vector<Point2f>& corners,
                                       bool patternWasFound );    
    
enum
{
    CALIB_USE_INTRINSIC_GUESS = 1,
    CALIB_FIX_ASPECT_RATIO = 2,
    CALIB_FIX_PRINCIPAL_POINT = 4,
    CALIB_ZERO_TANGENT_DIST = 8,
    CALIB_FIX_FOCAL_LENGTH = 16,
    CALIB_FIX_K1 = CV_CALIB_FIX_K1,
    CALIB_FIX_K2 = CV_CALIB_FIX_K2,
    CALIB_FIX_K3 = CV_CALIB_FIX_K3,
    CALIB_FIX_K4 = CV_CALIB_FIX_K4,
    CALIB_FIX_K5 = CV_CALIB_FIX_K5,
    CALIB_FIX_K6 = CV_CALIB_FIX_K6,
    CALIB_RATIONAL_MODEL = CV_CALIB_RATIONAL_MODEL,
    // only for stereo
    CALIB_FIX_INTRINSIC = CV_CALIB_FIX_INTRINSIC,
    CALIB_SAME_FOCAL_LENGTH = CV_CALIB_SAME_FOCAL_LENGTH,
    // for stereo rectification
    CALIB_ZERO_DISPARITY = CV_CALIB_ZERO_DISPARITY
};

//! finds intrinsic and extrinsic camera parameters from several fews of a known calibration pattern.
CV_EXPORTS_W double calibrateCamera( const vector<vector<Point3f> >& objectPoints,
                                     const vector<vector<Point2f> >& imagePoints,
                                     Size imageSize,
                                     CV_IN_OUT Mat& cameraMatrix,
                                     CV_IN_OUT Mat& distCoeffs,
                                     CV_OUT vector<Mat>& rvecs, CV_OUT vector<Mat>& tvecs,
                                     int flags=0 );

//! computes several useful camera characteristics from the camera matrix, camera frame resolution and the physical sensor size.
CV_EXPORTS_W void calibrationMatrixValues( const Mat& cameraMatrix,
                                Size imageSize,
                                double apertureWidth,
                                double apertureHeight,
                                CV_OUT double& fovx,
                                CV_OUT double& fovy,
                                CV_OUT double& focalLength,
                                CV_OUT Point2d& principalPoint,
                                CV_OUT double& aspectRatio );

//! finds intrinsic and extrinsic parameters of a stereo camera
CV_EXPORTS_W double stereoCalibrate( const vector<vector<Point3f> >& objectPoints,
                                     const vector<vector<Point2f> >& imagePoints1,
                                     const vector<vector<Point2f> >& imagePoints2,
                                     CV_IN_OUT Mat& cameraMatrix1, CV_IN_OUT Mat& distCoeffs1,
                                     CV_IN_OUT Mat& cameraMatrix2, CV_IN_OUT Mat& distCoeffs2,
                                     Size imageSize, CV_OUT Mat& R, CV_OUT Mat& T,
                                     CV_OUT Mat& E, CV_OUT Mat& F,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+
                                         TermCriteria::EPS, 30, 1e-6),
                                     int flags=CALIB_FIX_INTRINSIC );

    
//! computes the rectification transformation for a stereo camera from its intrinsic and extrinsic parameters
CV_EXPORTS void stereoRectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                               const Mat& cameraMatrix2, const Mat& distCoeffs2,
                               Size imageSize, const Mat& R, const Mat& T,
                               CV_OUT Mat& R1, CV_OUT Mat& R2,
                               CV_OUT Mat& P1, CV_OUT Mat& P2, CV_OUT Mat& Q,
                               int flags=CALIB_ZERO_DISPARITY );

//! computes the rectification transformation for a stereo camera from its intrinsic and extrinsic parameters
CV_EXPORTS_W void stereoRectify( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                                 const Mat& cameraMatrix2, const Mat& distCoeffs2,
                                 Size imageSize, const Mat& R, const Mat& T,
                                 CV_OUT Mat& R1, CV_OUT Mat& R2,
                                 CV_OUT Mat& P1, CV_OUT Mat& P2, CV_OUT Mat& Q,
                                 double alpha, Size newImageSize=Size(),
                                 CV_OUT Rect* validPixROI1=0, CV_OUT Rect* validPixROI2=0,
                                 int flags=CALIB_ZERO_DISPARITY );

//! computes the rectification transformation for an uncalibrated stereo camera (zero distortion is assumed)
CV_EXPORTS_W bool stereoRectifyUncalibrated( const Mat& points1, const Mat& points2,
                                             const Mat& F, Size imgSize,
                                             CV_OUT Mat& H1, CV_OUT Mat& H2,
                                             double threshold=5 );

//! computes the rectification transformations for 3-head camera, where all the heads are on the same line.
CV_EXPORTS_W float rectify3Collinear( const Mat& cameraMatrix1, const Mat& distCoeffs1,
                                      const Mat& cameraMatrix2, const Mat& distCoeffs2,
                                      const Mat& cameraMatrix3, const Mat& distCoeffs3,
                                      const vector<vector<Point2f> >& imgpt1,
                                      const vector<vector<Point2f> >& imgpt3,
                                      Size imageSize, const Mat& R12, const Mat& T12,
                                      const Mat& R13, const Mat& T13,
                                      CV_OUT Mat& R1, CV_OUT Mat& R2, CV_OUT Mat& R3,
                                      CV_OUT Mat& P1, CV_OUT Mat& P2, CV_OUT Mat& P3, CV_OUT Mat& Q,
                                      double alpha, Size newImgSize,
                                      CV_OUT Rect* roi1, CV_OUT Rect* roi2, int flags );
    
//! returns the optimal new camera matrix
CV_EXPORTS_W Mat getOptimalNewCameraMatrix( const Mat& cameraMatrix, const Mat& distCoeffs,
                                            Size imageSize, double alpha, Size newImgSize=Size(),
                                            CV_OUT Rect* validPixROI=0);

//! converts point coordinates from normal pixel coordinates to homogeneous coordinates ((x,y)->(x,y,1))
CV_EXPORTS void convertPointsHomogeneous( const Mat& src, CV_OUT vector<Point3f>& dst );
//! converts point coordinates from homogeneous to normal pixel coordinates ((x,y,z)->(x/z, y/z))
CV_EXPORTS void convertPointsHomogeneous( const Mat& src, CV_OUT vector<Point2f>& dst );

//! the algorithm for finding fundamental matrix
enum
{ 
    FM_7POINT = 1, //!< 7-point algorithm
    FM_8POINT = 2, //!< 8-point algorithm
    FM_LMEDS = 4,  //!< least-median algorithm
    FM_RANSAC = 8  //!< RANSAC algorithm
};

//! finds fundamental matrix from a set of corresponding 2D points
CV_EXPORTS Mat findFundamentalMat( const Mat& points1, const Mat& points2,
                                     CV_OUT vector<uchar>& mask, int method=FM_RANSAC,
                                     double param1=3., double param2=0.99 );

//! finds fundamental matrix from a set of corresponding 2D points
CV_EXPORTS_W Mat findFundamentalMat( const Mat& points1, const Mat& points2,
                                     int method=FM_RANSAC,
                                     double param1=3., double param2=0.99 );

//! finds coordinates of epipolar lines corresponding the specified points
CV_EXPORTS void computeCorrespondEpilines( const Mat& points1,
                                             int whichImage, const Mat& F,
                                             CV_OUT vector<Vec3f>& lines );

template<> CV_EXPORTS void Ptr<CvStereoBMState>::delete_obj();

/*!
 Block Matching Stereo Correspondence Algorithm
 
 The class implements BM stereo correspondence algorithm by K. Konolige.
*/
class CV_EXPORTS_W StereoBM
{
public:
    enum { PREFILTER_NORMALIZED_RESPONSE = 0, PREFILTER_XSOBEL = 1,
        BASIC_PRESET=0, FISH_EYE_PRESET=1, NARROW_PRESET=2 };

    //! the default constructor
    CV_WRAP StereoBM();
    //! the full constructor taking the camera-specific preset, number of disparities and the SAD window size
    CV_WRAP StereoBM(int preset, int ndisparities=0, int SADWindowSize=21);
    //! the method that reinitializes the state. The previous content is destroyed
    void init(int preset, int ndisparities=0, int SADWindowSize=21);
    //! the stereo correspondence operator. Finds the disparity for the specified rectified stereo pair
    CV_WRAP_AS(compute) void operator()( const Mat& left, const Mat& right, Mat& disparity, int disptype=CV_16S );

    //! pointer to the underlying CvStereoBMState
    Ptr<CvStereoBMState> state;
};


/*!
 Semi-Global Block Matching Stereo Correspondence Algorithm
 
 The class implements the original SGBM stereo correspondence algorithm by H. Hirschmuller and some its modification.
 */
class CV_EXPORTS_W StereoSGBM
{
public:
    enum { DISP_SHIFT=4, DISP_SCALE = (1<<DISP_SHIFT) };

    //! the default constructor
    CV_WRAP StereoSGBM();
    
    //! the full constructor taking all the necessary algorithm parameters
    CV_WRAP StereoSGBM(int minDisparity, int numDisparities, int SADWindowSize,
               int P1=0, int P2=0, int disp12MaxDiff=0,
               int preFilterCap=0, int uniquenessRatio=0,
               int speckleWindowSize=0, int speckleRange=0,
               bool fullDP=false);
    //! the destructor
    virtual ~StereoSGBM();

    //! the stereo correspondence operator that computes disparity map for the specified rectified stereo pair
    CV_WRAP_AS(compute) virtual void operator()(const Mat& left, const Mat& right, Mat& disp);

    CV_PROP_RW int minDisparity;
    CV_PROP_RW int numberOfDisparities;
    CV_PROP_RW int SADWindowSize;
    CV_PROP_RW int preFilterCap;
    CV_PROP_RW int uniquenessRatio;
    CV_PROP_RW int P1;
    CV_PROP_RW int P2;
    CV_PROP_RW int speckleWindowSize;
    CV_PROP_RW int speckleRange;
    CV_PROP_RW int disp12MaxDiff;
    CV_PROP_RW bool fullDP;

protected:
    Mat buffer;
};

//! filters off speckles (small regions of incorrectly computed disparity)
CV_EXPORTS_W void filterSpeckles( Mat& img, double newVal, int maxSpeckleSize, double maxDiff, Mat& buf );

//! computes valid disparity ROI from the valid ROIs of the rectified images (that are returned by cv::stereoRectify())
CV_EXPORTS_W Rect getValidDisparityROI( Rect roi1, Rect roi2,
                                        int minDisparity, int numberOfDisparities,
                                        int SADWindowSize );

//! validates disparity using the left-right check. The matrix "cost" should be computed by the stereo correspondence algorithm
CV_EXPORTS_W void validateDisparity( Mat& disparity, const Mat& cost,
                                     int minDisparity, int numberOfDisparities,
                                     int disp12MaxDisp=1 );

//! reprojects disparity image to 3D: (x,y,d)->(X,Y,Z) using the matrix Q returned by cv::stereoRectify
CV_EXPORTS_W void reprojectImageTo3D( const Mat& disparity,
                                      CV_OUT Mat& _3dImage, const Mat& Q,
                                      bool handleMissingValues=false );
    
}

#endif

#endif
