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
// Copyright( C) 2000, Intel Corporation, all rights reserved.
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
//(including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_COMPAT_HPP__
#define __OPENCV_COMPAT_HPP__

#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/types_c.h"

#include <math.h>
#include <string.h>

#define CV_NOOP(...)

#ifdef __cplusplus
extern "C" {
#endif

typedef int CvMatType;
typedef int CvDisMaskType;
typedef CvMat CvMatArray;

typedef int CvThreshType;
typedef int CvAdaptiveThreshMethod;
typedef int CvCompareMethod;
typedef int CvFontFace;
typedef int CvPolyApproxMethod;
typedef int CvContoursMatchMethod;
typedef int CvContourTreesMatchMethod;
typedef int CvCoeffType;
typedef int CvRodriguesType;
typedef int CvElementShape;
typedef int CvMorphOp;
typedef int CvTemplMatchMethod;

typedef CvPoint2D64f CvPoint2D64d;
typedef CvPoint3D64f CvPoint3D64d;

enum
{
    CV_MAT32F      = CV_32FC1,
    CV_MAT3x1_32F  = CV_32FC1,
    CV_MAT4x1_32F  = CV_32FC1,
    CV_MAT3x3_32F  = CV_32FC1,
    CV_MAT4x4_32F  = CV_32FC1,

    CV_MAT64D      = CV_64FC1,
    CV_MAT3x1_64D  = CV_64FC1,
    CV_MAT4x1_64D  = CV_64FC1,
    CV_MAT3x3_64D  = CV_64FC1,
    CV_MAT4x4_64D  = CV_64FC1
};

enum
{
    IPL_GAUSSIAN_5x5 = 7
};

typedef CvBox2D  CvBox2D32f;

/* allocation/deallocation macros */
#define cvCreateImageData   cvCreateData
#define cvReleaseImageData  cvReleaseData
#define cvSetImageData      cvSetData
#define cvGetImageRawData   cvGetRawData

#define cvmAlloc            cvCreateData
#define cvmFree             cvReleaseData
#define cvmAllocArray       cvCreateData
#define cvmFreeArray        cvReleaseData

#define cvIntegralImage     cvIntegral
#define cvMatchContours     cvMatchShapes

CV_EXPORTS CvMat cvMatArray( int rows, int cols, int type,
                            int count, void* data CV_DEFAULT(0));

#define cvUpdateMHIByTime  cvUpdateMotionHistory

#define cvAccMask cvAcc
#define cvSquareAccMask cvSquareAcc
#define cvMultiplyAccMask cvMultiplyAcc
#define cvRunningAvgMask(imgY, imgU, mask, alpha) cvRunningAvg(imgY, imgU, alpha, mask)

#define cvSetHistThresh  cvSetHistBinRanges
#define cvCalcHistMask(img, mask, hist, doNotClear) cvCalcHist(img, hist, doNotClear, mask)

CV_EXPORTS double cvMean( const CvArr* image, const CvArr* mask CV_DEFAULT(0));
CV_EXPORTS double cvSumPixels( const CvArr* image );
CV_EXPORTS void  cvMean_StdDev( const CvArr* image, double* mean, double* sdv,
                                const CvArr* mask CV_DEFAULT(0));

CV_EXPORTS void cvmPerspectiveProject( const CvMat* mat, const CvArr* src, CvArr* dst );
CV_EXPORTS void cvFillImage( CvArr* mat, double color );

#define cvCvtPixToPlane cvSplit
#define cvCvtPlaneToPix cvMerge

typedef struct CvRandState
{
    CvRNG     state;    /* RNG state (the current seed and carry)*/
    int       disttype; /* distribution type */
    CvScalar  param[2]; /* parameters of RNG */
} CvRandState;

/* Changes RNG range while preserving RNG state */
CV_EXPORTS void  cvRandSetRange( CvRandState* state, double param1,
                                 double param2, int index CV_DEFAULT(-1));

CV_EXPORTS void  cvRandInit( CvRandState* state, double param1,
                             double param2, int seed,
                             int disttype CV_DEFAULT(CV_RAND_UNI));

/* Fills array with random numbers */
CV_EXPORTS void cvRand( CvRandState* state, CvArr* arr );

#define cvRandNext( _state ) cvRandInt( &(_state)->state )

CV_EXPORTS void cvbRand( CvRandState* state, float* dst, int len );

CV_EXPORTS void  cvbCartToPolar( const float* y, const float* x,
                                 float* magnitude, float* angle, int len );
CV_EXPORTS void  cvbFastArctan( const float* y, const float* x, float* angle, int len );
CV_EXPORTS void  cvbSqrt( const float* x, float* y, int len );
CV_EXPORTS void  cvbInvSqrt( const float* x, float* y, int len );
CV_EXPORTS void  cvbReciprocal( const float* x, float* y, int len );
CV_EXPORTS void  cvbFastExp( const float* x, double* y, int len );
CV_EXPORTS void  cvbFastLog( const double* x, float* y, int len );

CV_EXPORTS CvRect  cvContourBoundingRect( void* point_set, int update CV_DEFAULT(0));

CV_EXPORTS double cvPseudoInverse( const CvArr* src, CvArr* dst );
#define cvPseudoInv cvPseudoInverse

#define cvContourMoments( contour, moments ) cvMoments( contour, moments, 0 )

#define cvGetPtrAt              cvPtr2D
#define cvGetAt                 cvGet2D
#define cvSetAt(arr,val,y,x)    cvSet2D((arr),(y),(x),(val))

#define cvMeanMask  cvMean
#define cvMean_StdDevMask(img,mask,mean,sdv) cvMean_StdDev(img,mean,sdv,mask)

#define cvNormMask(imgA,imgB,mask,normType) cvNorm(imgA,imgB,normType,mask)

#define cvMinMaxLocMask(img, mask, min_val, max_val, min_loc, max_loc) \
        cvMinMaxLoc(img, min_val, max_val, min_loc, max_loc, mask)

#define cvRemoveMemoryManager  CV_NOOP
#define cvSetMemoryManager     CV_NOOP

#define cvmSetZero( mat )               cvSetZero( mat )
#define cvmSetIdentity( mat )           cvSetIdentity( mat )
#define cvmAdd( src1, src2, dst )       cvAdd( src1, src2, dst, 0 )
#define cvmSub( src1, src2, dst )       cvSub( src1, src2, dst, 0 )
#define cvmCopy( src, dst )             cvCopy( src, dst, 0 )
#define cvmMul( src1, src2, dst )       cvMatMulAdd( src1, src2, 0, dst )
#define cvmTranspose( src, dst )        cvT( src, dst )
#define cvmInvert( src, dst )           cvInv( src, dst )
#define cvmMahalanobis(vec1, vec2, mat) cvMahalanobis( vec1, vec2, mat )
#define cvmDotProduct( vec1, vec2 )     cvDotProduct( vec1, vec2 )
#define cvmCrossProduct(vec1, vec2,dst) cvCrossProduct( vec1, vec2, dst )
#define cvmTrace( mat )                 (cvTrace( mat )).val[0]
#define cvmMulTransposed( src, dst, order ) cvMulTransposed( src, dst, order )
#define cvmEigenVV( mat, evec, eval, eps)   cvEigenVV( mat, evec, eval, eps )
#define cvmDet( mat )                   cvDet( mat )
#define cvmScale( src, dst, scale )     cvScale( src, dst, scale )

#define cvCopyImage( src, dst )         cvCopy( src, dst, 0 )
#define cvReleaseMatHeader              cvReleaseMat

/* Calculates exact convex hull of 2d point set */
CV_EXPORTS void cvConvexHull( CvPoint* points, int num_points,
                             CvRect* bound_rect,
                             int orientation, int* hull, int* hullsize );


CV_EXPORTS void cvMinAreaRect( CvPoint* points, int n,
                              int left, int bottom,
                              int right, int top,
                              CvPoint2D32f* anchor,
                              CvPoint2D32f* vect1,
                              CvPoint2D32f* vect2 );

typedef int CvDisType;
typedef int CvChainApproxMethod;
typedef int CvContourRetrievalMode;

CV_EXPORTS  void  cvFitLine3D( CvPoint3D32f* points, int count, int dist,
                    void *param, float reps, float aeps, float* line );

/* Fits a line into set of 2d points in a robust way (M-estimator technique) */
CV_EXPORTS  void  cvFitLine2D( CvPoint2D32f* points, int count, int dist,
                    void *param, float reps, float aeps, float* line );

CV_EXPORTS  void  cvFitEllipse( const CvPoint2D32f* points, int count, CvBox2D* box );

/* Projects 2d points to one of standard coordinate planes
   (i.e. removes one of coordinates) */
CV_EXPORTS  void  cvProject3D( CvPoint3D32f* points3D, int count,
                              CvPoint2D32f* points2D,
                              int xIndx CV_DEFAULT(0),
                              int yIndx CV_DEFAULT(1));

/* Retrieves value of the particular bin
   of x-dimensional (x=1,2,3,...) histogram */
#define cvQueryHistValue_1D( hist, idx0 ) \
    ((float)cvGetReal1D( (hist)->bins, (idx0)))
#define cvQueryHistValue_2D( hist, idx0, idx1 ) \
    ((float)cvGetReal2D( (hist)->bins, (idx0), (idx1)))
#define cvQueryHistValue_3D( hist, idx0, idx1, idx2 ) \
    ((float)cvGetReal3D( (hist)->bins, (idx0), (idx1), (idx2)))
#define cvQueryHistValue_nD( hist, idx ) \
    ((float)cvGetRealND( (hist)->bins, (idx)))

/* Returns pointer to the particular bin of x-dimesional histogram.
   For sparse histogram the bin is created if it didn't exist before */
#define cvGetHistValue_1D( hist, idx0 ) \
    ((float*)cvPtr1D( (hist)->bins, (idx0), 0))
#define cvGetHistValue_2D( hist, idx0, idx1 ) \
    ((float*)cvPtr2D( (hist)->bins, (idx0), (idx1), 0))
#define cvGetHistValue_3D( hist, idx0, idx1, idx2 ) \
    ((float*)cvPtr3D( (hist)->bins, (idx0), (idx1), (idx2), 0))
#define cvGetHistValue_nD( hist, idx ) \
    ((float*)cvPtrND( (hist)->bins, (idx), 0))


#define CV_IS_SET_ELEM_EXISTS CV_IS_SET_ELEM


CV_EXPORTS  int  cvHoughLines( CvArr* image, double rho,
                              double theta, int threshold,
                              float* lines, int linesNumber );

CV_EXPORTS  int  cvHoughLinesP( CvArr* image, double rho,
                               double theta, int threshold,
                               int lineLength, int lineGap,
                               int* lines, int linesNumber );


CV_EXPORTS  int  cvHoughLinesSDiv( CvArr* image, double rho, int srn,
                                  double theta, int stn, int threshold,
                                  float* lines, int linesNumber );

CV_EXPORTS  float  cvCalcEMD( const float* signature1, int size1,
                             const float* signature2, int size2,
                             int dims, int dist_type CV_DEFAULT(CV_DIST_L2),
                             CvDistanceFunction dist_func CV_DEFAULT(0),
                             float* lower_bound CV_DEFAULT(0),
                             void* user_param CV_DEFAULT(0));

CV_EXPORTS  void  cvKMeans( int num_clusters, float** samples,
                           int num_samples, int vec_size,
                           CvTermCriteria termcrit, int* cluster_idx );

CV_EXPORTS void  cvStartScanGraph( CvGraph* graph, CvGraphScanner* scanner,
                                  CvGraphVtx* vtx CV_DEFAULT(NULL),
                                  int mask CV_DEFAULT(CV_GRAPH_ALL_ITEMS));

CV_EXPORTS  void  cvEndScanGraph( CvGraphScanner* scanner );


/* old drawing functions */
CV_EXPORTS void  cvLineAA( CvArr* img, CvPoint pt1, CvPoint pt2,
                            double color, int scale CV_DEFAULT(0));

CV_EXPORTS void  cvCircleAA( CvArr* img, CvPoint center, int radius,
                            double color, int scale CV_DEFAULT(0) );

CV_EXPORTS void  cvEllipseAA( CvArr* img, CvPoint center, CvSize axes,
                              double angle, double start_angle,
                              double end_angle, double color,
                              int scale CV_DEFAULT(0) );

CV_EXPORTS void  cvPolyLineAA( CvArr* img, CvPoint** pts, int* npts, int contours,
                              int is_closed, double color, int scale CV_DEFAULT(0) );

/****************************************************************************************\
*                                   Pixel Access Macros                                  *
\****************************************************************************************/

typedef struct _CvPixelPosition8u
{
    uchar*  currline;      /* pointer to the start of the current pixel line   */
    uchar*  topline;       /* pointer to the start of the top pixel line       */
    uchar*  bottomline;    /* pointer to the start of the first line           */
                                    /* which is below the image                         */
    int     x;                      /* current x coordinate ( in pixels )               */
    int     width;                  /* width of the image  ( in pixels )                */
    int     height;                 /* height of the image  ( in pixels )               */
    int     step;                   /* distance between lines ( in elements of single   */
                                    /* plane )                                          */
    int     step_arr[3];            /* array: ( 0, -step, step ). It is used for        */
                                    /* vertical moving                                  */
} CvPixelPosition8u;

/* this structure differs from the above only in data type */
typedef struct _CvPixelPosition8s
{
    schar*  currline;
    schar*  topline;
    schar*  bottomline;
    int     x;
    int     width;
    int     height;
    int     step;
    int     step_arr[3];
} CvPixelPosition8s;

/* this structure differs from the CvPixelPosition8u only in data type */
typedef struct _CvPixelPosition32f
{
    float*  currline;
    float*  topline;
    float*  bottomline;
    int     x;
    int     width;
    int     height;
    int     step;
    int     step_arr[3];
} CvPixelPosition32f;


/* Initialize one of the CvPixelPosition structures.   */
/*  pos    - initialized structure                     */
/*  origin - pointer to the left-top corner of the ROI */
/*  step   - width of the whole image in bytes         */
/*  roi    - width & height of the ROI                 */
/*  x, y   - initial position                          */
#define CV_INIT_PIXEL_POS(pos, origin, _step, roi, _x, _y, orientation)    \
    (                                                                        \
    (pos).step = (_step)/sizeof((pos).currline[0]) * (orientation ? -1 : 1), \
    (pos).width = (roi).width,                                               \
    (pos).height = (roi).height,                                             \
    (pos).bottomline = (origin) + (pos).step*(pos).height,                   \
    (pos).topline = (origin) - (pos).step,                                   \
    (pos).step_arr[0] = 0,                                                   \
    (pos).step_arr[1] = -(pos).step,                                         \
    (pos).step_arr[2] = (pos).step,                                          \
    (pos).x = (_x),                                                          \
    (pos).currline = (origin) + (pos).step*(_y) )


/* Move to specified point ( absolute shift ) */
/*  pos    - position structure               */
/*  x, y   - coordinates of the new position  */
/*  cs     - number of the image channels     */
#define CV_MOVE_TO( pos, _x, _y, cs )                                                   \
((pos).currline = (_y) >= 0 && (_y) < (pos).height ? (pos).topline + ((_y)+1)*(pos).step : 0, \
 (pos).x = (_x) >= 0 && (_x) < (pos).width ? (_x) : 0, (pos).currline + (_x) * (cs) )

/* Get current coordinates                    */
/*  pos    - position structure               */
/*  x, y   - coordinates of the new position  */
/*  cs     - number of the image channels     */
#define CV_GET_CURRENT( pos, cs )  ((pos).currline + (pos).x * (cs))

/* Move by one pixel relatively to current position */
/*  pos    - position structure                     */
/*  cs     - number of the image channels           */

/* left */
#define CV_MOVE_LEFT( pos, cs ) \
 ( --(pos).x >= 0 ? (pos).currline + (pos).x*(cs) : 0 )

/* right */
#define CV_MOVE_RIGHT( pos, cs ) \
 ( ++(pos).x < (pos).width ? (pos).currline + (pos).x*(cs) : 0 )

/* up */
#define CV_MOVE_UP( pos, cs ) \
 (((pos).currline -= (pos).step) != (pos).topline ? (pos).currline + (pos).x*(cs) : 0 )

/* down */
#define CV_MOVE_DOWN( pos, cs ) \
 (((pos).currline += (pos).step) != (pos).bottomline ? (pos).currline + (pos).x*(cs) : 0 )

/* left up */
#define CV_MOVE_LU( pos, cs ) ( CV_MOVE_LEFT(pos, cs), CV_MOVE_UP(pos, cs))

/* right up */
#define CV_MOVE_RU( pos, cs ) ( CV_MOVE_RIGHT(pos, cs), CV_MOVE_UP(pos, cs))

/* left down */
#define CV_MOVE_LD( pos, cs ) ( CV_MOVE_LEFT(pos, cs), CV_MOVE_DOWN(pos, cs))

/* right down */
#define CV_MOVE_RD( pos, cs ) ( CV_MOVE_RIGHT(pos, cs), CV_MOVE_DOWN(pos, cs))



/* Move by one pixel relatively to current position with wrapping when the position     */
/* achieves image boundary                                                              */
/*  pos    - position structure                                                         */
/*  cs     - number of the image channels                                               */

/* left */
#define CV_MOVE_LEFT_WRAP( pos, cs ) \
 ((pos).currline + ( --(pos).x >= 0 ? (pos).x : ((pos).x = (pos).width-1))*(cs))

/* right */
#define CV_MOVE_RIGHT_WRAP( pos, cs ) \
 ((pos).currline + ( ++(pos).x < (pos).width ? (pos).x : ((pos).x = 0))*(cs) )

/* up */
#define CV_MOVE_UP_WRAP( pos, cs ) \
    ((((pos).currline -= (pos).step) != (pos).topline ? \
    (pos).currline : ((pos).currline = (pos).bottomline - (pos).step)) + (pos).x*(cs) )

/* down */
#define CV_MOVE_DOWN_WRAP( pos, cs ) \
    ((((pos).currline += (pos).step) != (pos).bottomline ? \
    (pos).currline : ((pos).currline = (pos).topline + (pos).step)) + (pos).x*(cs) )

/* left up */
#define CV_MOVE_LU_WRAP( pos, cs ) ( CV_MOVE_LEFT_WRAP(pos, cs), CV_MOVE_UP_WRAP(pos, cs))
/* right up */
#define CV_MOVE_RU_WRAP( pos, cs ) ( CV_MOVE_RIGHT_WRAP(pos, cs), CV_MOVE_UP_WRAP(pos, cs))
/* left down */
#define CV_MOVE_LD_WRAP( pos, cs ) ( CV_MOVE_LEFT_WRAP(pos, cs), CV_MOVE_DOWN_WRAP(pos, cs))
/* right down */
#define CV_MOVE_RD_WRAP( pos, cs ) ( CV_MOVE_RIGHT_WRAP(pos, cs), CV_MOVE_DOWN_WRAP(pos, cs))

/* Numeric constants which used for moving in arbitrary direction  */
enum
{
    CV_SHIFT_NONE = 2,
    CV_SHIFT_LEFT = 1,
    CV_SHIFT_RIGHT = 3,
    CV_SHIFT_UP = 6,
    CV_SHIFT_DOWN = 10,
    CV_SHIFT_LU = 5,
    CV_SHIFT_RU = 7,
    CV_SHIFT_LD = 9,
    CV_SHIFT_RD = 11
};

/* Move by one pixel in specified direction                                     */
/*  pos    - position structure                                                 */
/*  shift  - direction ( it's value must be one of the CV_SHIFT_Ö constants ) */
/*  cs     - number of the image channels                                       */
#define CV_MOVE_PARAM( pos, shift, cs )                                             \
    ( (pos).currline += (pos).step_arr[(shift)>>2], (pos).x += ((shift)&3)-2,       \
    ((pos).currline != (pos).topline && (pos).currline != (pos).bottomline &&       \
    (pos).x >= 0 && (pos).x < (pos).width) ? (pos).currline + (pos).x*(cs) : 0 )

/* Move by one pixel in specified direction with wrapping when the               */
/* position achieves image boundary                                              */
/*  pos    - position structure                                                  */
/*  shift  - direction ( it's value must be one of the CV_SHIFT_Ö constants )  */
/*  cs     - number of the image channels                                        */
#define CV_MOVE_PARAM_WRAP( pos, shift, cs )                                        \
    ( (pos).currline += (pos).step_arr[(shift)>>2],                                 \
    (pos).currline = ((pos).currline == (pos).topline ?                             \
    (pos).bottomline - (pos).step :                                                 \
    (pos).currline == (pos).bottomline ?                                            \
    (pos).topline + (pos).step : (pos).currline),                                   \
                                                                                    \
    (pos).x += ((shift)&3)-2,                                                       \
    (pos).x = ((pos).x < 0 ? (pos).width-1 : (pos).x >= (pos).width ? 0 : (pos).x), \
                                                                                    \
    (pos).currline + (pos).x*(cs) )


typedef float*   CvVect32f;
typedef float*   CvMatr32f;
typedef double*  CvVect64d;
typedef double*  CvMatr64d;

CV_EXPORTS void cvUnDistortOnce( const CvArr* src, CvArr* dst,
                                const float* intrinsic_matrix,
                                const float* distortion_coeffs,
                                int interpolate );

/* the two functions below have quite hackerish implementations, use with care
   (or, which is better, switch to cvUndistortInitMap and cvRemap instead */
CV_EXPORTS void cvUnDistortInit( const CvArr* src,
                                CvArr* undistortion_map,
                                const float* A, const float* k,
                                int interpolate );

CV_EXPORTS void  cvUnDistort( const CvArr* src, CvArr* dst,
                             const CvArr* undistortion_map,
                             int interpolate );

/* Find fundamental matrix */
CV_EXPORTS void  cvFindFundamentalMatrix( int* points1, int* points2,
    int numpoints, int method, float* matrix );


CV_EXPORTS int cvFindChessBoardCornerGuesses( const void* arr, void* thresharr,
                               CvMemStorage* storage,
                               CvSize pattern_size, CvPoint2D32f * corners,
                               int *corner_count );

/* Calibrates camera using multiple views of calibration pattern */
CV_EXPORTS void cvCalibrateCamera( int image_count, int* _point_counts,
    CvSize image_size, CvPoint2D32f* _image_points, CvPoint3D32f* _object_points,
    float* _distortion_coeffs, float* _camera_matrix, float* _translation_vectors,
    float* _rotation_matrices, int flags );


CV_EXPORTS void cvCalibrateCamera_64d( int image_count, int* _point_counts,
    CvSize image_size, CvPoint2D64f* _image_points, CvPoint3D64f* _object_points,
    double* _distortion_coeffs, double* _camera_matrix, double* _translation_vectors,
    double* _rotation_matrices, int flags );


/* Find 3d position of object given intrinsic camera parameters,
   3d model of the object and projection of the object into view plane */
CV_EXPORTS void cvFindExtrinsicCameraParams( int point_count,
    CvSize image_size, CvPoint2D32f* _image_points,
    CvPoint3D32f* _object_points, float* focal_length,
    CvPoint2D32f principal_point, float* _distortion_coeffs,
    float* _rotation_vector, float* _translation_vector );

/* Variant of the previous function that takes double-precision parameters */
CV_EXPORTS void cvFindExtrinsicCameraParams_64d( int point_count,
    CvSize image_size, CvPoint2D64f* _image_points,
    CvPoint3D64f* _object_points, double* focal_length,
    CvPoint2D64f principal_point, double* _distortion_coeffs,
    double* _rotation_vector, double* _translation_vector );

/* Rodrigues transform */
enum
{
    CV_RODRIGUES_M2V = 0,
    CV_RODRIGUES_V2M = 1
};

/* Converts rotation_matrix matrix to rotation_matrix vector or vice versa */
CV_EXPORTS void  cvRodrigues( CvMat* rotation_matrix, CvMat* rotation_vector,
                              CvMat* jacobian, int conv_type );

/* Does reprojection of 3d object points to the view plane */
CV_EXPORTS void  cvProjectPoints( int point_count, CvPoint3D64f* _object_points,
    double* _rotation_vector, double*  _translation_vector,
    double* focal_length, CvPoint2D64f principal_point,
    double* _distortion, CvPoint2D64f* _image_points,
    double* _deriv_points_rotation_matrix,
    double* _deriv_points_translation_vect,
    double* _deriv_points_focal,
    double* _deriv_points_principal_point,
    double* _deriv_points_distortion_coeffs );


/* Simpler version of the previous function */
CV_EXPORTS void  cvProjectPointsSimple( int point_count, CvPoint3D64f* _object_points,
    double* _rotation_matrix, double*  _translation_vector,
    double* _camera_matrix, double* _distortion, CvPoint2D64f* _image_points );


#define cvMake2DPoints cvConvertPointsHomogeneous
#define cvMake3DPoints cvConvertPointsHomogeneous

#define cvWarpPerspectiveQMatrix cvGetPerspectiveTransform

#define cvConvertPointsHomogenious cvConvertPointsHomogeneous


//////////////////////////////////// feature extractors: obsolete API //////////////////////////////////

typedef struct CvSURFPoint
{
    CvPoint2D32f pt;

    int          laplacian;
    int          size;
    float        dir;
    float        hessian;

} CvSURFPoint;

CV_INLINE CvSURFPoint cvSURFPoint( CvPoint2D32f pt, int laplacian,
                                  int size, float dir CV_DEFAULT(0),
                                  float hessian CV_DEFAULT(0))
{
    CvSURFPoint kp;

    kp.pt        = pt;
    kp.laplacian = laplacian;
    kp.size      = size;
    kp.dir       = dir;
    kp.hessian   = hessian;

    return kp;
}

typedef struct CvSURFParams
{
    int    extended;
    int    upright;
    double hessianThreshold;

    int    nOctaves;
    int    nOctaveLayers;

} CvSURFParams;

CVAPI(CvSURFParams) cvSURFParams( double hessianThreshold, int extended CV_DEFAULT(0) );

// If useProvidedKeyPts!=0, keypoints are not detected, but descriptors are computed
//  at the locations provided in keypoints (a CvSeq of CvSURFPoint).
CVAPI(void) cvExtractSURF( const CvArr* img, const CvArr* mask,
                          CvSeq** keypoints, CvSeq** descriptors,
                          CvMemStorage* storage, CvSURFParams params,
                             int useProvidedKeyPts CV_DEFAULT(0)  );

/*!
 Maximal Stable Regions Parameters
 */
typedef struct CvMSERParams
{
    //! delta, in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta}
    int delta;
    //! prune the area which bigger than maxArea
    int maxArea;
    //! prune the area which smaller than minArea
    int minArea;
    //! prune the area have simliar size to its children
    float maxVariation;
    //! trace back to cut off mser with diversity < min_diversity
    float minDiversity;

    /////// the next few params for MSER of color image

    //! for color image, the evolution steps
    int maxEvolution;
    //! the area threshold to cause re-initialize
    double areaThreshold;
    //! ignore too small margin
    double minMargin;
    //! the aperture size for edge blur
    int edgeBlurSize;
} CvMSERParams;

CVAPI(CvMSERParams) cvMSERParams( int delta CV_DEFAULT(5), int min_area CV_DEFAULT(60),
                                 int max_area CV_DEFAULT(14400), float max_variation CV_DEFAULT(.25f),
                                 float min_diversity CV_DEFAULT(.2f), int max_evolution CV_DEFAULT(200),
                                 double area_threshold CV_DEFAULT(1.01),
                                 double min_margin CV_DEFAULT(.003),
                                 int edge_blur_size CV_DEFAULT(5) );

// Extracts the contours of Maximally Stable Extremal Regions
CVAPI(void) cvExtractMSER( CvArr* _img, CvArr* _mask, CvSeq** contours, CvMemStorage* storage, CvMSERParams params );


typedef struct CvStarKeypoint
{
    CvPoint pt;
    int size;
    float response;
} CvStarKeypoint;

CV_INLINE CvStarKeypoint cvStarKeypoint(CvPoint pt, int size, float response)
{
    CvStarKeypoint kpt;
    kpt.pt = pt;
    kpt.size = size;
    kpt.response = response;
    return kpt;
}

typedef struct CvStarDetectorParams
{
    int maxSize;
    int responseThreshold;
    int lineThresholdProjected;
    int lineThresholdBinarized;
    int suppressNonmaxSize;
} CvStarDetectorParams;

CV_INLINE CvStarDetectorParams cvStarDetectorParams(
                                                    int maxSize CV_DEFAULT(45),
                                                    int responseThreshold CV_DEFAULT(30),
                                                    int lineThresholdProjected CV_DEFAULT(10),
                                                    int lineThresholdBinarized CV_DEFAULT(8),
                                                    int suppressNonmaxSize CV_DEFAULT(5))
{
    CvStarDetectorParams params;
    params.maxSize = maxSize;
    params.responseThreshold = responseThreshold;
    params.lineThresholdProjected = lineThresholdProjected;
    params.lineThresholdBinarized = lineThresholdBinarized;
    params.suppressNonmaxSize = suppressNonmaxSize;

    return params;
}

CVAPI(CvSeq*) cvGetStarKeypoints( const CvArr* img, CvMemStorage* storage,
                                 CvStarDetectorParams params CV_DEFAULT(cvStarDetectorParams()));

#ifdef __cplusplus
}
#endif

#endif
