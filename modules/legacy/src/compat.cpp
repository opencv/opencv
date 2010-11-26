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
// Copyright (C) 2008-2010, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

CvMat cvMatArray( int rows, int cols, int type,
                  int count, void* data)
{
    return cvMat( rows*count, cols, type, data );
}


double cvMean( const CvArr* image, const CvArr* mask )
{
    CvScalar mean = cvAvg( image, mask );
    return mean.val[0];
}


double  cvSumPixels( const CvArr* image )
{
    CvScalar scalar = cvSum( image );
    return scalar.val[0];
}

void  cvMean_StdDev( const CvArr* image, double* mean, double* sdv, const CvArr* mask)
{
    CvScalar _mean, _sdv;
    cvAvgSdv( image, &_mean, &_sdv, mask );

    if( mean )
        *mean = _mean.val[0];

    if( sdv )
        *sdv = _sdv.val[0];
}


void cvmPerspectiveProject( const CvMat* mat, const CvArr* src, CvArr* dst )
{
    CvMat tsrc, tdst;

    cvReshape( src, &tsrc, 3, 0 );
    cvReshape( dst, &tdst, 3, 0 );

    cvPerspectiveTransform( &tsrc, &tdst, mat );
}


void cvFillImage( CvArr* mat, double color )
{
    cvSet( mat, cvColorToScalar(color, cvGetElemType(mat)), 0 );
}


/* Changes RNG range while preserving RNG state */
void  cvRandSetRange( CvRandState* state, double param1, double param2, int index)
{
    if( !state )
    {
        cvError( CV_StsNullPtr, "cvRandSetRange", "Null pointer to RNG state", "cvcompat.h", 0 );
        return;
    }

    if( (unsigned)(index + 1) > 4 )
    {
        cvError( CV_StsOutOfRange, "cvRandSetRange", "index is not in -1..3", "cvcompat.h", 0 );
        return;
    }

    if( index < 0 )
    {
        state->param[0].val[0] = state->param[0].val[1] =
        state->param[0].val[2] = state->param[0].val[3] = param1;
        state->param[1].val[0] = state->param[1].val[1] =
        state->param[1].val[2] = state->param[1].val[3] = param2;
    }
    else
    {
        state->param[0].val[index] = param1;
        state->param[1].val[index] = param2;
    }
}


void  cvRandInit( CvRandState* state, double param1, double param2,
                  int seed, int disttype)
{
    if( !state )
    {
        cvError( CV_StsNullPtr, "cvRandInit", "Null pointer to RNG state", "cvcompat.h", 0 );
        return;
    }

    if( disttype != CV_RAND_UNI && disttype != CV_RAND_NORMAL )
    {
        cvError( CV_StsBadFlag, "cvRandInit", "Unknown distribution type", "cvcompat.h", 0 );
        return;
    }

    state->state = (uint64)(seed ? seed : -1);
    state->disttype = disttype;
    cvRandSetRange( state, param1, param2, -1 );
}


/* Fills array with random numbers */
void cvRand( CvRandState* state, CvArr* arr )
{
    if( !state )
    {
        cvError( CV_StsNullPtr, "cvRand", "Null pointer to RNG state", "cvcompat.h", 0 );
        return;
    }
    cvRandArr( &state->state, arr, state->disttype, state->param[0], state->param[1] );
}

void cvbRand( CvRandState* state, float* dst, int len )
{
    CvMat mat = cvMat( 1, len, CV_32F, (void*)dst );
    cvRand( state, &mat );
}


void  cvbCartToPolar( const float* y, const float* x, float* magnitude, float* angle, int len )
{
    CvMat mx = cvMat( 1, len, CV_32F, (void*)x );
    CvMat my = mx;
    CvMat mm = mx;
    CvMat ma = mx;

    my.data.fl = (float*)y;
    mm.data.fl = (float*)magnitude;
    ma.data.fl = (float*)angle;

    cvCartToPolar( &mx, &my, &mm, angle ? &ma : NULL, 1 );
}


void  cvbFastArctan( const float* y, const float* x, float* angle, int len )
{
    CvMat mx = cvMat( 1, len, CV_32F, (void*)x );
    CvMat my = mx;
    CvMat ma = mx;

    my.data.fl = (float*)y;
    ma.data.fl = (float*)angle;

    cvCartToPolar( &mx, &my, NULL, &ma, 1 );
}


 void  cvbSqrt( const float* x, float* y, int len )
{
    CvMat mx = cvMat( 1, len, CV_32F, (void*)x );
    CvMat my = mx;
    my.data.fl = (float*)y;

    cvPow( &mx, &my, 0.5 );
}


 void  cvbInvSqrt( const float* x, float* y, int len )
{
    CvMat mx = cvMat( 1, len, CV_32F, (void*)x );
    CvMat my = mx;
    my.data.fl = (float*)y;

    cvPow( &mx, &my, -0.5 );
}


 void  cvbReciprocal( const float* x, float* y, int len )
{
    CvMat mx = cvMat( 1, len, CV_32F, (void*)x );
    CvMat my = mx;
    my.data.fl = (float*)y;

    cvPow( &mx, &my, -1 );
}


 void  cvbFastExp( const float* x, double* y, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        y[i] = exp((double)x[i]);
}


 void  cvbFastLog( const double* x, float* y, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        y[i] = (float)log(x[i]);
}


CvRect  cvContourBoundingRect( void* point_set, int update)
{
    return cvBoundingRect( point_set, update );
}


double cvPseudoInverse( const CvArr* src, CvArr* dst )
{
    return cvInvert( src, dst, CV_SVD );
}


/* Calculates exact convex hull of 2d point set */
void cvConvexHull( CvPoint* points, int num_points, CvRect*,
                   int orientation, int* hull, int* hullsize )
{
    CvMat points1 = cvMat( 1, num_points, CV_32SC2, points );
    CvMat hull1 = cvMat( 1, num_points, CV_32SC1, hull );

    cvConvexHull2( &points1, &hull1, orientation, 0 );
    *hullsize = hull1.cols;
}

void cvMinAreaRect( CvPoint* points, int n, int, int, int, int,
                    CvPoint2D32f* anchor, CvPoint2D32f* vect1, CvPoint2D32f* vect2 )
{
    CvMat mat = cvMat( 1, n, CV_32SC2, points );
    CvBox2D box = cvMinAreaRect2( &mat, 0 );
    CvPoint2D32f pt[4];

    cvBoxPoints( box, pt );
    *anchor = pt[0];
    vect1->x = pt[1].x - pt[0].x;
    vect1->y = pt[1].y - pt[0].y;
    vect2->x = pt[3].x - pt[0].x;
    vect2->y = pt[3].y - pt[0].y;
}

void  cvFitLine3D( CvPoint3D32f* points, int count, int dist,
                   void *param, float reps, float aeps, float* line )
{
    CvMat mat = cvMat( 1, count, CV_32FC3, points );
    float _param = param != NULL ? *(float*)param : 0.f;
    assert( dist != CV_DIST_USER );
    cvFitLine( &mat, dist, _param, reps, aeps, line );
}

/* Fits a line into set of 2d points in a robust way (M-estimator technique) */
void  cvFitLine2D( CvPoint2D32f* points, int count, int dist,
                   void *param, float reps, float aeps, float* line )
{
    CvMat mat = cvMat( 1, count, CV_32FC2, points );
    float _param = param != NULL ? *(float*)param : 0.f;
    assert( dist != CV_DIST_USER );
    cvFitLine( &mat, dist, _param, reps, aeps, line );
}


void cvFitEllipse( const CvPoint2D32f* points, int count, CvBox2D* box )
{
    CvMat mat = cvMat( 1, count, CV_32FC2, (void*)points );
    *box = cvFitEllipse2( &mat );
}

/* Projects 2d points to one of standard coordinate planes
   (i.e. removes one of coordinates) */
void  cvProject3D( CvPoint3D32f* points3D, int count,
                   CvPoint2D32f* points2D, int xIndx, int yIndx)
{
    CvMat src = cvMat( 1, count, CV_32FC3, points3D );
    CvMat dst = cvMat( 1, count, CV_32FC2, points2D );
    float m[6] = {0,0,0,0,0,0};
    CvMat M = cvMat( 2, 3, CV_32F, m );

    assert( (unsigned)xIndx < 3 && (unsigned)yIndx < 3 );
    m[xIndx] = m[yIndx+3] = 1.f;

    cvTransform( &src, &dst, &M, NULL );
}


int  cvHoughLines( CvArr* image, double rho,
                   double theta, int threshold,
                   float* lines, int linesNumber )
{
    CvMat linesMat = cvMat( 1, linesNumber, CV_32FC2, lines );
    cvHoughLines2( image, &linesMat, CV_HOUGH_STANDARD,
                   rho, theta, threshold, 0, 0 );

    return linesMat.cols;
}


int  cvHoughLinesP( CvArr* image, double rho,
                    double theta, int threshold,
                    int lineLength, int lineGap,
                    int* lines, int linesNumber )
{
    CvMat linesMat = cvMat( 1, linesNumber, CV_32SC4, lines );
    cvHoughLines2( image, &linesMat, CV_HOUGH_PROBABILISTIC,
                   rho, theta, threshold, lineLength, lineGap );

    return linesMat.cols;
}


int  cvHoughLinesSDiv( CvArr* image, double rho, int srn,
                       double theta, int stn, int threshold,
                       float* lines, int linesNumber )
{
    CvMat linesMat = cvMat( 1, linesNumber, CV_32FC2, lines );
    cvHoughLines2( image, &linesMat, CV_HOUGH_MULTI_SCALE,
                   rho, theta, threshold, srn, stn );

    return linesMat.cols;
}


float  cvCalcEMD( const float* signature1, int size1, const float* signature2, int size2,
                  int dims, int dist_type, CvDistanceFunction dist_func,
                  float* lower_bound, void* user_param)
{
    CvMat sign1 = cvMat( size1, dims + 1, CV_32FC1, (void*)signature1 );
    CvMat sign2 = cvMat( size2, dims + 1, CV_32FC1, (void*)signature2 );

    return cvCalcEMD2( &sign1, &sign2, dist_type, dist_func, 0, 0, lower_bound, user_param );
}


void  cvKMeans( int num_clusters, float** samples,
                int num_samples, int vec_size,
                CvTermCriteria termcrit, int* cluster_idx )
{
    CvMat* samples_mat = cvCreateMat( num_samples, vec_size, CV_32FC1 );
    CvMat cluster_idx_mat = cvMat( num_samples, 1, CV_32SC1, cluster_idx );
    int i;
    for( i = 0; i < num_samples; i++ )
        memcpy( samples_mat->data.fl + i*vec_size, samples[i], vec_size*sizeof(float));
    cvKMeans2( samples_mat, num_clusters, &cluster_idx_mat, termcrit, 1, 0, 0, 0, 0 );
    cvReleaseMat( &samples_mat );
}


void  cvStartScanGraph( CvGraph* graph, CvGraphScanner* scanner,
                        CvGraphVtx* vtx, int mask)
{
    CvGraphScanner* temp_scanner;

    if( !scanner )
        cvError( CV_StsNullPtr, "cvStartScanGraph", "Null scanner pointer", "cvcompat.h", 0 );

    temp_scanner = cvCreateGraphScanner( graph, vtx, mask );
    *scanner = *temp_scanner;
    cvFree( &temp_scanner );
}


void  cvEndScanGraph( CvGraphScanner* scanner )
{
    if( !scanner )
        cvError( CV_StsNullPtr, "cvEndScanGraph", "Null scanner pointer", "cvcompat.h", 0 );

    if( scanner->stack )
    {
        CvGraphScanner* temp_scanner = (CvGraphScanner*)cvAlloc( sizeof(*temp_scanner) );
        *temp_scanner = *scanner;
        cvReleaseGraphScanner( &temp_scanner );
        memset( scanner, 0, sizeof(*scanner) );
    }
}


/* old drawing functions */
void  cvLineAA( CvArr* img, CvPoint pt1, CvPoint pt2, double color, int scale)
{
    cvLine( img, pt1, pt2, cvColorToScalar(color, cvGetElemType(img)), 1, CV_AA, scale );
}

void  cvCircleAA( CvArr* img, CvPoint center, int radius, double color, int scale)
{
    cvCircle( img, center, radius, cvColorToScalar(color, cvGetElemType(img)), 1, CV_AA, scale );
}

void  cvEllipseAA( CvArr* img, CvPoint center, CvSize axes,
                   double angle, double start_angle,
                   double end_angle, double color,
                   int scale)
{
    cvEllipse( img, center, axes, angle, start_angle, end_angle,
               cvColorToScalar(color, cvGetElemType(img)), 1, CV_AA, scale );
}

void  cvPolyLineAA( CvArr* img, CvPoint** pts, int* npts, int contours,
                    int is_closed, double color, int scale )
{
    cvPolyLine( img, pts, npts, contours, is_closed,
                cvColorToScalar(color, cvGetElemType(img)),
                1, CV_AA, scale );
}


void cvUnDistortOnce( const CvArr* src, CvArr* dst,
                      const float* intrinsic_matrix,
                      const float* distortion_coeffs,
                      int )
{
    CvMat _a = cvMat( 3, 3, CV_32F, (void*)intrinsic_matrix );
    CvMat _k = cvMat( 4, 1, CV_32F, (void*)distortion_coeffs );
    cvUndistort2( src, dst, &_a, &_k, 0 );
}


/* the two functions below have quite hackerish implementations, use with care
   (or, which is better, switch to cvUndistortInitMap and cvRemap instead */
void cvUnDistortInit( const CvArr*,
                      CvArr* undistortion_map,
                      const float* A, const float* k,
                      int)
{
    union { uchar* ptr; float* fl; } data;
    CvSize sz;
    cvGetRawData( undistortion_map, &data.ptr, 0, &sz );
    assert( sz.width >= 8 );
    /* just save the intrinsic parameters to the map */
    data.fl[0] = A[0]; data.fl[1] = A[4];
    data.fl[2] = A[2]; data.fl[3] = A[5];
    data.fl[4] = k[0]; data.fl[5] = k[1];
    data.fl[6] = k[2]; data.fl[7] = k[3];
}

void  cvUnDistort( const CvArr* src, CvArr* dst,
                   const CvArr* undistortion_map, int )
{
    union { uchar* ptr; float* fl; } data;
    float a[] = {0,0,0,0,0,0,0,0,1};
    CvSize sz;
    cvGetRawData( undistortion_map, &data.ptr, 0, &sz );
    assert( sz.width >= 8 );
    a[0] = data.fl[0]; a[4] = data.fl[1];
    a[2] = data.fl[2]; a[5] = data.fl[3];
    cvUnDistortOnce( src, dst, a, data.fl + 4, 1 );
}


/* Find fundamental matrix */
void  cvFindFundamentalMatrix( int* points1, int* points2, int numpoints, int, float* matrix )
{
    CvMat* pointsMat1;
    CvMat* pointsMat2;
    CvMat fundMatr = cvMat(3,3,CV_32F,matrix);
    int i, curr = 0;

    pointsMat1 = cvCreateMat(3,numpoints,CV_64F);
    pointsMat2 = cvCreateMat(3,numpoints,CV_64F);

    for( i = 0; i < numpoints; i++ )
    {
        cvmSet(pointsMat1,0,i,points1[curr]);//x
        cvmSet(pointsMat1,1,i,points1[curr+1]);//y
        cvmSet(pointsMat1,2,i,1.0);

        cvmSet(pointsMat2,0,i,points2[curr]);//x
        cvmSet(pointsMat2,1,i,points2[curr+1]);//y
        cvmSet(pointsMat2,2,i,1.0);
        curr += 2;
    }

    cvFindFundamentalMat(pointsMat1,pointsMat2,&fundMatr,CV_FM_RANSAC,1,0.99,0);

    cvReleaseMat(&pointsMat1);
    cvReleaseMat(&pointsMat2);
}


int cvFindChessBoardCornerGuesses( const void* arr, void*,
                                   CvMemStorage*, CvSize pattern_size,
                                   CvPoint2D32f* corners, int* corner_count )
{
    return cvFindChessboardCorners( arr, pattern_size, corners,
                                    corner_count, CV_CALIB_CB_ADAPTIVE_THRESH );
}


/* Calibrates camera using multiple views of calibration pattern */
void cvCalibrateCamera( int image_count, int* _point_counts,
    CvSize image_size, CvPoint2D32f* _image_points, CvPoint3D32f* _object_points,
    float* _distortion_coeffs, float* _camera_matrix, float* _translation_vectors,
    float* _rotation_matrices, int flags )
{
    int i, total = 0;
    CvMat point_counts = cvMat( image_count, 1, CV_32SC1, _point_counts );
    CvMat image_points, object_points;
    CvMat dist_coeffs = cvMat( 4, 1, CV_32FC1, _distortion_coeffs );
    CvMat camera_matrix = cvMat( 3, 3, CV_32FC1, _camera_matrix );
    CvMat rotation_matrices = cvMat( image_count, 9, CV_32FC1, _rotation_matrices );
    CvMat translation_vectors = cvMat( image_count, 3, CV_32FC1, _translation_vectors );

    for( i = 0; i < image_count; i++ )
        total += _point_counts[i];

    image_points = cvMat( total, 1, CV_32FC2, _image_points );
    object_points = cvMat( total, 1, CV_32FC3, _object_points );

    cvCalibrateCamera2( &object_points, &image_points, &point_counts, image_size,
        &camera_matrix, &dist_coeffs, &rotation_matrices, &translation_vectors,
        flags );
}


void cvCalibrateCamera_64d( int image_count, int* _point_counts,
    CvSize image_size, CvPoint2D64f* _image_points, CvPoint3D64f* _object_points,
    double* _distortion_coeffs, double* _camera_matrix, double* _translation_vectors,
    double* _rotation_matrices, int flags )
{
    int i, total = 0;
    CvMat point_counts = cvMat( image_count, 1, CV_32SC1, _point_counts );
    CvMat image_points, object_points;
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion_coeffs );
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, _camera_matrix );
    CvMat rotation_matrices = cvMat( image_count, 9, CV_64FC1, _rotation_matrices );
    CvMat translation_vectors = cvMat( image_count, 3, CV_64FC1, _translation_vectors );

    for( i = 0; i < image_count; i++ )
        total += _point_counts[i];

    image_points = cvMat( total, 1, CV_64FC2, _image_points );
    object_points = cvMat( total, 1, CV_64FC3, _object_points );

    cvCalibrateCamera2( &object_points, &image_points, &point_counts, image_size,
        &camera_matrix, &dist_coeffs, &rotation_matrices, &translation_vectors,
        flags );
}



/* Find 3d position of object given intrinsic camera parameters,
   3d model of the object and projection of the object into view plane */
void cvFindExtrinsicCameraParams( int point_count,
    CvSize, CvPoint2D32f* _image_points,
    CvPoint3D32f* _object_points, float* focal_length,
    CvPoint2D32f principal_point, float* _distortion_coeffs,
    float* _rotation_vector, float* _translation_vector )
{
    CvMat image_points = cvMat( point_count, 1, CV_32FC2, _image_points );
    CvMat object_points = cvMat( point_count, 1, CV_32FC3, _object_points );
    CvMat dist_coeffs = cvMat( 4, 1, CV_32FC1, _distortion_coeffs );
    float a[9];
    CvMat camera_matrix = cvMat( 3, 3, CV_32FC1, a );
    CvMat rotation_vector = cvMat( 1, 1, CV_32FC3, _rotation_vector );
    CvMat translation_vector = cvMat( 1, 1, CV_32FC3, _translation_vector );

    a[0] = focal_length[0]; a[4] = focal_length[1];
    a[2] = principal_point.x; a[5] = principal_point.y;
    a[1] = a[3] = a[6] = a[7] = 0.f;
    a[8] = 1.f;

    cvFindExtrinsicCameraParams2( &object_points, &image_points, &camera_matrix,
        &dist_coeffs, &rotation_vector, &translation_vector, 0 );
}


/* Variant of the previous function that takes double-precision parameters */
void cvFindExtrinsicCameraParams_64d( int point_count,
    CvSize, CvPoint2D64f* _image_points,
    CvPoint3D64f* _object_points, double* focal_length,
    CvPoint2D64f principal_point, double* _distortion_coeffs,
    double* _rotation_vector, double* _translation_vector )
{
    CvMat image_points = cvMat( point_count, 1, CV_64FC2, _image_points );
    CvMat object_points = cvMat( point_count, 1, CV_64FC3, _object_points );
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion_coeffs );
    double a[9];
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, a );
    CvMat rotation_vector = cvMat( 1, 1, CV_64FC3, _rotation_vector );
    CvMat translation_vector = cvMat( 1, 1, CV_64FC3, _translation_vector );

    a[0] = focal_length[0]; a[4] = focal_length[1];
    a[2] = principal_point.x; a[5] = principal_point.y;
    a[1] = a[3] = a[6] = a[7] = 0.;
    a[8] = 1.;

    cvFindExtrinsicCameraParams2( &object_points, &image_points, &camera_matrix,
        &dist_coeffs, &rotation_vector, &translation_vector, 0 );
}

/* Converts rotation_matrix matrix to rotation_matrix vector or vice versa */
void  cvRodrigues( CvMat* rotation_matrix, CvMat* rotation_vector,
                   CvMat* jacobian, int conv_type )
{
    if( conv_type == CV_RODRIGUES_V2M )
        cvRodrigues2( rotation_vector, rotation_matrix, jacobian );
    else
        cvRodrigues2( rotation_matrix, rotation_vector, jacobian );
}


/* Does reprojection of 3d object points to the view plane */
void  cvProjectPoints( int point_count, CvPoint3D64f* _object_points,
    double* _rotation_vector, double*  _translation_vector,
    double* focal_length, CvPoint2D64f principal_point,
    double* _distortion, CvPoint2D64f* _image_points,
    double* _deriv_points_rotation_matrix,
    double* _deriv_points_translation_vect,
    double* _deriv_points_focal,
    double* _deriv_points_principal_point,
    double* _deriv_points_distortion_coeffs )
{
    CvMat object_points = cvMat( point_count, 1, CV_64FC3, _object_points );
    CvMat image_points = cvMat( point_count, 1, CV_64FC2, _image_points );
    CvMat rotation_vector = cvMat( 3, 1, CV_64FC1, _rotation_vector );
    CvMat translation_vector = cvMat( 3, 1, CV_64FC1, _translation_vector );
    double a[9];
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, a );
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion );
    CvMat dpdr = cvMat( 2*point_count, 3, CV_64FC1, _deriv_points_rotation_matrix );
    CvMat dpdt = cvMat( 2*point_count, 3, CV_64FC1, _deriv_points_translation_vect );
    CvMat dpdf = cvMat( 2*point_count, 2, CV_64FC1, _deriv_points_focal );
    CvMat dpdc = cvMat( 2*point_count, 2, CV_64FC1, _deriv_points_principal_point );
    CvMat dpdk = cvMat( 2*point_count, 4, CV_64FC1, _deriv_points_distortion_coeffs );

    a[0] = focal_length[0]; a[4] = focal_length[1];
    a[2] = principal_point.x; a[5] = principal_point.y;
    a[1] = a[3] = a[6] = a[7] = 0.;
    a[8] = 1.;

    cvProjectPoints2( &object_points, &rotation_vector, &translation_vector,
                      &camera_matrix, &dist_coeffs, &image_points,
                      &dpdr, &dpdt, &dpdf, &dpdc, &dpdk, 0 );
}


/* Simpler version of the previous function */
void  cvProjectPointsSimple( int point_count, CvPoint3D64f* _object_points,
    double* _rotation_matrix, double*  _translation_vector,
    double* _camera_matrix, double* _distortion, CvPoint2D64f* _image_points )
{
    CvMat object_points = cvMat( point_count, 1, CV_64FC3, _object_points );
    CvMat image_points = cvMat( point_count, 1, CV_64FC2, _image_points );
    CvMat rotation_matrix = cvMat( 3, 3, CV_64FC1, _rotation_matrix );
    CvMat translation_vector = cvMat( 3, 1, CV_64FC1, _translation_vector );
    CvMat camera_matrix = cvMat( 3, 3, CV_64FC1, _camera_matrix );
    CvMat dist_coeffs = cvMat( 4, 1, CV_64FC1, _distortion );

    cvProjectPoints2( &object_points, &rotation_matrix, &translation_vector,
                      &camera_matrix, &dist_coeffs, &image_points,
                      0, 0, 0, 0, 0, 0 );
}
